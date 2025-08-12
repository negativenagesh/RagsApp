import os
import asyncio
from io import BytesIO
from typing import AsyncGenerator, Dict, Any, List, Tuple, Optional
import yaml
from pathlib import Path
import hashlib
import copy
import xml.etree.ElementTree as ET
import re
import traceback
import json
import shutil
import random
import sys
import filetype

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from openai import AsyncOpenAI
from elasticsearch.helpers import async_bulk, BulkIndexError

from parsers.pdf_parser import PDFParser
from parsers.ocr_parser import OCRParser
from parsers.csv_parser import CSVParser
from parsers.xlsx_parser import XLSXParser
from parsers.img_parser import ImageParser
from parsers.text_parser import TextParser
from parsers.docx_parser import DOCXParser
from parsers.odt_parser import ODTParser
from parsers.doc_parser import DOCParser

from .es_client import ensure_es_index_exists
from .es_client import CHUNKED_PDF_MAPPINGS

SHARED_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "shared" / "prompts"

OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4.1-nano")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", 3072))
CHUNK_SIZE_TOKENS = 20000
CHUNK_OVERLAP_TOKENS = 0
FORWARD_CHUNKS = 3
BACKWARD_CHUNKS = 3
CHARS_PER_TOKEN_ESTIMATE = 4
SUMMARY_MAX_TOKENS = 1024

def tiktoken_len(text: str) -> int:
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text, disallowed_special=())
        return len(tokens)
    except Exception as e:
        print(f"Failed to calculate tiktoken_len: {e}")
        return 0

def generate_doc_id_from_content(content_bytes: bytes) -> str:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content_bytes)
    return sha256_hash.hexdigest()

class IngestionProcessor:
    """
    Orchestrates the file processing pipeline: parsing, chunking, embedding, and indexing.
    """
    def __init__(self, params: Any, config: Any, aclient_openai: Optional[AsyncOpenAI], file_extension: str):
        self.params = params
        self.config = config
        self.aclient_openai = aclient_openai
        self.embedding_dims = OPENAI_EMBEDDING_DIMENSIONS

        if file_extension in [".docx", ".doc", ".odt", ".txt"]:
            chunk_size = 1024
            chunk_overlap = 512
        elif file_extension in [".csv", ".xlsx"]:
            chunk_size = 2000000
            chunk_overlap = 0
        else:
            chunk_size = CHUNK_SIZE_TOKENS
            chunk_overlap = CHUNK_OVERLAP_TOKENS

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=tiktoken_len,
            separators=["\n|", "\n", "|", ". "," ", ""],
        )
        self.enrich_prompt_template = self._load_prompt_template("chunk_enrichment")
        self.graph_extraction_prompt_template = self._load_prompt_template("graph_extraction")
        self.summary_prompt_template = self._load_prompt_template("summary")

    def _load_prompt_template(self, prompt_name: str) -> str:
        try:
            prompt_file_path = SHARED_PROMPTS_DIR / f"{prompt_name}.yaml"
            with open(prompt_file_path, 'r') as f:
                prompt_data = yaml.safe_load(f)
            if prompt_data and prompt_name in prompt_data and "template" in prompt_data[prompt_name]:
                return prompt_data[prompt_name]["template"]
            else:
                raise ValueError(f"Invalid prompt structure for {prompt_name}")
        except Exception as e:
            print(f"Error loading prompt '{prompt_name}': {e}")
            raise
    
    async def _call_openai_api(
        self,
        model_name: str,
        payload_messages: List[Dict[str, Any]],
        is_vision_call: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> str:
        if not self.aclient_openai:
            print("OpenAI client not configured. Cannot make API call.")
            return ""
        max_retries = 5
        base_delay_seconds = 3
        for attempt in range(max_retries):
            try:
                response = await self.aclient_openai.chat.completions.create(
                    model=model_name,
                    messages=payload_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                print(f"OpenAI API call failed (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt + 1 < max_retries:
                delay = base_delay_seconds * (2 ** attempt)
                await asyncio.sleep(delay)
        print("Max retries reached for OpenAI API call. Returning empty string.")
        return ""

    async def _generate_document_summary(self, full_document_text: str) -> str:
        if not self.aclient_openai or not self.summary_prompt_template:
            return "Summary generation skipped due to missing configuration."
        if not full_document_text.strip():
            return "Document is empty, no summary generated."
        formatted_prompt = self.summary_prompt_template.format(document=full_document_text)
        messages = [{"role": "user", "content": formatted_prompt}]
        summary_text_content = await self._call_openai_api(
            model_name=OPENAI_SUMMARY_MODEL,
            payload_messages=messages,
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=0.3
        )
        return summary_text_content.strip() if summary_text_content else "Summary generation resulted in empty content."

    def _clean_xml_string(self, xml_string: str) -> str:
        """Cleans the XML string from common LLM artifacts and prepares it for parsing."""
        if not isinstance(xml_string, str):
            print(f"XML input is not a string, type: {type(xml_string)}. Returning empty string.")
            return ""

        cleaned_xml = xml_string.strip()

        if cleaned_xml.startswith("```xml"):
            cleaned_xml = cleaned_xml[len("```xml"):].strip()
        elif cleaned_xml.startswith("```"):
            cleaned_xml = cleaned_xml[len("```"):].strip()
        
        if cleaned_xml.endswith("```"):
            cleaned_xml = cleaned_xml[:-len("```")].strip()

        if cleaned_xml.startswith("<?xml"):
            end_decl = cleaned_xml.find("?>")
            if end_decl != -1:
                cleaned_xml = cleaned_xml[end_decl + 2:].lstrip()
        
        first_angle_bracket = cleaned_xml.find("<")
        last_angle_bracket = cleaned_xml.rfind(">")

        if first_angle_bracket != -1 and last_angle_bracket != -1 and last_angle_bracket > first_angle_bracket:
            cleaned_xml = cleaned_xml[first_angle_bracket : last_angle_bracket + 1]
        elif first_angle_bracket == -1 :
            print(f"No XML tags found in the string after initial cleaning. Original: {xml_string[:200]}")
            return ""


        cleaned_xml = re.sub(r'&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', cleaned_xml)

        common_prefixes = ["Sure, here is the XML:", "Here's the XML output:", "Okay, here's the XML:"]
        for prefix in common_prefixes:
            if cleaned_xml.lower().startswith(prefix.lower()):
                cleaned_xml = cleaned_xml[len(prefix):].lstrip()
                break
        
        cleaned_xml = re.sub(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]', '', cleaned_xml)

        return cleaned_xml

    def _parse_graph_xml(self, xml_string: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        entities = []
        relationships = []
        
        cleaned_xml = self._clean_xml_string(xml_string)

        if not cleaned_xml:
            print("XML string is empty after cleaning. Cannot parse.")
            return entities, relationships

        has_known_root = False
        known_roots = ["<graph>", "<root>", "<entities>", "<response>", "<data>"]
        for root_tag_start in known_roots:
            if cleaned_xml.startswith(root_tag_start):
                root_tag_name = root_tag_start[1:-1]
                if cleaned_xml.endswith(f"</{root_tag_name}>"):
                    has_known_root = True
                    break
        
        string_to_parse = cleaned_xml
        if not has_known_root:
            if (cleaned_xml.startswith("<entity") or cleaned_xml.startswith("<relationship")) and \
               (cleaned_xml.endswith("</entity>") or cleaned_xml.endswith("</relationship>")):
                string_to_parse = f"<root_wrapper>{cleaned_xml}</root_wrapper>"
                print("Wrapping multiple top-level entity/relationship tags with <root_wrapper>.")
            elif not (cleaned_xml.count("<") > 1 and cleaned_xml.count(">") > 1 and cleaned_xml.find("</") > 0 and cleaned_xml.startswith("<") and cleaned_xml.endswith(">") and cleaned_xml[1:cleaned_xml.find(">") if cleaned_xml.find(">") > 1 else 0].strip() == cleaned_xml[cleaned_xml.rfind("</")+2:-1].strip() ):
                 string_to_parse = f"<root_wrapper>{cleaned_xml}</root_wrapper>"
                 print("Wrapping content with <root_wrapper> as it doesn't appear to have a single root or matching end tag.")
        
        try:
            root = ET.fromstring(string_to_parse)
            
            for entity_elem in root.findall(".//entity"): 
                name_val = entity_elem.get("name")
                if not name_val:
                    name_elem = entity_elem.find("name")
                    name_val = name_elem.text.strip() if name_elem is not None and name_elem.text else None
                
                ent_type_elem = entity_elem.find("type")
                ent_desc_elem = entity_elem.find("description")
                
                ent_type = ent_type_elem.text.strip() if ent_type_elem is not None and ent_type_elem.text else "Unknown"
                ent_desc = ent_desc_elem.text.strip() if ent_desc_elem is not None and ent_desc_elem.text else ""
                
                if name_val:
                    entities.append({"name": name_val.strip(), "type": ent_type, "description": ent_desc})

            for rel_elem in root.findall(".//relationship"): 
                source_elem = rel_elem.find("source")
                target_elem = rel_elem.find("target")
                rel_type_elem = rel_elem.find("type")
                rel_desc_elem = rel_elem.find("description")
                rel_weight_elem = rel_elem.find("weight")
                
                source = source_elem.text.strip() if source_elem is not None and source_elem.text else None
                target = target_elem.text.strip() if target_elem is not None and target_elem.text else None
                rel_type = rel_type_elem.text.strip() if rel_type_elem is not None and rel_type_elem.text else "RELATED_TO"
                rel_desc = rel_desc_elem.text.strip() if rel_desc_elem is not None and rel_desc_elem.text else ""
                weight = None
                if rel_weight_elem is not None and rel_weight_elem.text:
                    try:
                        weight = float(rel_weight_elem.text.strip())
                    except ValueError:
                        print(f"Could not parse relationship weight '{rel_weight_elem.text}' as float.")
                
                if source and target:
                    relationships.append({
                        "source_entity": source, "target_entity": target, "relation": rel_type,
                        "relationship_description": rel_desc, "relationship_weight": weight
                    })
            
            print(f"Successfully parsed {len(entities)} entities and {len(relationships)} relationships using ET.fromstring.")

        except ET.ParseError as e:
            err_line, err_col = e.position if hasattr(e, 'position') else (-1, -1)
            log_message = (
                f"XML parsing error with ET.fromstring: {e}\n"
                f"Error at line {err_line}, column {err_col} (approximate). Trying regex-based extraction as fallback.\n"
                f"Cleaned XML snippet attempted (first 1000 chars):\n{string_to_parse[:1000]}"
            )
            print(log_message)
            
            entities = [] 
            relationships = [] 

            entity_pattern_attr = r'<entity\s+name\s*=\s*"([^"]*)"\s*>\s*(?:<type>([^<]*)</type>)?\s*(?:<description>([^<]*)</description>)?\s*</entity>'
            entity_pattern_tag = r'<entity>\s*<name>([^<]+)</name>\s*(?:<type>([^<]*)</type>)?\s*(?:<description>([^<]*)</description>)?\s*</entity>'


            for pattern in [entity_pattern_attr, entity_pattern_tag]:
                for match in re.finditer(pattern, string_to_parse): 
                    name, entity_type, description = match.groups()
                    if name:
                        entities.append({
                            "name": name.strip(),
                            "type": entity_type.strip() if entity_type and entity_type.strip() else "Unknown",
                            "description": description.strip() if description and description.strip() else ""
                        })
            
            rel_pattern = r'<relationship>\s*(?:<source>([^<]+)</source>)?\s*(?:<target>([^<]+)</target>)?\s*(?:<type>([^<]*)</type>)?\s*(?:<description>([^<]*)</description>)?\s*(?:<weight>([^<]*)</weight>)?\s*</relationship>'
            for match in re.finditer(rel_pattern, string_to_parse): 
                source, target, rel_type, description, weight_str = match.groups()
                if source and target:
                    weight = None
                    if weight_str and weight_str.strip():
                        try:
                            weight = float(weight_str.strip())
                        except ValueError:
                            print(f"Regex fallback: Could not parse weight '{weight_str}' for relationship.")
                    
                    relationships.append({
                        "source_entity": source.strip(),
                        "target_entity": target.strip(),
                        "relation": rel_type.strip() if rel_type and rel_type.strip() else "RELATED_TO",
                        "relationship_description": description.strip() if description and description.strip() else "",
                        "relationship_weight": weight
                    })
            if entities or relationships:
                 print(f"Regex fallback extracted {len(entities)} entities and {len(relationships)} relationships.")
            else:
                 print("Regex fallback also failed to extract any entities or relationships.")
        
        except Exception as final_e: 
            print(f"An unexpected error occurred during XML parsing (after ET.ParseError or during regex): {final_e}\n"
                        f"Original XML content from LLM (first 500 chars):\n{xml_string[:500]}\n"
                        f"Cleaned XML attempted for parsing (first 500 chars):\n{string_to_parse[:500]}")
        
        return entities, relationships

    async def _extract_knowledge_graph(
        self, chunk_text: str, document_summary: str 
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        
        if not self.aclient_openai or not self.graph_extraction_prompt_template:
            print("OpenAI client or graph extraction prompt not available. Skipping graph extraction.")
            return [], []
        
        formatted_prompt = self.graph_extraction_prompt_template.format(
            document_summary=document_summary, 
            input=chunk_text, 
            entity_types=str([]), 
            relation_types=str([]) 
        )
        print(f"Formatted prompt for graph extraction (chunk-level, to {OPENAI_SUMMARY_MODEL}): First 200 chars: {formatted_prompt[:200]}...")
        
        messages=[
            {"role": "system", "content": "You are an expert assistant that extracts entities and relationships from text and formats them as XML according to the provided schema. Ensure all tags are correctly opened and closed. Use <entity name=\"...\"><type>...</type><description>...</description></entity> and <relationship><source>...</source><target>...</target><type>...</type><description>...</description><weight>...</weight></relationship> format. Wrap multiple entities and relationships in a single <root> or <graph> tag."},
            {"role": "user", "content": formatted_prompt}
        ]

        xml_response_content = await self._call_openai_api(
            model_name=OPENAI_SUMMARY_MODEL,
            payload_messages=messages,
            max_tokens=4000,
            temperature=0.1
        )
        
        if not xml_response_content:
            print("LLM returned empty content for graph extraction.")
            return [], []
        
        print(f"Raw XML response from LLM for chunk-level graph extraction (first 500 chars):\n{xml_response_content[:500]}")
        return self._parse_graph_xml(xml_response_content)

    async def _enrich_chunk_content(
        self, chunk_text: str, document_summary: str, 
        preceding_chunks_texts: List[str], succeeding_chunks_texts: List[str],
    ) -> str:

        if not self.aclient_openai or not self.enrich_prompt_template:
            print("OpenAI client or enrichment prompt not available. Skipping enrichment.")
            return chunk_text 
        
        preceding_context = "\n---\n".join(preceding_chunks_texts)
        succeeding_context = "\n---\n".join(succeeding_chunks_texts)
        max_output_chars = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN_ESTIMATE
        
        formatted_prompt = self.enrich_prompt_template.format(
            document_summary=document_summary, preceding_chunks=preceding_context,
            succeeding_chunks=succeeding_context, chunk=chunk_text, chunk_size=max_output_chars 
        )
        print(f"Formatted prompt for enrichment (to be sent to {OPENAI_CHAT_MODEL}): ...")
        
        messages = [
            {"role": "system", "content": "You are an expert assistant that refines and enriches text chunks according to specific guidelines."},
            {"role": "user", "content": formatted_prompt}
        ]

        enriched_text_content = await self._call_openai_api(
            model_name=OPENAI_CHAT_MODEL,
            payload_messages=messages,
            max_tokens=min(CHUNK_SIZE_TOKENS + CHUNK_OVERLAP_TOKENS, 4000),
            temperature=0.3
        )

        if not enriched_text_content:
            print("LLM returned empty content for chunk enrichment. Using original chunk.")
            return chunk_text
        
        enriched_text = enriched_text_content.strip()
        print(f"Chunk enriched. Original length: {len(chunk_text)}, Enriched length: {len(enriched_text)}")
        return enriched_text
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        if not self.aclient_openai:
            return [[] for _ in texts]
        all_embeddings = []
        openai_batch_size = 2048
        try:
            for i in range(0, len(texts), openai_batch_size):
                batch_texts = texts[i:i + openai_batch_size]
                processed_batch_texts = [text if text.strip() else " " for text in batch_texts]
                response = await self.aclient_openai.embeddings.create(
                    input=processed_batch_texts,
                    model=OPENAI_EMBEDDING_MODEL,
                    dimensions=OPENAI_EMBEDDING_DIMENSIONS
                )
                all_embeddings.extend([item.embedding for item in response.data])
            return all_embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [[] for _ in texts]

    async def _generate_all_raw_chunks_from_doc(
        self,
        doc_text: str,
        file_name: str,
        doc_id: str
    ) -> List[Dict[str, Any]]:
        all_raw_chunks_with_meta: List[Dict[str, Any]] = []
        if not doc_text or not doc_text.strip():
            print(f"Skipping empty document {file_name} for raw chunk generation.")
            return []

        raw_chunks = self.text_splitter.split_text(doc_text)
        print(f'***raw chunks{raw_chunks}')
        for chunk_idx, raw_chunk_text in enumerate(raw_chunks):
            doc_file_name = self.params.get('file_name', file_name)
            print(f"RAW CHUNK (File: {file_name}, Original File Name: {doc_file_name}, Idx: {chunk_idx}, Len {len(raw_chunk_text)}): '''{raw_chunk_text[:100].strip()}...'''")
            all_raw_chunks_with_meta.append({
                "text": raw_chunk_text,
                "page_num": 1,
                "chunk_idx_on_page": chunk_idx,
                "file_name": doc_file_name,
                "doc_id": doc_id
            })
        print(f"Generated {len(all_raw_chunks_with_meta)} raw chunks from {file_name}.")
        return all_raw_chunks_with_meta

    async def _process_individual_chunk_pipeline(
        self, 
        raw_chunk_info: Dict[str, Any], 
        user_provided_doc_summary: str, 
        llm_generated_doc_summary: str, 
        all_raw_texts: List[str], 
        global_idx: int, 
        file_name: str, 
        doc_id: str,
        params: Any
    ) -> Dict[str, Any] | None: 
        chunk_text = raw_chunk_info["text"]
        page_num = raw_chunk_info["page_num"]
        chunk_idx_on_page = raw_chunk_info["chunk_idx_on_page"]
        
        print(f"Starting pipeline for chunk: File {file_name}, Page {page_num}, Index {chunk_idx_on_page}")

        file_extension = os.path.splitext(file_name)[1].lower()
        is_tabular_file = file_extension in ['.csv', '.xlsx']

        if is_tabular_file:
          print(f"Tabular file ({file_extension}) detected. Skipping KG extraction and enrichment.")
          enriched_text = chunk_text
          chunk_entities, chunk_relationships = [], []
        
        else:
            preceding_indices = range(max(0, global_idx - BACKWARD_CHUNKS), global_idx)
            succeeding_indices = range(global_idx + 1, min(len(all_raw_texts), global_idx + 1 + FORWARD_CHUNKS))
            preceding_texts = [all_raw_texts[i] for i in preceding_indices]
            succeeding_texts = [all_raw_texts[i] for i in succeeding_indices]

            contextual_summary = llm_generated_doc_summary
            if not llm_generated_doc_summary or \
            llm_generated_doc_summary == "Document is empty, no summary generated." or \
            llm_generated_doc_summary.startswith("Error during summary generation") or \
            llm_generated_doc_summary == "Summary generation skipped due to missing configuration.":
                contextual_summary = user_provided_doc_summary

            kg_task = asyncio.create_task(
                self._extract_knowledge_graph(chunk_text, contextual_summary)
            )
            enrich_task = asyncio.create_task(
                self._enrich_chunk_content(
                    chunk_text, contextual_summary, preceding_texts, succeeding_texts,
                )
            )

            results = await asyncio.gather(kg_task, enrich_task, return_exceptions=True)
            
            kg_result_or_exc = results[0]
            enrich_result_or_exc = results[1]

            chunk_entities, chunk_relationships = [], []
            if isinstance(kg_result_or_exc, Exception):
                print(f"KG extraction failed for chunk (Page {page_num}, Index {chunk_idx_on_page}) for '{file_name}': {kg_result_or_exc}")
            elif kg_result_or_exc: 
                chunk_entities, chunk_relationships = kg_result_or_exc
                print(f"KG extracted for chunk (Page {page_num}, Index {chunk_idx_on_page}): {len(chunk_entities)} entities, {len(chunk_relationships)} relationships.")
                entity_descriptions_to_embed = [
                    entity['description'] for entity in chunk_entities if entity.get('description', '').strip()
                ]
                entity_indices_with_description = [
                    i for i, entity in enumerate(chunk_entities) if entity.get('description', '').strip()
                ]
                
                if entity_descriptions_to_embed:
                    print(f"Generating embeddings for {len(entity_descriptions_to_embed)} entity descriptions.")
                    try:
                        description_embeddings = await self._generate_embeddings(entity_descriptions_to_embed)
                        
                        if description_embeddings and len(description_embeddings) == len(entity_indices_with_description):
                            for original_index, embedding in zip(entity_indices_with_description, description_embeddings):
                                if embedding:
                                    chunk_entities[original_index]['description_embedding'] = embedding
                            print(f"Successfully assigned {len(description_embeddings)} embeddings to entity descriptions.")
                        else:
                            print(
                                f"Mismatch between number of descriptions ({len(entity_indices_with_description)}) and "
                                f"generated embeddings ({len(description_embeddings) if description_embeddings else 0}). Skipping assignment."
                            )
                    except Exception as e:
                        print(f"Failed to generate or assign embeddings for entity descriptions: {e}")
                
            enriched_text: str
            if isinstance(enrich_result_or_exc, Exception):
                print(f"Enrichment failed for chunk (Page {page_num}, Index {chunk_idx_on_page}) for '{file_name}': {enrich_result_or_exc}. Using original text.")
                enriched_text = chunk_text 
            else:
                enriched_text = enrich_result_or_exc
                print(f"Enrichment successful for chunk (Page {page_num}, Index {chunk_idx_on_page}).")
        
        embedding_list = await self._generate_embeddings([enriched_text]) 
        
        embedding_vector = []
        if embedding_list and embedding_list[0]: 
            embedding_vector = embedding_list[0]
        
        if not embedding_vector: 
            print(f"Skipping chunk from page {page_num}, index {chunk_idx_on_page} for '{file_name}' due to missing embedding.")
            return None

        es_doc_id = f"{doc_id}_p{page_num}_c{chunk_idx_on_page}"
        doc_file_name = self.params.get('file_name', file_name)
        print(f"Original file name: {doc_file_name}")
        metadata_payload = {
            "file_name": doc_file_name, 
            "doc_id": doc_id,
            "page_number": page_num,
            "chunk_index_in_page": chunk_idx_on_page,
            "document_summary": llm_generated_doc_summary,
            "entities": chunk_entities, 
            "relationships": chunk_relationships 
        }
        
        index_name = params.get('index_name')
        print('index name:--', index_name)
        action = {
            "_index": index_name, 
            "_id": es_doc_id,
            "_source": {
                "chunk_text": enriched_text, 
                "embedding": embedding_vector, 
                "metadata": metadata_payload
            }
        }
        print(f"Pipeline complete for chunk (Page {page_num}, Index {chunk_idx_on_page}). ES action prepared.")
        return action

    async def process_pdf(
        self, data: bytes, file_name: str, doc_id: str, user_provided_document_summary: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        print(f"Processing PDF: {file_name} (Doc ID: {doc_id})")
        
        use_ocr = self.params.get('is_ocr_pdf', False)

        try:
            pages_with_tables = 0
            with pdfplumber.open(BytesIO(data)) as pdf:
                if not pdf.pages:
                    print("PDF has no pages. Aborting.")
                    return
                
                for page in pdf.pages:
                    if page.extract_tables():
                        pages_with_tables += 1

                if pages_with_tables > 1:
                    print(f"PDF contains tables on {pages_with_tables} pages. Adjusting chunk size to 2048.")
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2048,
                    chunk_overlap=1024,
                    length_function=tiktoken_len,
                    separators=["\n|", "\n", "|", ". "," ", ""],
                )
        except Exception as e:
            print(f"Could not pre-check PDF with pdfplumber, defaulting to OCR. Error: {e}")
            use_ocr = True

        full_document_text = ""
        if use_ocr:
            print("Processing with OCR parser.")
            ocr_parser = OCRParser()
            ocr_texts = [page_text async for page_text in ocr_parser.ingest(data)]
            full_document_text = " ".join(ocr_texts)
        else:
            print("Processing with standard PDF parser.")
            parser = PDFParser(self.aclient_openai,self)
            all_content_parts = [part async for part in parser.ingest(data)]
            full_document_text = " ".join(all_content_parts)

        if not full_document_text.strip():
            print(f"No text or tables extracted from '{file_name}'. Aborting processing.")
            yield {"status": "error", "message": f"No text or tables extracted from '{file_name}'."}
            return

        llm_generated_doc_summary = await self._generate_document_summary(full_document_text)
        
        all_raw_chunks_with_meta = await self._generate_all_raw_chunks_from_doc(
            full_document_text, file_name, doc_id
        )

        if not all_raw_chunks_with_meta:
            print(f"No raw chunks were generated from '{file_name}'. Aborting further processing.")
            yield {"status": "error", "message": f"No raw chunks generated from '{file_name}'."}
            return
        
        all_raw_texts = [chunk["text"] for chunk in all_raw_chunks_with_meta]
        
        print(f"Starting concurrent processing for {len(all_raw_chunks_with_meta)} raw chunks from '{file_name}'.")

        processing_tasks = []
        for i, raw_chunk_info_item in enumerate(all_raw_chunks_with_meta):
            task = asyncio.create_task(
                self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info_item,
                    user_provided_doc_summary=user_provided_document_summary, 
                    llm_generated_doc_summary=llm_generated_doc_summary, 
                    all_raw_texts=all_raw_texts,
                    global_idx=i, file_name=file_name, doc_id=doc_id,
                    params=self.params        
                )
            )
            processing_tasks.append(task)
        
        num_successfully_processed = 0
        for future in asyncio.as_completed(processing_tasks):
            try:
                es_action = await future 
                if es_action: 
                    yield es_action
                    num_successfully_processed += 1
            except Exception as e:
                print(f"Error processing a chunk future for '{file_name}': {e}")
        
        print(f"Finished processing for '{file_name}'. Successfully processed and yielded {num_successfully_processed}/{len(all_raw_chunks_with_meta)} chunks.")

    async def process_doc(
        self, data: bytes, file_name: str, doc_id: str, user_provided_document_summary: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        print(f"Processing DOC: {file_name} (Doc ID: {doc_id})")
    
        parser = DOCParser(self.aclient_openai,self.Server_type,self)
        full_document_text = ""
        try:
            # The DOCParser's ingest method is an async generator. We consume it fully
            # to get all paragraphs and then join them to form the complete document text.
            all_parts = [p async for p in parser.ingest(data)]
            full_document_text = " ".join(all_parts)
            print(f"Successfully parsed .doc file '{file_name}'. Total length: {len(full_document_text)} characters.")
        except Exception as e:
            print(f"Failed to parse .doc file '{file_name}': {e}")
            return

        if not full_document_text.strip():
            print(f"No text extracted from DOC file '{file_name}'. Aborting processing.")
            return

        llm_generated_doc_summary = await self._generate_document_summary(full_document_text)
        
        all_raw_chunks_with_meta = await self._generate_all_raw_chunks_from_doc(
            full_document_text, file_name, doc_id
        )

        if not all_raw_chunks_with_meta:
            print(f"No raw chunks were generated from DOC file '{file_name}'. Aborting further processing.")
            return

        all_raw_texts = [chunk["text"] for chunk in all_raw_chunks_with_meta]
        
        print(f"Starting concurrent processing for {len(all_raw_chunks_with_meta)} raw chunks from DOC file '{file_name}'.")

        processing_tasks = []
        for i, raw_chunk_info_item in enumerate(all_raw_chunks_with_meta):
            task = asyncio.create_task(
                self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info_item,
                    user_provided_doc_summary=user_provided_document_summary, 
                    llm_generated_doc_summary=llm_generated_doc_summary, 
                    all_raw_texts=all_raw_texts,
                    global_idx=i, file_name=file_name, doc_id=doc_id,
                    params=self.params        
                )
            )
            processing_tasks.append(task)
        
        num_successfully_processed = 0
        for future in asyncio.as_completed(processing_tasks):
            try:
                es_action = await future 
                if es_action: 
                    yield es_action
                    num_successfully_processed += 1
            except Exception as e:
                print(f"Error processing a chunk future for DOC file '{file_name}': {e}")
        
        print(f"Finished processing for DOC file '{file_name}'. Successfully processed and yielded {num_successfully_processed}/{len(all_raw_chunks_with_meta)} chunks.")
     
    async def process_docx(
        self, data: bytes, file_name: str, doc_id: str, user_provided_document_summary: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        print(f"Processing DOCX: {file_name} (Doc ID: {doc_id})")
    
        parser = DOCXParser(self.aclient_openai,self.Server_type,self)
        
        full_document_text = ""
        try:
            all_parts = [p async for p in parser.ingest(data)]
            full_document_text = " ".join(all_parts)
            print(f"Successfully parsed .docx file '{file_name}'. Total length: {len(full_document_text)} characters.")
        except Exception as e:
            print(f"Failed to parse .docx file '{file_name}': {e}")
            return

        if not full_document_text.strip():
            print(f"No text extracted from DOCX file '{file_name}'. Aborting processing.")
            return

        llm_generated_doc_summary = await self._generate_document_summary(full_document_text)
        
        all_raw_chunks_with_meta = await self._generate_all_raw_chunks_from_doc(
            full_document_text, file_name, doc_id
        )

        if not all_raw_chunks_with_meta:
            print(f"No raw chunks were generated from DOCX file '{file_name}'. Aborting further processing.")
            return

        all_raw_texts = [chunk["text"] for chunk in all_raw_chunks_with_meta]
        
        print(f"Starting concurrent processing for {len(all_raw_chunks_with_meta)} raw chunks from DOCX file '{file_name}'.")

        processing_tasks = []
        for i, raw_chunk_info_item in enumerate(all_raw_chunks_with_meta):
            task = asyncio.create_task(
                self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info_item,
                    user_provided_doc_summary=user_provided_document_summary, 
                    llm_generated_doc_summary=llm_generated_doc_summary, 
                    all_raw_texts=all_raw_texts,
                    global_idx=i, file_name=file_name, doc_id=doc_id,
                    params=self.params        
                )
            )
            processing_tasks.append(task)
        
        num_successfully_processed = 0
        for future in asyncio.as_completed(processing_tasks):
            try:
                es_action = await future 
                if es_action: 
                    yield es_action
                    num_successfully_processed += 1
            except Exception as e:
                print(f"Error processing a chunk future for DOCX file '{file_name}': {e}")
        
        print(f"Finished processing for DOCX file '{file_name}'. Successfully processed and yielded {num_successfully_processed}/{len(all_raw_chunks_with_meta)} chunks.")

    async def process_odt(
        self, data: bytes, file_name: str, doc_id: str, user_provided_document_summary: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        print(f"Processing ODT: {file_name} (Doc ID: {doc_id})")

        parser = ODTParser(self.aclient_openai, self.Server_type, self)

        full_document_text = ""
        try:
            all_parts = [p async for p in parser.ingest(data)]
            full_document_text = " ".join(all_parts)
            print(f"Successfully parsed .odt file '{file_name}'. Total length: {len(full_document_text)} characters.")
        except Exception as e:
            print(f"Failed to parse .odt file '{file_name}': {e}")
            return

        if not full_document_text.strip():
            print(f"No content extracted from ODT file '{file_name}'. Aborting processing.")
            return

        llm_generated_doc_summary = await self._generate_document_summary(full_document_text)

        all_raw_chunks_with_meta = await self._generate_all_raw_chunks_from_doc(
            full_document_text, file_name, doc_id
        )

        if not all_raw_chunks_with_meta:
            return

        all_raw_texts = [chunk["text"] for chunk in all_raw_chunks_with_meta]

        processing_tasks = []
        for i, raw_chunk_info_item in enumerate(all_raw_chunks_with_meta):
            task = asyncio.create_task(
                self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info_item,
                    user_provided_doc_summary=user_provided_document_summary,
                    llm_generated_doc_summary=llm_generated_doc_summary,
                    all_raw_texts=all_raw_texts,
                    global_idx=i, file_name=file_name, doc_id=doc_id,
                    params=self.params
                )
            )
            processing_tasks.append(task)

        for future in asyncio.as_completed(processing_tasks):
            es_action = await future
            if es_action:
                yield es_action
    
    async def process_txt(
        self, data: bytes, file_name: str, doc_id: str, user_provided_document_summary: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        print(f"Processing TXT: {file_name} (Doc ID: {doc_id})")
        try:
            txt_parser = TextParser()
            full_document_text_parts = [text_part async for text_part in txt_parser.ingest(data)]
            full_document_text = " ".join(filter(None, full_document_text_parts))

            if not full_document_text.strip():
                print(f"No text extracted from '{file_name}'. Aborting processing.")
                return

            llm_generated_doc_summary = await self._generate_document_summary(full_document_text)

            all_raw_chunks_with_meta = await self._generate_all_raw_chunks_from_doc(
                full_document_text, file_name, doc_id
            )

            if not all_raw_chunks_with_meta:
                print(f"No raw chunks were generated from '{file_name}'. Aborting.")
                return

            all_raw_texts = [chunk["text"] for chunk in all_raw_chunks_with_meta]
            print(f"Starting concurrent processing for {len(all_raw_chunks_with_meta)} raw chunks from '{file_name}'.")

            processing_tasks = []
            for i, raw_chunk_info_item in enumerate(all_raw_chunks_with_meta):
                task = asyncio.create_task(
                    self._process_individual_chunk_pipeline(
                        raw_chunk_info=raw_chunk_info_item,
                        user_provided_doc_summary=user_provided_document_summary,
                        llm_generated_doc_summary=llm_generated_doc_summary,
                        all_raw_texts=all_raw_texts,
                        global_idx=i,
                        file_name=file_name,
                        doc_id=doc_id,
                        params=self.params
                    )
                )
                processing_tasks.append(task)

            num_successfully_processed = 0
            for future in asyncio.as_completed(processing_tasks):
                try:
                    es_action = await future
                    if es_action:
                        yield es_action
                        num_successfully_processed += 1
                except Exception as e:
                    print(f"Error processing a chunk future for '{file_name}': {e}")

            print(f"Finished processing for '{file_name}'. Successfully processed and yielded {num_successfully_processed}/{len(all_raw_chunks_with_meta)} chunks.")

        except Exception as e:
            print(f"Major failure in process_txt for '{file_name}': {e}")
    
    async def process_csv_semantic_chunking(
        self, data: bytes, file_name: str, doc_id: str, user_provided_document_summary: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Enhanced CSV processing with semantic chunking that preserves context and structure.
        """
        print(f"Processing CSV with semantic chunking: {file_name} (Doc ID: {doc_id})")
    
        try:
            csv_parser = CSVParser()
            rows = [row_text async for row_text in csv_parser.ingest(data)]
        
            if not rows:
                print(f"No data extracted from CSV '{file_name}'. Aborting processing.")
                return
        
            # Parse CSV structure
            header_row = rows[0] if rows else ""
            data_rows = rows[1:] if len(rows) > 1 else []
        
            # Create semantic chunks
            chunks = await self._create_semantic_csv_chunks(
                header_row, data_rows, file_name
            )
        
            if not chunks:
                print(f"No chunks generated from CSV '{file_name}'. Aborting.")
                return
        
            # Process each chunk through the pipeline
            for chunk_idx, chunk_data in enumerate(chunks):
                chunk_text = chunk_data["text"]
                chunk_context = chunk_data.get("context", "")
            
                # Combine chunk with context for enrichment
                full_chunk_text = f"{chunk_context}\n\n{chunk_text}" if chunk_context else chunk_text
            
                # Generate document summary
                if chunk_idx == 0:
                    full_document_text = "\n".join(rows)
                    llm_generated_doc_summary = await self._generate_document_summary(full_document_text)
            
                # Process through individual chunk pipeline
                raw_chunk_info = {
                    "text": full_chunk_text,
                    "page_num": 1,
                    "chunk_idx_on_page": chunk_idx,
                    "file_name": file_name,
                    "doc_id": doc_id
                }
            
                es_action = await self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info,
                    user_provided_doc_summary=user_provided_document_summary,
                    llm_generated_doc_summary=llm_generated_doc_summary,
                    all_raw_texts=[chunk["text"] for chunk in chunks],
                    global_idx=chunk_idx,
                    file_name=file_name,
                    doc_id=doc_id,
                    params=self.params
                )
            
                if es_action:
                    yield es_action
                
        except Exception as e:
            print(f"Error in semantic CSV processing for '{file_name}': {e}")

    async def _create_semantic_csv_chunks(
        self, header_row: str, data_rows: List[str], file_name: str
    ) -> List[Dict[str, Any]]:
        """
        Creates semantically meaningful chunks from CSV data.
        """
        chunks = []
    
        #Strategy 1: Header + batch of rows
        rows_per_chunk = self._calculate_optimal_rows_per_chunk(header_row, data_rows)
    
        for i in range(0, len(data_rows), rows_per_chunk):
            batch_rows = data_rows[i:i + rows_per_chunk]
        
            # Create chunk with header context
            chunk_text = f"CSV Structure:\n{header_row}\n\nData:\n" + "\n".join(batch_rows)
        
            # Add metadata context
            context = f"This is part {(i // rows_per_chunk) + 1} of CSV file '{file_name}' containing rows {i+1} to {min(i + rows_per_chunk, len(data_rows))}."
        
            chunks.append({
                "text": chunk_text,
                "context": context,
                "start_row": i + 1,
                "end_row": min(i + rows_per_chunk, len(data_rows)),
                "total_rows": len(batch_rows)
            })
    
        return chunks
    
    def _calculate_optimal_rows_per_chunk(self, header_row: str, data_rows: List[str]) -> int:
        """
        Calculate optimal number of rows per chunk based on token limits.
        """
        if not data_rows:
            return 1
    
        # Estimate tokens for header and average row
        header_tokens = tiktoken_len(header_row)
        avg_row_tokens = tiktoken_len(data_rows[0]) if data_rows else 1
    
        # Reserve space for context and formatting
        available_tokens = CHUNK_SIZE_TOKENS - header_tokens - 100  # 100 for context/formatting
    
        # Calculate how many rows fit
        rows_per_chunk = max(1, available_tokens // avg_row_tokens)
    
        # Cap at reasonable limits
        return min(rows_per_chunk, 1)  #Max 50 rows per chunk for readability

    async def process_xlsx_semantic_chunking(
        self, data: bytes, file_name: str, doc_id: str, user_provided_document_summary: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Processes XLSX files using a row-by-row, text-based chunking approach.
        """
        print(f"Processing XLSX with text-based chunking: {file_name} (Doc ID: {doc_id})")
        try:
            xlsx_parser = XLSXParser()
            # The parser now yields a list of lists of strings
            rows = [row_list async for row_list in xlsx_parser.ingest(data)]

            if not rows:
                print(f"No data extracted from CSV '{file_name}'. Aborting processing.")
                yield {"status": "error", "message": f"No data extracted from CSV '{file_name}'."}
                return

            # Create chunks with the corrected mapping logic
            chunks = await self._create_semantic_xlsx_chunks(rows, file_name)

            if not chunks:
                print(f"No chunks generated from CSV '{file_name}'. Aborting.")
                yield {"status": "error", "message": f"No chunks generated from CSV '{file_name}'."}
                return

            # CHANGED: Properly join the list of lists into a single string for the summary
            full_document_text = "\n".join([", ".join(row) for row in rows])
            llm_generated_doc_summary = await self._generate_document_summary(full_document_text)

            # Process each generated chunk through the standard pipeline
            for chunk_idx, chunk_data in enumerate(chunks):
                chunk_text = chunk_data["text"]
                raw_chunk_info = {
                    "text": chunk_text,
                    "page_num": 1,  # XLSX is treated as a single "page"
                    "chunk_idx_on_page": chunk_idx,
                    "file_name": file_name,
                    "doc_id": doc_id
                }
                es_action = await self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info,
                    user_provided_doc_summary=user_provided_document_summary,
                    llm_generated_doc_summary=llm_generated_doc_summary,
                    all_raw_texts=[c["text"] for c in chunks],
                    global_idx=chunk_idx,
                    file_name=file_name,
                    doc_id=doc_id,
                    params=self.params
                )
                if es_action:
                    yield es_action
        except Exception as e:
            print(f"Error in semantic XLSX processing for '{file_name}': {e}")
            traceback.print_exc()
            
    async def _create_semantic_xlsx_chunks(
        self, data_rows: List[List[str]], file_name: str
    ) -> List[Dict[str, Any]]:
        """
        Creates semantically meaningful chunks from XLSX data. Processes a batch of rows
        at a time, handling blank values correctly. If a batch's data exceeds the token 
        limit, it is split into multiple, connected chunks.
        """
        chunks = []
        if not data_rows or len(data_rows) < 2:
            print(f"XLSX file '{file_name}' has no data rows or is missing a header. Skipping.")
            return chunks

        # CHANGED: Headers and rows are now lists, no splitting needed.
        headers = [str(h).strip() for h in data_rows[0]]
        content_rows = data_rows[1:]
            
        format_description = "This chunk contains data from a row where each value is mapped to its column header in a 'Header : Value' format."

        rows_per_chunk = self._calculate_optimal_xlsx_rows_per_chunk(data_rows)
        print(f"Processing XLSX in batches of {rows_per_chunk} rows.")

        for i in range(0, len(content_rows), rows_per_chunk):
            batch_of_rows = content_rows[i:i + rows_per_chunk]
            start_row = i + 1
            end_row = i + len(batch_of_rows)

            all_batch_items = []
            for row_values in batch_of_rows:
                # Pad/truncate values to match header count, ensuring alignment
                num_headers = len(headers)
                row_values.extend([''] * (num_headers - len(row_values)))
                row_values = row_values[:num_headers]
                
                if all_batch_items:
                    all_batch_items.append(("---", "New Row ---"))
                all_batch_items.extend(list(zip(headers, row_values)))

            if not all_batch_items:
                continue

            current_chunk_items = []
            is_first_chunk_for_batch = True

            for header, value in all_batch_items:
                if not header: # Skip empty headers from trailing commas
                    continue

                item_str = f"--- {value} ---" if header == "---" else f"{header} : {value.strip() if value else 'N/A'}"

                prospective_items = current_chunk_items + [item_str]
                
                if is_first_chunk_for_batch:
                    context = f"This is part {start_row} of XLSX file '{file_name}' containing rows {start_row} to {end_row}."
                else:
                    context = f"Continuation of data from rows {start_row} to {end_row} in XLSX file '{file_name}'."
                
                row_data_content = ", ".join(prospective_items)
                prospective_text = f"{context}\n\n{format_description}\n\nRow Data: {row_data_content}"
                
                if tiktoken_len(prospective_text) > CHUNK_SIZE_TOKENS and current_chunk_items:
                    final_context = (f"This is part {start_row} of XLSX file '{file_name}' containing rows {start_row} to {end_row}."
                                     if is_first_chunk_for_batch else f"Continuation of data from rows {start_row} to {end_row} in XLSX file '{file_name}'.")
                    final_row_data = ", ".join(current_chunk_items)
                    chunks.append({"text": f"{final_context}\n\n{format_description}\n\nRow Data: {final_row_data}"})
                    
                    current_chunk_items = [item_str]
                    is_first_chunk_for_batch = False
                else:
                    current_chunk_items = prospective_items

            if current_chunk_items:
                final_context = (f"This is part {start_row} of XLSX file '{file_name}' containing rows {start_row} to {end_row}."
                                 if is_first_chunk_for_batch else f"Continuation of data from rows {start_row} to {end_row} in XLSX file '{file_name}'.")
                final_row_data = ", ".join(current_chunk_items)
                chunks.append({"text": f"{final_context}\n\n{format_description}\n\nRow Data: {final_row_data}"})

        print(f"Created {len(chunks)} text-based chunks from {len(content_rows)} rows in '{file_name}'.")
        return chunks

    def _calculate_optimal_xlsx_rows_per_chunk(self, data_rows: List[List[str]]) -> int:
        """
        Calculate optimal number of rows per chunk for XLSX based on token limits.
        """
        if not data_rows:
            return 1

        # CORRECTED: Join the list of header strings into a single string before tokenizing.
        header_string = ", ".join(data_rows[0])
        avg_row_tokens = tiktoken_len(header_string) if header_string else 1
        
        available_tokens = CHUNK_SIZE_TOKENS - 100  # Reserve space for context/formatting
        rows_per_chunk = max(1, available_tokens // (avg_row_tokens if avg_row_tokens > 0 else 1))
        
        return min(rows_per_chunk, 1)

async def process_and_ingest_file(
    file_data: bytes,
    original_file_name: str,
    index_name: str,
    es_client: Any,
    aclient_openai: Any,
    params: Any,
    user_provided_doc_summary: str = "",
    mime_type: str = None
):
    """
    High-level function to run the full ingestion pipeline for a single file.
    """
    index_name = "ragsapp"
    if not await ensure_es_index_exists(es_client, index_name, CHUNKED_PDF_MAPPINGS):
        return {"status": "error", "message": "Elasticsearch index setup failed."}

    document_id = generate_doc_id_from_content(file_data)
    file_extension = os.path.splitext(original_file_name)[1].lower()
    params = {"index_name": index_name, "file_name": original_file_name}
    config = {}
    
    file_extension = os.path.splitext(original_file_name)[1].lower()
    if not file_extension or file_extension == "":
        kind = filetype.guess(file_data)
        if kind and kind.mime.startswith("image/"):
            file_extension = "." + kind.extension
            print(f"Guessed image extension: {file_extension}")
        elif mime_type and mime_type.startswith("image/"):
            file_extension = "." + mime_type.split("/")[-1]
            print(f"Guessed image extension from mime_type: {file_extension}")

    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"]
    if file_extension in image_extensions:
        print(f"Processing image file: {original_file_name}")
        parser = ImageParser(aclient_openai)
        description = ""
        async for desc in parser.ingest(file_data, filename=original_file_name):
            description = desc
            break  # Only one result expected

        if not description:
            return {"status": "error", "message": "Failed to process image with vision model."}

        # Use the vision output as both the chunk and the summary
        chunk_text = description
        doc_summary = description

        # Generate embedding for the description
        embedding_list = await IngestionProcessor(params, config, aclient_openai, file_extension)._generate_embeddings([chunk_text])
        embedding_vector = embedding_list[0] if embedding_list and embedding_list[0] else []

        if not embedding_vector:
            return {"status": "error", "message": "Failed to generate embedding for image description."}

        es_doc_id = f"{document_id}_img"
        metadata_payload = {
            "file_name": original_file_name,
            "doc_id": document_id,
            "page_number": 1,
            "chunk_index_in_page": 0,
            "document_summary": doc_summary,
            "entities": [],
            "relationships": [],
            "image_file": True
        }
        action = {
            "_index": index_name,
            "_id": es_doc_id,
            "_source": {
                "chunk_text": chunk_text,
                "embedding": embedding_vector,
                "metadata": metadata_payload
            }
        }
        try:
            successes, response = await async_bulk(es_client, [action], raise_on_error=False)
            print(f"Elasticsearch image ingestion: {successes} successes.")
            return {"status": "success", "message": "Image ingestion complete.", "num_chunks": 1}
        except Exception as e:
            print(f"Error indexing image: {e}")
            return {"status": "error", "message": f"Failed to index image: {e}"}

    processor = IngestionProcessor(params, config, aclient_openai, file_extension)
    actions_for_es = []

    try:
        doc_iterator = None
        if file_extension == ".pdf":
            doc_iterator = processor.process_pdf(file_data, original_file_name, document_id, user_provided_doc_summary)
        elif file_extension == ".csv":
            doc_iterator = processor.process_csv_semantic_chunking(file_data, original_file_name, document_id, user_provided_doc_summary)
        elif file_extension == ".xlsx":
            doc_iterator = processor.process_xlsx_semantic_chunking(file_data, original_file_name, document_id, user_provided_doc_summary)
        elif file_extension == ".doc":
            doc_iterator = processor.process_doc(file_data, original_file_name, document_id, user_provided_doc_summary)
        elif file_extension == ".docx":
            doc_iterator = processor.process_docx(file_data, original_file_name, document_id, user_provided_doc_summary)
        elif file_extension == ".odt":
            doc_iterator = processor.process_odt(file_data, original_file_name, document_id, user_provided_doc_summary)
        elif file_extension == ".txt":
            doc_iterator = processor.process_txt(file_data, original_file_name, document_id, user_provided_doc_summary)
        else:
            print(f"Unsupported file type: '{file_extension}'. Only .pdf, .csv and .xlsx are supported.")
            return {"status": "error", "message": "Unsupported file type."}

        if doc_iterator:
            async for action in doc_iterator:
                if action:
                    actions_for_es.append(action)

        if actions_for_es:
            print(f"Collected {len(actions_for_es)} actions for bulk ingestion into '{index_name}'.")
            
            if actions_for_es: 
                print("Sample document to be indexed (first one, embedding vector omitted if long):")
                sample_action_copy = copy.deepcopy(actions_for_es[0]) 
                if "_source" in sample_action_copy and "embedding" in sample_action_copy["_source"]:
                    embedding_val = sample_action_copy["_source"]["embedding"]
                    if isinstance(embedding_val, list) and embedding_val:
                        sample_action_copy["_source"]["embedding"] = f"<embedding_vector_dim_{len(embedding_val)}>"
                    elif not embedding_val: 
                         sample_action_copy["_source"]["embedding"] = "<empty_embedding_vector>"
                    else: 
                        sample_action_copy["_source"]["embedding"] = f"<embedding_vector_unexpected_format: {type(embedding_val).__name__}>"

            errors = []
            try:
                successes, response = await async_bulk(es_client, actions_for_es, raise_on_error=False)
                print(f"Elasticsearch bulk ingestion: {successes} successes.")
                
                failed = [r for r in response if not r[0]]
                if failed:
                    print(f"{len(failed)} document(s) failed to index. Showing first error:")
                    errors = failed

            except BulkIndexError as e:
                errors = e.errors
                print("BulkIndexError occurred:")
                print(json.dumps(e.errors, indent=2, default=str))

            if errors:
                print(f"Elasticsearch bulk ingestion errors ({len(errors)}):")
                for i, err_info in enumerate(errors):
                    error_item = err_info.get('index', err_info.get('create', err_info.get('update', err_info.get('delete', {}))))
                    status = error_item.get('status', 'N/A')
                    error_details = error_item.get('error', {})
                    error_type = error_details.get('type', 'N/A')
                    error_reason = error_details.get('reason', 'N/A')
                    doc_id_errored = error_item.get('_id', 'N/A')
                    print(f"Error {i+1}: Doc ID '{doc_id_errored}', Status {status}, Type '{error_type}', Reason: {error_reason}")
        else:
            print(f"No chunks generated or processed for ingestion from '{original_file_name}'.")
            return {"status": "error", "message": f"No chunks generated or processed for ingestion from '{original_file_name}'."}
            
    except Exception as e:
        print(f"An error occurred during the example run for '{original_file_name}': {e}")
    print(f"--- Finished PDF Processing for: {original_file_name} ---\n")
    # After bulk ingest
    return {"status": "success", "message": "Ingestion complete.", "num_chunks": len(actions_for_es)}
