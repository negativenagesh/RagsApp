import os
import asyncio
from typing import AsyncGenerator, Dict, Any, List, Tuple
import yaml 
from pathlib import Path 
import copy 
import xml.etree.ElementTree as ET 
import re
import hashlib 
import json
import sys
import shutil
import random
import traceback
import fitz
from urllib.parse import urlparse

from typing import Optional
from dotenv import load_dotenv
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk, BulkIndexError
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from openai import AsyncOpenAI
from datetime import datetime, timezone
try:
    from sdk.response import FunctionResponse, Messages
    from sdk.message import Message
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from sdk.response import FunctionResponse, Messages
    from sdk.message import Message

from parsers.doc_parser import DOCParser
from parsers.docx_parser import DOCXParser
from parsers.odt_parser import ODTParser
from parsers.text_parser import TextParser
from parsers.csv_parser import CSVParser
from parsers.xlsx_parser import XLSXParser
from parsers.pdf_parser import PDFParser
from parsers.ocr_parser import OCRParser
from parsers.mistral_ocr_parser import MistralOCRParser, MISTRAL_SDK_AVAILABLE

load_dotenv()

OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_SUMMARY_MODEL = "gpt-4.1-nano"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_EMBEDDING_DIMENSIONS = 3072

CHUNK_SIZE_TOKENS = 20000
CHUNK_OVERLAP_TOKENS = 0
FORWARD_CHUNKS = 3
BACKWARD_CHUNKS = 3
CHARS_PER_TOKEN_ESTIMATE = 4 
SUMMARY_MAX_TOKENS = 1024

MISTRAL_API_KEY = (os.getenv("MISTRAL_API_KEY") or "").strip().strip('"').strip("'")

def init_async_openai_client(text_model) -> Optional[AsyncOpenAI]:
    try:
        openai_api_key = os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("❌ OPENAI_API_KEY not found in .env. OpenAI client will not be functional.")
            return None

        client = AsyncOpenAI(api_key=openai_api_key)
        print("✅ AsyncOpenAI client initialized.")
        return client
    except Exception as e:
        print(f"❌ Failed to initialize AsyncOpenAI client: {e}")
        return None

ELASTICSEARCH_URL = os.getenv("RAG_UPLOAD_ELASTIC_URL")                    
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")

async def check_async_elasticsearch_connection() -> Optional[AsyncElasticsearch]:
    try:
        print('Connecting to Elsatic client ELASTICSEARCH_URL:-', ELASTICSEARCH_URL)
        es_client = None
        if ELASTICSEARCH_API_KEY:
            es_client = AsyncElasticsearch(
                ELASTICSEARCH_URL,
                api_key=ELASTICSEARCH_API_KEY,
                request_timeout=60,
                retry_on_timeout=True
            )
        else:
            es_client = AsyncElasticsearch(
                hosts=[ELASTICSEARCH_URL],
                request_timeout=60,
                retry_on_timeout=True
            )
        
        if not await es_client.ping():
            print("❌ Ping to Elasticsearch cluster failed. URL may be incorrect or server is down.")
            return None

        print("✅ AsyncElasticsearch client initialized.")
        return es_client
    except Exception as e:
       print(f"❌ Failed to initialize AsyncElasticsearch client: {e}")
       return None

def tiktoken_len(text: str) -> int:
  try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)
  except Exception as e:
    print(f"❌ Failed to calc tiktoken_len: {e}")
    return 0

CHUNKED_PDF_MAPPINGS = {
    "mappings": {
        "properties": {
            "chunk_text": {"type": "text"}, 
            "embedding": {
                "type": "dense_vector",
                "dims": 3072,
                "index": True,
                "similarity": "cosine"
            },
            "metadata": {
                "properties": {
                    "file_name": {"type": "keyword"},
                    "doc_id": {"type": "keyword"}, 
                    "page_number": {"type": "integer"},
                    "chunk_index_in_page": {"type": "integer"},
                }
            }
        }
    }
}


ATTACHMENT_MAPPINGS = {
    "mappings": {
        "properties": {
            "attachment_description": {"type": "text"},
            "metadata": {
                "properties": {
                    "file_name": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "upload_timestamp": {"type": "date"},
                    "file_extension": {"type": "keyword"}
                }
            }
        }
    }
}

class ChunkingEmbeddingPDFProcessor:
    def __init__(self, params: Any, config: Any, aclient_openai: Optional[AsyncOpenAI], file_extension: str):
        self.params = params
        self.config = config
        self.aclient_openai = aclient_openai
        self.embedding_model = None
        self.embedding_dims = OPENAI_EMBEDDING_DIMENSIONS
      
        if file_extension in [".docx", ".doc", ".odt"]:
            chunk_size = 2048
            chunk_overlap = 1024
            print(f"Using DOCX specific chunking: size={chunk_size}, overlap={chunk_overlap}")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=tiktoken_len,
                separators=["\n|", "\n", "|", ". "," ", ""],
            )
        else:
            chunk_size = CHUNK_SIZE_TOKENS
            chunk_overlap = CHUNK_OVERLAP_TOKENS
            print(f"Using default chunking: size={chunk_size}, overlap={chunk_overlap}")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE_TOKENS,
                chunk_overlap=CHUNK_OVERLAP_TOKENS,
                length_function=tiktoken_len,
                separators=["\n|", "\n", "|", ". "," ", ""],
            )
        
        self.embedding_dims = OPENAI_EMBEDDING_DIMENSIONS

        self.enrich_prompt_template = self._load_prompt_template("chunk_enrichment")

    def _load_prompt_template(self, prompt_name: str) -> str:
        try:
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            prompt_file_path = base_dir / "shared" / "prompts" / f"{prompt_name}.yaml"
            if not prompt_file_path.exists():
                prompt_file_path = Path("./prompts") / f"{prompt_name}.yaml"
            print('prompt_file_path:----', prompt_file_path)
            with open(prompt_file_path, 'r') as f:
                prompt_data = yaml.safe_load(f)
            
            if prompt_data and prompt_name in prompt_data and "template" in prompt_data[prompt_name]:
                template_content = prompt_data[prompt_name]["template"]
                print(f"Successfully loaded prompt template for '{prompt_name}'.")
                return template_content
            else:
                print(f"Prompt template for '{prompt_name}' not found or invalid in {prompt_file_path}.")
                raise ValueError(f"Invalid prompt structure for {prompt_name}")
        except FileNotFoundError:
            print(f"Prompt file not found: {prompt_file_path}")
            raise
        except Exception as e:
            print(f"Error loading prompt '{prompt_name}': {e}")
            raise

    def _clean_ocr_repetitions(self, text: str) -> str:
        """Remove repetitive patterns from OCR text."""
        if not text:
            return text
            
        words = text.split()
        cleaned_words = []
        i = 0
        
        while i < len(words):
            cleaned_words.append(words[i])
            
            for phrase_len in range(5, min(16, len(words) - i)):
                phrase = words[i:i+phrase_len]
                phrase_str = ' '.join(phrase)
                
                next_pos = i + phrase_len
                repeat_count = 0
                
                while next_pos + phrase_len <= len(words):
                    next_phrase = words[next_pos:next_pos+phrase_len]
                    next_phrase_str = ' '.join(next_phrase)
                    
                    if next_phrase_str == phrase_str:
                        repeat_count += 1
                        next_pos += phrase_len
                    else:
                        break
                
                if repeat_count > 0:
                    print(f"Found {repeat_count} repetitions of phrase: '{phrase_str}'")
                    i = next_pos - 1  # -1 because we'll increment i at the end of the loop
                    break
            
            i += 1
        
        return ' '.join(cleaned_words)

    async def _call_openai_api(
        self,
        model_name: str,
        payload_messages: List[Dict[str, Any]],
        is_vision_call: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> str:
        """A unified async method to call OpenAI text and vision models with retry logic."""
        if not self.aclient_openai:
            print("OpenAI client not configured. Cannot make API call.")
            return ""

        max_retries = 5
        base_delay_seconds = 10

        for attempt in range(max_retries):
            try:
                start_time = datetime.now(timezone.utc)
                
                response = await self.aclient_openai.chat.completions.create(
                    model=model_name,
                    messages=payload_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                
                content = response.choices[0].message.content
                

                if content:
                    print(f"OpenAI API call successful. Preview: {content[:100].strip()}...")
                    return content
                else:
                    print(f"OpenAI API returned empty content. Attempt {attempt + 1}/{max_retries}")

            except Exception as e:
                print(f"OpenAI API call failed (Attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt + 1 < max_retries:
                delay = base_delay_seconds * (2 ** attempt)
                print(f"Waiting for {delay} seconds before retrying...")
                await asyncio.sleep(delay)

        print("Max retries reached for OpenAI API call. Returning empty string.")
        return ""



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
            print("OpenAI client not available. Cannot generate embeddings.")
            raise RuntimeError("OpenAI client not available and no alternative embedding method configured")
        
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
            traceback.print_exc()
            raise RuntimeError(f"Failed to generate embeddings using OpenAI: {e}") from e

    async def _generate_all_raw_chunks_from_doc(
        self,
        doc_text: str,
        file_name: str,
        doc_id: str,
        page_breaks: List[int] = None
    ) -> List[Dict[str, Any]]:
        all_raw_chunks_with_meta: List[Dict[str, Any]] = []
        if not doc_text or not doc_text.strip():
            print(f"Skipping empty document {file_name} for raw chunk generation.")
            return []

        raw_chunks = self.text_splitter.split_text(doc_text)
        
        if not page_breaks or len(page_breaks) <= 1:
            for chunk_idx, raw_chunk_text in enumerate(raw_chunks):
                doc_file_name = self.params.get('file_name', file_name)
                print(f"RAW CHUNK (File: {file_name}, Original File Name: {doc_file_name}, Page: 1, Idx: {chunk_idx}, Len {len(raw_chunk_text)}): '''{raw_chunk_text[:100].strip()}...'''")
                all_raw_chunks_with_meta.append({
                    "text": raw_chunk_text,
                    "page_num": 1,
                    "chunk_idx_on_page": chunk_idx,
                    "file_name": doc_file_name,
                    "doc_id": doc_id
                })
        
        else:
            current_page = 1
            next_page_break_idx = 0
            current_pos = 0
            
            for chunk_idx, raw_chunk_text in enumerate(raw_chunks):
                chunk_start_pos = current_pos
                chunk_end_pos = current_pos + len(raw_chunk_text)
                
                while next_page_break_idx < len(page_breaks) and chunk_start_pos >= page_breaks[next_page_break_idx]:
                    current_page += 1
                    next_page_break_idx += 1
                
                doc_file_name = self.params.get('file_name', file_name)
                print(f"RAW CHUNK (File: {file_name}, Original File Name: {doc_file_name}, Page: {current_page}, Idx: {chunk_idx}, Len {len(raw_chunk_text)}): '''{raw_chunk_text[:100].strip()}...'''")
                
                all_raw_chunks_with_meta.append({
                    "text": raw_chunk_text,
                    "page_num": current_page,
                    "chunk_idx_on_page": chunk_idx,
                    "file_name": doc_file_name,
                    "doc_id": doc_id
                })
                
                current_pos = chunk_end_pos
        
        print(f"Generated {len(all_raw_chunks_with_meta)} raw chunks from {file_name} across multiple pages.")
        return all_raw_chunks_with_meta

    async def _process_individual_chunk_pipeline(
        self, 
        raw_chunk_info: Dict[str, Any], 
        user_provided_doc_summary: str, 
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
        is_docx_file = file_extension == '.docx'
        is_ocr_pdf = params.get('is_ocr_pdf', False)
        
        enriched_text = chunk_text

        if is_tabular_file:
          print(f"Tabular file ({file_extension}) detected. Skipping enrichment.")
        
        elif is_ocr_pdf:
            print(f"OCR PDF detected. Using raw chunks without enrichment for '{file_name}'.")
            enriched_text = chunk_text
        
        elif is_docx_file:
            print(f"DOCX file detected. Using raw chunks without enrichment for '{file_name}'.")
            enriched_text = chunk_text

        else:
            preceding_indices = range(max(0, global_idx - BACKWARD_CHUNKS), global_idx)
            succeeding_indices = range(global_idx + 1, min(len(all_raw_texts), global_idx + 1 + FORWARD_CHUNKS))
            preceding_texts = [all_raw_texts[i] for i in preceding_indices]
            succeeding_texts = [all_raw_texts[i] for i in succeeding_indices]

            contextual_summary = user_provided_doc_summary


            try:
                enriched_text = await self._enrich_chunk_content(
                    chunk_text, contextual_summary, preceding_texts, succeeding_texts
                )
                print(f"Enrichment successful for chunk (Page {page_num}, Index {chunk_idx_on_page}).")
            except Exception as e:
                print(f"Enrichment failed for chunk (Page {page_num}, Index {chunk_idx_on_page}) for '{file_name}': {e}. Using original text.")
                enriched_text = chunk_text

        
        doc_file_name = self.params.get('file_name', file_name)
        enriched_text = f"[File: {doc_file_name}]\n\n{enriched_text}"
        
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
        }
        
        index_name = params.get('index_name')
        print('Index Name:--', index_name)
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
        """
        Orchestrates PDF processing, splitting large documents into 100-page batches.
        """
        print(f"Processing PDF: {file_name} (Doc ID: {doc_id})")
        batch_size = 100

        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                total_pages = len(doc)
                print(f"PDF has {total_pages} pages. Processing in batches of up to {batch_size} pages.")

            if total_pages <= batch_size:
                print("Document is small enough to be processed in a single batch.")
                async for action in self._process_pdf_batch(data, file_name, doc_id, user_provided_document_summary, 0, total_pages):
                    yield action
                return

            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                print(f"\n--- Processing batch of pages {batch_start + 1} to {batch_end} of {total_pages} ---")

                with fitz.open(stream=data, filetype="pdf") as original_doc:
                    with fitz.open() as batch_doc:
                        batch_doc.insert_pdf(original_doc, from_page=batch_start, to_page=batch_end - 1)
                        batch_data = batch_doc.write()

                try:
                    async for action in self._process_pdf_batch(batch_data, file_name, doc_id, user_provided_document_summary, batch_start, batch_end):
                        yield action

                    if batch_end < total_pages:
                        print(f"Completed batch for pages {batch_start + 1}-{batch_end}. Waiting 5 seconds before next batch...")
                        await asyncio.sleep(5)

                except Exception as e:
                    print(f"❌ Error processing batch for pages {batch_start + 1}-{batch_end}: {e}")
                    print("Attempting to continue with the next batch after a 15-second delay...")
                    await asyncio.sleep(15)
                    continue

        except Exception as e:
            print(f"❌ A critical error occurred during PDF batching setup: {e}")
            print("Falling back to processing the document as a single unit, which may fail for large files.")
            async for action in self._process_pdf_batch(data, file_name, doc_id, user_provided_document_summary, 0, -1):
                 yield action

    async def _process_pdf_batch(
        self, data: bytes, file_name: str, doc_id: str, user_provided_document_summary: str, batch_start_page: int, total_pages_in_batch: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        print(f"Processing PDF: {file_name} (Doc ID: {doc_id})")
        
        use_ocr = self.params.get('is_ocr_pdf', False)

        try:
                

            print(f"PDF processing: Setting consistent chunk size to 2048 with overlap 1024 for '{file_name}'.")
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
            ocr_engine = self.params.get('ocr_engine', 'mistral')
            print(f"Processing with OCR parser, engine: {ocr_engine}.")
            
            try:
                mistral_api_key = (os.getenv("MISTRAL_API_KEY") or MISTRAL_API_KEY or "").strip().strip('"').strip("'")

                if ocr_engine == 'mistral' and mistral_api_key and MISTRAL_SDK_AVAILABLE:
                    print("Using Mistral OCR.")
                    mistral_parser = MistralOCRParser(api_key=mistral_api_key)
                    
                    numbered_pages = []
                    async for page_text, page_num_in_batch in mistral_parser.ingest(data):
                        absolute_page_num = batch_start_page + page_num_in_batch
                        
                        cleaned_page_text = self._clean_ocr_repetitions(page_text)
                        
                        paragraphs = [p.strip() for p in cleaned_page_text.split("\n\n") if p.strip()]
                        
                        numbered_paragraphs = []
                        for para_idx, paragraph in enumerate(paragraphs, 1):
                            numbered_paragraph = f"[Page {absolute_page_num}, Paragraph {para_idx}] {paragraph}"
                            numbered_paragraphs.append(numbered_paragraph)
                        
                        if numbered_paragraphs:
                            numbered_pages.append("\n\n".join(numbered_paragraphs))

                    full_document_text = "\n\n\n".join(numbered_pages)
                    print(f"Successfully parsed and numbered OCR'd file '{file_name}' with {len(numbered_pages)} pages.")

                else: # Fallback for other OCR engines or if Mistral is unavailable
                    if ocr_engine == 'mistral' and not mistral_api_key:
                        print("Mistral OCR requested but MISTRAL_API_KEY is missing or empty; falling back to Tesseract OCR.")
                    elif ocr_engine == 'mistral' and not MISTRAL_SDK_AVAILABLE:
                        print("Mistral OCR requested but mistralai SDK is not installed; falling back to Tesseract OCR.")
                    else:
                        print("Using Tesseract OCR or fallback.")
                    ocr_parser = OCRParser()
                    ocr_texts = [page_text async for page_text in ocr_parser.ingest(data)]
                    full_document_text = " ".join(ocr_texts)

            except Exception as ocr_error:
                print(f"An error occurred during OCR with '{ocr_engine}': {ocr_error}. Aborting batch.")
                traceback.print_exc()
                return

        else:
            print("Processing with standard PDF parser.")
            parser = PDFParser(self.aclient_openai, self.config, self)
            content_by_page = {}
            async for content, page_num_in_batch in parser.ingest(data):
                absolute_page_num = batch_start_page + page_num_in_batch
                if absolute_page_num not in content_by_page:
                    content_by_page[absolute_page_num] = []
                content_by_page[absolute_page_num].append(content)

            numbered_pages = []
            for page_num in sorted(content_by_page.keys()):
                page_blocks = content_by_page[page_num]
                numbered_paragraphs = []
                for para_idx, block in enumerate(page_blocks, 1):
                    numbered_paragraph = f"[Page {page_num}, Paragraph {para_idx}] {block}"
                    numbered_paragraphs.append(numbered_paragraph)
                
                if numbered_paragraphs:
                    numbered_pages.append("\n\n".join(numbered_paragraphs))

            full_document_text = "\n\n\n".join(numbered_pages)
            print(f"Successfully parsed and numbered standard PDF file '{file_name}' with {len(numbered_pages)} pages.")

        if not full_document_text.strip():
            print(f"No text or tables extracted from '{file_name}'. Aborting processing.")
            return

        
        all_raw_chunks_with_meta = await self._generate_all_raw_chunks_from_doc(
            full_document_text, file_name, doc_id
        )

        if not all_raw_chunks_with_meta:
            print(f"No raw chunks were generated from '{file_name}'. Aborting further processing.")
            return

        all_raw_texts = [chunk["text"] for chunk in all_raw_chunks_with_meta]
        
        print(f"Starting concurrent processing for {len(all_raw_chunks_with_meta)} raw chunks from '{file_name}'.")

        processing_tasks = []
        for i, raw_chunk_info_item in enumerate(all_raw_chunks_with_meta):
            raw_chunk_info_item["page_num"] += batch_start_page
            task = asyncio.create_task(
                self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info_item,
                    user_provided_doc_summary=user_provided_document_summary, 
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
    
        parser = DOCParser(self.aclient_openai, self)
        
        full_document_text = ""
        page_breaks = []
        current_position = 0
        
        try:
            all_parts = []
            async for part in parser.ingest(data):
                all_parts.append(part)
                if "\f" in part or "[PAGE BREAK]" in part:
                    page_breaks.append(current_position)
                current_position += len(part)
            
            full_document_text = " ".join(all_parts)
            
            if not page_breaks and len(full_document_text) > 3000:  # If document is substantial
                avg_chars_per_page = 3000  # Estimate chars per page
                for i in range(1, len(full_document_text) // avg_chars_per_page + 1):
                    page_breaks.append(i * avg_chars_per_page)
            
            print(f"Successfully parsed .docx file '{file_name}' with {len(page_breaks) + 1} detected pages.")
            
        except Exception as e:
            print(f"Failed to parse .docx file '{file_name}': {e}")
            return

        if not full_document_text.strip():
            print(f"No text extracted from DOCX file '{file_name}'. Aborting processing.")
            return

        
        all_raw_chunks_with_meta = await self._generate_all_raw_chunks_from_doc(
            full_document_text, file_name, doc_id, page_breaks
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

        parser = DOCXParser(self.aclient_openai, self)
        
        try:
            all_parts = [part async for part in parser.ingest(data)]
            
            page_break_marker = "[[--PAGE_BREAK--]]"
            reconstructed_text = ""
            for part in all_parts:
                if "\f" in part or "[PAGE BREAK]" in part:
                    reconstructed_text += page_break_marker
                else:
                    reconstructed_text += part.strip() + "\n\n"

            pages_text = reconstructed_text.split(page_break_marker)
            
            numbered_pages = []
            for i, page_content in enumerate(pages_text):
                page_number = i + 1
                paragraphs = [p.strip() for p in page_content.split("\n\n") if p.strip()]
                
                numbered_paragraphs = []
                for para_idx, paragraph in enumerate(paragraphs, 1):
                    numbered_paragraph = f"[Page {page_number}, Paragraph {para_idx}] {paragraph}"
                    numbered_paragraphs.append(numbered_paragraph)
                
                if numbered_paragraphs:
                    numbered_pages.append("\n\n".join(numbered_paragraphs))
            
            full_document_text = "\n\n\n".join(numbered_pages)
            
            print(f"Successfully parsed and numbered .docx file '{file_name}' with {len(numbered_pages)} pages.")
            
        except Exception as e:
            print(f"Failed to parse .docx file '{file_name}': {e}")
            traceback.print_exc()
            return

        if not full_document_text.strip():
            print(f"No text extracted from DOCX file '{file_name}'. Aborting processing.")
            return

        
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

        parser = ODTParser(self.aclient_openai, self)

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
        
            header_row = rows[0] if rows else ""
            data_rows = rows[1:] if len(rows) > 1 else []
        
            chunks = await self._create_semantic_csv_chunks(
                header_row, data_rows, file_name
            )
        
            if not chunks:
                print(f"No chunks generated from CSV '{file_name}'. Aborting.")
                return
        
            for chunk_idx, chunk_data in enumerate(chunks):
                chunk_text = chunk_data["text"]
                chunk_context = chunk_data.get("context", "")
            
                full_chunk_text = f"{chunk_context}\n\n{chunk_text}" if chunk_context else chunk_text
            
            
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
    
        rows_per_chunk = self._calculate_optimal_rows_per_chunk(header_row, data_rows)
    
        for i in range(0, len(data_rows), rows_per_chunk):
            batch_rows = data_rows[i:i + rows_per_chunk]
        
            chunk_text = f"CSV Structure:\n{header_row}\n\nData:\n" + "\n".join(batch_rows)
        
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
        """ Calculate optimal number of rows per chunk based on token limits """
        if not data_rows:
            return 1
    
        header_tokens = tiktoken_len(header_row)
        avg_row_tokens = tiktoken_len(data_rows[0]) if data_rows else 1
        available_tokens = CHUNK_SIZE_TOKENS - header_tokens - 100  # 100 for context/formatting
        rows_per_chunk = max(1, available_tokens // avg_row_tokens)
    
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
            rows = [row_list async for row_list in xlsx_parser.ingest(data)]

            if not rows:
                print(f"No data extracted from XLSX '{file_name}'. Aborting processing.")
                return

            chunks = await self._create_semantic_xlsx_chunks(rows, file_name)

            if not chunks:
                print(f"No chunks generated from XLSX '{file_name}'. Aborting.")
                return

            full_document_text = "\n".join([", ".join(row) for row in rows])

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

        header_string = ", ".join(data_rows[0])
        avg_row_tokens = tiktoken_len(header_string) if header_string else 1
        
        available_tokens = CHUNK_SIZE_TOKENS - 100  # Reserve space for context/formatting
        rows_per_chunk = max(1, available_tokens // (avg_row_tokens if avg_row_tokens > 0 else 1))
        
        return min(rows_per_chunk, 1)


async def ensure_es_index_exists(client: Any, index_name: str, mappings_body: Dict, config: Any):
    try:
        if not await client.indices.exists(index=index_name):
            updated_mappings = copy.deepcopy(mappings_body)
            dims = OPENAI_EMBEDDING_DIMENSIONS
            
            if "embedding" in updated_mappings["mappings"]["properties"]:
                updated_mappings["mappings"]["properties"]["embedding"]["dims"] = dims
            
            if "metadata" in updated_mappings["mappings"]["properties"]:
                metadata_props= updated_mappings["mappings"]["properties"]["metadata"].get("properties", {})
                if "entities" in metadata_props and "properties" in metadata_props["entities"]:
                    entities_props= metadata_props["entities"]["properties"]
                    if "description_embedding" in entities_props:
                        entities_props["description_embedding"]["dims"] = dims
            
            if "entities" in updated_mappings["mappings"]["properties"]:
                entities_props= updated_mappings["mappings"]["properties"]["entities"].get("properties", {})
                if "description_embedding" in entities_props:
                    entities_props["description_embedding"]["dims"] = dims
            
            await client.indices.create(index=index_name, body=updated_mappings)
            print(f"Created index '{index_name}' created with specified mappings")
            return True
        else:
            print('Index already exists in ensure_es_index_exists')
            
            current_mapping_response = await client.indices.get_mapping(index=index_name)
            current_mapping = current_mapping_response.get(index_name, {}).get('mappings', {}).get('properties', {})
            current_metadata = current_mapping.get('metadata', {}).get('properties', {})
            current_entities = current_metadata.get('entities', {})
            
            entities_exists = 'entities' in current_metadata
            entities_is_nested = current_entities.get('type') == 'nested'
            
            has_description_embedding = False
            if entities_is_nested and 'properties' in current_entities:
                has_description_embedding = 'description_embedding' in current_entities.get('properties', {})
            
            if entities_exists and not entities_is_nested:
                print(f"⚠️ Warning: Index '{index_name}' has entities field as non-nested, but we need it to be nested.")
                print("This requires recreating the index or creating a new one with a different name.")
                print("Options:")
                print("1. Delete the existing index and recreate it (will lose all data)")
                print("2. Use a different index name")
                print("3. Continue with limited functionality (no entity embeddings)")
                
                print("Continuing with existing mapping - entity embeddings will not be available.")
                return True
            
            if not entities_exists:
                print(f"Adding entities field as nested to index '{index_name}'")
                
                dims = OPENAI_EMBEDDING_DIMENSIONS
                
                try:
                    update_mapping = {
                        "properties": {
                            "metadata": {
                                "properties": {
                                    "entities": {
                                        "type": "nested",
                                        "properties": {
                                            "name": {"type": "keyword"},
                                            "type": {"type": "keyword"},
                                            "description": {"type": "text"},
                                            "description_embedding": {
                                                "type": "dense_vector",
                                                "dims": dims,
                                                "index": True,
                                                "similarity": "cosine"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    await client.indices.put_mapping(index=index_name, body=update_mapping)
                    print(f"Successfully added nested entities field to index '{index_name}'")
                    return True
                    
                except Exception as e:
                    print(f"Failed to add entities mapping to index '{index_name}': {e}")
                    return False
            
            if entities_is_nested and not has_description_embedding:
                print(f"Adding description_embedding field to nested entities in index '{index_name}'")
                
                dims = OPENAI_EMBEDDING_DIMENSIONS
                
                try:
                    update_mapping = {
                        "properties": {
                            "metadata": {
                                "properties": {
                                    "entities": {
                                        "type": "nested",
                                        "properties": {
                                            "description_embedding": {
                                                "type": "dense_vector",
                                                "dims": dims,
                                                "index": True,
                                                "similarity": "cosine"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    await client.indices.put_mapping(index=index_name, body=update_mapping)
                    print(f"Successfully added description_embedding to entities in index '{index_name}'")
                    return True
                    
                except Exception as e:
                    print(f"Failed to update entities mapping in index '{index_name}': {e}")
                    return False
            
            expected_top_level_props = mappings_body.get('mappings', {}).get('properties', {})
            missing_fields = []
            
            for field, expected_field_mapping in expected_top_level_props.items():
                if field == "metadata":
                    continue  # Already handled above
                if field not in current_mapping:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"Adding missing top-level fields: {missing_fields}")
                update_body = {field: expected_top_level_props[field] for field in missing_fields}
                
                if "embedding" in update_body:
                    dims = OPENAI_EMBEDDING_DIMENSIONS
                    update_body["embedding"]["dims"] = dims
                
                try:
                    await client.indices.put_mapping(index=index_name, body={"properties": update_body})
                    print(f"Successfully added missing fields to index '{index_name}'")
                except Exception as e:
                    print(f"Failed to add missing fields to index '{index_name}': {e}")
            
            print(f"Index '{index_name}' mapping verification completed.")
            return True
    except Exception as e:
        print(f"❌ Error with Elasticsearch index '{index_name}': {e}")
        traceback.print_exc()
        if hasattr(e, 'info'):
            print("🔎 Error details:", json.dumps(e.info, indent=2))
        return False

async def example_run_file_processing(file_data: str | bytes, original_file_name: str, document_id: str, user_provided_doc_summary: str,es_client: Any, aclient_openai: Any, params: Any, config: Any):
    index_name = params.get('index_name')
    expected_mappings = copy.deepcopy(CHUNKED_PDF_MAPPINGS)
    print('elastic search index_name:--', index_name)
    dims = OPENAI_EMBEDDING_DIMENSIONS
    
    expected_mappings["mappings"]["properties"]["embedding"]["dims"] = dims


    print(f"Mode: Standard, Embedding Dimensions: {dims}")
    
    main_index_task = ensure_es_index_exists(es_client, index_name, expected_mappings, config)
    
    main_index_result = await main_index_task
    
    if not main_index_result:
        print("Failed to ensure Elasticsearch index '{index_name}' exists or is compatible. Aborting.")
        return


    file_extension = os.path.splitext(original_file_name)[1].lower()
    try:
        s3_url_path = urlparse(params.get("s3Path")).path
        file_extension = os.path.splitext(s3_url_path)[1].lower()
    except Exception:
        pass

    processor = ChunkingEmbeddingPDFProcessor(params, config, aclient_openai, file_extension)
    actions_for_es = []
    
    file_extension = file_extension or os.path.splitext(original_file_name)[1].lower()
    print(f"\n--- Starting Processing for: {original_file_name} (Ext: {file_extension}) ---")
    
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tiff"}
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
    MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS

    if file_extension in MEDIA_EXTS:
        media_type= "video" if file_extension in VIDEO_EXTS else "image"
        try:
            media_description_text = ""
            mud = params.get("map_upload_data") or {}

            if isinstance(mud, dict):
                media_description_text = mud.get("image_description_data") or ""
            
            if not media_description_text:
                media_description_text = params.get("image_description_data") or ""

            media_description_text = (media_description_text or "").strip()
            if not media_description_text:
                print(f"No {media_type} description text provided; skipping indexing for {media_type} file.")
                return

            doc_file_name = params.get("file_name", original_file_name)
            enriched_text = f"[File: {doc_file_name}]\n\n{media_description_text}"



            embedding_list = await processor._generate_embeddings([enriched_text])
            embedding_vector = embedding_list[0] if (embedding_list and embedding_list[0]) else []
            if not embedding_vector:
                print(f"Skipping {media_type} document due to missing embedding.")
                return

            es_doc_id = f"{document_id}_p1_c0"
            metadata_payload = {
                "file_name": doc_file_name,
                "doc_id": document_id,
                "page_number": 1,
                "chunk_index_in_page": 0,
                "media_type": media_type,
            }

            main_action = {
                "_index": index_name,
                "_id": es_doc_id,
                "_source": {
                    "chunk_text": enriched_text,
                    "embedding": embedding_vector,
                    "metadata": metadata_payload,
                },
            }


            bulk_tasks = [async_bulk(es_client, [main_action], raise_on_error=False)]

            results = await asyncio.gather(*bulk_tasks, return_exceptions=True)
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"❌ {media_type.capitalize()} document bulk indexing failed: {result}")
                else:
                    successes, _ = result
                    print(f"✅ MAIN bulk ingestion ({media_type}): {successes} successes.")

            return
        except Exception as e:
            print(f"Error during {media_type.capitalize()} ingestion path: {e}")
            return

    try:
        doc_iterator = None
        if file_extension == ".pdf":
            doc_iterator = processor.process_pdf(file_data, original_file_name, document_id,user_provided_doc_summary)
        elif file_extension == ".doc":
            doc_iterator = processor.process_doc(file_data, original_file_name, document_id, user_provided_doc_summary)
        elif file_extension == ".docx":
            doc_iterator = processor.process_docx(file_data, original_file_name, document_id, user_provided_doc_summary)
        elif file_extension == ".odt":
            doc_iterator = processor.process_odt(file_data, original_file_name, document_id, user_provided_doc_summary)
        elif file_extension == ".txt":
            doc_iterator = processor.process_txt(file_data, original_file_name, document_id, user_provided_doc_summary)    
        elif file_extension == ".csv":
            doc_iterator = processor.process_csv_semantic_chunking(file_data, original_file_name, document_id, user_provided_doc_summary)
        elif file_extension == ".xlsx":
            doc_iterator = processor.process_xlsx_semantic_chunking(file_data, original_file_name, document_id, user_provided_doc_summary)
        else:
            print(f"Unsupported file type: '{file_extension}'. Only .pdf, .doc, .docx, .csv and .txt are supported.")
            return None

        if doc_iterator:
            main_actions_for_es = []
            
            async for result in doc_iterator:
                if result: 
                    main_actions_for_es.append(result)

        if main_actions_for_es:
            print(f"Collected {len(main_actions_for_es)} main chunk actions for bulk ingestion.")
            
            if main_actions_for_es: 
                print("Sample main chunk document to be indexed (first one, embedding vector omitted if long):")
                sample_action_copy = copy.deepcopy(main_actions_for_es[0]) 
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
                
                bulk_tasks = []
                
                if main_actions_for_es:
                    main_bulk_task = async_bulk(es_client, main_actions_for_es, raise_on_error=False)
                    bulk_tasks.append(('main', main_bulk_task))
                
                
                if bulk_tasks:
                    results = await asyncio.gather(*[task for _, task in bulk_tasks], return_exceptions=True)
                    
                    for idx, (task_name, _) in enumerate(bulk_tasks):
                        result = results[idx]
                        if isinstance(result, Exception):
                            print(f"❌ {task_name.upper()} bulk indexing failed: {result}")
                            errors.append({task_name: str(result)})
                        else:
                            successes, response = result
                            print(f"✅ {task_name.upper()} bulk ingestion: {successes} successes.")

                            failed = [r for r in response if not r[0]]
                            if failed:
                                print(f"❌ {len(failed)} {task_name} document(s) failed to index.")
                                errors.append(failed)
                
            except BulkIndexError as e:
                errors.extend(e.errors)
                print("BulkIndexError occurred:")
                print(json.dumps(e.errors[:5], indent=2, default=str))  # Show first 5 errors only

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
            
    except Exception as e:
        print(f"An error occurred during the example run for '{original_file_name}': {e}")

def _generate_doc_id_from_content(content_bytes: bytes) -> str:
    """Generates a SHA256 hash for the given byte content."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content_bytes)
    return sha256_hash.hexdigest()

def parse_azure_blob_path(url):
    parsed = urlparse(url)
    path_parts = parsed.path.lstrip("/").split("/", 1)
    container_name = path_parts[0]
    blob_name = path_parts[1] if len(path_parts) > 1 else ""
    return container_name, blob_name

def read_file(file_content, file_key):
    print("here", file_key)
    file_parts = file_key.split("/")
    filename = file_parts[-1]
    random_number = random.randint(1000, 9999)
    folder_path = f"./data/{random_number}"
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)

    try:
        with open(file_path, "wb") as f:
            f.write(file_content)

        if filename.lower().endswith(".pdf"):
            document = {"filename": filename, "text": "DEPRECATED_LOGIC"}
        else:
            return {"error": f"Unsupported file format: {filename}"}

    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}

    finally:
        try:
            shutil.rmtree(folder_path)
        except Exception as cleanup_error:
            print("Cleanup error:", cleanup_error)

    return document, filename

async def handle_attachment_request(data: Message) -> FunctionResponse:
    """
    Handle attachment-based ingestion requests.
    Uses example_run_file_processing for document processing, then adds S3 upload and custom fields.
    """
    try:
        print('\n ATTACHMENT MODE ENABLED')
        print('Incoming Attachment Data:', data)
        
        params = data.params
        config = data.config
        
        if not params.get('file_path'):
            return FunctionResponse(False, "file_path is required for attachment mode")
        
        if not params.get('index_name'):
            return FunctionResponse(False, "index_name is required for attachment mode")
        
        es_client = await check_async_elasticsearch_connection()
        if not es_client:
            return FunctionResponse(False, "Could not connect to Elasticsearch")
        
        text_model = params.get('text_model') or config.get('text_model') or 'gpt-4o-mini'
        aclient_openai = init_async_openai_client(text_model)
        if not aclient_openai:
            return FunctionResponse(False, "Could not connect to OpenAI")
        
        file_data = None
        original_file_name = None
        
        if file_data is None and params.get('file_path'):
            doc_path_input = params.get("file_path")
            file_path = doc_path_input
            
            try:
                with open(file_path, "rb") as f:
                    file_data = f.read()
                    print(f"Successfully read file: {file_path}")
            except Exception as e:
                return FunctionResponse(False, f"Failed to read file '{file_path}': {e}")
            
            original_file_name = params.get("file_name") or os.path.basename(file_path)
        
        if file_data is None:
            return FunctionResponse(False, "Could not read file from local path")
        
        if params.get('fileId'):
            generated_doc_id = params.get('fileId')
            print(f"Using provided fileId as Document ID: {generated_doc_id}")
        else:
            generated_doc_id = _generate_doc_id_from_content(file_data)
            print(f"Generated Document ID from content hash: {generated_doc_id}")

        custom_fields = {}
        map_upload_data = params.get('map_upload_data', {})
        if isinstance(map_upload_data, dict) and 'rag_form_data' in map_upload_data:
            custom_fields = map_upload_data['rag_form_data'].copy()
        if isinstance(map_upload_data, dict) and 'image_description_data' in map_upload_data:
            custom_fields['image_description'] = map_upload_data['image_description_data']
        
        print(f"Custom fields to be added: {custom_fields}")
        
        user_provided_summary = params.get('description') or f"Content from {original_file_name}"
        
        print("\n--- Starting file processing with example_run_file_processing ---")
        await example_run_file_processing(
            file_data=file_data,
            original_file_name=original_file_name,
            document_id=generated_doc_id,
            user_provided_doc_summary=user_provided_summary,
            es_client=es_client,
            aclient_openai=aclient_openai,
            params=params,
            config=config
        )
        
        index_name = params.get('index_name')
        print(f"\n--- Updating documents with custom fields ---")
        
        await es_client.indices.refresh(index=index_name)
        
        update_body = {
            "script": {
                "source": """
                    for (entry in params.custom_fields.entrySet()) {
                        ctx._source[entry.getKey()] = entry.getValue();
                    }
                """,
                "params": {
                    "custom_fields": custom_fields
                }
            },
            "query": {
                "term": {
                    "metadata.doc_id": generated_doc_id
                }
            }
        }
        
        try:
            response = await es_client.update_by_query(
                index=index_name,
                body=update_body,
                refresh=True
            )
            updated_count = response.get('updated', 0)
            print(f"Updated {updated_count} document(s) with custom fields")
            
        except Exception as e:
            print(f"Warning: Failed to update documents with custom fields: {e}")
        
        if es_client:
            await es_client.close()
        if aclient_openai and hasattr(aclient_openai, "aclose"):
            try:
                await aclient_openai.aclose()
            except Exception as e:
                print(f"Error closing OpenAI client: {e}")
                
        print('Map upload rag data successfully uploaded')
        return FunctionResponse(message=Messages("success"))
        
    except Exception as e:
        print(f"Error during attachment processing: {e}")
        traceback.print_exc()
        return FunctionResponse(message=Messages(str(e)))
    
async def handle_request(data: Message) -> FunctionResponse:
  try:
    print('Incoming Data:--', data)
    params = data.params
    config = data.config

    if params.get('enable_attachment_mode'):
        print('Routing to Map Upload One')
        return await handle_attachment_request(data)

    if params.get('is_ocr_pdf'):
        params['ocr_engine'] = params.get('ocr_engine', 'mistral')

    es_client=None
    aclient_openai=None
    es_client = await check_async_elasticsearch_connection()
    if not es_client:
      return FunctionResponse(False, "Could not connect to Elasticsearch.")

    text_model = ( params.get('text_model') or config.get('text_model') or 'gpt-4o-mini' )
    aclient_openai = init_async_openai_client(text_model)
    if not aclient_openai:
        return FunctionResponse(False, "Could not connect to Open Ai.")

    doc_path_input = params.get("file_path")
    doc_path = Path(doc_path_input)
    original_file_name = doc_path.name
    print('doc_path:----', doc_path)

    try:
        with open(doc_path, "rb") as f:
            doc_bytes_data = f.read()
            print(f"Successfully read PDF file: {doc_path}")
    except Exception as e:
        print(f"Failed to read PDF file '{doc_path}': {e}")
        return

    generated_doc_id = _generate_doc_id_from_content(doc_bytes_data)
    print(f"Generated Document ID (SHA256 of content) for '{original_file_name}': {generated_doc_id}")
    
    user_provided_summary_input = params.get('description') or f"Content from {original_file_name}" 

    await example_run_file_processing(
      file_data=doc_bytes_data,
      original_file_name=original_file_name,
      document_id=generated_doc_id, 
      user_provided_doc_summary=user_provided_summary_input,
      es_client=es_client,
      aclient_openai=aclient_openai,
      params=params,
      config=config,
    )
        
    if es_client:
        await es_client.close()
        print("Elasticsearch client closed.")
    if aclient_openai and hasattr(aclient_openai, "aclose"): 
        try:
            await aclient_openai.aclose() 
            print("OpenAI client closed.")
        except Exception as e:
            print(f"Error closing OpenAI client: {e}")

    print('Rag unstructured file successfully uploaded')
    return FunctionResponse(message=Messages("success"))
  except Exception as e:
    print(f"❌ Error during indexing: {e}")
    return FunctionResponse(message=Messages(str(e)))

async def process_and_ingest_file(
    file_data: str | bytes,
    original_file_name: str,
    index_name: str,
    es_client: Any,
    aclient_openai: Any,
    user_provided_doc_summary: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """FastAPI-facing ingestion entrypoint used by app.main."""
    if isinstance(file_data, str):
        file_data = file_data.encode("utf-8")

    if not isinstance(file_data, (bytes, bytearray)):
        return {"status": "failed", "error": "file_data must be bytes"}

    final_params = dict(params or {})
    final_params["index_name"] = index_name
    final_params.setdefault("file_name", original_file_name)

    config = {
        "text_model": final_params.get("text_model", OPENAI_CHAT_MODEL),
    }

    try:
        doc_id = _generate_doc_id_from_content(bytes(file_data))
        await example_run_file_processing(
            file_data=bytes(file_data),
            original_file_name=original_file_name,
            document_id=doc_id,
            user_provided_doc_summary=user_provided_doc_summary,
            es_client=es_client,
            aclient_openai=aclient_openai,
            params=final_params,
            config=config,
        )
        return {
            "status": "success",
            "doc_id": doc_id,
            "index_name": index_name,
            "file_name": original_file_name,
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}