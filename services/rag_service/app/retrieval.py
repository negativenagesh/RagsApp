import os
import asyncio
import yaml
import logging
import traceback
import requests
import time
import httpx
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Awaitable, AsyncGenerator
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from datetime import datetime, timezone
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import TransportError

try:
    from sdk.response import FunctionResponse, Messages
    from sdk.message import Message
except ModuleNotFoundError:
    # Allow running from services/rag_service without manually exporting PYTHONPATH.
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from sdk.response import FunctionResponse, Messages
    from sdk.message import Message

load_dotenv()

OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_DIMENSIONS = 3072
RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL_ID", "BAAI/bge-reranker-base")

ELASTICSEARCH_URL = os.getenv("RAG_UPLOAD_ELASTIC_URL")                    
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")


def _extract_files_cited_in_answer(answer: str, candidate_files: set[str]) -> List[str]:
    """Extract cited file names from bracket citations and keep only known retrieved files."""
    if not answer or not candidate_files:
        return []

    cited: set[str] = set()
    for bracket_content in re.findall(r"\[([^\[\]]+)\]", answer):
        for raw_part in bracket_content.split(","):
            token = raw_part.strip().strip("`\"'")
            if not token:
                continue
            # Allow exact match first.
            if token in candidate_files:
                cited.add(token)
                continue
            # Then allow basename match.
            token_base = os.path.basename(token)
            for candidate in candidate_files:
                if token_base == os.path.basename(candidate):
                    cited.add(candidate)
                    break

    return sorted(cited)


def _sanitize_whatsapp_text(text: str, strip_leading_markdown: bool = False) -> str:
    """Normalize model output to WhatsApp-friendly formatting."""
    if not text:
        return ""

    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n")

    # Remove markdown heading syntax and horizontal rules.
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", cleaned)

    # Convert markdown bold (**text**) to WhatsApp-friendly bold (*text*).
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"*\1*", cleaned)

    if strip_leading_markdown:
        cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned)
        cleaned = re.sub(r"^\s*[-*_]{3,}\s*", "", cleaned)

    # Avoid emitting orphan heading fragments in streaming starts.
    if cleaned.strip() in {"#", "##", "###", "----", "***"}:
        return ""

    return cleaned

def safe_fire_and_forget(coro):
  try:
    loop = asyncio.get_running_loop()
    loop.create_task(ignore_exceptions(coro))
  except RuntimeError:
    pass  # No event loop available; skip silently

async def ignore_exceptions(coro):
  try:
    await coro
  except Exception:
    pass

async def init_async_openai_client(text_model) -> Optional[AsyncOpenAI]:
    openai_api_key = os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in .env. OpenAI client will not be functional.")
        return None

    try:
        client = AsyncOpenAI(api_key=openai_api_key)
        print("✅ AsyncOpenAI client initialized.")
        return client
    except Exception as e:
        print(f"❌ Failed to initialize AsyncOpenAI client: {e}")
        return None

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


class RAGFusionRetriever:
    def __init__(self, params: Any, config: Any, es_client: Any, aclient_openai: Optional[AsyncOpenAI], token: Any):
        self.aclient_openai = aclient_openai
        self.params = params
        self.config = config
        self.es_client = es_client
        self.token = token
        self.reranker = None
        self.provence_pruner = None
        self.embedding_dims = OPENAI_EMBEDDING_DIMENSIONS
        self.deep_research = self.params.get('deep_research', False)

        # Index configuration — default to 'ragsapp'
        index_name_config = config.get('index_name', 'ragsapp')
        self.index_names = [idx.strip() for idx in index_name_config.split(',') if idx.strip()]
        if not self.index_names:
            self.index_names = ['ragsapp']
        self.index_name = self.index_names[0]
        print(f"Configured indexes: {self.index_names}")

    def _get_conversation_history(self) -> List[Dict[str, str]]:
        raw_history = self.params.get("conversation_history") or []
        if not isinstance(raw_history, list):
            return []

        normalized: List[Dict[str, str]] = []
        for item in raw_history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user")).strip().lower()
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    def _render_conversation_history(self, max_turns: int = 8, max_chars: int = 1600) -> str:
        history = self._get_conversation_history()
        if not history:
            return "(none)"

        lines: List[str] = []
        for turn in history[-max_turns:]:
            role = turn.get("role", "user")
            prefix = "User" if role == "user" else "Assistant"
            content = re.sub(r"\s+", " ", turn.get("content", "")).strip()
            if len(content) > 260:
                content = content[:257].rstrip() + "..."
            lines.append(f"{prefix}: {content}")

        rendered = "\n".join(lines) if lines else "(none)"
        if len(rendered) > max_chars:
            rendered = rendered[-max_chars:]
        return rendered

    def _load_prompt_template(self, prompt_name: str) -> str:
        try:
            prompt_file_path = Path("./prompts") / f"{prompt_name}.yaml"
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


    async def _call_openai_api(
        self,
        model_name: str,
        payload_messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.1,
        stream_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        """A unified async method to call OpenAI text models with retry logic."""
        if not self.aclient_openai:
            print("❌ OpenAI client not configured. Cannot make API call.")
            return ""

        print(f"*** openai messages: {payload_messages}")
        max_retries = 10
        base_delay_seconds = 10

        for attempt in range(max_retries):
            try:
                start_time = datetime.now(timezone.utc)
                api_call_start = time.time()
                if stream_callback:
                    response = await self.aclient_openai.chat.completions.create(
                        model=model_name,
                        messages=payload_messages,
                        max_tokens=max_tokens,
                        stream=True,
                    )
                    content_parts: List[str] = []
                    async for event in response:
                        if not event.choices:
                            continue
                        delta = event.choices[0].delta.content
                        if not delta:
                            continue
                        content_parts.append(delta)
                        try:
                            await stream_callback(delta)
                        except Exception as callback_error:
                            print(f"Stream callback failed: {callback_error}")
                    content = "".join(content_parts)
                else:
                    response = await self.aclient_openai.chat.completions.create(
                        model=model_name,
                        messages=payload_messages,
                        max_tokens=max_tokens,
                    )
                    content = response.choices[0].message.content

                api_call_duration = time.time() - api_call_start
                
                # if response.usage:
                #     safe_fire_and_forget(calculatePriceByApi(self.config, self.params, response.usage, start_time, other_details=None, token=self.token))

                if content:
                    print(f"✅ OpenAI API call successful in {api_call_duration:.2f}s. Preview: {content[:100].strip()}...")
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


    async def _prune_documents(self, query: str, documents: List[Dict[str, Any]], doc_type: str) -> List[Dict[str, Any]]:
        if not self.provence_pruner:
            print("Provence pruner not initialized. Skipping pruning.")
            return documents
        if not documents:
            print(f"No {doc_type} documents to prune for query: '{query[:50]}...'")
            return []

        print(f"Pruning {len(documents)} {doc_type} documents with Provence for query: '{query[:50]}...'")
        
        text_key = "text" if doc_type == "chunk" else "chunk_text"
        
        pruned_docs = []
        for doc_idx,doc in enumerate(documents):
            original_text = doc.get(text_key, "")
            if not original_text:
                pruned_docs.append(doc) # Keep doc as is if no text
                continue

            try:
                # Run synchronous model inference in a thread
                provence_output = await asyncio.to_thread(
                    self.provence_pruner.process,
                    question=query,
                    context=original_text,
                    threshold=0.1, # Recommended conservative threshold
                    always_select_title=True
                )
                pruned_text = provence_output.get('pruned_context', original_text)
                
                doc_copy = doc.copy()
                doc_copy[text_key] = pruned_text
                # Optionally store the provence score if needed later
                doc_copy['provence_score'] = provence_output.get('reranking_score')
                pruned_docs.append(doc_copy)
                
                if len(original_text) != len(pruned_text):
                    print(f"  - Pruned content for doc #{doc_idx+1} (File: {doc.get('file_name', 'N/A')}, Page: {doc.get('page_number', 'N/A')}). Original len: {len(original_text)}, Pruned len: {len(pruned_text)}")
                else:
                    print(f"  - No content pruned for doc #{doc_idx+1} (File: {doc.get('file_name', 'N/A')}, Page: {doc.get('page_number', 'N/A')}).")
                
                print(f"    - Pruned Text: '{pruned_text}'")
                
            except Exception as e:
                print(f"  - Error pruning document with Provence: {e}. Using original text.")
                pruned_docs.append(doc) # Append original doc on error

        print(f"Successfully pruned {len(documents)} {doc_type} documents.")
        return pruned_docs
    
    async def _process_single_subquery(
        self,
        sq_text: str,
        initial_candidate_pool_size: int,
        top_k_kg_entities: int
    ) -> List[Dict[str, Any]]:
        """Process a single subquery: retrieves chunks via semantic + keyword search,
        then performs internal RRF fusion to produce ONE normalized ranked list.
        The caller collects fused lists from all subqueries and performs cross-subquery RRF.
        """
        print(f"\n--- Processing Subquery: '{sq_text}' ---")
        ranked_lists: List[List[Dict[str, Any]]] = []  # Raw lists collected, fused before return
        
        try:
            # Generate embedding for the subquery
            embedding_result = await self._generate_embedding([sq_text])
            if isinstance(embedding_result, Exception):
                embedding_result = []
            
            if isinstance(embedding_result, Exception) or not embedding_result or not embedding_result[0]:
                print(f"⚠️ Failed to generate embedding for subquery: '{sq_text}'. Keyword-only retrieval.")
                # No semantic search possible — keyword search only
                keyword_tasks = [
                    self._keyword_search_chunks(sq_text, initial_candidate_pool_size)
                ]
                keyword_results = await asyncio.gather(*keyword_tasks, return_exceptions=True)
                for res in keyword_results:
                    if isinstance(res, list) and res:
                        ranked_lists.append(res)
                # Internal RRF: normalize keyword-only signals into one fused list
                if ranked_lists:
                    fused = self._fuse_ranked_lists_with_rrf(ranked_lists, top_k=initial_candidate_pool_size)
                    print(f"Internal RRF fused {len(ranked_lists)} keyword-only lists into {len(fused)} chunks for subquery: '{sq_text[:50]}...'")
                    return fused
                return []
                
            query_embedding = embedding_result[0]

            try:
                # Run semantic search and single raw keyword search concurrently
                semantic_task = self._semantic_search_chunks(query_embedding, initial_candidate_pool_size)
                keyword_tasks = [
                    self._keyword_search_chunks(sq_text, initial_candidate_pool_size)
                ]
                
                all_tasks = [semantic_task] + keyword_tasks
                results = await asyncio.gather(*all_tasks, return_exceptions=True)
                
                # First result is semantic search
                if isinstance(results[0], list) and results[0]:
                    ranked_lists.append(results[0])
                    print(f"Semantic search returned {len(results[0])} chunks for subquery: '{sq_text[:50]}...'")
                elif isinstance(results[0], Exception):
                    print(f"⚠️ Semantic search failed for subquery '{sq_text[:50]}...': {results[0]}")
                
                # Remaining result is the unified raw keyword search
                if isinstance(results[1], list) and results[1]:
                    ranked_lists.append(results[1])
                    print(f"Keyword search returned {len(results[1])} chunks for subquery: '{sq_text[:50]}...'")
                elif isinstance(results[1], Exception):
                    print(f"⚠️ Keyword search failed for subquery '{sq_text[:50]}...': {results[1]}")
                        
            except Exception as e:
                print(f"⚠️ Search failed for subquery '{sq_text}': {e}")
                    
        except Exception as e:
            print(f"⚠️ Error processing subquery '{sq_text}': {e}")
        
        # Normalize semantic + keyword signals into ONE fused list for this subquery.
        if ranked_lists:
            fused = self._fuse_ranked_lists_with_rrf(ranked_lists, top_k=initial_candidate_pool_size)
            print(f"Internal RRF fused {len(ranked_lists)} lists into {len(fused)} chunks for subquery: '{sq_text[:50]}...'")
            return fused
        
        print(f"Subquery '{sq_text[:50]}...' produced no results")
        return []

    async def _generate_subqueries(
        self,
        original_query: str,
        num_subqueries: int = 2
    ) -> List[Dict[str, Any]]:
        """Generate subqueries via a fast LLM call and extract keywords locally.
        Returns a list of dicts: [{"query": str, "keywords": [str, ...]}, ...]
        """
        system_prompt = f"""You are an expert at search query decomposition for information retrieval.
Given the user's original query, break down the intent and generate exactly {num_subqueries} distinct search subqueries.
- Each subquery must target a distinct aspect/intent of the original query.
- Each subquery should be self-contained and searchable independently.
- Output ONLY the subqueries, separated by a newline. Do not output anything else. Do not use bullets or numbers.
"""
        user_prompt = f"""**Original User Query:** "{original_query}" """

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]

        llm_response_content = ""
        print(f"Generating {num_subqueries} subqueries via text endpoint for: '{original_query}'")
        llm_response_content = await self._call_openai_api(
            model_name=OPENAI_CHAT_MODEL,
            payload_messages=messages,
            max_tokens=150,
            temperature=0.3
        )

        if not llm_response_content:
            print("⚠️ LLM returned empty content for subqueries.")
            return []

        # Parse plaintext response
        lines = [line.strip() for line in llm_response_content.split('\n') if line.strip()]
        
        # Clean potential bullets/numberings if the LLM adds them despite instructions
        clean_lines = []
        for line in lines:
            # removing '1.', '-', '*', etc. at the start
            cleaned = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
            # remove surrounding double quotes if LLM added them
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1].strip()
            if cleaned:
                clean_lines.append(cleaned)
                
        results: List[Dict[str, Any]] = []
        seen_queries = set()
        for query_text in clean_lines[:num_subqueries]:
            q_key = query_text.lower()
            if q_key in seen_queries:
                continue
            seen_queries.add(q_key)

            results.append({
                "query": query_text
            })

        if results:
            print(f"✅ Generated {len(results)} subqueries directly:")
            for i, r in enumerate(results):
                print(f"   Subquery {i+1}: '{r['query']}'")
        else:
            print("⚠️ Could not parse subqueries text. Returning empty.")

        return results

    async def _generate_embedding(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        
        if not self.aclient_openai:
            print("OpenAI client not available. Cannot generate embeddings.")
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
                    dimensions=self.embedding_dims  #changed
                )
                all_embeddings.extend([item.embedding for item in response.data])
            return all_embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [[] for _ in texts]

    async def _semantic_search_chunks(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        if not query_embedding:
            print("Semantic search skipped: No query embedding provided.")
            return []

        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": top_k * 10
        }
        
        async def search_single_index(index_name: str) -> List[Dict[str, Any]]:
            """Perform semantic search on a single index and retrieve all fields."""
            # Known fields to exclude from custom_fields
            KNOWN_FIELDS = {'chunk_text', 'embedding', 'metadata'}
            
            try:
                response = await self.es_client.search(
                    index=index_name,
                    knn=knn_query,
                    size=top_k,
                    _source_excludes=["embedding"]  # Retrieve all fields except embedding
                )
                results = []
                for hit in response.get('hits', {}).get('hits', []):
                    source = hit.get('_source', {})
                    metadata = source.get('metadata', {})
                    
                    # Extract custom fields (fields not in KNOWN_FIELDS)
                    custom_fields = {k: v for k, v in source.items() if k not in KNOWN_FIELDS}
                    
                    results.append({
                        "id": hit.get('_id'),
                        "source_index": index_name,  # Track which index this came from
                        "text": source.get('chunk_text'),
                        "score": hit.get('_score'),
                        "file_name": metadata.get('file_name'),
                        "doc_id": metadata.get('doc_id'),
                        "page_number": metadata.get('page_number'),
                        "chunk_index_in_page": metadata.get('chunk_index_in_page'),
                        "custom_fields": custom_fields
                    })
                print(f"Semantic search found {len(results)} chunks in {index_name}.")
                return results
            except TransportError as e:
                print(f"Elasticsearch semantic search error on index {index_name}: {e}")
                return []
        
        print(f'Performing semantic search across indexes: {self.index_names}')
        
        # Search all indexes concurrently
        tasks = [search_single_index(idx) for idx in self.index_names]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results - use composite key (source_index + id) for deduplication
        # This ensures same document in different indexes is kept as separate results
        all_results = []
        seen_composite_ids = set()
        for result in results_list:
            if isinstance(result, list):
                for item in result:
                    # Use composite key: source_index + document_id
                    composite_key = f"{item.get('source_index', '')}:{item['id']}"
                    if composite_key not in seen_composite_ids:
                        seen_composite_ids.add(composite_key)
                        all_results.append(item)
        
        # Sort by score and return top_k
        all_results.sort(key=lambda x: x.get('score', 0) or 0, reverse=True)
        print(f"Semantic search found {len(all_results)} chunks across {len(self.index_names)} indexes.")
        return all_results[:top_k]

    async def _structured_kg_search(self, query_embedding: List[float], top_k: int, top_k_entities: int) -> List[Dict[str, Any]]:
        """
        Search for entities in the knowledge graph using embeddings of their descriptions.
        
        Args:
            query_embedding: Vector embedding for the query
            top_k: Number of results to return
            top_k_entities: Number of entities to return
            
        Returns:
            List of matching entity information with scores
        """
        try:
            index_name = self.config.get('index_name')
                        
            # Create the main query body
            query_body = {
                "size": top_k,
                "_source": {
                    "includes": [
                        "metadata.entities",
                        "metadata.file_name",
                        "metadata.doc_id",
                        "metadata.page_number",
                        "metadata.chunk_index_in_page",
                        "chunk_text"
                    ]
                },
                "query": {
                    "nested": {
                        "path": "metadata.entities",
                        "score_mode": "max",
                        "query": {
                            "bool": {
                                "must": [
                                    {
                                        "exists": {
                                            "field": "metadata.entities.description_embedding"
                                        }
                                    },
                                    {
                                        "script_score": {
                                            "query": {"match_all": {}},
                                            "script": {
                                                "source": "cosineSimilarity(params.query_vector, 'metadata.entities.description_embedding') + 1.0",
                                                "params": {
                                                    "query_vector": query_embedding
                                                }
                                            }
                                        }
                                    }
                                ]
                            }
                        },
                        "inner_hits": {
                            "size": 3,
                            "_source": ["name", "type", "description"]
                        }
                    }
                }
            }
            
            # Execute search using the Elasticsearch client
            response = await self.es_client.search(
                index=index_name,
                body=query_body
            )
            
            results = []
            
            if response["hits"]["total"]["value"] > 0:
                print(f"Found {response['hits']['total']['value']} documents with entity embeddings in index.")
                
                # Process results
                for hit in response["hits"]["hits"]:
                    score = hit["_score"]
                    source = hit["_source"]
                    metadata = source.get("metadata", {})
                    doc_id = metadata.get("doc_id", "unknown")
                    file_name = metadata.get("file_name", "unknown")
                    page_number = metadata.get("page_number")
                    chunk_index_in_page = metadata.get("chunk_index_in_page")
                    chunk_text = source.get("chunk_text", "")
                    
                    # Process inner hits (matching entities)
                    if "inner_hits" in hit and "metadata.entities" in hit["inner_hits"]:
                        entity_hits = hit["inner_hits"]["metadata.entities"]["hits"]["hits"]
                        
                        entities = []
                        relationships = metadata.get("relationships", [])
                        
                        for entity_hit in entity_hits:
                            entity = entity_hit["_source"]
                            entities.append({
                                "name": entity.get("name", ""),
                                "type": entity.get("type", ""),
                                "description": entity.get("description", "")
                            })
                        
                        result_item = {
                            "id": hit["_id"],
                            "score": score,
                            "doc_id": doc_id,
                            "file_name": file_name,
                            "page_number": page_number,
                            "chunk_index_in_page": chunk_index_in_page,
                            "chunk_text": chunk_text,
                            "entities": entities,
                            "relationships": relationships
                        }
                        
                        results.append(result_item)
            else:
                print("No documents with entity embeddings found. Check your data processing pipeline.")
                
            # Sort by score (highest first)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Return top-k results
            final_results = results[:top_k_entities]
            print(f"KG search returning {len(final_results)} results out of {len(results)} found.")
            
            return final_results
            
        except Exception as e:
            print(f"Error in structured KG search: {e}")
            traceback.print_exc()
            return []

    def _fuse_ranked_lists_with_rrf(self, ranked_lists: List[List[Dict[str, Any]]], k_rrf: int = 60, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fuse multiple ranked lists with Reciprocal Rank Fusion (RRF) using composite keys.
        """
        if not ranked_lists:
            return []

        def get_composite_key(doc: Dict[str, Any]) -> str:
            source_index = doc.get('source_index', '')
            doc_id = doc.get('id', '')
            return f"{source_index}:{doc_id}"

        all_docs: Dict[str, Dict[str, Any]] = {}
        rank_maps: List[Dict[str, int]] = []

        for ranked_list in ranked_lists:
            rank_map: Dict[str, int] = {}
            for idx, doc in enumerate(ranked_list):
                composite_key = get_composite_key(doc)
                rank_map[composite_key] = idx + 1
                if composite_key not in all_docs:
                    all_docs[composite_key] = doc
            rank_maps.append(rank_map)

        for composite_key, doc in all_docs.items():
            score = 0.0
            for rank_map in rank_maps:
                if composite_key in rank_map:
                    score += 1.0 / (k_rrf + rank_map[composite_key])
            doc['score'] = score

        fused_docs = sorted(all_docs.values(), key=lambda x: x.get('score', 0) or 0, reverse=True)
        return fused_docs[:top_k] if top_k else fused_docs



    async def _keyword_search_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a precise keyword (phrase) search on the 'chunk_text' field.
        Searches all indexes concurrently when multiple indexes are configured.
        """
        if not query:
            print("Keyword search skipped: No query provided.")
            return []

        keyword_query = {
            "match": {
                "chunk_text": {
                    "query": query,
                    "operator": "or",
                    "fuzziness": "AUTO"
                }
            }
        }
        print(f"Performing keyword (match) search with top_k={top_k}. Query: {query}")

        async def search_single_index(index_name: str) -> List[Dict[str, Any]]:
            """Search a single index and return results with all fields."""
            # Known fields to exclude from custom_fields
            KNOWN_FIELDS = {'chunk_text', 'embedding', 'metadata'}
            
            try:
                response = await self.es_client.search(
                    index=index_name,
                    query=keyword_query,
                    size=top_k,
                    _source_excludes=["embedding"]  # Retrieve all fields except embedding
                )
                results = []
                for hit in response.get('hits', {}).get('hits', []):
                    source = hit.get('_source', {})
                    metadata = source.get('metadata', {})
                    
                    # Extract custom fields (fields not in KNOWN_FIELDS)
                    custom_fields = {k: v for k, v in source.items() if k not in KNOWN_FIELDS}
                    
                    results.append({
                        "id": hit.get('_id'),
                        "source_index": index_name,  # Track which index this came from
                        "text": source.get('chunk_text'),
                        "score": hit.get('_score'),
                        "file_name": metadata.get('file_name'),
                        "doc_id": metadata.get('doc_id'),
                        "page_number": metadata.get('page_number'),
                        "chunk_index_in_page": metadata.get('chunk_index_in_page'),
                        "custom_fields": custom_fields
                    })
                print(f"Keyword search found {len(results)} chunks in {index_name}.")
                return results
            except TransportError as e:
                print(f"Elasticsearch keyword search error for {index_name}: {e}")
                return []
        
        # Search all indexes concurrently
        search_tasks = [search_single_index(idx) for idx in self.index_names]
        results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Merge results - use composite key (source_index + id) for deduplication
        all_results = []
        seen_composite_ids = set()
        for result in results_lists:
            if isinstance(result, Exception):
                print(f"Keyword search task failed: {result}")
                continue
            if isinstance(result, list):
                for item in result:
                    # Use composite key: source_index + document_id
                    composite_key = f"{item.get('source_index', '')}:{item['id']}"
                    if composite_key not in seen_composite_ids:
                        seen_composite_ids.add(composite_key)
                        all_results.append(item)
        
        # Sort by score and return top_k
        all_results.sort(key=lambda x: x.get('score', 0) or 0, reverse=True)
        print(f"Keyword search found {len(all_results)} chunks across {len(self.index_names)} indexes.")
        return all_results[:top_k]

    async def _keyword_search_kg(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a keyword search and retrieves full knowledge graph data.
        """
        if not query:
            return []

        keyword_query = {"match": {"chunk_text": {"query": query, "fuzziness": "AUTO"}}}
        index_name = self.config.get('index_name')
        print(f"Performing KG keyword search with top_k={top_k}. Query: {query}")

        try:
            response = await self.es_client.search(
                index=index_name,
                query=keyword_query,
                size=top_k,
                _source_includes=["chunk_text", "metadata"]
            )
            results = []
            for hit in response.get('hits', {}).get('hits', []):
                source = hit.get('_source', {})
                metadata = source.get('metadata', {})
                
                result_item = {
                    "id": hit.get('_id'),
                    "chunk_text": source.get('chunk_text'),
                    "entities": metadata.get('entities', []),
                    "relationships": metadata.get('relationships', []),
                    "score": hit.get('_score'),
                    "file_name": metadata.get('file_name'),
                    "doc_id": metadata.get('doc_id'),
                    "page_number": metadata.get('page_number'),
                    "chunk_index_in_page": metadata.get('chunk_index_in_page')
                }
                
                results.append(result_item)
                
            print(f"KG keyword search found {len(results)} chunks.")
            return results
        except TransportError as e:
            print(f"Elasticsearch KG keyword search error: {e}")
        return []

    async def _rerank_documents(self, query: str, documents: List[Dict[str, Any]], doc_type: str, absolute_score_floor: float = 0.3) -> List[Dict[str, Any]]:
        if not documents:
            print(f"No {doc_type} documents to rerank for query: '{query[:50]}...'")
            return []
        if not self.reranker:
            print(f"Reranker not initialized. Skipping reranking for {doc_type} documents.")
            for doc in documents:
                if 'rerank_score' not in doc:
                    doc['rerank_score'] = doc.get('score') 
            return documents

        print(f"Reranking {len(documents)} {doc_type} documents for query: '{query[:50]}...'")

        if doc_type == "chunk":
            pairs = [(query, doc.get("text", "")) for doc in documents]
            text_key_for_logging = "text"
        elif doc_type == "kg":
            pairs = [(query, doc.get("chunk_text", "")) for doc in documents]
            text_key_for_logging = "chunk_text"
        else:
            print(f"Unknown document type '{doc_type}' for reranking. Skipping.")
            return documents

        try:
            scores = await asyncio.to_thread(self.reranker.predict, pairs, batch_size=8)
            print(f"Successfully got scores for {len(pairs)} pairs for {doc_type} reranking.")
        except Exception as e:
            print(f"Error during reranking {doc_type} documents with CrossEncoder: {e}")
            # If reranking fails, ensure 'rerank_score' is present, set to original score or None
            for doc in documents:
                if 'rerank_score' not in doc:
                    doc['rerank_score'] = doc.get('score')
            return documents

        docs_with_scores = [{'doc': doc, 'score': scores[i]} for i, doc in enumerate(documents)]
        docs_with_scores.sort(key=lambda x: x['score'], reverse=True)

        # 1. Print all document scores in descending order
        print(f"\n--- Initial Reranked Scores for all {len(docs_with_scores)} documents ---")
        for i, item in enumerate(docs_with_scores):
            content_to_log = item['doc'].get(text_key_for_logging, '')
            print(f"  Rank {i+1}: Score={item['score']:.4f} | Content: '{content_to_log[:70]}...'")
        print("--- End of Initial Scores ---\n")

       # --- MODIFIED LOGIC: APPLY ABSOLUTE FLOOR FIRST ---
        # 1. Apply Absolute Score Cutoff as a primary quality gate.
        docs_passing_floor = [
            item for item in docs_with_scores if item['score'] >= absolute_score_floor
        ]
        print(f"Applied absolute score floor of {absolute_score_floor}. {len(docs_passing_floor)} of {len(docs_with_scores)} documents passed the quality gate.")

        if not docs_passing_floor:
            print("No documents met the absolute score floor. Returning empty list.")
            return []

        # --- HYBRID ELBOW METHOD LOGIC on the filtered set ---
        if len(docs_passing_floor) <= 2:
            print("Fewer than 3 documents passed the floor, returning all of them.")
            final_docs_with_scores = docs_passing_floor
        else:
            sorted_scores = [item['score'] for item in docs_passing_floor]
            
            score_diffs = [
                (sorted_scores[i] - sorted_scores[i+1]) / (sorted_scores[i] + 1e-9)
                for i in range(len(sorted_scores) - 1)
            ]

            elbow_index = 0
            if score_diffs:
                elbow_index = np.argmax(score_diffs)
            
            if score_diffs:
                elbow_doc = docs_passing_floor[elbow_index]
                elbow_score = elbow_doc['score']
                next_score = docs_passing_floor[elbow_index + 1]['score'] if (elbow_index + 1) < len(docs_passing_floor) else -1
                elbow_content = elbow_doc['doc'].get(text_key_for_logging, '')
                print(f"Elbow point detected after document at Rank {elbow_index + 1} (among docs that passed the floor).")
                print(f"  - Elbow Doc Score: {elbow_score:.4f} -> Next Doc Score: {next_score:.4f}")
                print(f"  - Elbow Doc Content: '{elbow_content[:70]}...'")

            num_to_keep = elbow_index + 1
            
            min_docs_to_keep = 3
            if num_to_keep < min_docs_to_keep and len(docs_passing_floor) >= min_docs_to_keep:
                print(f"Elbow method suggested keeping {num_to_keep}, but minimum is {min_docs_to_keep}. Adjusting to keep top {min_docs_to_keep}.")
                num_to_keep = min_docs_to_keep
            
            print(f"Dynamically selecting top {num_to_keep} documents after elbow analysis.")
            final_docs_with_scores = docs_passing_floor[:num_to_keep]
        
        reranked_docs = []
        print(f"\n--- Final Selected Documents (Top {len(final_docs_with_scores)} selected by hybrid method) ---")
        for i, item in enumerate(final_docs_with_scores):
            doc = item['doc']
            score = float(item['score'])
            doc['rerank_score'] = score
            content_to_log = doc.get(text_key_for_logging, '')
            print(f"  Final Rank {i+1}: Score={doc['rerank_score']:.4f} | Source: {doc.get('file_name', 'N/A')}, Page: {doc.get('page_number', 'N/A')} | Content: '{content_to_log[:70]}...'")
            reranked_docs.append(doc)
        print("--- End of Final Selected Documents ---\n")

        print(f"Successfully reranked and selected {len(reranked_docs)} {doc_type} documents using the hybrid elbow method after applying score floor.")
        return reranked_docs

    async def _perform_semantic_search_for_subquery(self, subquery_text: str, top_k: int) -> List[Dict[str, Any]]:
        print(f"Performing semantic search for subquery: '{subquery_text}'")
        embedding_list = await self._generate_embedding([subquery_text])
        if not embedding_list or not embedding_list[0]:
            print(f"Could not generate embedding for subquery: '{subquery_text}'. Semantic search will yield no results.")
            return []
        query_embedding= embedding_list[0]  # Get the first embedding if multiple texts were passed
        return await self._semantic_search_chunks(query_embedding, top_k)

    async def _perform_kg_search_for_subquery(self, subquery_text: str, top_k: int, top_k_entities:int) -> List[Dict[str, Any]]:
        print(f"Performing KG search for subquery: '{subquery_text}'")
        embedding_list= await self._generate_embedding([subquery_text])
        if not embedding_list or not embedding_list[0]:
            print(f"Could not generate embedding for subquery: '{subquery_text}'. KG search will yield no results.")
            return []
        query_embedding = embedding_list[0]  # Get the first embedding if multiple texts were passed
        # return await self._structured_kg_search(query_embedding, top_k, top_k_entities)
        return []

    def _generate_shorthand_id(self, item: Dict[str, Any], prefix: str, index: int) -> str:
        doc_id_part = "unknown"
        if item.get("doc_id"):
            doc_id_part = str(item["doc_id"]).replace('-', '')[:6]
        
        page_num_val = item.get("page_number")
        page_num_part = str(page_num_val) if page_num_val is not None else "NA"
        
        chunk_idx_val = item.get("chunk_index_in_page")
        chunk_idx_part = str(chunk_idx_val) if chunk_idx_val is not None else str(index)
        
        return f"{prefix}_{doc_id_part}_p{page_num_part}_i{chunk_idx_part}"

    def _format_search_results_for_llm(self, original_query: str, sub_queries_results: List[Dict[str, Any]]) -> str:
        lines = [
            "=== ORIGINAL USER QUERY ===",
            original_query,
            "",
            "=== RETRIEVAL CONTEXT GROUPED BY SUBQUERY ===",
            "",
        ]
        
        if not sub_queries_results:
            lines.append("No search results found.")
            return "\n".join(lines)

        for sq_idx, sq_result in enumerate(sub_queries_results):
            sub_query_text = sq_result.get("sub_query_text", f"Sub-query {sq_idx + 1}")
            reranked_chunks = sq_result.get("reranked_chunks", [])
            retrieved_kg_data = sq_result.get("retrieved_kg_data", [])

            lines.append(f"--- SUBQUERY {sq_idx + 1} ---")
            lines.append(f"Subquery: {sub_query_text}")
            lines.append(f"Retrieved chunk count: {len(reranked_chunks)}")
            lines.append(f"Retrieved KG item count: {len(retrieved_kg_data)}")

            
            # Create a mapping of KG data by (source_index, file_name, page_number, chunk_index_in_page)
            # Include source_index to preserve KG data from different indexes
            kg_by_location = {}
            for kg_item in retrieved_kg_data:
                if not isinstance(kg_item, dict):
                    continue
                source_index = kg_item.get('source_index', '')
                file_name = kg_item.get('file_name', '')
                page_number = kg_item.get('page_number')
                chunk_index = kg_item.get('chunk_index_in_page')
                # Include source_index in key for multi-index support
                key = (source_index, file_name, page_number, chunk_index)
                kg_by_location[key] = kg_item
            
            if reranked_chunks:
                lines.append("\nVector Search Results (Chunks):")
                for chunk_idx, chunk in enumerate(reranked_chunks):
                    if not isinstance(chunk, dict):
                        print(f"Skipping non-dict chunk item during formatting: {chunk}")
                        continue

                    shorthand_id = self._generate_shorthand_id(chunk, "c", chunk_idx)
                    score_val = chunk.get('rerank_score', chunk.get('score'))
                    score_str = f"{score_val:.4f}" if score_val is not None else "N/A"
                    lines.append(f"Source ID [{shorthand_id}]: (Score: {score_str})")
                    
                    text_content = chunk.get("text") or chunk.get("chunk_text", "N/A")
                    lines.append(text_content)
                    lines.append(f"  File: {chunk.get('file_name', 'N/A')}, Page: {chunk.get('page_number', 'N/A')}, Chunk Index in Page: {chunk.get('chunk_index_in_page', 'N/A')}")
                    
                    # Display custom fields if present
                    custom_fields = chunk.get('custom_fields', {})
                    if custom_fields:
                        lines.append("  Custom Fields:")
                        for field_name, field_value in custom_fields.items():
                            lines.append(f"    {field_name}: {field_value}")
                    
                    # Check if there's KG data for this chunk
                    # The chunk's source_index is the main index (e.g., 'fields_test')
                    # The KG's source_index is the KG index (e.g., 'fields_test_kg_index')
                    # We need to map chunk's index to corresponding KG index name
                    chunk_source_index = chunk.get('source_index', '')
                    # Derive the expected KG index name from chunk's source index
                    expected_kg_source_index = f"{chunk_source_index}_kg_index" if chunk_source_index else ''
                    chunk_file = chunk.get('file_name', '')
                    chunk_page = chunk.get('page_number')
                    chunk_index = chunk.get('chunk_index_in_page')
                    # Use the derived KG index name for lookup
                    chunk_key = (expected_kg_source_index, chunk_file, chunk_page, chunk_index)
                    
                    if chunk_key in kg_by_location:
                        kg_item = kg_by_location[chunk_key]
                        score_val = kg_item.get('rerank_score', kg_item.get('score'))
                        score_str = f"{score_val:.4f}" if score_val is not None else "N/A"
                        
                        lines.append(f"  Knowledge Graph for this chunk (Score: {score_str}):")
                        
                        entities = kg_item.get("entities", [])
                        if entities:
                            lines.append("    Entities:")
                            for entity in entities:
                                if not isinstance(entity, dict): continue
                                lines.append(f"      - Name: {entity.get('name', 'N/A')}, Type: {entity.get('type', 'N/A')}")
                                entity_desc = entity.get('description', '')
                                if entity_desc:
                                    lines.append(f"        Description: {entity_desc}")
                        
                        relationships = kg_item.get("relationships", [])
                        if relationships:
                            lines.append("    Relationships:")
                            for rel in relationships:
                                if not isinstance(rel, dict): continue
                                lines.append(f"      - {rel.get('source_entity', 'S')} -> {rel.get('relation', 'R')} -> {rel.get('target_entity', 'T')} (Weight: {rel.get('relationship_weight', 'N/A')})")
                                rel_desc = rel.get("relationship_description", "")
                                if rel_desc:
                                    lines.append(f"        Description: {rel_desc}")
                    
                    lines.append("")  # Blank line between chunks
            else:
                lines.append("\nNo vector search results for this sub-query.")
            
            lines.append("")
        
        return "\n".join(lines)

    async def search(
        self,
        user_query: str,
        num_subqueries: int = 2,
        initial_candidate_pool_size: int = 50,
        top_k_kg_entities: int = 8,
        absolute_score_floor: float = 0.3,
        stream_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        print(f"\n{'='*80}")
        print(f"⏱️  Starting RAG Fusion search for user query: '{user_query[:60]}...'")
        print(f"{'='*80}\n")
        retrieval_start_time = time.time()

        setup_start_time = time.time()
        
        # Direct retrieval — no subquery decomposition needed for WhatsApp-grade latency
        subquery_data = [{"query": user_query}]
        
        setup_duration = time.time() - setup_start_time
        print(f"✅ Setup completed in {setup_duration:.2f}s\n")
        
        print(f"⏱️  [2/3] Processing {len(subquery_data)} subqueries concurrently...")
        subquery_processing_start = time.time()
        
        # Process all subqueries concurrently — each returns ONE internally-fused ranked list
        all_ranked_lists: List[List[Dict[str, Any]]] = []
        
        if subquery_data:
            subquery_tasks = [
                self._process_single_subquery(
                    sq_entry["query"],
                    initial_candidate_pool_size,
                    top_k_kg_entities
                )
                for sq_entry in subquery_data
            ]
            
            # Execute all subquery processing tasks concurrently
            subquery_results = await asyncio.gather(*subquery_tasks, return_exceptions=True)
            
            for idx, result in enumerate(subquery_results):
                sq_text = subquery_data[idx]["query"]
                if isinstance(result, Exception):
                    print(f"⚠️ Subquery '{sq_text[:50]}...' processing failed: {result}")
                    continue
                if isinstance(result, list) and result:
                    all_ranked_lists.append(result)
                    print(f"Subquery '{sq_text[:50]}...' contributed 1 fused list ({len(result)} chunks)")

        subquery_processing_duration = time.time() - subquery_processing_start
        print(f"✅ Subquery processing completed in {subquery_processing_duration:.2f}s")
        print(f"Total fused lists for cross-subquery RRF: {len(all_ranked_lists)}\n")

        # CROSS-SUBQUERY RRF FUSION (Tier 2)
        # Each subquery has already normalized its internal signals (semantic + keyword) via
        # per-subquery RRF (Tier 1). Now fuse across subqueries so documents appearing in
        # multiple subquery perspectives get properly boosted.
        print("⏱️  Performing cross-subquery RRF fusion...")
        rrf_start_time = time.time()
        
        fused_chunks = self._fuse_ranked_lists_with_rrf(
            all_ranked_lists,
            top_k=initial_candidate_pool_size
        )
        
        rrf_fusion_duration = time.time() - rrf_start_time
        print(f"✅ Unified RRF fusion completed in {rrf_fusion_duration:.2f}s — {len(fused_chunks)} unique chunks after fusion")

        # Separate regular chunks from KG entries
        retrieved_chunks = []
        retrieved_kg_evidence_with_chunk_text = []
        
        for chunk_data in fused_chunks:
            if chunk_data.get('entities') or chunk_data.get('relationships'):
                retrieved_kg_evidence_with_chunk_text.append(chunk_data)
            else:
                retrieved_chunks.append(chunk_data)
        
        print(f"Separated {len(retrieved_chunks)} chunks and {len(retrieved_kg_evidence_with_chunk_text)} KG entries from unified RRF results")
                
        # Deep research: reranking and pruning
        rerank_start_time = time.time() 
        if self.reranker and self.deep_research:
            print(f"⏱️  [3/3] Deep research is ON. Reranking and pruning the candidate pool...")
            retrieved_chunks = await self._rerank_documents(user_query, retrieved_chunks, "chunk", absolute_score_floor)
            if self.provence_pruner:
                retrieved_chunks = await self._prune_documents(user_query, retrieved_chunks, "chunk")

            retrieved_kg_evidence_with_chunk_text = await self._rerank_documents(user_query, retrieved_kg_evidence_with_chunk_text, "kg", absolute_score_floor)
            if self.provence_pruner:
                 retrieved_kg_evidence_with_chunk_text = await self._prune_documents(user_query, retrieved_kg_evidence_with_chunk_text, "kg")
            rerank_duration = time.time() - rerank_start_time
            print(f"✅ Reranking and pruning completed in {rerank_duration:.2f}s\n")
        else:
            print("⏱️  [3/3] Deep research is OFF. Skipping reranking and pruning.\n")

        # Build unified results for LLM context formatting.
        # Since all results are globally ranked via a single unified RRF, present as one block.
        # The original user query is included as context for the LLM.
        final_kg_evidence_for_output = []
        for doc in retrieved_kg_evidence_with_chunk_text:
            doc_copy = doc.copy()
            doc_copy.pop("chunk_text", None)
            final_kg_evidence_for_output.append(doc_copy)
        
        processed_subquery_results = [{
            "sub_query_text": user_query,
            "reranked_chunks": retrieved_chunks,
            "retrieved_kg_data": final_kg_evidence_for_output
        }]
        

        print(f"Prepared {len(processed_subquery_results)} subquery result entries for LLM context")
        
        show_references = self.params.get('enable_references_citations', False)
        candidate_source_files: set[str] = set()
        for sq_result in processed_subquery_results:
            for chunk in sq_result.get("reranked_chunks", []):
                if chunk.get("file_name"):
                    candidate_source_files.add(chunk["file_name"])
            for kg_item in sq_result.get("retrieved_kg_data", []):
                if kg_item.get("file_name"):
                    candidate_source_files.add(kg_item["file_name"])

        final_results_dict = {
            "original_query": user_query,
            "sub_queries_results": processed_subquery_results,
            "refrences": ""
        }
        
        llm_formatted_context = self._format_search_results_for_llm(
            original_query=user_query,
            sub_queries_results=processed_subquery_results 
        )
        final_results_dict["llm_formatted_context"] = llm_formatted_context

        # Mark end of retrieval pipeline
        retrieval_end_time = time.time()
        retrieval_duration = retrieval_end_time - retrieval_start_time
        print(f"\n{'='*80}")
        print(f"✅ RETRIEVAL PIPELINE COMPLETED in {retrieval_duration:.2f}s")
        print(f"{'='*80}\n")

        # LLM Call for final Ans
        print(f"⏱️  [3/3] Generating final answer with LLM...") 
        llm_generation_start = time.time()
        cite_instruction = """
                5.  Cite sources inline at the end of each paragraph using square brackets with file names, e.g. [some_document.pdf]. If multiple files are used in one paragraph, cite all, e.g. [file_one.pdf, file_two.docx].
        """ if self.params.get("enable_references_citations", False) else ""

        final_call_system_message = f"""
            ROLE
            You are a precise assistant that answers using only provided context.

            TASK
            Provide a clear, complete, mobile-friendly WhatsApp answer.

            STRICT OUTPUT RULES (MUST FOLLOW)
            1. Never output markdown headers or separators. Forbidden: #, ##, ###, ---, ```.
            2. Use only plain text paragraphs and bullets that begin with "- ".
            3. WhatsApp emphasis only: *bold* and _italic_. Do not use **double-asterisk** markdown bold.
            4. Do not nest emphasis markers and do not leave unmatched markers.
            5. First visible characters must be normal text (not symbols like # or -).
            6. Keep paragraphs short and readable on chat screens.
            7. Stay strictly context-bound; if insufficient context, state that clearly.
            {cite_instruction}

            Return only the final user-facing answer text in WhatsApp style.
            """

        if not self.params.get("enable_references_citations", False):
            print("Citations are disabled. Using a WhatsApp-format prompt without citation instructions.")

        USER_PROMPT_TEMPLATE = """
            Follow the system rules exactly.
            ORIGINAL_QUESTION: "{original_query}"

            RECENT_CONVERSATION_START
            {conversation_history}
            RECENT_CONVERSATION_END

            CONTEXT_START
            {context}
            CONTEXT_END

            ANSWER:
            """
        user_prompt = USER_PROMPT_TEMPLATE.format(
            original_query=user_query,
            conversation_history=self._render_conversation_history(),
            context=final_results_dict["llm_formatted_context"]
        )

        messages = [
            {"role": "system", "content": final_call_system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"Generating final structured llm response: '{user_query}'")
        llm_response_content = ""
        stream_started = {"value": False}

        async def _safe_stream_callback(delta: str):
            if not stream_callback:
                return
            cleaned_delta = _sanitize_whatsapp_text(
                delta,
                strip_leading_markdown=not stream_started["value"],
            )
            if not cleaned_delta:
                return
            stream_started["value"] = True
            await stream_callback(cleaned_delta)

        llm_response_content = await self._call_openai_api(
            model_name=OPENAI_CHAT_MODEL,
            payload_messages=messages,
            max_tokens=8000,
            temperature=0.1,
            stream_callback=_safe_stream_callback if stream_callback else None,
        )

        llm_response_content = _sanitize_whatsapp_text(llm_response_content, strip_leading_markdown=True).strip()

        if show_references and llm_response_content:
            used_source_files = _extract_files_cited_in_answer(llm_response_content, candidate_source_files)
            if used_source_files:
                llm_response_content = (
                    llm_response_content.rstrip()
                    + "\n\n*Cited Sources:*\n"
                    + "\n".join(f"- {name}" for name in used_source_files)
                )

        llm_generation_duration = time.time() - llm_generation_start
        print(f"✅ Final answer generation completed in {llm_generation_duration:.2f}s\n")

        final_results_dict["final_answer"] = llm_response_content
        
        print(f"\n{'='*80}")
        print(f"📝 FINAL ANSWER:")
        print(f"{'='*80}")
        print(llm_response_content)
        print(f"{'='*80}\n")
        
        # Print final timing summary
        total_time = time.time() - retrieval_start_time
        print(f" RAG FUSION SEARCH COMPLETED")
        print(f"Query: '{user_query[:60]}...'")
        print(f"\n⏱️  TIMING BREAKDOWN:")
        print(f"  1. Setup (subqueries + keywords):          {setup_duration:.2f}s")
        print(f"  2. Subquery processing (concurrent):       {subquery_processing_duration:.2f}s")
        if self.reranker and self.deep_research:
            print(f"  3. Reranking & pruning:                     {rerank_duration:.2f}s")
        else:
            print(f"  3. Reranking & pruning:                     0.00s (skipped)")
        print(f"  4. LLM final answer generation:             {llm_generation_duration:.2f}s")
        print(f"\n RETRIEVAL PIPELINE ONLY:  {retrieval_duration:.2f}s")
        print(f" TOTAL TIME (with LLM):     {total_time:.2f}s")
        print(f" LLM API CALLS:             2 (setup + final answer)")
        print(f"\n")
        
        return final_results_dict

async def handle_request(data: Message) -> FunctionResponse:
  token = getattr(data, "token", None)
  params = data.params
  config = data.config
  try:
    es_client = None
    aclient_openai = None
    print('Incoming Data:--', data)
    params = data.params
    config = data.config
    token = getattr(data, "token", None)
    server_type = os.getenv('RAG_SERVER_TYPE')
    print('server_type:-', server_type)

    es_client = await check_async_elasticsearch_connection()
    if not es_client:
      return FunctionResponse(False, "Could not connect to Elasticsearch.")

    text_model = ( params.get('text_model') or config.get('text_model') or config.get('model') or 'gpt-4o-mini' )
    print('text_model:-', text_model)
    aclient_openai = await init_async_openai_client(text_model)
    if not aclient_openai:
        return FunctionResponse(False, "Could not connect to Open Ai.")

    retriever = RAGFusionRetriever(params, config, es_client, aclient_openai, token)
    user_query_input = params.get('question') or params.get('query')
    raw_top_k = params.get('top_k_chunks', 6)
    try:
        top_k_chunks = int(raw_top_k) if raw_top_k is not None else 6
    except (TypeError, ValueError):
        top_k_chunks = 6
    print(f"\n--- Running RAG Fusion Search for: '{user_query_input}' ---")
    search_results_dict = await retriever.search(
        user_query=user_query_input, initial_candidate_pool_size=top_k_chunks, top_k_kg_entities=top_k_chunks, absolute_score_floor=0.3
    )
    print("\n--- Search Results Dictionary (RAG Fusion: Chunks & KG Reranked if applicable) ---")
    
    print("\n--- LLM Formatted Context ---")
    # print(search_results_dict.get("llm_formatted_context", "No formatted context generated."))

    if es_client and hasattr(es_client, 'close'):
      await es_client.close()
      print("Elasticsearch client closed.")
    if aclient_openai and hasattr(aclient_openai, "aclose"):
        print('open ai clinet a close')
        try:
          await aclient_openai.aclose()
          print("OpenAI client closed.")
        except Exception as e:
          print(f"Error closing OpenAI client: {e}")

    return FunctionResponse(message=Messages(search_results_dict.get("final_answer", "No formatted context generated.")), failed=False) 
  except Exception as e:
    print(f"❌ Error during retrieval: {e}")
    return FunctionResponse(message=Messages(e))

async def handle_request_stream(data: Message) -> AsyncGenerator[str, None]:
    token = getattr(data, "token", None)
    params = data.params
    config = data.config
    es_client = None
    aclient_openai = None
    queue: asyncio.Queue = asyncio.Queue()

    async def _on_stream_delta(delta: str):
        await queue.put(delta)

    async def _run_search():
        nonlocal es_client, aclient_openai
        try:
            es_client = await check_async_elasticsearch_connection()
            if not es_client:
                await queue.put("❌ Could not connect to Elasticsearch.")
                return

            text_model = ( params.get('text_model') or config.get('text_model') or config.get('model') or 'gpt-4o-mini' )
            aclient_openai = await init_async_openai_client(text_model)
            if not aclient_openai:
                await queue.put("❌ Could not connect to OpenAI.")
                return

            retriever = RAGFusionRetriever(params, config, es_client, aclient_openai, token)
            user_query_input = params.get('question') or params.get('query')
            raw_top_k = params.get('top_k_chunks', 6)
            try:
                top_k_chunks = int(raw_top_k) if raw_top_k is not None else 6
            except (TypeError, ValueError):
                top_k_chunks = 6

            await retriever.search(
                user_query=user_query_input,
                initial_candidate_pool_size=top_k_chunks,
                top_k_kg_entities=top_k_chunks,
                absolute_score_floor=0.3,
                stream_callback=_on_stream_delta,
            )
        except Exception as e:
            print(f"❌ Streaming retrieval failed: {e}")
            await queue.put(f"❌ Error: {e}")
        finally:
            if es_client and hasattr(es_client, 'close'):
                await es_client.close()
            if aclient_openai and hasattr(aclient_openai, "aclose"):
                try:
                    await aclient_openai.aclose()
                except Exception as e:
                    print(f"Error closing OpenAI client: {e}")
            await queue.put(None)

    runner = asyncio.create_task(_run_search())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield str(item)
    finally:
        await runner

def test_query():
    params = {
        "question": "what was the audit observation by pavankumar for security",
        "top_k_chunks": 15,
        "enable_references_citations": False,
        "deep_research": False
    }
    config = {
        "index_name": "minera-audit",
    }
    message = Message(params=params, config=config)
    res = asyncio.run(handle_request(message))
    print('\n\n=== FINAL ANSWER ===\n', res.message.message)

if __name__ == "__main__":
    test_query()