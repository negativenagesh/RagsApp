import os
import copy
import json
import traceback
from typing import Dict, Optional, Any
from elasticsearch import AsyncElasticsearch

def get_env_or_fail(var):
    val = os.getenv(var)
    if not val:
        raise RuntimeError(f"Environment variable '{var}' is required but not set.")
    return val

async def get_es_client():
    try:
        ELASTICSEARCH_URL = get_env_or_fail("RAG_UPLOAD_ELASTIC_URL")
        ELASTICSEARCH_API_KEY = get_env_or_fail("ELASTICSEARCH_API_KEY")
        print(f'Connecting to Elasticsearch: {ELASTICSEARCH_URL}')
        es_client = AsyncElasticsearch(
            ELASTICSEARCH_URL,
            api_key=ELASTICSEARCH_API_KEY,
            request_timeout=60,
            retry_on_timeout=True
        )
        if not await es_client.ping():
            print("Ping to Elasticsearch failed. Check URL or server status.")
            return None
        print("Successfully connected to Elasticsearch.")
        return es_client
    except Exception as e:
        print(f"Failed to initialize Elasticsearch client: {e}")
        return None

OPENAI_EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", 3072))

CHUNKED_PDF_MAPPINGS = {
    "mappings": {
        "properties": {
            "chunk_text": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": OPENAI_EMBEDDING_DIMENSIONS,
                "index": True,
                "similarity": "cosine"
            },
            "metadata": {
                "properties": {
                    "file_name": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "chunk_index_in_page": {"type": "integer"},
                    "document_summary": {"type": "text"},
                    "entities": {
                        "type": "nested",
                        "properties": {
                            "name": {"type": "keyword"},
                            "type": {"type": "keyword"},
                            "description": {"type": "text"},
                            "description_embedding": {
                                "type": "dense_vector",
                                "dims": OPENAI_EMBEDDING_DIMENSIONS,
                                "index": True,
                                "similarity": "cosine"
                            }
                        }
                    },
                    "relationships": {
                        "type": "nested",
                        "properties": {
                            "source_entity": {"type": "keyword"},
                            "target_entity": {"type": "keyword"},
                            "relation": {"type": "keyword"},
                            "relationship_description": {"type": "text"},
                            "relationship_weight": {"type": "float"}
                        }
                    }
                }
            }
        }
    }
}

async def ensure_es_index_exists(client: Any, index_name: str, mappings_body: Dict):
    try:
        if not await client.indices.exists(index=index_name):
            updated_mappings = copy.deepcopy(mappings_body)
            updated_mappings["mappings"]["properties"]["embedding"]["dims"] = OPENAI_EMBEDDING_DIMENSIONS
            if "description_embedding" in updated_mappings["mappings"]["properties"]["metadata"]["properties"]["entities"]["properties"]:
                updated_mappings["mappings"]["properties"]["metadata"]["properties"]["entities"]["properties"]["description_embedding"]["dims"] = OPENAI_EMBEDDING_DIMENSIONS

            await client.indices.create(index=index_name, body=updated_mappings)
            print(f"Elasticsearch index '{index_name}' created with specified mappings.")
            return True
        else: 
            print('Index already exits in ensure_es_index_exists')
            current_mapping_response = await client.indices.get_mapping(index=index_name)
            current_top_level_props = current_mapping_response.get(index_name, {}).get('mappings', {}).get('properties', {})
            current_metadata_props = current_top_level_props.get('metadata', {}).get('properties', {})
            
            expected_top_level_props = mappings_body.get('mappings', {}).get('properties', {})
            expected_metadata_props = expected_top_level_props.get('metadata', {}).get('properties', {})
            
            missing_fields = []
            different_fields = []

            for field, expected_field_mapping in expected_top_level_props.items():
                if field == "metadata": continue # Handled separately
                if field not in current_top_level_props:
                    missing_fields.append(field)
                elif current_top_level_props[field].get('type') != expected_field_mapping.get('type'):
                    if field == "embedding" and current_top_level_props[field].get('type') == 'dense_vector' and expected_field_mapping.get('type') == 'dense_vector':
                        if current_top_level_props[field].get('dims') != expected_field_mapping.get('dims'):
                            different_fields.append(f"{field} (dims: {current_top_level_props[field].get('dims')} vs {expected_field_mapping.get('dims')})")
                    else:
                        different_fields.append(f"{field} (type: {current_top_level_props[field].get('type')} vs {expected_field_mapping.get('type')})")
            
            if expected_metadata_props:
                for field, expected_meta_mapping in expected_metadata_props.items():
                    if field not in current_metadata_props:
                        missing_fields.append(f"metadata.{field}")
                    elif current_metadata_props[field].get('type') != expected_meta_mapping.get('type'):
                        different_fields.append(f"metadata.{field} (type: {current_metadata_props[field].get('type')} vs {expected_meta_mapping.get('type')})")


            if missing_fields:
                print(f"Fields {missing_fields} missing in index '{index_name}'. Attempting to update mapping.")
                update_body_props_for_put_mapping = {}
                metadata_updates = {}

                for field_path_to_add in missing_fields:
                    if field_path_to_add.startswith("metadata."):
                        field_name = field_path_to_add.split("metadata.")[1]
                        if field_name in expected_metadata_props:
                            metadata_updates[field_name] = expected_metadata_props[field_name]
                    elif field_path_to_add in expected_top_level_props:
                         update_body_props_for_put_mapping[field_path_to_add] = expected_top_level_props[field_path_to_add]
                
                if metadata_updates:
                    update_body_props_for_put_mapping["metadata"] = {"properties": metadata_updates}

                if update_body_props_for_put_mapping:
                    try:
                        await client.indices.put_mapping(index=index_name, body={"properties": update_body_props_for_put_mapping})
                        print(f"Successfully attempted to add missing fields {missing_fields} to mapping of index '{index_name}'.")
                    except Exception as map_e:
                        print(f"Failed to update mapping for index '{index_name}' to add fields: {map_e}. This might cause issues.")
                else:
                    print(f"Could not prepare update body for missing fields in '{index_name}'.")
            
            if different_fields:
                print(f"Elasticsearch index '{index_name}' exists but mappings for fields {different_fields} differ. This might cause issues.")

            if not missing_fields and not different_fields:
                print(f"Elasticsearch index '{index_name}' already exists and critical fields appear consistent.")
            return True 
    except Exception as e:
        print(f"❌ Error with Elasticsearch index '{index_name}': {e}")
        traceback.print_exc()
        if hasattr(e, 'info'):
            print("🔎 Error details:", json.dumps(e.info, indent=2))
        return False