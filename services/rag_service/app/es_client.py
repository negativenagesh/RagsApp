import os
import traceback
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
        traceback.print_exc()
        return None