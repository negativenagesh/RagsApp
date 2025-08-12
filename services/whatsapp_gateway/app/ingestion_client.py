import os
import httpx
from dotenv import load_dotenv

load_dotenv()

INGESTION_API_URL = os.getenv("INGESTION_API_URL", "http://localhost:8002/ingest")

async def ingest_file(filename: str, file_bytes: bytes):
    files = {"file": (filename, file_bytes)}
    timeout = httpx.Timeout(300.0, connect=10.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(INGESTION_API_URL, files=files)
            
            resp.raise_for_status() 
            
            return resp.json()
        except Exception as e:
            print(f"Ingestion client error: {repr(e)}")
            raise e