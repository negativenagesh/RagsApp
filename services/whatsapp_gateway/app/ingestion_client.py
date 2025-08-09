import os
import httpx
from dotenv import load_dotenv

load_dotenv()

INGESTION_API_URL = os.getenv("INGESTION_API_URL", "http://localhost:8002/ingest")

async def ingest_file(filename: str, file_bytes: bytes):
    files = {"file": (filename, file_bytes)}
    async with httpx.AsyncClient() as client:
        resp = await client.post(INGESTION_API_URL, files=files)
        resp.raise_for_status()
        return resp.json()