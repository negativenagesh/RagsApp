import os
import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import Field, AnyUrl
from typing import Annotated
from mcp.types import TextContent, INVALID_PARAMS, INTERNAL_ERROR
from mcp import ErrorData, McpError

load_dotenv()
RAG_API_URL = os.environ.get("RAG_API_URL")
INGESTION_API_URL = os.environ.get("INGESTION_API_URL")
MY_NUMBER = os.environ.get("MY_NUMBER")

def get_env_or_error(var):
    v = os.environ.get(var)
    if not v:
        raise RuntimeError(f"Missing env var: {var}")
    return v

# --- Tool: validate (required by Puch) ---
async def validate() -> str:
    return MY_NUMBER

# --- Tool: ask_rag ---
async def ask_rag(
    question: Annotated[str, Field(description="Your question")],
    enable_references_citations: Annotated[bool, Field(description="Show references")] = True,
    deep_research: Annotated[bool, Field(description="Enable deep research")] = False,
) -> str:
    payload = {
        "question": question,
        "enable_references_citations": enable_references_citations,
        "deep_research": deep_research,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(RAG_API_URL, json=payload, timeout=120)
        data = resp.json()
        if "final_answer" in data:
            return data["final_answer"]
        if "result" in data:
            return data["result"]
        return str(data)

# --- Tool: ingest_file ---
async def ingest_file(
    file_url: Annotated[AnyUrl, Field(description="Public URL to the file")],
    description: Annotated[str, Field(description="File description")] = "",
) -> str:
    async with httpx.AsyncClient() as client:
        file_resp = await client.get(str(file_url))
        if file_resp.status_code != 200:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Could not download file"))
        files = {"file": ("uploaded_file", file_resp.content)}
        data = {"description": description}
        ingest_resp = await client.post(INGESTION_API_URL, data=data, files=files)
        return ingest_resp.text