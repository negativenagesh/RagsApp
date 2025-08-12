import os
import asyncio
import re
from base64 import b64decode
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INVALID_PARAMS
from pydantic import Field
from typing import Annotated
import httpx

# --- Environment Setup ---
load_dotenv()
TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
RAG_API_URL = os.environ.get("RAG_API_URL")
INGESTION_API_URL = os.environ.get("INGESTION_API_URL")

# --- Configuration Validation ---
assert TOKEN, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER, "Please set MY_NUMBER in your .env file"
assert RAG_API_URL, "Please set RAG_API_URL in your .env file"
assert INGESTION_API_URL, "Please set INGESTION_API_URL in your .env file"

# --- Authentication ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    """A simple authentication provider that checks for a static bearer token."""
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- MCP Server Initialization ---
mcp = FastMCP(
    "RagsApp MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- MCP Tools ---

@mcp.tool(description="Ask a question about your documents using RAG.")
async def ask_rag(
    question: Annotated[str, Field(description="Your question")],
    enable_references_citations: Annotated[bool, Field(description="Show references")] = True,
    deep_research: Annotated[bool, Field(description="Enable deep research")] = False,
) -> str:
    """Contacts the RAG API to answer a question."""
    payload = {
        "question": question,
        "enable_references_citations": enable_references_citations,
        "deep_research": deep_research,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(RAG_API_URL, json=payload, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
            if "final_answer" in data:
                return data["final_answer"]
            if "result" in data:
                return data["result"]
            return str(data)
    except httpx.ReadTimeout:
        return "Your request is taking longer than expected. You will receive the answer soon."
    except httpx.HTTPStatusError as e:
        return f"Error from RAG backend: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error contacting RAG backend: {e}"

@mcp.tool(description="Ingest a document into the knowledge base from a local upload (PDF, CSV, XLSX).")
async def ingest_file(
    filename: Annotated[str, Field(description="Original filename of the uploaded file")],
    file_data: Annotated[str, Field(description="Base64-encoded content of the file")],
    description: Annotated[str, Field(description="A brief description of the file's content")] = "",
) -> str:
    """
    Ingests a locally uploaded file by decoding it and sending it to the ingestion service.
    This is the correct tool for handling files from WhatsApp or other chat clients.
    """
    try:
        file_bytes = b64decode(file_data)
    except Exception as e:
        raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Invalid file data. Could not decode base64. Error: {e}"))

    files = {"file": (filename, file_bytes)}
    data = {"description": description}
    print(f"Forwarding file '{filename}' to ingestion service at {INGESTION_API_URL}")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(INGESTION_API_URL, data=data, files=files, timeout=300.0)
            resp.raise_for_status()
        try:
            response_json = resp.json()
            return f"✅ Successfully ingested file '{filename}'. Details: {response_json.get('details', 'N/A')}"
        except Exception:
            return f"✅ Successfully ingested file '{filename}'. Response: {resp.text}"
    except httpx.HTTPStatusError as e:
        return f"❌ Error from ingestion service: {e.response.status_code} - {e.response.text}"
    except httpx.RequestError as e:
        return f"❌ Could not connect to the ingestion service at {INGESTION_API_URL}. Please check the server logs and network configuration. Error: {e}"
    except Exception as e:
        return f"❌ An unexpected error occurred during ingestion: {e}"

@mcp.tool(description="Returns the server owner's phone number for validation.")
async def validate() -> str:
    """
    Returns the server owner's phone number in {country_code}{number} format.
    Required by the Puch platform for agent validation. Example: 919876543210
    """
    number = os.environ.get("MY_NUMBER")
    if not number:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="MY_NUMBER is not set in the environment."))
    if not re.fullmatch(r"\d{10,15}", number):
        raise McpError(ErrorData(code=INVALID_PARAMS, message="MY_NUMBER must be in {country_code}{number} format, containing only digits."))
    return number

# --- Server Execution ---
async def main():
    """Starts the MCP server."""
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    print(f"   - RAG API URL: {RAG_API_URL}")
    print(f"   - Ingestion API URL: {INGESTION_API_URL}")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())