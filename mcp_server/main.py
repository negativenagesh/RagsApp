import os
import asyncio
from base64 import b64decode
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from pydantic import Field, AnyUrl
from typing import Annotated
import httpx
from mcp import ErrorData, McpError
from mcp.types import INVALID_PARAMS

# --- Load environment variables ---
load_dotenv()
TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
RAG_API_URL = os.environ.get("RAG_API_URL")
INGESTION_API_URL = os.environ.get("INGESTION_API_URL")

assert TOKEN, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER, "Please set MY_NUMBER in your .env file"
assert RAG_API_URL, "Please set RAG_API_URL in your .env file"
assert INGESTION_API_URL, "Please set INGESTION_API_URL in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
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

# --- MCP Server ---
mcp = FastMCP(
    "RagsApp MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: ask_rag ---
@mcp.tool(description="Ask a question about your documents using RAG.")
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
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(RAG_API_URL, json=payload, timeout=60)  # 60s max
            data = resp.json()
            if "final_answer" in data:
                return data["final_answer"]
            if "result" in data:
                return data["result"]
            return str(data)
    except httpx.ReadTimeout:
        # Optionally, trigger background processing here and notify user later by other means
        return "Your request is taking longer than expected. You will receive the answer soon."
    except Exception as e:
        return f"Error contacting RAG backend: {e}"

@mcp.tool(description="Ingest a document into the knowledge base.")
async def ingest_file(
    file_data: Annotated[str | None, Field(description="Base64-encoded file data")] = None,
    file_url: Annotated[AnyUrl | None, Field(description="Public URL to the file")] = None,
    filename: Annotated[str | None, Field(description="Original filename")] = None,
    description: Annotated[str, Field(description="File description")] = "",
) -> str:
    if file_data:
        # Decode base64 file data
        file_bytes = b64decode(file_data)
        files = {"file": (filename or "uploaded_file", file_bytes)}
    elif file_url:
        async with httpx.AsyncClient() as client:
            file_resp = await client.get(str(file_url))
            if file_resp.status_code != 200:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="Could not download file"))
            files = {"file": ("uploaded_file", file_resp.content)}
    else:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="No file data or URL provided."))

    data = {"description": description}
    async with httpx.AsyncClient() as client:
        ingest_resp = await client.post(INGESTION_API_URL, data=data, files=files)
        return ingest_resp.text

@mcp.tool(description="Returns the server owner's phone number in {country_code}{number} format.")
async def validate() -> str:
    """
    Returns the server owner's phone number in {country_code}{number} format.
    Example: 919876543210 for +91-9876543210
    """
    import re
    number = os.environ.get("MY_NUMBER")
    if not number:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="MY_NUMBER is not set in the environment."))
    # Ensure only digits, and length is reasonable (10-15 digits)
    if not re.fullmatch(r"\d{10,15}", number):
        raise McpError(ErrorData(code=INVALID_PARAMS, message="MY_NUMBER must be in {country_code}{number} format, digits only."))
    return number

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())