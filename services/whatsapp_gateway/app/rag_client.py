import os
import httpx
import json
from typing import AsyncGenerator, Dict, List, Optional

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8001/rag-search")
RAG_STREAM_API_URL = os.getenv("RAG_STREAM_API_URL", "http://localhost:8001/rag-search/stream")
RAG_SYNC_TIMEOUT_SECONDS = float(os.getenv("RAG_SYNC_TIMEOUT_SECONDS", "120"))
RAG_STREAM_TIMEOUT_SECONDS = float(os.getenv("RAG_STREAM_TIMEOUT_SECONDS", "180"))

async def ask_rag(
    question: str,
    top_k_chunks: int = None,
    enable_references_citations: bool = True,
    deep_research: bool = False,
    timeout_seconds: float = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
):
    rag_payload = {
        "question": question,
        "enable_references_citations": enable_references_citations,
        "deep_research": deep_research,
        "conversation_history": conversation_history or [],
    }
    if top_k_chunks is not None:
        rag_payload["top_k_chunks"] = top_k_chunks
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                RAG_API_URL,
                json=rag_payload,
                timeout=timeout_seconds if timeout_seconds is not None else RAG_SYNC_TIMEOUT_SECONDS,
            )
            data = resp.json()
            if isinstance(data, dict):
                if data.get("failed") is True:
                    failed_answer = data.get("final_answer")
                    if failed_answer:
                        return f"❌ {failed_answer}"
                    return "❌ RAG service is currently unavailable. Please try again in a moment."
                if "final_answer" in data:
                    final_answer = data["final_answer"]
                    if isinstance(final_answer, bool):
                        return "❌ I could not get an answer right now. Please try again."
                    if final_answer is None:
                        return "❌ I could not get an answer right now. Please try again."
                    text_answer = str(final_answer).strip()
                    if not text_answer or text_answer.lower() in {"false", "none", "null"}:
                        return "❌ I could not get an answer right now. Please try again."
                    return text_answer
                if "result" in data:
                    return data["result"]
            return "No answer found."
    except httpx.ReadTimeout:
        print("RAG service timed out.")
        return "❌ RAG service timed out. Please try again later."
    except Exception as e:
        print(f"ask_rag error: {e}")
        return f"❌ Error: {e}"


async def stream_rag(
    question: str,
    top_k_chunks: int = None,
    enable_references_citations: bool = True,
    deep_research: bool = False,
    timeout_seconds: float = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[str, None]:
    rag_payload = {
        "question": question,
        "enable_references_citations": enable_references_citations,
        "deep_research": deep_research,
        "conversation_history": conversation_history or [],
    }
    if top_k_chunks is not None:
        rag_payload["top_k_chunks"] = top_k_chunks

    async with httpx.AsyncClient(
        timeout=timeout_seconds if timeout_seconds is not None else RAG_STREAM_TIMEOUT_SECONDS
    ) as client:
        async with client.stream("POST", RAG_STREAM_API_URL, json=rag_payload, headers={"Accept": "text/event-stream"}) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw_payload = line[len("data:"):].strip()
                if not raw_payload:
                    continue
                try:
                    event = json.loads(raw_payload)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "done":
                    break
                if event.get("type") == "delta":
                    delta = event.get("delta") or ""
                    if delta:
                        yield str(delta)