import os
import httpx

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8001/rag-search")

async def ask_rag(
    question: str,
    top_k_chunks: int = None,
    enable_references_citations: bool = True,
    deep_research: bool = False
):
    rag_payload = {
        "question": question,
        "enable_references_citations": enable_references_citations,
        "deep_research": deep_research,
    }
    if top_k_chunks is not None:
        rag_payload["top_k_chunks"] = top_k_chunks
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(RAG_API_URL, json=rag_payload, timeout=120)
            data = resp.json()
            if isinstance(data, dict):
                if "final_answer" in data:
                    return data["final_answer"]
                if "result" in data:
                    return data["result"]
            return "No answer found."
    except httpx.ReadTimeout:
        print("RAG service timed out.")
        return "❌ RAG service timed out. Please try again later."
    except Exception as e:
        print(f"ask_rag error: {e}")
        return f"❌ Error: {e}"