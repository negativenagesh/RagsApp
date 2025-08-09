from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from app.retrieval import handle_request, Message, FunctionResponse

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    top_k_chunks: Optional[int] = None
    enable_references_citations: bool = True
    deep_research: bool = False

@app.post("/rag-search")
async def rag_search(request: QueryRequest):
    params = request.dict()
    config = {"index_name": "ragsapp"}
    message = Message(params=params, config=config)
    response: FunctionResponse = await handle_request(message)
    # Ensure final_answer is a string
    final_answer = ""
    if isinstance(response.message, dict):
        final_answer = response.message.get("final_answer", "")
    elif hasattr(response.message, "get"):
        final_answer = response.message.get("final_answer", "")
    else:
        final_answer = str(response.message)
    return {"final_answer": final_answer, "failed": response.failed}