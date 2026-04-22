from app.retrieval import init_async_openai_client
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import os
import json
from app.retrieval import handle_request, handle_request_stream, Message, FunctionResponse

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    top_k_chunks: Optional[int] = None
    enable_references_citations: bool = True
    deep_research: bool = False
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)

@app.post("/rag-search")
async def rag_search(request: QueryRequest):
    params = request.dict()
    config = {"index_name": "ragsapp"}
    message = Message(params=params, config=config)
    response: FunctionResponse = await handle_request(message)
    # response.message is a Messages object — extract the actual text from .message attribute
    final_answer = ""
    msg = response.message
    if hasattr(msg, "message"):
        # It's a Messages object — unwrap it
        final_answer = msg.message if isinstance(msg.message, str) else str(msg.message)
    elif isinstance(msg, dict):
        final_answer = msg.get("final_answer", str(msg))
    else:
        final_answer = str(msg)
    return {"final_answer": final_answer, "failed": response.failed}


@app.post("/rag-search/stream")
async def rag_search_stream(request: QueryRequest):
    params = request.dict()
    config = {"index_name": "ragsapp"}
    message = Message(params=params, config=config)

    async def event_generator():
        async for delta in handle_request_stream(message):
            payload = {"type": "delta", "delta": delta}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield "data: {\"type\":\"done\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "whatsapp_provider": os.getenv("WHATSAPP_PROVIDER", "twilio"),
    }


@app.get("/config")
def config_check():
    return {
        "whatsapp_provider": os.getenv("WHATSAPP_PROVIDER", "twilio"),
    }