import os
from unittest import result
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from openai import AsyncOpenAI

from .es_client import get_es_client
from .processor import process_and_ingest_file

app = FastAPI(
    title="RagsApp Ingestion Service",
    description="An API to process and ingest documents into Elasticsearch."
)

app.state.es_client = None
app.state.openai_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize external clients on startup."""
    app.state.es_client = await get_es_client()
    if not app.state.es_client:
        print("FATAL: Could not connect to Elasticsearch. Service will not be functional.")
    
    openai_api_key = os.getenv("OPEN_AI_KEY")
    if openai_api_key:
        app.state.openai_client = AsyncOpenAI(api_key=openai_api_key)
        print("OpenAI client initialized.")
    else:
        print("WARNING: OPEN_AI_KEY not found. OpenAI-dependent features will fail.")

@app.on_event("shutdown")
async def shutdown_event():
    """Close client connections on shutdown."""
    if app.state.es_client:
        await app.state.es_client.close()
        print("Elasticsearch client closed.")
    if app.state.openai_client:
        print("OpenAI client shutdown.")

@app.post("/ingest/")
async def ingest_file(
    description: str = Form(""),
    is_ocr_pdf: bool = Form(False),
    file: UploadFile = File(...)
):
    """
    Endpoint to upload a file for processing and ingestion.
    """
    if not app.state.es_client or not app.state.openai_client:
        raise HTTPException(status_code=503, detail="Service is not ready. Clients not initialized.")

    file_content = await file.read()
    original_filename = file.filename
    index_name = "ragsapp"

    print(f"Received file '{original_filename}' for ingestion into index '{index_name}'.")

    params = {
        "index_name": index_name,
        "file_name": original_filename,
        "is_ocr_pdf": is_ocr_pdf,
    }

    try:
        result = await process_and_ingest_file(
            file_data=file_content,
            original_file_name=original_filename,
            index_name=index_name,
            es_client=app.state.es_client,
            aclient_openai=app.state.openai_client,
            user_provided_doc_summary=description,
            params=params
        )
        if not result:
            raise HTTPException(status_code=500, detail="Ingestion failed with no result returned.")
        if result.get("status") != "success":
            raise HTTPException(status_code=400, detail=result)
        return {"status": "success", "filename": original_filename, "details": result}
    except Exception as e:
        print(f"An error occurred during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/config")
def config_check():
    return {
        "RAG_UPLOAD_ELASTIC_URL": os.getenv("RAG_UPLOAD_ELASTIC_URL"),
        "ELASTICSEARCH_API_KEY": bool(os.getenv("ELASTICSEARCH_API_KEY")),
        "OPEN_AI_KEY": bool(os.getenv("OPEN_AI_KEY")),
    }