<p align="center">
	<img src="RagsApp-logo/cover.png" alt="RagsApp Logo" width="800"/>
</p>

<p align="center">
  <a href="https://github.com/negativenagesh/RagsApp/stargazers">
    <img src="https://img.shields.io/github/stars/negativenagesh/RagsApp?style=flat&logo=github" alt="Stars">
  </a>
  <a href="https://github.com/negativenagesh/RagsApp/network/members">
    <img src="https://img.shields.io/github/forks/negativenagesh/RagsApp?style=flat&logo=github" alt="Forks">
  </a>
  <a href="https://github.com/negativenagesh/RagsApp/pulls">
    <img src="https://img.shields.io/github/issues-pr/negativenagesh/RagsApp?style=flat&logo=github" alt="Pull Requests">
  </a>
  <a href="https://github.com/negativenagesh/RagsApp/issues">
    <img src="https://img.shields.io/github/issues/negativenagesh/RagsApp?style=flat&logo=github" alt="Issues">
  </a>
  <a href="https://github.com/negativenagesh/RagsApp/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/negativenagesh/RagsApp?style=flat&logo=github" alt="License">
  </a>
</p>

/mcp use OX2pXc0xOM

WhatsApp-first RAG with Elasticsearch Cloud, split into Python microservices:

- Ingestion Service (FastAPI): extracts text from user files, chunks, embeds with OpenAI, and indexes vectors into Elasticsearch dense_vector.
- Retrieval Service (FastAPI): ANN search in Elasticsearch and LLM answering via OpenAI chat completions.
- WhatsApp Webhook Service (FastAPI): receives WhatsApp messages (Whapi.cloud), routes documents to ingestion and text to retrieval, and replies to users.

## Directory layout

```text
src/
	common/                # shared utilities
		config.py            # env-driven settings (ES, OpenAI, WhatsApp)
		es_client.py         # Elasticsearch client factory
		embeddings.py        # OpenAI embeddings wrapper
		chunking.py          # simple text chunking helpers
	core/base/parsers/     # existing parsers reused by ingestion
	ingestion_service/
		app.py               # /ingest file endpoint
	retrieval_service/
		app.py               # /search and /answer
	whatsapp_service/
		app.py               # /whapi/webhook for Whapi.cloud
```

Entry point: `main.py` dispatches based on SERVICE env (`ingestion|retrieval|whatsapp`).

## Configure environment

Set these variables (e.g., in a `.env` or shell):

- Elasticsearch Cloud
	- ES_CLOUD_ID and ES_API_KEY (preferred), or ES_HOST, ES_USERNAME, ES_PASSWORD
	- ES_INDEX_PREFIX (optional, default: ragsapp)

- OpenAI
	- OPENAI_API_KEY
	- OPENAI_EMBEDDING_MODEL (default: text-embedding-3-small)
	- OPENAI_CHAT_MODEL (default: gpt-4o-mini)

- WhatsApp (Whapi.cloud example)
	- WHATSAPP_PROVIDER=whapi
	- WHAPI_TOKEN=<your_whapi_token>

- Service URLs for cross-calls (used by whatsapp service)
	- INGESTION_URL=http://localhost:8001
	- RETRIEVAL_URL=http://localhost:8002

## Run locally

Use your virtualenv. Then run services in separate terminals:

```sh
# Ingestion
SERVICE=ingestion PORT=8001 python -m main

# Retrieval
SERVICE=retrieval PORT=8002 python -m main

# WhatsApp webhook
SERVICE=whatsapp PORT=8003 python -m main
```

Test ingestion (cURL or Postman):

- POST /ingest (multipart form)
	- file: upload a pdf/csv/xlsx/txt
	- user_id: a phone number or user key
	- namespace: optional; defaults to user_id. Retrieval searches over `{namespace}-*` indices.

Test retrieval:

- GET /search?q=...&namespace=<user_id>&k=5
- POST /answer?q=...&namespace=<user_id>&k=5

## WhatsApp integration

Using Whapi.cloud:
1) Create an account, connect your WhatsApp device, and get a token.
2) Expose your local webhook via a public URL (e.g., ngrok). Point Whapi webhook to `POST /whapi/webhook` of the WhatsApp service.
3) Send a document to your number; the webhook downloads media via Whapi and forwards to the ingestion service. Then send a text query; it will be answered via retrieval + LLM.

Alternatives: Twilio WhatsApp API is also supported with custom wiring (add TWILIO_* envs and extend whatsapp_service to send/receive accordingly).

## Elasticsearch Cloud notes

This project uses `dense_vector` + KNN search with cosine similarity. The ingestion service creates per-upload indices named `<namespace>-<ulid>-<filename>`. Retrieval searches over `<namespace>-*` to isolate each user.

For large scale, consider a single index with a `namespace` field and use filters in KNN queries, or leverage Elasticsearch vector tiles and routing.

## Extending to MCP and Puch AI

- Wrap retrieval endpoints as MCP tools (e.g., `search`, `answer`) using the MCP Python SDK, then register the server with Puch AI. The WhatsApp service can remain a thin adapter.

## Security & Ops

- Keep API keys in env vars or a secret manager.
- Enforce per-user namespaces to avoid data leakage.
- Add rate limiting (e.g., slowapi) and logging. For production, add auth to ingestion/retrieval.

## Roadmap

- Add OCR for images (pytesseract) on ingestion.
- Add CSV/XLSX header-aware chunking.
- Add hybrid retrieval (BM25 + vector) via ES `rank_features` or ELSER.
- Swap OpenAI for local models via vLLM/ollama if needed.
