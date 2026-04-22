# RAG Skill Code Map

## WhatsApp Gateway
- `services/whatsapp_gateway/app/supervisor.py`
  - LLM-first routing decision.
  - Non-RAG direct LLM response path.
  - RAG confirmation gate for retrieval-only scenarios.
- `services/whatsapp_gateway/app/skills/rag_skill.py`
  - RAG skill implementation calling RAG service sync/stream methods.
- `services/whatsapp_gateway/app/retrieval_tool.py`
  - Thin compatibility adapter delegating to `RagSkill`.
- `services/whatsapp_gateway/app/main.py`
  - Webhook handlers.
  - Sends thinking indicator only on RAG route.

## RAG Service
- `services/rag_service/app/main.py`
  - HTTP endpoints for `rag-search` and streaming variant.
- `services/rag_service/app/retrieval.py`
  - Retrieval workflow over Elasticsearch and ranking.
- `services/rag_service/app/llm_handler.py`
  - Grounded final answer generation using retrieved context.

## Runtime Expectations
- Casual chat should remain in non-RAG path.
- RAG should run only for document-grounded requests.
- Thinking indicator should be emitted only for RAG path.
