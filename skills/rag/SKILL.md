---
name: rag
description: "Use when a user asks for document-grounded answers, citations, semantic retrieval, or uploaded-file facts. Trigger for: summarize file, what does the PDF say, find in docs, cite sources, retrieval, vector search, RAG."
argument-hint: "Question that may require retrieval from uploaded docs"
user-invocable: true
disable-model-invocation: false
---

# RAG Skill

## Purpose
Use retrieval-augmented generation when the user request needs factual grounding from uploaded files or indexed knowledge.

## When To Use
- The user asks about uploaded documents, PDFs, DOCX, CSV, XLSX, or indexed content.
- The answer must include citations or evidence from source chunks.
- The user asks to summarize, compare, extract, or verify facts from documents.

## Do Not Use
- Greeting, small talk, social chat, or casual conversation.
- Generic conversational questions that do not require stored documents.
- Requests that can be answered directly by a normal LLM response.

## Procedure
1. Classify whether the user query requires external document grounding.
2. If grounding is required, route to retrieval in the WhatsApp gateway and call RAG service.
3. Generate final grounded answer using retrieved context.
4. Return concise user-facing text and include citations when enabled.

## Integration Notes
- Retrieval entrypoint: `RetrievalTool.ask` / `RetrievalTool.stream`.
- Gateway orchestration and route handling live in the WhatsApp service.
- RAG service executes retrieval and grounded generation.
- Thinking indicator/sticker should be sent only when this skill is being used.

## References
- [Code map](./references/code-map.md)
