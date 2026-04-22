import os
from typing import AsyncGenerator, Dict, List, Optional

from app.rag_client import ask_rag, stream_rag


class RagSkill:
    def __init__(self) -> None:
        self.default_top_k = int(os.getenv("RETRIEVAL_TOP_K_DEFAULT", "6"))
        self.min_top_k = int(os.getenv("RETRIEVAL_TOP_K_MIN", "2"))
        self.max_top_k = int(os.getenv("RETRIEVAL_TOP_K_MAX", "20"))
        self.enable_references_citations = self._as_bool(
            os.getenv("RETRIEVAL_ENABLE_REFERENCES_CITATIONS"),
            True,
        )
        self.deep_research_default = self._as_bool(os.getenv("RETRIEVAL_DEEP_RESEARCH_DEFAULT"), False)

    async def ask(
        self,
        question: str,
        top_k_chunks: Optional[int] = None,
        enable_references_citations: Optional[bool] = None,
        deep_research: Optional[bool] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        return await ask_rag(
            question=question,
            top_k_chunks=self.normalize_top_k(top_k_chunks),
            enable_references_citations=self.enable_references_citations
            if enable_references_citations is None
            else bool(enable_references_citations),
            deep_research=self.deep_research_default if deep_research is None else bool(deep_research),
            conversation_history=conversation_history or [],
        )

    async def stream(
        self,
        question: str,
        top_k_chunks: Optional[int] = None,
        enable_references_citations: Optional[bool] = None,
        deep_research: Optional[bool] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        async for token in stream_rag(
            question=question,
            top_k_chunks=self.normalize_top_k(top_k_chunks),
            enable_references_citations=self.enable_references_citations
            if enable_references_citations is None
            else bool(enable_references_citations),
            deep_research=self.deep_research_default if deep_research is None else bool(deep_research),
            conversation_history=conversation_history or [],
        ):
            yield token

    def normalize_top_k(self, value: Optional[int]) -> int:
        try:
            candidate = int(value) if value is not None else self.default_top_k
        except (TypeError, ValueError):
            candidate = self.default_top_k
        return max(self.min_top_k, min(candidate, self.max_top_k))

    @staticmethod
    def _as_bool(value: Optional[str], default: bool) -> bool:
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}
