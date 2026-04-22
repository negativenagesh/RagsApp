from typing import AsyncGenerator, Dict, List, Optional

from app.skills.rag_skill import RagSkill


class RetrievalTool:
    def __init__(self) -> None:
        self._skill = RagSkill()

    async def ask(
        self,
        question: str,
        top_k_chunks: Optional[int] = None,
        enable_references_citations: Optional[bool] = None,
        deep_research: Optional[bool] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        return await self._skill.ask(
            question=question,
            top_k_chunks=top_k_chunks,
            enable_references_citations=enable_references_citations,
            deep_research=deep_research,
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
        async for token in self._skill.stream(
            question=question,
            top_k_chunks=top_k_chunks,
            enable_references_citations=enable_references_citations,
            deep_research=deep_research,
            conversation_history=conversation_history or [],
        ):
            yield token

    def _normalize_top_k(self, value: Optional[int]) -> int:
        return self._skill.normalize_top_k(value)

