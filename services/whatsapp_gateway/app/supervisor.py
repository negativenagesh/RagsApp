import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from app.conversation_memory import ConversationMemoryStore


ALLOWED_ROUTE_TYPES = {
    "non_rag_reply",
    "rag_retrieval",
    "ask_clarification",
    "arecanut_price",
    "news_search",
    "meme_generation",
}


@dataclass
class SupervisorDecision:
    route_type: str
    final_text: str
    confidence: float
    reason_code: str
    memory_hit: bool = False
    used_rag: bool = False


class MessageSupervisor:
    def __init__(self) -> None:
        self.enabled = self._as_bool(os.getenv("SUPERVISOR_ENABLED"), False)
        self.fail_open = self._as_bool(os.getenv("SUPERVISOR_FAIL_OPEN"), True)
        self.use_llm_router = self._as_bool(os.getenv("SUPERVISOR_USE_LLM_ROUTER"), True)
        self.memory_llm_validation = self._as_bool(
            os.getenv("SUPERVISOR_MEMORY_LLM_VALIDATION"),
            True,
        )
        self.confidence_threshold = float(os.getenv("SUPERVISOR_CONFIDENCE_THRESHOLD", "0.65"))

        self.non_rag_model = os.getenv("SUPERVISOR_NONRAG_MODEL", "gpt-4o-mini")
        self.openai_api_key = os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY")
        self._openai_client: Optional[AsyncOpenAI] = None
        self.memory = ConversationMemoryStore()
        self.runtime_context_enabled = self._as_bool(
            os.getenv("SUPERVISOR_RUNTIME_CONTEXT_ENABLED"),
            True,
        )
        self.runtime_context_max_turns = max(2, min(self.memory.context_max_turns, 30))
        self._runtime_history: Dict[str, List[Dict[str, str]]] = {}

    async def startup(self) -> None:
        if self.openai_api_key:
            self._openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        await self.memory.startup()

    async def shutdown(self) -> None:
        if self._openai_client and hasattr(self._openai_client, "aclose"):
            try:
                await self._openai_client.aclose()
            except Exception as exc:
                print(f"Supervisor OpenAI client close failed: {exc}")
        self._openai_client = None
        self._runtime_history.clear()
        await self.memory.shutdown()

    async def get_recent_conversation_history(self, provider: str, user_id: str) -> List[Dict[str, str]]:
        memory_history = await self.memory.get_recent_conversation_history(provider=provider, user_id=user_id)
        runtime_history = self._get_runtime_history(provider=provider, user_id=user_id)
        if not runtime_history:
            return memory_history
        if not memory_history:
            return runtime_history
        combined = (memory_history + runtime_history)[-self.runtime_context_max_turns :]
        return combined

    async def decide(
        self,
        provider: str,
        user_id: str,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> SupervisorDecision:
        text = (user_message or "").strip()
        history = conversation_history or []
        if not text:
            return SupervisorDecision(
                route_type="ask_clarification",
                final_text=self._clarification_text(text=text, conversation_history=history),
                confidence=0.99,
                reason_code="empty_message",
            )

        if not self.enabled:
            return SupervisorDecision(
                route_type="rag_retrieval",
                final_text="",
                confidence=1.0,
                reason_code="supervisor_disabled",
            )

        memory_hit = await self.memory.get_cached_answer(provider=provider, user_id=user_id, query=text)
        if memory_hit:
            use_memory = await self._llm_should_use_cached_answer(
                text=text,
                conversation_history=history,
                cached_doc=memory_hit,
            )
            if use_memory:
                return SupervisorDecision(
                    route_type="non_rag_reply",
                    final_text=memory_hit.get("final_answer", ""),
                    confidence=float(memory_hit.get("confidence", 1.0)),
                    reason_code="memory_hit",
                    memory_hit=True,
                )

        llm_decision = await self._llm_route(text, conversation_history=history)
        if llm_decision:
            if llm_decision.route_type == "non_rag_reply":
                reply_text = await self._llm_non_rag_reply(text, conversation_history=history)
                return SupervisorDecision(
                    route_type="non_rag_reply",
                    final_text=reply_text,
                    confidence=llm_decision.confidence,
                    reason_code=llm_decision.reason_code,
                )

            if llm_decision.route_type == "rag_retrieval":
                rag_check = await self._llm_requires_rag(text, conversation_history=history)
                if rag_check is not None:
                    requires_rag, rag_confidence, rag_reason = rag_check
                    if not requires_rag:
                        reply_text = await self._llm_non_rag_reply(text, conversation_history=history)
                        return SupervisorDecision(
                            route_type="non_rag_reply",
                            final_text=reply_text,
                            confidence=rag_confidence,
                            reason_code=rag_reason,
                        )
                    llm_decision.confidence = rag_confidence
                    llm_decision.reason_code = rag_reason

            if llm_decision.confidence < self.confidence_threshold:
                if self.fail_open and llm_decision.route_type == "rag_retrieval":
                    return SupervisorDecision(
                        route_type="rag_retrieval",
                        final_text="",
                        confidence=llm_decision.confidence,
                        reason_code="low_confidence_fail_open",
                    )
                return SupervisorDecision(
                    route_type="ask_clarification",
                    final_text=self._clarification_text(text=text, conversation_history=history),
                    confidence=llm_decision.confidence,
                    reason_code="low_confidence_clarify",
                )

            if llm_decision.route_type == "ask_clarification":
                return SupervisorDecision(
                    route_type="ask_clarification",
                    final_text=self._clarification_text(text=text, conversation_history=history),
                    confidence=llm_decision.confidence,
                    reason_code=llm_decision.reason_code,
                )

            return llm_decision

        reply_text = await self._llm_non_rag_reply(text, conversation_history=history)
        return SupervisorDecision(
            route_type="non_rag_reply",
            final_text=reply_text,
            confidence=0.4,
            reason_code="router_unavailable_non_rag_fallback",
        )

    async def record_answer(
        self,
        provider: str,
        user_id: str,
        user_query: str,
        answer_text: str,
        decision: SupervisorDecision,
        used_rag: bool,
    ) -> None:
        self._append_runtime_turn(
            provider=provider,
            user_id=user_id,
            user_query=user_query,
            answer_text=answer_text,
            route_type=decision.route_type,
            reason_code=decision.reason_code,
        )

        if not self.enabled:
            return
        if decision.memory_hit:
            return
        if decision.route_type == "ask_clarification":
            return
        if decision.confidence < self.confidence_threshold:
            return
        await self.memory.save_answer(
            provider=provider,
            user_id=user_id,
            user_query=user_query,
            final_answer=answer_text,
            route_type=decision.route_type,
            reason_code=decision.reason_code,
            confidence=decision.confidence,
            used_rag=used_rag,
        )

    async def _llm_route(
        self,
        text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[SupervisorDecision]:
        if not self.use_llm_router or not self._openai_client:
            return None

        history_block = self._build_context_snippet(conversation_history)

        prompt = (
            "Route this user message for a WhatsApp assistant. "
            "Return only JSON with keys: route_type, confidence, reason_code. "
            "route_type must be one of non_rag_reply, rag_retrieval, ask_clarification, arecanut_price, news_search, meme_generation. "
            "Use arecanut_price when user asks for arecanut, betelnut, or supari mandi market prices and rates. "
            "Use news_search when user asks for latest news, headlines, breaking updates, or topic-based current events from the web. "
            "Use meme_generation when user asks to create, generate, or make memes, viral image jokes, brainrot-style images, or funny captioned images. "
            "If user asks for memes based on trending/current events, still choose meme_generation so the meme workflow can fetch and present selectable news first. "
            "If the current message is short but clearly continues prior arecanut slot-filling (state/market), choose arecanut_price. "
            "If the assistant previously asked for arecanut state/market and the user now replies with only a state or market name (any language/script), choose arecanut_price. "
            "Use non_rag_reply for greetings, chitchat, social conversation, casual questions, and non-document requests. "
            "Use rag_retrieval only when the user likely needs facts from uploaded files or enterprise knowledge base. "
            "Use ask_clarification when intent is unclear and neither route is safe. "
            f"Recent conversation context:\n{history_block}\n"
            f"User message: {text}"
        )

        try:
            response = await self._openai_client.chat.completions.create(
                model=self.non_rag_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a strict JSON router."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=180,
            )
            raw = (response.choices[0].message.content or "{}").strip()
            data = json.loads(raw)

            route_type = str(data.get("route_type", "")).strip()
            if route_type not in ALLOWED_ROUTE_TYPES:
                return None

            confidence = float(data.get("confidence", 0.5))
            reason_code = str(data.get("reason_code", "llm_router"))[:80]

            return SupervisorDecision(
                route_type=route_type,
                final_text="",
                confidence=max(0.0, min(confidence, 1.0)),
                reason_code=reason_code,
            )
        except Exception as exc:
            print(f"LLM router failed: {exc}")
            return None

    async def _llm_non_rag_reply(
        self,
        text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        if not self._openai_client:
            return "I can help with that."

        history_block = self._build_context_snippet(conversation_history)

        try:
            response = await self._openai_client.chat.completions.create(
                model=self.non_rag_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise WhatsApp assistant. "
                            "Answer directly and naturally in plain text. "
                            "Do not invent document facts. "
                            "If the user asks for information that needs uploaded documents, "
                            "ask them to ask a document-specific question."
                        ),
                    },
                    {
                        "role": "system",
                        "content": f"Recent conversation context:\n{history_block}",
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.4,
                max_tokens=220,
            )
            content = (response.choices[0].message.content or "").strip()
            return content or "I can help with that."
        except Exception as exc:
            print(f"LLM non-rag response failed: {exc}")
            return "I can help with that."

    async def _llm_requires_rag(
        self,
        text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[tuple[bool, float, str]]:
        if not self._openai_client:
            return None

        history_block = self._build_context_snippet(conversation_history)

        prompt = (
            "Decide if this message requires retrieval from uploaded documents or enterprise knowledge base. "
            "Return only JSON with keys: requires_rag (boolean), confidence (0 to 1), reason_code (short string). "
            "Set requires_rag=false for greetings, casual chat, opinions, or generic conversational messages. "
            "Set requires_rag=true only when the user is asking for factual content likely stored in documents or knowledge base. "
            f"Recent conversation context:\n{history_block}\n"
            f"User message: {text}"
        )

        try:
            response = await self._openai_client.chat.completions.create(
                model=self.non_rag_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a strict JSON classifier."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=140,
            )
            raw = (response.choices[0].message.content or "{}").strip()
            data = json.loads(raw)

            requires_rag = bool(data.get("requires_rag", False))
            confidence = float(data.get("confidence", 0.5))
            reason_code = str(data.get("reason_code", "rag_check"))[:80]
            return requires_rag, max(0.0, min(confidence, 1.0)), reason_code
        except Exception as exc:
            print(f"LLM rag-requirement check failed: {exc}")
            return None

    async def _llm_should_use_cached_answer(
        self,
        text: str,
        conversation_history: Optional[List[Dict[str, str]]],
        cached_doc: Dict[str, str],
    ) -> bool:
        if not self.memory_llm_validation:
            return True
        if not self._openai_client:
            return False

        history_block = self._build_context_snippet(conversation_history)
        cached_query = str(cached_doc.get("user_query", "")).strip()
        cached_answer = str(cached_doc.get("final_answer", "")).strip()
        cached_route = str(cached_doc.get("route_type", "")).strip()

        prompt = (
            "Decide whether a cached answer should be reused for the current user message. "
            "Return only JSON with keys: use_memory (boolean), confidence (0 to 1), reason_code (short string). "
            "Set use_memory=false if the cached answer is stale, too generic, mismatched, or if the current message continues a different ongoing thread. "
            "If the cached answer is a clarification question and the current user message supplies new specific information, set use_memory=false. "
            f"Current user message: {text}\n"
            f"Recent conversation context:\n{history_block}\n"
            f"Cached user query: {cached_query}\n"
            f"Cached answer: {cached_answer}\n"
            f"Cached route_type: {cached_route}"
        )

        try:
            response = await self._openai_client.chat.completions.create(
                model=self.non_rag_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a strict JSON validator for conversational cache reuse."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=140,
            )
            raw = (response.choices[0].message.content or "{}").strip()
            data = json.loads(raw)
            return bool(data.get("use_memory", False))
        except Exception as exc:
            print(f"LLM memory validation failed: {exc}")
            return False

    @staticmethod
    def _as_bool(value: Optional[str], default: bool) -> bool:
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _build_context_snippet(
        conversation_history: Optional[List[Dict[str, str]]],
        max_turns: int = 6,
        max_chars: int = 1200,
    ) -> str:
        if not conversation_history:
            return "(none)"

        turns = conversation_history[-max_turns:]
        lines: List[str] = []
        for turn in turns:
            role = str(turn.get("role", "user")).strip().lower()
            prefix = "User" if role == "user" else "Assistant"
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            content = re.sub(r"\s+", " ", content)
            if len(content) > 220:
                content = content[:217].rstrip() + "..."
            lines.append(f"{prefix}: {content}")

        rendered = "\n".join(lines) if lines else "(none)"
        if len(rendered) > max_chars:
            rendered = rendered[-max_chars:]
        return rendered

    def _runtime_key(self, provider: str, user_id: str) -> str:
        return f"{provider}::{user_id}"

    def _get_runtime_history(self, provider: str, user_id: str) -> List[Dict[str, str]]:
        if not self.runtime_context_enabled:
            return []
        key = self._runtime_key(provider=provider, user_id=user_id)
        return list(self._runtime_history.get(key, []))

    def _append_runtime_turn(
        self,
        provider: str,
        user_id: str,
        user_query: str,
        answer_text: str,
        route_type: str,
        reason_code: str,
    ) -> None:
        if not self.runtime_context_enabled:
            return

        query = str(user_query or "").strip()
        answer = str(answer_text or "").strip()
        if not query or not answer:
            return

        key = self._runtime_key(provider=provider, user_id=user_id)
        turns = self._runtime_history.setdefault(key, [])
        turns.append(
            {
                "role": "user",
                "content": query,
                "route_type": route_type,
                "reason_code": reason_code,
            }
        )
        turns.append(
            {
                "role": "assistant",
                "content": answer,
                "route_type": route_type,
                "reason_code": reason_code,
            }
        )

        if len(turns) > self.runtime_context_max_turns:
            self._runtime_history[key] = turns[-self.runtime_context_max_turns :]

    def _clarification_text(
        self,
        text: str,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> str:
        if self._prefer_kannada(text=text, conversation_history=conversation_history):
            return "ದಯವಿಟ್ಟು ನೀವು ಯಾವ ಮಾಹಿತಿ ತಿಳಿದುಕೊಳ್ಳಬೇಕು ಎಂದು ಸ್ವಲ್ಪ ಸ್ಪಷ್ಟಪಡಿಸಿ."
        return "Could you clarify what you want to know?"

    @staticmethod
    def _prefer_kannada(text: str, conversation_history: Optional[List[Dict[str, str]]]) -> bool:
        if re.search(r"[\u0C80-\u0CFF]", text or ""):
            return True
        if not conversation_history:
            return False
        for turn in reversed(conversation_history[-6:]):
            content = str(turn.get("content", ""))
            if re.search(r"[\u0C80-\u0CFF]", content):
                return True
        return False
