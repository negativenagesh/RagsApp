import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.supervisor import MessageSupervisor, SupervisorDecision


class SupervisorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._previous_env = dict(os.environ)
        os.environ["SUPERVISOR_ENABLED"] = "true"
        os.environ["SUPERVISOR_FAIL_OPEN"] = "true"
        os.environ["SUPERVISOR_USE_LLM_ROUTER"] = "true"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._previous_env)

    async def test_non_rag_route_uses_llm_reply(self):
        supervisor = MessageSupervisor()
        supervisor._llm_route = AsyncMock(
            return_value=SupervisorDecision(
                route_type="non_rag_reply",
                final_text="",
                confidence=0.82,
                reason_code="social_chat",
            )
        )
        supervisor._llm_non_rag_reply = AsyncMock(return_value="Hey there! How can I help?")

        decision = await supervisor.decide(provider="meta", user_id="user-1", user_message="Wassup")

        self.assertEqual(decision.route_type, "non_rag_reply")
        self.assertEqual(decision.reason_code, "social_chat")
        self.assertEqual(decision.final_text, "Hey there! How can I help?")

    async def test_low_confidence_non_rag_still_avoids_rag(self):
        supervisor = MessageSupervisor()
        supervisor._llm_route = AsyncMock(
            return_value=SupervisorDecision(
                route_type="non_rag_reply",
                final_text="",
                confidence=0.2,
                reason_code="uncertain_social",
            )
        )
        supervisor._llm_non_rag_reply = AsyncMock(return_value="Hi! I am here.")

        decision = await supervisor.decide(provider="meta", user_id="user-1", user_message="hey")

        self.assertEqual(decision.route_type, "non_rag_reply")
        self.assertEqual(decision.reason_code, "uncertain_social")
        self.assertEqual(decision.final_text, "Hi! I am here.")

    async def test_document_prompt_routes_to_rag(self):
        supervisor = MessageSupervisor()
        supervisor._llm_route = AsyncMock(
            return_value=SupervisorDecision(
                route_type="rag_retrieval",
                final_text="",
                confidence=0.91,
                reason_code="needs_documents",
            )
        )
        supervisor._llm_requires_rag = AsyncMock(return_value=(True, 0.94, "kb_required"))
        decision = await supervisor.decide(
            provider="meta",
            user_id="user-1",
            user_message="summarize the uploaded pdf report",
        )

        self.assertEqual(decision.route_type, "rag_retrieval")
        self.assertEqual(decision.reason_code, "kb_required")

    async def test_memory_hit_short_circuits(self):
        supervisor = MessageSupervisor()
        supervisor.memory.get_cached_answer = AsyncMock(
            return_value={"final_answer": "cached answer", "confidence": 0.99}
        )
        supervisor._llm_should_use_cached_answer = AsyncMock(return_value=True)

        decision = await supervisor.decide(provider="meta", user_id="user-1", user_message="repeat this")

        self.assertEqual(decision.route_type, "non_rag_reply")
        self.assertEqual(decision.reason_code, "memory_hit")
        self.assertTrue(decision.memory_hit)
        self.assertEqual(decision.final_text, "cached answer")

    async def test_memory_hit_rejected_by_llm_continues_to_router(self):
        supervisor = MessageSupervisor()
        supervisor.memory.get_cached_answer = AsyncMock(
            return_value={"final_answer": "Could you clarify what you want to know?", "confidence": 0.5}
        )
        supervisor._llm_should_use_cached_answer = AsyncMock(return_value=False)
        supervisor._llm_route = AsyncMock(
            return_value=SupervisorDecision(
                route_type="arecanut_price",
                final_text="",
                confidence=0.9,
                reason_code="arecanut_price_lookup",
            )
        )

        history = [
            {"role": "user", "content": "What is today's arecanut mandi price?"},
            {"role": "assistant", "content": "Please tell me the state to fetch arecanut mandi price."},
        ]
        decision = await supervisor.decide(
            provider="meta",
            user_id="user-1",
            user_message="Karnataka",
            conversation_history=history,
        )

        self.assertEqual(decision.route_type, "arecanut_price")
        self.assertEqual(decision.reason_code, "arecanut_price_lookup")

    async def test_low_confidence_fail_open_routes_to_rag(self):
        supervisor = MessageSupervisor()
        supervisor._llm_route = AsyncMock(
            return_value=SupervisorDecision(
                route_type="rag_retrieval",
                final_text="",
                confidence=0.2,
                reason_code="uncertain_docs",
            )
        )
        supervisor._llm_requires_rag = AsyncMock(return_value=(True, 0.2, "uncertain_docs"))

        decision = await supervisor.decide(provider="meta", user_id="user-1", user_message="ambiguous question")

        self.assertEqual(decision.route_type, "rag_retrieval")
        self.assertEqual(decision.reason_code, "low_confidence_fail_open")

    async def test_rag_route_rejected_by_confirmation_uses_non_rag(self):
        supervisor = MessageSupervisor()
        supervisor._llm_route = AsyncMock(
            return_value=SupervisorDecision(
                route_type="rag_retrieval",
                final_text="",
                confidence=0.88,
                reason_code="possible_docs",
            )
        )
        supervisor._llm_requires_rag = AsyncMock(return_value=(False, 0.92, "social_chat"))
        supervisor._llm_non_rag_reply = AsyncMock(return_value="Not much, how are you?")

        decision = await supervisor.decide(provider="meta", user_id="user-1", user_message="Wassup")

        self.assertEqual(decision.route_type, "non_rag_reply")
        self.assertEqual(decision.reason_code, "social_chat")
        self.assertEqual(decision.final_text, "Not much, how are you?")

    async def test_router_unavailable_falls_back_to_non_rag_reply(self):
        supervisor = MessageSupervisor()
        supervisor._llm_route = AsyncMock(return_value=None)
        supervisor._llm_non_rag_reply = AsyncMock(return_value="I can help with that.")

        decision = await supervisor.decide(provider="meta", user_id="user-1", user_message="hello")

        self.assertEqual(decision.route_type, "non_rag_reply")
        self.assertEqual(decision.reason_code, "router_unavailable_non_rag_fallback")

    async def test_arecanut_price_route_passes_through(self):
        supervisor = MessageSupervisor()
        supervisor._llm_route = AsyncMock(
            return_value=SupervisorDecision(
                route_type="arecanut_price",
                final_text="",
                confidence=0.92,
                reason_code="arecanut_price_lookup",
            )
        )

        decision = await supervisor.decide(
            provider="meta",
            user_id="user-1",
            user_message="What is arecanut price in karnataka yellapur?",
        )

        self.assertEqual(decision.route_type, "arecanut_price")
        self.assertEqual(decision.reason_code, "arecanut_price_lookup")

    async def test_news_search_route_passes_through(self):
        supervisor = MessageSupervisor()
        supervisor._llm_route = AsyncMock(
            return_value=SupervisorDecision(
                route_type="news_search",
                final_text="",
                confidence=0.93,
                reason_code="news_search_route",
            )
        )

        decision = await supervisor.decide(
            provider="meta",
            user_id="user-1",
            user_message="latest technology headlines",
        )

        self.assertEqual(decision.route_type, "news_search")
        self.assertEqual(decision.reason_code, "news_search_route")

    async def test_meme_generation_route_passes_through(self):
        supervisor = MessageSupervisor()
        supervisor._llm_route = AsyncMock(
            return_value=SupervisorDecision(
                route_type="meme_generation",
                final_text="",
                confidence=0.9,
                reason_code="meme_generation_route",
            )
        )

        decision = await supervisor.decide(
            provider="meta",
            user_id="user-1",
            user_message="create 3 memes about startup life",
        )

        self.assertEqual(decision.route_type, "meme_generation")
        self.assertEqual(decision.reason_code, "meme_generation_route")

    async def test_record_answer_skips_ask_clarification(self):
        supervisor = MessageSupervisor()
        supervisor.memory.save_answer = AsyncMock()

        decision = SupervisorDecision(
            route_type="ask_clarification",
            final_text="Could you clarify what you want to know?",
            confidence=0.9,
            reason_code="clarify",
        )

        await supervisor.record_answer(
            provider="meta",
            user_id="user-1",
            user_query="Karnataka",
            answer_text=decision.final_text,
            decision=decision,
            used_rag=False,
        )

        supervisor.memory.save_answer.assert_not_awaited()

    async def test_runtime_history_fallback_is_available_when_memory_is_empty(self):
        supervisor = MessageSupervisor()
        supervisor.memory.get_recent_conversation_history = AsyncMock(return_value=[])
        supervisor.memory.save_answer = AsyncMock()

        decision = SupervisorDecision(
            route_type="arecanut_price",
            final_text="",
            confidence=0.95,
            reason_code="arecanut_price_lookup",
        )

        await supervisor.record_answer(
            provider="meta",
            user_id="user-1",
            user_query="ಇಂದಿನ ಅಡಿಕೆ ಬೆಲೆ",
            answer_text="ಅಡಕೆ ಮಾರುಕಟ್ಟೆ ಬೆಲೆ ಪಡೆಯಲು ದಯವಿಟ್ಟು ರಾಜ್ಯವನ್ನು ತಿಳಿಸಿ.",
            decision=decision,
            used_rag=False,
        )

        history = await supervisor.get_recent_conversation_history(provider="meta", user_id="user-1")

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertIn("ಇಂದಿನ ಅಡಿಕೆ ಬೆಲೆ", history[0]["content"])


if __name__ == "__main__":
    unittest.main()
