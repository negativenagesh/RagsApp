import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app import main as gateway_main
from app.supervisor import SupervisorDecision
from app.skills.meme_generation_skill import MemeGenerationOutput


class MainRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._previous_env = dict(os.environ)
        os.environ["WHATSAPP_PROVIDER"] = "meta"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._previous_env)

    async def test_non_rag_route_does_not_send_thinking_indicator(self):
        gateway_main.SUPERVISOR.get_recent_conversation_history = AsyncMock(return_value=[])
        gateway_main.SUPERVISOR.decide = AsyncMock(
            return_value=SupervisorDecision(
                route_type="non_rag_reply",
                final_text="Hey there!",
                confidence=0.95,
                reason_code="social_chat",
            )
        )
        gateway_main.SUPERVISOR.record_answer = AsyncMock()
        gateway_main._send_text_in_chunks = AsyncMock()
        gateway_main.send_thinking_indicator = AsyncMock(return_value=True)
        gateway_main._try_stream_answer = AsyncMock(return_value=(False, ""))
        gateway_main.ask_rag_with_progress = AsyncMock(return_value="should not run")

        await gateway_main._process_meta_text_message("919000000000", "Wassup", "m-1")

        gateway_main.send_thinking_indicator.assert_not_awaited()
        gateway_main._try_stream_answer.assert_not_awaited()
        gateway_main.ask_rag_with_progress.assert_not_awaited()

    async def test_rag_route_sends_thinking_indicator(self):
        gateway_main.SUPERVISOR.get_recent_conversation_history = AsyncMock(return_value=[])
        gateway_main.SUPERVISOR.decide = AsyncMock(
            return_value=SupervisorDecision(
                route_type="rag_retrieval",
                final_text="",
                confidence=0.9,
                reason_code="kb_required",
            )
        )
        gateway_main.SUPERVISOR.record_answer = AsyncMock()
        gateway_main.send_thinking_indicator = AsyncMock(return_value=True)
        gateway_main.send_whatsapp_message = AsyncMock(return_value=True)
        gateway_main._send_text_in_chunks = AsyncMock()
        gateway_main._try_stream_answer = AsyncMock(return_value=(True, "streamed output"))

        await gateway_main._process_meta_text_message("919000000000", "summarize uploaded pdf", "m-2")

        gateway_main.send_thinking_indicator.assert_awaited_once()
        gateway_main._try_stream_answer.assert_awaited_once()

    async def test_arecanut_route_uses_tool_without_thinking_indicator(self):
        history = [
            {"role": "user", "content": "what is arecanut price today"},
            {"role": "assistant", "content": "Please tell me the state to fetch arecanut mandi price."},
        ]
        gateway_main.SUPERVISOR.get_recent_conversation_history = AsyncMock(return_value=history)
        gateway_main.SUPERVISOR.decide = AsyncMock(
            return_value=SupervisorDecision(
                route_type="arecanut_price",
                final_text="",
                confidence=0.9,
                reason_code="arecanut_price_lookup",
            )
        )
        gateway_main.SUPERVISOR.record_answer = AsyncMock()
        gateway_main.ARECANUT_PRICE_TOOL.ask = AsyncMock(
            return_value=(
                "Arecanut mandi price in yellapur, karnataka\n"
                "Price updated: 14 Apr '26, 10:05 am\n"
                "Source: https://www.commodityonline.com/mandiprices/arecanut-betelnutsupari/karnataka/yellapur"
            )
        )
        gateway_main.send_whatsapp_message = AsyncMock(return_value=True)
        gateway_main._send_text_in_chunks = AsyncMock()
        gateway_main.send_thinking_indicator = AsyncMock(return_value=True)
        gateway_main._try_stream_answer = AsyncMock(return_value=(False, ""))
        gateway_main.ask_rag_with_progress = AsyncMock(return_value="should not run")

        await gateway_main._process_meta_text_message("919000000000", "arecanut price in yellapur", "m-3")

        gateway_main.ARECANUT_PRICE_TOOL.ask.assert_awaited_once_with(
            "arecanut price in yellapur",
            conversation_history=history,
        )
        gateway_main.send_whatsapp_message.assert_awaited_once_with(
            "919000000000",
            "Fetching latest arecanut mandi prices... Just a moment.",
        )
        gateway_main.send_thinking_indicator.assert_not_awaited()
        gateway_main._try_stream_answer.assert_not_awaited()
        gateway_main.ask_rag_with_progress.assert_not_awaited()

    async def test_news_route_uses_skill_without_thinking_indicator(self):
        history = [{"role": "user", "content": "latest ai news"}]
        gateway_main.SUPERVISOR.get_recent_conversation_history = AsyncMock(return_value=history)
        gateway_main.SUPERVISOR.decide = AsyncMock(
            return_value=SupervisorDecision(
                route_type="news_search",
                final_text="",
                confidence=0.91,
                reason_code="news_search_route",
            )
        )
        gateway_main.SUPERVISOR.record_answer = AsyncMock()
        gateway_main.NEWS_SEARCH_SKILL.ask = AsyncMock(return_value="1. Headline\n2. Headline")
        gateway_main.send_whatsapp_message = AsyncMock(return_value=True)
        gateway_main._send_text_in_chunks = AsyncMock()
        gateway_main.send_thinking_indicator = AsyncMock(return_value=True)
        gateway_main._try_stream_answer = AsyncMock(return_value=(False, ""))
        gateway_main.ask_rag_with_progress = AsyncMock(return_value="should not run")

        await gateway_main._process_meta_text_message("919000000000", "latest ai news", "m-4")

        gateway_main.NEWS_SEARCH_SKILL.ask.assert_awaited_once_with(
            "latest ai news",
            conversation_history=history,
        )
        gateway_main.send_whatsapp_message.assert_awaited_once_with(
            "919000000000",
            "Fetching latest news updates... Just a moment.",
        )
        gateway_main.send_thinking_indicator.assert_not_awaited()
        gateway_main._try_stream_answer.assert_not_awaited()
        gateway_main.ask_rag_with_progress.assert_not_awaited()

    async def test_meme_route_uses_skill_without_thinking_indicator(self):
        history = [{"role": "user", "content": "create 2 memes about ai layoffs"}]
        gateway_main.SUPERVISOR.get_recent_conversation_history = AsyncMock(return_value=history)
        gateway_main.SUPERVISOR.decide = AsyncMock(
            return_value=SupervisorDecision(
                route_type="meme_generation",
                final_text="",
                confidence=0.89,
                reason_code="meme_generation_route",
            )
        )
        gateway_main.SUPERVISOR.record_answer = AsyncMock()
        gateway_main.MEME_GENERATION_SKILL.has_pending_selection = Mock(return_value=False)
        gateway_main.MEME_GENERATION_SKILL.ask = AsyncMock(
            return_value=MemeGenerationOutput(
                text="1. Meme one\n2. Meme two",
                image_paths=["/tmp/meme1.png", "/tmp/meme2.png"],
                image_captions=["cap1", "cap2"],
                awaiting_user_input=False,
            )
        )
        gateway_main._send_generated_images = AsyncMock(return_value=2)
        gateway_main.send_whatsapp_message = AsyncMock(return_value=True)
        gateway_main._send_text_in_chunks = AsyncMock()
        gateway_main.send_thinking_indicator = AsyncMock(return_value=True)
        gateway_main._try_stream_answer = AsyncMock(return_value=(False, ""))
        gateway_main.ask_rag_with_progress = AsyncMock(return_value="should not run")

        await gateway_main._process_meta_text_message("919000000000", "create 2 memes about ai layoffs", "m-5")

        gateway_main.MEME_GENERATION_SKILL.ask.assert_awaited_once()
        gateway_main.send_thinking_indicator.assert_not_awaited()
        gateway_main._try_stream_answer.assert_not_awaited()
        gateway_main.ask_rag_with_progress.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
