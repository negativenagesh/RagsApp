import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.skills.news_search_skill import NewsSearchSkill


class NewsSearchSkillTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._previous_env = dict(os.environ)
        os.environ["NEWS_SEARCH_TOP_K_DEFAULT"] = "5"
        os.environ["NEWS_SEARCH_TOP_K_MIN"] = "3"
        os.environ["NEWS_SEARCH_TOP_K_MAX"] = "10"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._previous_env)

    def test_dynamic_top_k_respects_explicit_count(self):
        skill = NewsSearchSkill()
        self.assertEqual(skill._dynamic_top_k("top 8 startup news"), 8)

    def test_fetching_message_uses_kannada_from_context(self):
        skill = NewsSearchSkill()
        history = [{"role": "user", "content": "ನನಗೆ ಇವತ್ತಿನ ಟೆಕ್ ಸುದ್ದಿ ಬೇಕು"}]

        msg = skill.get_fetching_message("latest ai news", conversation_history=history)

        self.assertEqual(msg, "ಇತ್ತೀಚಿನ ಸುದ್ದಿಗಳನ್ನು ಪಡೆಯಲಾಗುತ್ತಿದೆ... ಒಂದು ಕ್ಷಣ.")

    async def test_ask_falls_back_with_results_without_openai(self):
        skill = NewsSearchSkill()
        skill._openai_client = None
        skill._search_news = AsyncMock(
            return_value=[
                {
                    "title": "AI startup raises funding",
                    "snippet": "Round details",
                    "url": "https://example.com/ai-startup",
                    "source": "Example News",
                    "date": "2026-01-10",
                }
            ]
        )

        result = await skill.ask("latest ai startup news")

        self.assertIn("Top news for", result)
        self.assertIn("AI startup raises funding", result)
        self.assertIn("https://example.com/ai-startup", result)

    async def test_ask_returns_no_results_message_in_kannada(self):
        skill = NewsSearchSkill()
        skill._openai_client = None
        skill._search_news = AsyncMock(return_value=[])

        result = await skill.ask("ಇವತ್ತಿನ ರಾಜಕೀಯ ಸುದ್ದಿ")

        self.assertIn("ಹೊಸ ಸುದ್ದಿಗಳು ಸಿಗಲಿಲ್ಲ", result)

    async def test_get_news_candidates_respects_force_top_k(self):
        skill = NewsSearchSkill()
        skill._openai_client = None
        skill._search_news = AsyncMock(
            return_value=[
                {"title": "A", "snippet": "", "url": "https://a", "source": "S", "date": "2026-01-01"},
                {"title": "B", "snippet": "", "url": "https://b", "source": "S", "date": "2026-01-01"},
                {"title": "C", "snippet": "", "url": "https://c", "source": "S", "date": "2026-01-01"},
            ]
        )

        items, rewritten_query, top_k, language = await skill.get_news_candidates(
            user_query="latest ai news",
            force_top_k=2,
        )

        self.assertEqual(top_k, 3)
        self.assertEqual(len(items), 3)
        self.assertEqual(rewritten_query, "latest ai news")
        self.assertEqual(language, "en")

    def test_format_selection_prompt_has_numbered_choices(self):
        skill = NewsSearchSkill()

        text = skill.format_selection_prompt(
            original_query="latest ai news",
            items=[
                {"title": "OpenAI update", "source": "Example"},
                {"title": "Chip race", "source": "Example"},
            ],
            language="en",
        )

        self.assertIn("1. OpenAI update", text)
        self.assertIn("2. Chip race", text)


if __name__ == "__main__":
    unittest.main()
