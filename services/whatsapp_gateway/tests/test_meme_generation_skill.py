import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.skills.meme_generation_skill import (
    MemeGenerationOutput,
    MemeGenerationSkill,
    PendingNewsSelection,
)


class MemeGenerationSkillTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._previous_env = dict(os.environ)
        os.environ["MEME_MAX_IMAGES"] = "3"
        os.environ["MEME_DEFAULT_IMAGES"] = "1"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._previous_env)

    async def test_news_selection_flow_clamps_image_count_to_three(self):
        mock_news_skill = Mock()
        mock_news_skill.get_news_candidates = AsyncMock(
            return_value=(
                [
                    {
                        "title": "AI layoffs surge",
                        "snippet": "Tech workforce update",
                        "source": "News A",
                        "url": "https://example.com/a",
                    },
                    {
                        "title": "Funding rebounds",
                        "snippet": "VC trend",
                        "source": "News B",
                        "url": "https://example.com/b",
                    },
                ],
                "ai layoffs latest",
                5,
                "en",
            )
        )
        mock_news_skill.format_selection_prompt = Mock(return_value="Choose one item")

        skill = MemeGenerationSkill(news_skill=mock_news_skill)
        skill._requires_news_selection = AsyncMock(return_value=True)

        result = await skill.ask(
            user_id="user-1",
            user_query="make 5 memes from latest ai news",
            conversation_history=[],
        )

        self.assertTrue(result.awaiting_user_input)
        self.assertIn("maximum allowed is 3", result.text.lower())
        self.assertTrue(skill.has_pending_selection("user-1"))

    async def test_pending_selection_allows_natural_language_pick(self):
        skill = MemeGenerationSkill(news_skill=Mock())
        skill._pending_by_user["user-2"] = PendingNewsSelection(
            created_at=9999999999,
            original_query="make trending meme",
            language="en",
            num_images=1,
            rewritten_query="latest trending news",
            news_items=[
                {"title": "OpenAI launches feature", "snippet": "Details", "source": "Source A"},
                {"title": "Market crash update", "snippet": "Details", "source": "Source B"},
            ],
        )
        skill._generate_memes = AsyncMock(
            return_value=MemeGenerationOutput(
                text="done",
                image_paths=["/tmp/a.png"],
                image_captions=["caption"],
                awaiting_user_input=False,
            )
        )

        result = await skill.ask(
            user_id="user-2",
            user_query="the OpenAI one and make 2 images",
            conversation_history=[],
        )

        self.assertFalse(result.awaiting_user_input)
        args = skill._generate_memes.await_args.kwargs
        self.assertEqual(args["num_images"], 2)

    def test_extract_image_count_defaults_to_one(self):
        skill = MemeGenerationSkill(news_skill=Mock())
        count, explicit = skill._extract_image_count("make meme about coding life")

        self.assertEqual(count, 1)
        self.assertFalse(explicit)


if __name__ == "__main__":
    unittest.main()
