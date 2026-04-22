# News Search Skill Code Map

## WhatsApp Gateway

- `services/whatsapp_gateway/app/skills/news_search_skill.py`
  - DuckDuckGo news retrieval.
  - LLM query rewrite and dynamic top-k planning.
  - Language-preserving answer synthesis.
  - Reusable candidate listing and numbered selection prompt for downstream meme workflow.
- `services/whatsapp_gateway/app/skills/meme_generation_skill.py`
  - Uses news skill candidate mode for trending/news meme requests.
  - Accepts number or natural-language item selection before image generation.
- `services/whatsapp_gateway/app/supervisor.py`
  - Adds `news_search` route type in LLM routing policy.
  - Adds `meme_generation` route type and intent policy.
- `services/whatsapp_gateway/app/main.py`
  - Dispatches `news_search` route in Meta and Twilio flows.
  - Sends a lightweight fetching message and no thinking indicator.
  - Dispatches `meme_generation` route with background progress updates and media delivery.

## Tests

- `services/whatsapp_gateway/tests/test_news_search_skill.py`
  - Dynamic top-k and language/fallback behavior checks.
  - Candidate list and selection prompt helper checks.
- `services/whatsapp_gateway/tests/test_main_routing.py`
  - Verifies route dispatch and no RAG-thinking behavior, including meme route.
- `services/whatsapp_gateway/tests/test_supervisor.py`
  - Verifies `news_search` and `meme_generation` routes pass through supervisor decisions.
- `services/whatsapp_gateway/tests/test_meme_generation_skill.py`
  - Verifies news-selection handoff, natural-language selection, and max-3 image clamping.

## Runtime Expectations

- Should keep response language aligned with user language.
- Should use dynamic top-k (not fixed-only top 5).
- Should avoid RAG retrieval path for web news requests.
