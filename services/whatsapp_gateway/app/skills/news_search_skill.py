import asyncio
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI


LANG_EN = "en"
LANG_KN = "kn"


class NewsSearchSkill:
    def __init__(self) -> None:
        self.default_top_k = int(os.getenv("NEWS_SEARCH_TOP_K_DEFAULT", "5"))
        self.min_top_k = int(os.getenv("NEWS_SEARCH_TOP_K_MIN", "3"))
        self.max_top_k = int(os.getenv("NEWS_SEARCH_TOP_K_MAX", "10"))
        self.search_pool_multiplier = int(os.getenv("NEWS_SEARCH_POOL_MULTIPLIER", "2"))
        self.region = os.getenv("NEWS_SEARCH_REGION", "wt-wt")
        self.safesearch = os.getenv("NEWS_SEARCH_SAFESEARCH", "moderate")
        self.timeout_seconds = float(os.getenv("NEWS_SEARCH_TIMEOUT_SECONDS", "20"))
        self.model = os.getenv("NEWS_SEARCH_LLM_MODEL", "gpt-4o-mini")

        openai_api_key = os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY")
        self._openai_client: Optional[AsyncOpenAI] = (
            AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None
        )

    async def ask(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        question = (user_query or "").strip()
        if not question:
            return self._msg(LANG_EN, "empty_query")

        selected_items, rewritten_query, top_k, language = await self.get_news_candidates(
            user_query=question,
            conversation_history=conversation_history,
        )
        if not selected_items:
            return self._msg(language, "no_results", query=question)

        synthesized = await self._synthesize_answer(
            original_query=question,
            rewritten_query=rewritten_query,
            language=language,
            top_k=top_k,
            items=selected_items,
        )
        if synthesized:
            return synthesized

        return self._fallback_answer(
            original_query=question,
            language=language,
            top_k=top_k,
            items=selected_items,
        )

    def get_fetching_message(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        language = self._resolve_language(user_query, conversation_history=conversation_history)
        return self._msg(language, "fetching")

    async def get_news_candidates(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        force_top_k: Optional[int] = None,
    ) -> Tuple[List[Dict[str, str]], str, int, str]:
        question = (user_query or "").strip()
        language = self._resolve_language(question, conversation_history=conversation_history)
        if not question:
            return [], "", self.default_top_k, language

        rewritten_query, top_k = await self._plan_query(
            question=question,
            language=language,
            conversation_history=conversation_history,
        )
        if force_top_k is not None:
            top_k = self._normalize_top_k(force_top_k, fallback=top_k)

        search_limit = max(top_k, top_k * max(1, self.search_pool_multiplier))
        news_items = await self._search_news(rewritten_query, max_results=search_limit)
        if not news_items and rewritten_query.strip().lower() != question.strip().lower():
            news_items = await self._search_news(question, max_results=search_limit)

        return news_items[:top_k], rewritten_query, top_k, language

    def format_selection_prompt(
        self,
        original_query: str,
        items: List[Dict[str, str]],
        language: str,
    ) -> str:
        if not items:
            return self._msg(language, "no_results", query=original_query)

        lines = [self._msg(language, "selection_heading", query=original_query)]
        for index, item in enumerate(items, start=1):
            source = item.get("source") or self._msg(language, "unknown_source")
            title = (item.get("title") or "").strip()
            lines.append(f"{index}. {title} ({source})")
        lines.append(self._msg(language, "selection_hint"))
        return "\n".join(lines)

    async def _plan_query(
        self,
        question: str,
        language: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, int]:
        fallback_top_k = self._dynamic_top_k(question)
        fallback_query = question

        if not self._openai_client:
            return fallback_query, fallback_top_k

        history_block = self._build_context_snippet(conversation_history)
        prompt = (
            "You are preparing a web-news search plan. "
            "Return strict JSON with keys rewritten_query and top_k. "
            "Rules: rewritten_query must be concise and optimized for finding latest news on DuckDuckGo news search. "
            f"top_k must be an integer between {self.min_top_k} and {self.max_top_k}. "
            "Use user intent: for broad topics return a larger top_k, for narrow topics return smaller top_k. "
            "If user explicitly asks for N headlines/news, honor it within range. "
            "Do not translate named entities incorrectly.\n"
            f"Language hint: {language}.\n"
            f"Recent context:\n{history_block}\n"
            f"User query: {question}"
        )

        try:
            response = await asyncio.wait_for(
                self._openai_client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are a strict JSON planner."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=180,
                ),
                timeout=self.timeout_seconds,
            )
            raw = (response.choices[0].message.content or "{}").strip()
            data = json.loads(raw)
            rewritten_query = str(data.get("rewritten_query", "")).strip() or fallback_query
            planned_top_k = self._normalize_top_k(data.get("top_k"), fallback=fallback_top_k)
            return rewritten_query, planned_top_k
        except Exception as exc:
            print(f"News planner fallback triggered: {exc}")
            return fallback_query, fallback_top_k

    async def _search_news(self, query: str, max_results: int) -> List[Dict[str, str]]:
        if not query.strip():
            return []

        try:
            from duckduckgo_search import DDGS
        except Exception as exc:
            print(f"duckduckgo-search import failed: {exc}")
            return []

        max_results = max(self.min_top_k, min(max_results, self.max_top_k * 3))

        def _run_search() -> List[Dict[str, str]]:
            ddgs = DDGS()
            try:
                raw_items = list(
                    ddgs.news(
                        keywords=query,
                        region=self.region,
                        safesearch=self.safesearch,
                        max_results=max_results,
                    )
                )
            finally:
                close_fn = getattr(ddgs, "close", None)
                if callable(close_fn):
                    close_fn()

            normalized: List[Dict[str, str]] = []
            for item in raw_items:
                title = str(item.get("title") or "").strip()
                body = str(item.get("body") or "").strip()
                url = str(item.get("url") or item.get("link") or item.get("href") or "").strip()
                source = str(item.get("source") or item.get("publisher") or "").strip()
                date = str(item.get("date") or item.get("published") or "").strip()
                if not title or not url:
                    continue
                normalized.append(
                    {
                        "title": title,
                        "snippet": body,
                        "url": url,
                        "source": source,
                        "date": date,
                    }
                )
            return normalized

        try:
            return await asyncio.wait_for(asyncio.to_thread(_run_search), timeout=self.timeout_seconds)
        except Exception as exc:
            print(f"DuckDuckGo news search failed: {exc}")
            return []

    async def _synthesize_answer(
        self,
        original_query: str,
        rewritten_query: str,
        language: str,
        top_k: int,
        items: List[Dict[str, str]],
    ) -> str:
        if not self._openai_client:
            return ""

        language_label = "Kannada" if language == LANG_KN else "English"
        prompt = (
            "Use only the provided news items. "
            f"Respond in {language_label}, preserving the user's language preference. "
            "Structure: short summary paragraph, then numbered headlines with one-line context and source URL. "
            "Do not fabricate facts, dates, or URLs. "
            "If an item lacks date/source, continue without inventing.\n"
            f"Original user query: {original_query}\n"
            f"Search query used: {rewritten_query}\n"
            f"Requested headlines: {top_k}\n"
            f"News items JSON: {json.dumps(items, ensure_ascii=False)}"
        )

        try:
            response = await asyncio.wait_for(
                self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a concise multilingual news assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=750,
                ),
                timeout=self.timeout_seconds,
            )
            text = (response.choices[0].message.content or "").strip()
            return text
        except Exception as exc:
            print(f"News synthesis fallback triggered: {exc}")
            return ""

    def _fallback_answer(
        self,
        original_query: str,
        language: str,
        top_k: int,
        items: List[Dict[str, str]],
    ) -> str:
        heading = self._msg(language, "fallback_heading", query=original_query)
        lines = [heading, self._msg(language, "fallback_count", count=len(items[:top_k]))]
        for index, item in enumerate(items[:top_k], start=1):
            source = item.get("source") or self._msg(language, "unknown_source")
            date = item.get("date") or self._msg(language, "unknown_date")
            lines.append(f"{index}. {item.get('title', '')} ({source}, {date})")
            lines.append(f"   {item.get('url', '')}")
        return "\n".join(lines)

    def _dynamic_top_k(self, question: str) -> int:
        text = (question or "").strip().lower()

        explicit = re.search(r"\btop\s*(\d{1,2})\b", text)
        if not explicit:
            explicit = re.search(r"\b(\d{1,2})\s*(?:news|headlines|stories|articles|results|sources)\b", text)
        if explicit:
            return self._normalize_top_k(explicit.group(1), fallback=self.default_top_k)

        token_count = len(re.findall(r"\w+", text))
        score = self.default_top_k

        if token_count <= 3:
            score += 1
        elif token_count >= 14:
            score += 2

        deep_keywords = {
            "detailed",
            "analysis",
            "compare",
            "timeline",
            "background",
            "deep",
            "explain",
        }
        if any(keyword in text for keyword in deep_keywords):
            score += 2

        very_narrow_keywords = {"today", "latest", "now", "headline", "update"}
        if any(keyword in text for keyword in very_narrow_keywords):
            score += 1

        return self._normalize_top_k(score, fallback=self.default_top_k)

    def _normalize_top_k(self, value, fallback: int) -> int:
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            candidate = fallback
        return max(self.min_top_k, min(candidate, self.max_top_k))

    def _resolve_language(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        if self._detect_language(user_query) == LANG_KN:
            return LANG_KN

        for turn in reversed((conversation_history or [])[-6:]):
            content = str(turn.get("content", ""))
            if self._detect_language(content) == LANG_KN:
                return LANG_KN

        return LANG_EN

    @staticmethod
    def _detect_language(text: str) -> str:
        if re.search(r"[\u0C80-\u0CFF]", text or ""):
            return LANG_KN
        return LANG_EN

    @staticmethod
    def _build_context_snippet(
        conversation_history: Optional[List[Dict[str, str]]],
        max_turns: int = 4,
    ) -> str:
        if not conversation_history:
            return "(none)"
        turns = conversation_history[-max_turns:]
        lines: List[str] = []
        for turn in turns:
            role = str(turn.get("role", "user")).strip().lower()
            prefix = "User" if role == "user" else "Assistant"
            content = re.sub(r"\s+", " ", str(turn.get("content", "")).strip())
            if content:
                lines.append(f"{prefix}: {content[:220]}")
        return "\n".join(lines) if lines else "(none)"

    def _msg(self, language: str, key: str, **kwargs) -> str:
        en = {
            "empty_query": "Please ask what news you want.",
            "fetching": "Fetching latest news updates... Just a moment.",
            "no_results": "I could not find fresh news results for '{query}'. Try adding topic or location details.",
            "fallback_heading": "Top news for: {query}",
            "fallback_count": "Showing {count} headline(s):",
            "selection_heading": "I found these news options for '{query}'. Reply with a number or describe your pick.",
            "selection_hint": "Example replies: '2', 'second one', or 'the OpenAI headline'.",
            "unknown_source": "Unknown source",
            "unknown_date": "Unknown date",
        }
        kn = {
            "empty_query": "ದಯವಿಟ್ಟು ಯಾವ ಸುದ್ದಿಯನ್ನು ಬೇಕು ಎಂದು ಕೇಳಿ.",
            "fetching": "ಇತ್ತೀಚಿನ ಸುದ್ದಿಗಳನ್ನು ಪಡೆಯಲಾಗುತ್ತಿದೆ... ಒಂದು ಕ್ಷಣ.",
            "no_results": "'{query}' ಕುರಿತು ಹೊಸ ಸುದ್ದಿಗಳು ಸಿಗಲಿಲ್ಲ. ವಿಷಯ ಅಥವಾ ಸ್ಥಳವನ್ನು ಸೇರಿಸಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
            "fallback_heading": "ಈ ಪ್ರಶ್ನೆಗೆ ಪ್ರಮುಖ ಸುದ್ದಿ: {query}",
            "fallback_count": "{count} ಮುಖ್ಯಶೀರ್ಷಿಕೆಗಳನ್ನು ತೋರಿಸಲಾಗುತ್ತಿದೆ:",
            "selection_heading": "'{query}' ಕುರಿತಾಗಿ ಈ ಸುದ್ದಿಗಳು ಸಿಕ್ಕಿವೆ. ಸಂಖ್ಯೆ ಅಥವಾ ನಿಮ್ಮ ಆಯ್ಕೆ ವಿವರಿಸಿ ಉತ್ತರಿಸಿ.",
            "selection_hint": "ಉದಾಹರಣೆ: '2', 'ಎರಡನೆಯದು', ಅಥವಾ 'OpenAI ಬಗ್ಗೆ ಇರುವ ಸುದ್ದಿ'.",
            "unknown_source": "ಅಪರಿಚಿತ ಮೂಲ",
            "unknown_date": "ದಿನಾಂಕ ಲಭ್ಯವಿಲ್ಲ",
        }
        template = kn.get(key) if language == LANG_KN else en.get(key)
        if not template:
            template = en.get(key, key)
        return template.format(**kwargs)
