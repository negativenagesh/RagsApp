import asyncio
import datetime
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from app.skills.news_search_skill import NewsSearchSkill


LANG_EN = "en"
LANG_KN = "kn"

ProgressCallback = Callable[[str], Awaitable[None]]


@dataclass
class MemeGenerationOutput:
    text: str
    image_paths: List[str]
    image_captions: List[str]
    awaiting_user_input: bool = False


@dataclass
class PendingNewsSelection:
    created_at: float
    original_query: str
    language: str
    num_images: int
    rewritten_query: str
    news_items: List[Dict[str, str]]


class MemeGenerationSkill:
    def __init__(self, news_skill: Optional[NewsSearchSkill] = None) -> None:
        self.max_images = int(os.getenv("MEME_MAX_IMAGES", "3"))
        self.default_images = int(os.getenv("MEME_DEFAULT_IMAGES", "1"))
        self.pending_ttl_seconds = int(os.getenv("MEME_PENDING_TTL_SECONDS", "900"))
        self.timeout_seconds = float(os.getenv("MEME_GENERATION_TIMEOUT_SECONDS", "180"))
        self.model = os.getenv("MEME_LLM_MODEL", os.getenv("SUPERVISOR_NONRAG_MODEL", "gpt-4o-mini"))
        self.meme_output_dir = os.getenv("MEME_OUTPUT_DIR", "memes")

        openai_api_key = os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY")
        self._openai_client: Optional[AsyncOpenAI] = (
            AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None
        )

        self._news_skill = news_skill or NewsSearchSkill()
        self._pending_by_user: Dict[str, PendingNewsSelection] = {}

    async def ask(
        self,
        user_id: str,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> MemeGenerationOutput:
        question = (user_query or "").strip()
        if not question:
            return MemeGenerationOutput(text=self._msg(LANG_EN, "empty_query"), image_paths=[], image_captions=[])

        self._cleanup_pending()
        language = self._resolve_language(question, conversation_history=conversation_history)

        if self._has_pending(user_id):
            return await self._continue_from_pending(
                user_id=user_id,
                user_query=question,
                conversation_history=conversation_history,
                progress_callback=progress_callback,
            )

        requested_images, is_explicit = self._extract_image_count(question)
        needs_news = await self._requires_news_selection(question, conversation_history=conversation_history)

        if needs_news:
            if progress_callback:
                await progress_callback(self._msg(language, "progress_news_fetch"))

            news_items, rewritten_query, _, language = await self._news_skill.get_news_candidates(
                user_query=question,
                conversation_history=conversation_history,
                force_top_k=5,
            )

            if news_items:
                self._pending_by_user[user_id] = PendingNewsSelection(
                    created_at=time.time(),
                    original_query=question,
                    language=language,
                    num_images=requested_images,
                    rewritten_query=rewritten_query,
                    news_items=news_items,
                )
                intro = self._news_skill.format_selection_prompt(
                    original_query=question,
                    items=news_items,
                    language=language,
                )
                count_line = self._image_count_instruction(language, requested_images, is_explicit)
                return MemeGenerationOutput(
                    text=f"{intro}\n\n{count_line}",
                    image_paths=[],
                    image_captions=[],
                    awaiting_user_input=True,
                )

        if progress_callback:
            await progress_callback(self._image_count_instruction(language, requested_images, is_explicit))

        return await self._generate_memes(
            prompt=question,
            language=language,
            num_images=requested_images,
            progress_callback=progress_callback,
            context_news_item=None,
        )

    def get_fetching_message(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        language = self._resolve_language(user_query, conversation_history=conversation_history)
        return self._msg(language, "fetching")

    def has_pending_selection(self, user_id: str) -> bool:
        self._cleanup_pending()
        return self._has_pending(user_id)

    async def _continue_from_pending(
        self,
        user_id: str,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]],
        progress_callback: Optional[ProgressCallback],
    ) -> MemeGenerationOutput:
        pending = self._pending_by_user.get(user_id)
        if not pending:
            return MemeGenerationOutput(text=self._msg(LANG_EN, "selection_missing"), image_paths=[], image_captions=[])

        if self._is_cancel(user_query):
            self._pending_by_user.pop(user_id, None)
            return MemeGenerationOutput(text=self._msg(pending.language, "selection_cancelled"), image_paths=[], image_captions=[])

        num_images, is_explicit = self._extract_image_count(user_query)
        if is_explicit:
            pending.num_images = num_images

        selected_index = await self._resolve_selection_index(
            selection_text=user_query,
            items=pending.news_items,
        )
        if selected_index is None:
            reminder = self._news_skill.format_selection_prompt(
                original_query=pending.original_query,
                items=pending.news_items,
                language=pending.language,
            )
            count_line = self._image_count_instruction(
                pending.language,
                pending.num_images,
                is_explicit=True,
            )
            return MemeGenerationOutput(
                text=f"{reminder}\n\n{count_line}",
                image_paths=[],
                image_captions=[],
                awaiting_user_input=True,
            )

        chosen = pending.news_items[selected_index]
        self._pending_by_user.pop(user_id, None)

        merged_prompt = self._build_news_grounded_prompt(
            original_query=pending.original_query,
            selected_news=chosen,
            selection_text=user_query,
        )
        return await self._generate_memes(
            prompt=merged_prompt,
            language=pending.language,
            num_images=pending.num_images,
            progress_callback=progress_callback,
            context_news_item=chosen,
        )

    async def _generate_memes(
        self,
        prompt: str,
        language: str,
        num_images: int,
        progress_callback: Optional[ProgressCallback],
        context_news_item: Optional[Dict[str, str]],
    ) -> MemeGenerationOutput:
        num_images = max(1, min(num_images, self.max_images))

        if not os.getenv("NANO_BANANA_API_KEY"):
            return MemeGenerationOutput(
                text=self._msg(language, "missing_api_key"),
                image_paths=[],
                image_captions=[],
            )

        image_paths: List[str] = []
        caption_lines: List[str] = []

        for index in range(1, num_images + 1):
            if progress_callback:
                await progress_callback(self._msg(language, "progress_research", current=index, total=num_images))

            research_brief = await self._build_research_brief(prompt, context_news_item=context_news_item)

            if progress_callback:
                await progress_callback(self._msg(language, "progress_layout", current=index, total=num_images))

            layout_plan = await self._build_layout_plan(prompt, research_brief)
            generation_prompt = self._compose_generation_prompt(
                base_prompt=prompt,
                research_brief=research_brief,
                layout_plan=layout_plan,
                index=index,
                total=num_images,
            )

            if progress_callback:
                await progress_callback(self._msg(language, "progress_visual", current=index, total=num_images))

            try:
                image_path = await asyncio.wait_for(
                    asyncio.to_thread(self._generate_single_image_sync, generation_prompt),
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                if progress_callback:
                    await progress_callback(self._msg(language, "progress_failed", current=index, reason=str(exc)[:120]))
                continue

            image_paths.append(image_path)
            caption = await self._build_caption(
                base_prompt=prompt,
                research_brief=research_brief,
                image_index=index,
                total_images=num_images,
                language=language,
            )
            caption_lines.append(caption)

            if progress_callback:
                await progress_callback(self._msg(language, "progress_done", current=index, total=num_images))

        if not image_paths:
            return MemeGenerationOutput(
                text=self._msg(language, "generation_failed"),
                image_paths=[],
                image_captions=[],
            )

        if len(image_paths) < num_images:
            partial = self._msg(
                language,
                "partial_success",
                generated=len(image_paths),
                requested=num_images,
            )
            summary = "\n".join([partial] + caption_lines)
        else:
            summary = "\n".join(caption_lines)

        return MemeGenerationOutput(text=summary, image_paths=image_paths, image_captions=caption_lines)

    async def _requires_news_selection(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> bool:
        question = (user_query or "").strip().lower()
        keyword_hint = any(
            token in question
            for token in [
                "news",
                "headline",
                "trending",
                "current event",
                "breaking",
                "latest",
                "today",
                "update",
            ]
        )
        if keyword_hint:
            return True

        if not self._openai_client:
            return False

        history_block = self._build_context_snippet(conversation_history)
        prompt = (
            "Decide if a meme request should first fetch current web news for user selection. "
            "Return strict JSON with one key: requires_news (boolean). "
            "Set requires_news=true only if user explicitly implies trending/current/news context. "
            f"Recent context:\n{history_block}\n"
            f"User message: {user_query}"
        )

        try:
            response = await self._openai_client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a strict JSON classifier."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=80,
            )
            raw = (response.choices[0].message.content or "{}").strip()
            return bool(__import__("json").loads(raw).get("requires_news", False))
        except Exception:
            return False

    async def _resolve_selection_index(self, selection_text: str, items: List[Dict[str, str]]) -> Optional[int]:
        text = (selection_text or "").strip().lower()
        if not text or not items:
            return None

        number_match = re.search(r"\b([1-9]|10)\b", text)
        if number_match:
            idx = int(number_match.group(1)) - 1
            if 0 <= idx < len(items):
                return idx

        ordinal_map = {
            "first": 0,
            "1st": 0,
            "one": 0,
            "second": 1,
            "2nd": 1,
            "two": 1,
            "third": 2,
            "3rd": 2,
            "three": 2,
        }
        for key, value in ordinal_map.items():
            if key in text and value < len(items):
                return value

        best_idx = self._resolve_selection_by_keyword(text=text, items=items)
        if best_idx is not None:
            return best_idx

        if not self._openai_client:
            return None

        try:
            items_json = __import__("json").dumps(items, ensure_ascii=False)
            prompt = (
                "Choose which item user selected from this list. "
                "Return strict JSON with key selected_index (1-based integer) or 0 if unclear. "
                f"User selection text: {selection_text}\n"
                f"News items: {items_json}"
            )
            response = await self._openai_client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a strict JSON resolver."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=120,
            )
            data = __import__("json").loads((response.choices[0].message.content or "{}").strip())
            selected_index = int(data.get("selected_index", 0))
            if 1 <= selected_index <= len(items):
                return selected_index - 1
            return None
        except Exception:
            return None

    @staticmethod
    def _resolve_selection_by_keyword(text: str, items: List[Dict[str, str]]) -> Optional[int]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return None

        best_idx = None
        best_score = 0
        words = {w for w in re.findall(r"[a-z0-9]+", normalized) if len(w) >= 3}
        for idx, item in enumerate(items):
            haystack = " ".join(
                [
                    str(item.get("title") or "").lower(),
                    str(item.get("snippet") or "").lower(),
                    str(item.get("source") or "").lower(),
                ]
            )
            score = sum(1 for w in words if w in haystack)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score <= 0:
            return None
        return best_idx

    async def _build_research_brief(
        self,
        user_prompt: str,
        context_news_item: Optional[Dict[str, str]] = None,
    ) -> str:
        if not self._openai_client:
            return f"Focus on a high-contrast meme angle for: {user_prompt}"

        extra = ""
        if context_news_item:
            extra = (
                f"\nSelected news title: {context_news_item.get('title', '')}"
                f"\nSelected news snippet: {context_news_item.get('snippet', '')}"
            )

        prompt = (
            "You are the Researcher agent for meme generation. "
            "Return 3 concise bullets: (1) setup idea, (2) twist idea, (3) relatable punchline angle. "
            "Avoid emojis and keep under 120 words total."
            f"\nUser request: {user_prompt}{extra}"
        )
        try:
            response = await self._openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are concise and practical."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=220,
            )
            return (response.choices[0].message.content or "").strip() or user_prompt
        except Exception:
            return f"Use expectation-vs-reality with one sharp twist about: {user_prompt}"

    async def _build_layout_plan(self, user_prompt: str, research_brief: str) -> str:
        if not self._openai_client:
            return "1-2 panels, large bold text, top/bottom safe margin."

        prompt = (
            "You are Layout_Calculator. Return compact layout guidance for meme readability. "
            "Include panel count, text zones, and max words per panel. Keep it under 80 words."
            f"\nRequest: {user_prompt}\nResearch brief: {research_brief}"
        )
        try:
            response = await self._openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You optimize for mobile social readability."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=180,
            )
            return (response.choices[0].message.content or "").strip() or "1-2 panels, safe margins."
        except Exception:
            return "1-2 panels, keep text short, avoid covering faces."

    async def _build_caption(
        self,
        base_prompt: str,
        research_brief: str,
        image_index: int,
        total_images: int,
        language: str,
    ) -> str:
        if not self._openai_client:
            return self._msg(language, "caption_fallback", index=image_index, total=total_images)

        language_name = "Kannada" if language == LANG_KN else "English"
        prompt = (
            "Create one short WhatsApp caption for a generated meme image. "
            "Use witty Gen Z-friendly tone without emojis. "
            "Keep under 180 characters."
            f"\nLanguage: {language_name}"
            f"\nUser request: {base_prompt}"
            f"\nResearch brief: {research_brief}"
            f"\nImage index: {image_index}/{total_images}"
        )
        try:
            response = await self._openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You produce concise, punchy captions."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=100,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return f"{image_index}. {text}"
            return self._msg(language, "caption_fallback", index=image_index, total=total_images)
        except Exception:
            return self._msg(language, "caption_fallback", index=image_index, total=total_images)

    def _compose_generation_prompt(
        self,
        base_prompt: str,
        research_brief: str,
        layout_plan: str,
        index: int,
        total: int,
    ) -> str:
        return (
            "Create a highly shareable meme image with clean readable text overlays. "
            "No emojis in overlay text. Use bold uppercase white text with black stroke and safe margins. "
            "Keep the joke immediate with setup and punchline. "
            f"Generate meme variation {index} of {total}.\n"
            f"User request: {base_prompt}\n"
            f"Research brief: {research_brief}\n"
            f"Layout guidance: {layout_plan}"
        )

    def _generate_single_image_sync(self, prompt: str) -> str:
        api_key = os.getenv("NANO_BANANA_API_KEY")
        if not api_key:
            raise RuntimeError("NANO_BANANA_API_KEY is not set")

        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise RuntimeError(f"google-genai is not installed: {exc}") from exc

        client = genai.Client(api_key=api_key)
        result = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt],
            config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
        )

        for candidate in getattr(result, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                inline_data = getattr(part, "inline_data", None)
                if not inline_data:
                    continue
                mime_type = getattr(inline_data, "mime_type", "")
                if not mime_type.startswith("image/"):
                    continue

                image_bytes = inline_data.data
                extension = mime_type.split("/", 1)[-1] or "png"
                date_folder = datetime.datetime.now().strftime("%Y-%m-%d")
                output_dir = Path(self.meme_output_dir) / date_folder
                output_dir.mkdir(parents=True, exist_ok=True)
                file_name = f"generated_meme_{uuid.uuid4().hex[:10]}.{extension}"
                file_path = output_dir / file_name
                file_path.write_bytes(image_bytes)
                return str(file_path.resolve())

        text = getattr(result, "text", "")
        raise RuntimeError(f"Image model returned no image. Response: {text}")

    def _build_news_grounded_prompt(
        self,
        original_query: str,
        selected_news: Dict[str, str],
        selection_text: str,
    ) -> str:
        return (
            f"{original_query}\n"
            f"Selected news by user: {selection_text}\n"
            f"News title: {selected_news.get('title', '')}\n"
            f"News snippet: {selected_news.get('snippet', '')}\n"
            f"News source: {selected_news.get('source', '')}\n"
            "Generate memes grounded in this selected news context while keeping the tone funny and shareable."
        )

    def _extract_image_count(self, text: str) -> Tuple[int, bool]:
        lowered = (text or "").lower()

        explicit = re.search(r"\b([1-9]|10)\s*(?:images?|memes?|pics?)\b", lowered)
        if explicit:
            return self._clamp_images(int(explicit.group(1))), True

        ordinal_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "1": 1,
            "2": 2,
            "3": 3,
        }
        for key, value in ordinal_words.items():
            if re.search(rf"\b{re.escape(key)}\b", lowered) and (
                "image" in lowered or "meme" in lowered or "pic" in lowered
            ):
                return self._clamp_images(value), True

        generic_number = re.search(r"\b([1-9]|10)\b", lowered)
        if generic_number and ("image" in lowered or "meme" in lowered):
            return self._clamp_images(int(generic_number.group(1))), True

        return self._clamp_images(self.default_images), False

    def _clamp_images(self, count: int) -> int:
        return max(1, min(int(count), self.max_images))

    def _image_count_instruction(self, language: str, count: int, is_explicit: bool) -> str:
        if is_explicit:
            return self._msg(language, "count_selected", count=count, max_images=self.max_images)
        return self._msg(language, "count_default", count=count, max_images=self.max_images)

    def _cleanup_pending(self) -> None:
        now = time.time()
        expired_users = [
            user_id
            for user_id, pending in self._pending_by_user.items()
            if pending.created_at + self.pending_ttl_seconds < now
        ]
        for user_id in expired_users:
            self._pending_by_user.pop(user_id, None)

    def _has_pending(self, user_id: str) -> bool:
        return user_id in self._pending_by_user

    @staticmethod
    def _is_cancel(text: str) -> bool:
        value = (text or "").strip().lower()
        return value in {"cancel", "stop", "skip", "ಬಿಡು", "ರದ್ದು"}

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

    def _msg(self, language: str, key: str, **kwargs: Any) -> str:
        en = {
            "empty_query": "Tell me what meme you want and I can generate up to 3 images.",
            "fetching": "Starting meme workflow... I can generate up to 3 images.",
            "missing_api_key": "Meme generation is not configured yet. Please set NANO_BANANA_API_KEY.",
            "selection_missing": "I could not find your pending news selection. Please ask again with your meme topic.",
            "selection_cancelled": "Cancelled the meme request. Send a new meme prompt when ready.",
            "count_default": "I will generate {count} image by default. You can ask for up to {max_images} images.",
            "count_selected": "Great. I will generate {count} image(s). Maximum allowed is {max_images}.",
            "progress_news_fetch": "Researcher is collecting latest news options for your meme request...",
            "progress_research": "Researcher is shaping meme idea {current}/{total}...",
            "progress_layout": "Layout_Calculator is preparing text layout {current}/{total}...",
            "progress_visual": "Visualizer is generating image {current}/{total}...",
            "progress_done": "Image {current}/{total} is ready.",
            "progress_failed": "Image {current} failed: {reason}",
            "generation_failed": "I could not generate meme images this time. Please try again with a shorter or clearer prompt.",
            "partial_success": "Generated {generated}/{requested} images successfully.",
            "caption_fallback": "{index}. Meme {index}/{total} generated.",
        }
        kn = {
            "empty_query": "ನೀವು ಯಾವ ಮೀಮ್ ಬೇಕು ಎಂದು ಹೇಳಿ. ಗರಿಷ್ಠ 3 ಚಿತ್ರಗಳನ್ನು ರಚಿಸಬಹುದು.",
            "fetching": "ಮೀಮ್ ಕಾರ್ಯ ಪ್ರಾರಂಭವಾಗುತ್ತಿದೆ... ಗರಿಷ್ಠ 3 ಚಿತ್ರಗಳನ್ನು ರಚಿಸಬಹುದು.",
            "missing_api_key": "ಮೀಮ್ ರಚನೆ ಇನ್ನೂ ಸಿದ್ಧವಾಗಿಲ್ಲ. NANO_BANANA_API_KEY ಸೆಟ್ ಮಾಡಿ.",
            "selection_missing": "ನಿಮ್ಮ ಬಾಕಿ ಸುದ್ದಿ ಆಯ್ಕೆ ಸಿಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಮೀಮ್ ವಿಷಯ ಕಳುಹಿಸಿ.",
            "selection_cancelled": "ಮೀಮ್ ವಿನಂತಿ ರದ್ದುಮಾಡಲಾಗಿದೆ. ಬೇಕಾದಾಗ ಹೊಸ ಮೀಮ್ ವಿನಂತಿ ಕಳುಹಿಸಿ.",
            "count_default": "ಡಿಫಾಲ್ಟ್ ಆಗಿ {count} ಚಿತ್ರ ರಚಿಸುತ್ತೇನೆ. ಗರಿಷ್ಠ {max_images} ಚಿತ್ರಗಳು ಸಾಧ್ಯ.",
            "count_selected": "ಸರಿ. {count} ಚಿತ್ರ(ಗಳು) ರಚಿಸುತ್ತೇನೆ. ಗರಿಷ್ಠ {max_images} ಮಾತ್ರ.",
            "progress_news_fetch": "ನಿಮ್ಮ ಮೀಮ್ ವಿನಂತಿಗೆ ತಾಜಾ ಸುದ್ದಿಗಳನ್ನು ಹುಡುಕುತ್ತಿದೆ...",
            "progress_research": "Researcher ಮೀಮ್ ಐಡಿಯಾ {current}/{total} ರೂಪಿಸುತ್ತಿದೆ...",
            "progress_layout": "Layout_Calculator ಪಠ್ಯ ವಿನ್ಯಾಸ {current}/{total} ಸಿದ್ಧಪಡಿಸುತ್ತಿದೆ...",
            "progress_visual": "Visualizer ಚಿತ್ರ {current}/{total} ರಚಿಸುತ್ತಿದೆ...",
            "progress_done": "ಚಿತ್ರ {current}/{total} ಸಿದ್ಧವಾಗಿದೆ.",
            "progress_failed": "ಚಿತ್ರ {current} ವಿಫಲವಾಗಿದೆ: {reason}",
            "generation_failed": "ಈ ಬಾರಿ ಮೀಮ್ ಚಿತ್ರಗಳನ್ನು ರಚಿಸಲಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಚಿಕ್ಕ ಮತ್ತು ಸ್ಪಷ್ಟ ವಿನಂತಿ ನೀಡಿ.",
            "partial_success": "{requested} ಯಲ್ಲಿ {generated} ಚಿತ್ರಗಳು ಯಶಸ್ವಿಯಾಗಿ ರಚಿಸಲಾಗಿದೆ.",
            "caption_fallback": "{index}. ಮೀಮ್ {index}/{total} ರಚಿಸಲಾಗಿದೆ.",
        }
        template = kn.get(key) if language == LANG_KN else en.get(key)
        if not template:
            template = en.get(key, key)
        return template.format(**kwargs)
