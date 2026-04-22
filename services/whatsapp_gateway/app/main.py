import os
import asyncio
import time
import json
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
import httpx

load_dotenv()

from app.ingestion_client import ingest_file
from app.whatsapp_api import download_media, send_whatsapp_image, send_whatsapp_message, send_thinking_indicator
from app.supervisor import MessageSupervisor
from app.retrieval_tool import RetrievalTool
from app.arecanut_price_tool import ArecanutPriceTool
from app.skills.meme_generation_skill import MemeGenerationSkill
from app.skills.news_search_skill import NewsSearchSkill

WHATSAPP_PROVIDER = os.getenv("WHATSAPP_PROVIDER", "twilio").strip().lower()

PROCESSED_MESSAGE_IDS = set()
SUPPORTED_MEDIA_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
    "text/plain": "txt",
    "text/csv": "csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "image/jpeg": "jpg",
    "image/png": "png",
}

app = FastAPI(title="WhatsApp Gateway for RagsApp")
RAG_PROGRESS_TIMEOUT_SECONDS = 20
BACKGROUND_TASKS = set()
MAX_WHATSAPP_TEXT_LENGTH = int(os.getenv("WHATSAPP_MAX_TEXT_LENGTH", "1500"))
STREAM_FLUSH_MIN_CHARS = int(os.getenv("WHATSAPP_STREAM_FLUSH_MIN_CHARS", "260"))
STREAM_FLUSH_MAX_SECONDS = float(os.getenv("WHATSAPP_STREAM_FLUSH_MAX_SECONDS", "1.8"))
SUPERVISOR = MessageSupervisor()
RETRIEVAL_TOOL = RetrievalTool()
ARECANUT_PRICE_TOOL = ArecanutPriceTool()
NEWS_SEARCH_SKILL = NewsSearchSkill()
MEME_GENERATION_SKILL = MemeGenerationSkill(news_skill=NEWS_SEARCH_SKILL)


@app.on_event("startup")
async def startup_event():
    await SUPERVISOR.startup()


@app.on_event("shutdown")
async def shutdown_event():
    await SUPERVISOR.shutdown()


def choose_uploaded_filename(message_sid: str, body: str, ext: str) -> str:
    """Prefer user-visible Twilio document name (often in Body) and fallback to MessageSid."""
    body_text = (body or "").strip()
    if body_text:
        # Keep only the basename and replace dangerous path separators.
        safe_name = os.path.basename(body_text).replace("/", "_").replace("\\", "_").strip()
        if safe_name:
            # Ensure file extension exists and matches detected media type.
            name_root, name_ext = os.path.splitext(safe_name)
            if name_root:
                if name_ext.lower() != f".{ext.lower()}":
                    safe_name = f"{name_root}.{ext}"
                return safe_name
    return f"uploaded_{message_sid}.{ext}"

def normalize_rag_answer(raw_answer: str) -> str:
    """Strip known SDK wrapper strings if they leak through the RAG response."""
    if not isinstance(raw_answer, str):
        return str(raw_answer)

    # Example leaked format: Message(message=..., ignore='False')
    if raw_answer.startswith("Message(message=") and raw_answer.endswith(")"):
        inner = raw_answer[len("Message(message="):-1]
        marker = ", ignore="
        if marker in inner:
            inner = inner.split(marker, 1)[0]
        return inner.strip().strip('"').strip("'")

    return raw_answer


def format_answer_for_whatsapp(answer_text: str) -> str:
    """Preserve model structure so users receive heading/bullets as generated."""
    if not answer_text:
        return "I could not generate an answer right now."

    text = str(answer_text).replace("\r\n", "\n").strip()

    # Keep content as-is except for excessive blank-line normalization.
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


async def ask_rag_with_progress(
    question: str,
    progress_callback,
    conversation_history: Optional[List[Dict[str, str]]] = None,
):
    """Send periodic progress pings while waiting for the final RAG response."""
    task = asyncio.create_task(
        RETRIEVAL_TOOL.ask(question, conversation_history=conversation_history or [])
    )
    progress_messages = [
        "Still working on your answer...",
        "Almost done. Finalizing the response...",
    ]
    progress_index = 0

    while True:
        try:
            # Keep the underlying RAG task alive when periodic progress timeouts occur.
            return await asyncio.wait_for(asyncio.shield(task), timeout=RAG_PROGRESS_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            msg = progress_messages[progress_index] if progress_index < len(progress_messages) else "Working on it..."
            if progress_index < len(progress_messages):
                progress_index += 1
            await progress_callback(msg)
        except asyncio.CancelledError:
            print("ask_rag_with_progress received CancelledError; returning graceful fallback.")
            if not task.done():
                task.cancel()
            return "I could not finish your request due to a temporary interruption. Please send your question again."


def _spawn_background_task(coro, label: str):
    """Run webhook work out-of-band so request cancellation does not kill processing."""
    task = asyncio.create_task(coro)
    BACKGROUND_TASKS.add(task)

    def _on_done(done_task: asyncio.Task):
        BACKGROUND_TASKS.discard(done_task)
        try:
            done_task.result()
        except asyncio.CancelledError:
            print(f"Background task cancelled: {label}")
        except Exception as exc:
            print(f"Background task failed ({label}): {exc}")

    task.add_done_callback(_on_done)


def _log_supervisor_decision(
    provider: str,
    user_id: str,
    message_id: str,
    decision,
    context_turns_used: int = 0,
):
    payload = {
        "event": "supervisor_decision",
        "provider": provider,
        "user_id": user_id,
        "message_id": message_id,
        "route_type": decision.route_type,
        "confidence": decision.confidence,
        "reason_code": decision.reason_code,
        "memory_hit": decision.memory_hit,
        "used_rag": decision.used_rag,
        "context_turns_used": context_turns_used,
    }
    print(json.dumps(payload, ensure_ascii=True))


def _log_retrieval_outcome(
    provider: str,
    user_id: str,
    message_id: str,
    mode: str,
    latency_seconds: float,
):
    payload = {
        "event": "retrieval_outcome",
        "provider": provider,
        "user_id": user_id,
        "message_id": message_id,
        "mode": mode,
        "latency_seconds": round(latency_seconds, 3),
    }
    print(json.dumps(payload, ensure_ascii=True))


async def _process_meta_text_message(from_number: str, body: str, message_id: str = None):
    conversation_history = await SUPERVISOR.get_recent_conversation_history(
        provider=WHATSAPP_PROVIDER,
        user_id=from_number,
    )

    decision = await SUPERVISOR.decide(
        provider=WHATSAPP_PROVIDER,
        user_id=from_number,
        user_message=body,
        conversation_history=conversation_history,
    )
    _log_supervisor_decision(
        provider=WHATSAPP_PROVIDER,
        user_id=from_number,
        message_id=message_id,
        decision=decision,
        context_turns_used=len(conversation_history),
    )

    if decision.route_type == "arecanut_price":
        await send_whatsapp_message(
            from_number,
            ARECANUT_PRICE_TOOL.get_fetching_message(body, conversation_history=conversation_history),
        )
        price_answer = format_answer_for_whatsapp(
            await ARECANUT_PRICE_TOOL.ask(body, conversation_history=conversation_history)
        )
        await _send_text_in_chunks(from_number, price_answer)
        await SUPERVISOR.record_answer(
            provider=WHATSAPP_PROVIDER,
            user_id=from_number,
            user_query=body,
            answer_text=price_answer,
            decision=decision,
            used_rag=False,
        )
        return

    if decision.route_type == "news_search":
        await send_whatsapp_message(
            from_number,
            NEWS_SEARCH_SKILL.get_fetching_message(body, conversation_history=conversation_history),
        )
        news_answer = format_answer_for_whatsapp(
            await NEWS_SEARCH_SKILL.ask(body, conversation_history=conversation_history)
        )
        await _send_text_in_chunks(from_number, news_answer)
        await SUPERVISOR.record_answer(
            provider=WHATSAPP_PROVIDER,
            user_id=from_number,
            user_query=body,
            answer_text=news_answer,
            decision=decision,
            used_rag=False,
        )
        return

    if decision.route_type == "meme_generation":
        if not MEME_GENERATION_SKILL.has_pending_selection(from_number):
            await send_whatsapp_message(
                from_number,
                MEME_GENERATION_SKILL.get_fetching_message(body, conversation_history=conversation_history),
            )

        async def send_meme_progress(msg: str):
            await send_whatsapp_message(from_number, msg)

        meme_result = await MEME_GENERATION_SKILL.ask(
            user_id=from_number,
            user_query=body,
            conversation_history=conversation_history,
            progress_callback=send_meme_progress,
        )

        if meme_result.awaiting_user_input:
            await _send_text_in_chunks(from_number, format_answer_for_whatsapp(meme_result.text))
            answer_text = meme_result.text
        else:
            delivered = await _send_generated_images(
                to=from_number,
                image_paths=meme_result.image_paths,
                image_captions=meme_result.image_captions,
                summary_text=format_answer_for_whatsapp(meme_result.text),
            )
            answer_text = f"Generated {len(meme_result.image_paths)} image(s), delivered {delivered}.\n{meme_result.text}".strip()

        await SUPERVISOR.record_answer(
            provider=WHATSAPP_PROVIDER,
            user_id=from_number,
            user_query=body,
            answer_text=answer_text,
            decision=decision,
            used_rag=False,
        )
        return

    if decision.route_type in {"non_rag_reply", "ask_clarification"}:
        quick_reply = format_answer_for_whatsapp(decision.final_text)
        await _send_text_in_chunks(from_number, quick_reply)
        await SUPERVISOR.record_answer(
            provider=WHATSAPP_PROVIDER,
            user_id=from_number,
            user_query=body,
            answer_text=quick_reply,
            decision=decision,
            used_rag=False,
        )
        return

    sent_indicator = await send_thinking_indicator(from_number, meta_message_id=message_id)
    if not sent_indicator:
        await send_whatsapp_message(from_number, "Thinking...")
    start_time = time.time()

    async def send_progress(msg: str):
        await send_whatsapp_message(from_number, msg)

    try:
        streamed, streamed_text = await _try_stream_answer(
            from_number,
            body,
            conversation_history=conversation_history,
        )
        if not streamed:
            raw_answer = await ask_rag_with_progress(
                body,
                send_progress,
                conversation_history=conversation_history,
            )
            answer = format_answer_for_whatsapp(normalize_rag_answer(raw_answer))
            elapsed = time.time() - start_time
            print(f"Meta RAG fallback answered in {elapsed:.2f}s: {answer[:200]}...")
            _log_retrieval_outcome(
                provider=WHATSAPP_PROVIDER,
                user_id=from_number,
                message_id=message_id,
                mode="non_stream_fallback",
                latency_seconds=elapsed,
            )
            await _send_text_in_chunks(from_number, answer)
            await SUPERVISOR.record_answer(
                provider=WHATSAPP_PROVIDER,
                user_id=from_number,
                user_query=body,
                answer_text=answer,
                decision=decision,
                used_rag=True,
            )
        elif streamed_text.strip():
            elapsed = time.time() - start_time
            _log_retrieval_outcome(
                provider=WHATSAPP_PROVIDER,
                user_id=from_number,
                message_id=message_id,
                mode="stream",
                latency_seconds=elapsed,
            )
            await SUPERVISOR.record_answer(
                provider=WHATSAPP_PROVIDER,
                user_id=from_number,
                user_query=body,
                answer_text=format_answer_for_whatsapp(streamed_text),
                decision=decision,
                used_rag=True,
            )
    except asyncio.CancelledError:
        print("Meta background text flow cancelled.")
    except Exception as e:
        print(f"Meta background text flow failed: {e}")
        await _send_text_in_chunks(
            from_number,
            "Sorry, I hit a temporary error while generating your answer. Please try again.",
        )


async def _process_meta_media_message(from_number: str, message_sid: str, content_type: str, media_id: str, provided_name: str):
    ext = SUPPORTED_MEDIA_TYPES.get(content_type)
    if not ext:
        await send_whatsapp_message(
            from_number,
            "Unsupported file type. Please upload one of: PDF, DOC, DOCX, TXT, CSV, XLSX, JPG, PNG.",
        )
        return

    filename = _choose_uploaded_filename_meta(message_sid, provided_name, ext)
    await send_whatsapp_message(from_number, f"Processing your file *{filename}*, This may take a moment.")

    try:
        file_bytes = await download_media(media_id)
        result = await ingest_file(filename, file_bytes)

        if not isinstance(result, dict) or result.get("status") != "success":
            raise RuntimeError(f"Unexpected ingestion response: {result}")

        reply = f"✅ *File processed successfully!*\nYou can now ask me questions about the content of _{filename}_."
    except Exception as e:
        print(f"Error during Meta ingestion flow: {e}")
        reply = "Sorry, there was an error processing your file. Please try uploading it again."

    await send_whatsapp_message(from_number, reply)

def split_message(text, max_length=MAX_WHATSAPP_TEXT_LENGTH):
    """Split long messages into WhatsApp-friendly chunks.
    WhatsApp supports up to 4096 chars, but shorter chunks are more readable.
    """
    paragraphs = text.split('\n\n')
    messages = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_length:
            current += (("\n\n" if current else "") + para)
        else:
            if current:
                messages.append(current)
            if len(para) > max_length:
                for i in range(0, len(para), max_length):
                    messages.append(para[i:i+max_length])
                current = ""
            else:
                current = para
    if current:
        messages.append(current)
    return [m for m in messages if m and m.strip()]


async def _send_text_in_chunks(to: str, text: str, from_number_override: str = None):
    for chunk in split_message(text, max_length=MAX_WHATSAPP_TEXT_LENGTH):
        if len(chunk) > MAX_WHATSAPP_TEXT_LENGTH:
            for i in range(0, len(chunk), MAX_WHATSAPP_TEXT_LENGTH):
                part = chunk[i:i + MAX_WHATSAPP_TEXT_LENGTH]
                if part.strip():
                    await send_whatsapp_message(to, part, from_number_override=from_number_override)
        else:
            await send_whatsapp_message(to, chunk, from_number_override=from_number_override)


async def _send_generated_images(
    to: str,
    image_paths: List[str],
    image_captions: Optional[List[str]] = None,
    summary_text: str = "",
    from_number_override: str = None,
) -> int:
    delivered = 0
    captions = image_captions or []

    for idx, image_path in enumerate(image_paths):
        caption = captions[idx] if idx < len(captions) else f"Meme {idx + 1}"
        sent = await send_whatsapp_image(
            to=to,
            image_path=image_path,
            caption=caption,
            from_number_override=from_number_override,
        )
        if sent:
            delivered += 1
            continue

        await _send_text_in_chunks(
            to,
            f"Image {idx + 1} was generated but delivery failed.",
            from_number_override=from_number_override,
        )

    if summary_text.strip():
        await _send_text_in_chunks(to, summary_text, from_number_override=from_number_override)

    return delivered


async def _try_stream_answer(
    to: str,
    question: str,
    from_number_override: str = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[bool, str]:
    buffer = ""
    sent_any = False
    full_text = ""
    last_flush = time.monotonic()

    async for delta in RETRIEVAL_TOOL.stream(
        question,
        conversation_history=conversation_history or [],
    ):
        if not delta:
            continue
        full_text += delta
        buffer += delta

        now = time.monotonic()
        should_flush = len(buffer) >= STREAM_FLUSH_MIN_CHARS or (buffer and now - last_flush >= STREAM_FLUSH_MAX_SECONDS)
        if not should_flush:
            continue

        await _send_text_in_chunks(to, format_answer_for_whatsapp(buffer), from_number_override=from_number_override)
        sent_any = True
        buffer = ""
        last_flush = now

    if buffer.strip():
        await _send_text_in_chunks(to, format_answer_for_whatsapp(buffer), from_number_override=from_number_override)
        sent_any = True

    return sent_any, full_text


def _parse_meta_message(payload: dict):
    """Extract normalized fields from a Meta webhook message payload."""
    entry_list = payload.get("entry") or []
    for entry in entry_list:
        changes = entry.get("changes") or []
        for change in changes:
            value = change.get("value") or {}
            messages = value.get("messages") or []
            if not messages:
                continue

            contacts = value.get("contacts") or []
            wa_id = None
            if contacts and isinstance(contacts[0], dict):
                wa_id = contacts[0].get("wa_id")

            metadata = value.get("metadata") or {}
            business_phone_id = metadata.get("phone_number_id")

            message = messages[0]
            message_type = message.get("type")
            message_id = message.get("id")
            from_number = message.get("from") or wa_id

            body = ""
            media_id = None
            mime_type = ""
            filename = ""

            if message_type == "text":
                body = ((message.get("text") or {}).get("body") or "").strip()
            elif message_type in {"document", "image"}:
                media_obj = message.get(message_type) or {}
                media_id = media_obj.get("id")
                mime_type = media_obj.get("mime_type") or ""
                filename = media_obj.get("filename") or ""

            return {
                "message_id": message_id,
                "from_number": from_number,
                "to_number": business_phone_id,
                "body": body,
                "message_type": message_type,
                "media_id": media_id,
                "mime_type": mime_type,
                "filename": filename,
            }
    return None


def _choose_uploaded_filename_meta(message_id: str, provided_name: str, ext: str) -> str:
    base_name = os.path.basename((provided_name or "").strip()).replace("/", "_").replace("\\", "_")
    if base_name:
        name_root, name_ext = os.path.splitext(base_name)
        if name_root:
            if name_ext.lower() != f".{ext.lower()}":
                return f"{name_root}.{ext}"
            return base_name
    return f"uploaded_{message_id}.{ext}"

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    if WHATSAPP_PROVIDER == "meta":
        try:
            payload = await request.json()
        except Exception as e:
            print(f"Failed to parse Meta payload: {e}")
            raise HTTPException(status_code=400, detail="Invalid Meta webhook payload")

        parsed = _parse_meta_message(payload)
        if not parsed:
            return {"status": "ignored"}

        message_sid = parsed["message_id"]
        if not message_sid:
            return {"status": "no user message"}

        if message_sid in PROCESSED_MESSAGE_IDS:
            print(f"Duplicate Meta message_id {message_sid}, ignoring.")
            return {"status": "duplicate ignored"}
        PROCESSED_MESSAGE_IDS.add(message_sid)

        if len(PROCESSED_MESSAGE_IDS) > 10000:
            PROCESSED_MESSAGE_IDS.clear()

        from_number = parsed["from_number"]
        body = (parsed["body"] or "").strip()
        message_type = parsed["message_type"]
        media_id = parsed["media_id"]
        content_type = parsed["mime_type"]

        print(f"Incoming Meta message: from={from_number}, type={message_type}, body='{body[:120]}'")

        if message_type in {"document", "image"} and media_id:
            _spawn_background_task(
                _process_meta_media_message(
                    from_number=from_number,
                    message_sid=message_sid,
                    content_type=content_type,
                    media_id=media_id,
                    provided_name=parsed.get("filename", ""),
                ),
                label=f"meta-media-{message_sid}",
            )
            return {"status": "accepted", "mode": "background"}

        if body:
            _spawn_background_task(
                _process_meta_text_message(from_number=from_number, body=body, message_id=message_sid),
                label=f"meta-text-{message_sid}",
            )
            return {"status": "accepted", "mode": "background"}

        return {"status": "ignored"}

    try:
        form_data = await request.form()
    except Exception as e:
        print(f"Failed to parse form data: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook payload structure")

    print(f"Incoming Twilio webhook data: {dict(form_data)}")

    message_sid = form_data.get("MessageSid")
    if not message_sid:
        return {"status": "no user message"}

    # Deduplicate: Only process if message_sid is new
    if message_sid in PROCESSED_MESSAGE_IDS:
        print(f"Duplicate message_sid {message_sid}, ignoring.")
        return {"status": "duplicate ignored"}
    PROCESSED_MESSAGE_IDS.add(message_sid)

    # Keep dedup set from growing indefinitely
    if len(PROCESSED_MESSAGE_IDS) > 10000:
        PROCESSED_MESSAGE_IDS.clear()

    from_number = form_data.get("From")
    to_number = form_data.get("To")
    body = form_data.get("Body", "").strip()
    num_media = int(form_data.get("NumMedia", "0"))

    print(f"From: {from_number}, Body: '{body}', NumMedia: {num_media}")

    # --- FILE UPLOAD FLOW ---
    if num_media > 0:
        media_url = form_data.get("MediaUrl0")
        content_type = form_data.get("MediaContentType0", "")

        if not media_url:
            await _send_text_in_chunks(
                from_number,
                "❌ I could not find the uploaded file URL from Twilio. Please try again.",
                from_number_override=to_number,
            )
            return {"status": "missing media url"}

        ext = SUPPORTED_MEDIA_TYPES.get(content_type)
        if not ext:
            supported_types = ", ".join(sorted(SUPPORTED_MEDIA_TYPES.keys()))
            await _send_text_in_chunks(
                from_number,
                "❌ Unsupported file type. Please upload one of: PDF, DOC, DOCX, TXT, CSV, XLSX, JPG, PNG.",
                from_number_override=to_number,
            )
            print(f"Unsupported media type from Twilio: {content_type}. Supported: {supported_types}")
            return {"status": "unsupported media type", "content_type": content_type}

        filename = choose_uploaded_filename(message_sid=message_sid, body=body, ext=ext)

        print(f"File upload detected: {filename} ({content_type})")
        
        # 1. Notify user that processing has started
        await _send_text_in_chunks(
            from_number,
            f"Processing your file *{filename}*, This may take a moment.",
            from_number_override=to_number,
        )
        
        try:
            file_bytes = await download_media(media_url)
            print(f"Downloaded {len(file_bytes)} bytes from Twilio media URL")
            
            # Send to ingestion service
            result = await ingest_file(filename, file_bytes)
            print(f"Ingestion result: {result}")

            if not isinstance(result, dict) or result.get("status") != "success":
                raise RuntimeError(f"Unexpected ingestion response: {result}")

            reply = f"✅ *File processed successfully!*\nYou can now ask me questions about the content of _{filename}_."
        except Exception as e:
            print(f"Error during file ingestion: {e}")
            reply = "❌ Sorry, there was an error processing your file. Please try uploading it again."
        
        await _send_text_in_chunks(from_number, reply, from_number_override=to_number)
        return {"status": "file processing finished"}
    
    elif body:
        print(f"User question: {body}")

        conversation_history = await SUPERVISOR.get_recent_conversation_history(
            provider=WHATSAPP_PROVIDER,
            user_id=from_number,
        )

        decision = await SUPERVISOR.decide(
            provider=WHATSAPP_PROVIDER,
            user_id=from_number,
            user_message=body,
            conversation_history=conversation_history,
        )
        _log_supervisor_decision(
            provider=WHATSAPP_PROVIDER,
            user_id=from_number,
            message_id=message_sid,
            decision=decision,
            context_turns_used=len(conversation_history),
        )

        if decision.route_type == "arecanut_price":
            await _send_text_in_chunks(
                from_number,
                ARECANUT_PRICE_TOOL.get_fetching_message(body, conversation_history=conversation_history),
                from_number_override=to_number,
            )
            price_answer = format_answer_for_whatsapp(
                await ARECANUT_PRICE_TOOL.ask(body, conversation_history=conversation_history)
            )
            await _send_text_in_chunks(from_number, price_answer, from_number_override=to_number)
            await SUPERVISOR.record_answer(
                provider=WHATSAPP_PROVIDER,
                user_id=from_number,
                user_query=body,
                answer_text=price_answer,
                decision=decision,
                used_rag=False,
            )
            return {"status": "sent", "route": decision.route_type}

        if decision.route_type == "news_search":
            await _send_text_in_chunks(
                from_number,
                NEWS_SEARCH_SKILL.get_fetching_message(body, conversation_history=conversation_history),
                from_number_override=to_number,
            )
            news_answer = format_answer_for_whatsapp(
                await NEWS_SEARCH_SKILL.ask(body, conversation_history=conversation_history)
            )
            await _send_text_in_chunks(from_number, news_answer, from_number_override=to_number)
            await SUPERVISOR.record_answer(
                provider=WHATSAPP_PROVIDER,
                user_id=from_number,
                user_query=body,
                answer_text=news_answer,
                decision=decision,
                used_rag=False,
            )
            return {"status": "sent", "route": decision.route_type}

        if decision.route_type == "meme_generation":
            if not MEME_GENERATION_SKILL.has_pending_selection(from_number):
                await _send_text_in_chunks(
                    from_number,
                    MEME_GENERATION_SKILL.get_fetching_message(body, conversation_history=conversation_history),
                    from_number_override=to_number,
                )

            async def send_meme_progress(msg: str):
                await _send_text_in_chunks(from_number, msg, from_number_override=to_number)

            meme_result = await MEME_GENERATION_SKILL.ask(
                user_id=from_number,
                user_query=body,
                conversation_history=conversation_history,
                progress_callback=send_meme_progress,
            )

            if meme_result.awaiting_user_input:
                await _send_text_in_chunks(
                    from_number,
                    format_answer_for_whatsapp(meme_result.text),
                    from_number_override=to_number,
                )
                answer_text = meme_result.text
            else:
                delivered = await _send_generated_images(
                    to=from_number,
                    image_paths=meme_result.image_paths,
                    image_captions=meme_result.image_captions,
                    summary_text=format_answer_for_whatsapp(meme_result.text),
                    from_number_override=to_number,
                )
                answer_text = f"Generated {len(meme_result.image_paths)} image(s), delivered {delivered}.\n{meme_result.text}".strip()

            await SUPERVISOR.record_answer(
                provider=WHATSAPP_PROVIDER,
                user_id=from_number,
                user_query=body,
                answer_text=answer_text,
                decision=decision,
                used_rag=False,
            )
            return {"status": "sent", "route": decision.route_type}

        if decision.route_type in {"non_rag_reply", "ask_clarification"}:
            quick_reply = format_answer_for_whatsapp(decision.final_text)
            await _send_text_in_chunks(from_number, quick_reply, from_number_override=to_number)
            await SUPERVISOR.record_answer(
                provider=WHATSAPP_PROVIDER,
                user_id=from_number,
                user_query=body,
                answer_text=quick_reply,
                decision=decision,
                used_rag=False,
            )
            return {"status": "sent", "route": decision.route_type}

        sent_indicator = await send_thinking_indicator(from_number, from_number_override=to_number)
        if not sent_indicator:
            await _send_text_in_chunks(from_number, "Thinking...", from_number_override=to_number)

        start_time = time.time()

        async def send_progress(msg: str):
            await _send_text_in_chunks(from_number, msg, from_number_override=to_number)

        try:
            streamed, streamed_text = await _try_stream_answer(
                from_number,
                body,
                from_number_override=to_number,
                conversation_history=conversation_history,
            )
            if not streamed:
                raw_answer = await ask_rag_with_progress(
                    body,
                    send_progress,
                    conversation_history=conversation_history,
                )
                answer = format_answer_for_whatsapp(normalize_rag_answer(raw_answer))
                elapsed = time.time() - start_time
                print(f"RAG fallback answered in {elapsed:.2f}s: {answer[:200]}...")
                _log_retrieval_outcome(
                    provider=WHATSAPP_PROVIDER,
                    user_id=from_number,
                    message_id=message_sid,
                    mode="non_stream_fallback",
                    latency_seconds=elapsed,
                )
                await _send_text_in_chunks(from_number, answer, from_number_override=to_number)
                await SUPERVISOR.record_answer(
                    provider=WHATSAPP_PROVIDER,
                    user_id=from_number,
                    user_query=body,
                    answer_text=answer,
                    decision=decision,
                    used_rag=True,
                )
            elif streamed_text.strip():
                elapsed = time.time() - start_time
                _log_retrieval_outcome(
                    provider=WHATSAPP_PROVIDER,
                    user_id=from_number,
                    message_id=message_sid,
                    mode="stream",
                    latency_seconds=elapsed,
                )
                await SUPERVISOR.record_answer(
                    provider=WHATSAPP_PROVIDER,
                    user_id=from_number,
                    user_query=body,
                    answer_text=format_answer_for_whatsapp(streamed_text),
                    decision=decision,
                    used_rag=True,
                )
            return {"status": "sent"}
        except asyncio.CancelledError:
            print("Twilio text flow cancelled; returning handled status.")
            return {"status": "cancelled handled"}
        except Exception as e:
            print(f"Twilio text flow failed: {e}")
            await _send_text_in_chunks(
                from_number,
                "Sorry, I hit a temporary error while generating your answer. Please try again.",
                from_number_override=to_number,
            )
            return {"status": "error handled"}

    return {"status": "ignored"}


@app.get("/webhook")
async def whatsapp_webhook_verify(request: Request):
    if WHATSAPP_PROVIDER != "meta":
        return {"status": "ok", "provider": WHATSAPP_PROVIDER}

    mode = request.query_params.get("hub.mode")
    verify_token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    expected = os.getenv("WHATSAPP_META_VERIFY_TOKEN", "")

    if mode == "subscribe" and verify_token and expected and verify_token == expected and challenge:
        return PlainTextResponse(content=str(challenge))

    raise HTTPException(status_code=403, detail="Meta webhook verification failed")

@app.get("/health")
def health():
    return {"status": "ok", "provider": WHATSAPP_PROVIDER}