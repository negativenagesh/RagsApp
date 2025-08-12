import os
from pydoc import text
from fastapi import FastAPI, Request, HTTPException, Query
from dotenv import load_dotenv
from app.rag_client import ask_rag
from app.ingestion_client import ingest_file
import httpx

load_dotenv()

import time

PROCESSED_MESSAGE_IDS = set()
MESSAGE_EXPIRY_SECONDS = 10

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

RAG_INTENT_URL = os.getenv("RAG_INTENT_URL", "http://localhost:8001/classify-intent")

app = FastAPI(title="WhatsApp Gateway for RagsApp")

async def send_whatsapp_message(to: str, text: str):
    whatsapp_url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    wa_payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }
    async with httpx.AsyncClient() as client:
        wa_resp = await client.post(whatsapp_url, headers=headers, json=wa_payload,)
        if wa_resp.status_code not in (200, 201):
            print(f"Failed to send WhatsApp message: {wa_resp.text}")

def split_message(text, max_length=4096):
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
    return messages

async def get_intent(question: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(RAG_INTENT_URL, json={"question": question}, timeout=300)
            data = resp.json()
            return data.get("intent", "retrieval")
    except Exception as e:
        print(f"get_intent error: {e}")
        return "retrieval"

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    print("Incoming webhook data:", data)

    try:
        entry = data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        messages = value.get("messages")
        if not messages:
            return {"status": "no user message"}

        message = messages[0]
        from_number = message["from"]
        message_id = message["id"]
        timestamp = int(message.get("timestamp", "0"))

        # Deduplicate: Only process if message_id is new
        if message_id in PROCESSED_MESSAGE_IDS:
            print(f"Duplicate message_id {message_id}, ignoring.")
            return {"status": "duplicate ignored"}
        PROCESSED_MESSAGE_IDS.add(message_id)

        # Filter: Only process recent messages
        now = int(time.time())
        if abs(now - timestamp) > MESSAGE_EXPIRY_SECONDS:
            print(f"Old message (timestamp {timestamp}), ignoring.")
            return {"status": "old message ignored"}

        if message.get("type") == "text" and from_number != WHATSAPP_PHONE_NUMBER_ID:
            text = message["text"]["body"]
            print(f"User text: {text}")

            # 1. Send "thinking..." message immediately
            intent = await get_intent(text)
            print(f"User intent: {intent}")

            # 2. Only send "thinking..." if intent is retrieval
            if intent == "retrieval":
                await send_whatsapp_message(from_number, "🤖 Thinking... Please wait while I process your request.")

            # 3. Now call RAG/LLM (always, for all intents)
            answer = await ask_rag(text)
            if isinstance(answer, dict):
                if "final_answer" in answer:
                    answer = answer["final_answer"]
                elif "result" in answer:
                    answer = answer["result"]
                else:
                    answer = str(answer)
            print(f"RAG answer: {answer}")

            # 3. Send the final answer
            for chunk in split_message(answer):
                await send_whatsapp_message(from_number, chunk)
            return {"status": "sent"}

        # Handle file uploads (document, image, etc.)
        for media_type in ["document", "image", "video", "audio"]:
            if media_type in message:
                media_info = message[media_type]
                media_id = media_info["id"]
                filename = media_info.get("filename", f"uploaded_{media_type}")
                
                from app.whatsapp_api import download_media
                file_bytes = await download_media(media_id)
                
                # 1. Notify user that processing has started
                await send_whatsapp_message(from_number, f"⏳ Processing the file: {filename}")
                
                try:
                    # This waits for the ingestion service to finish
                    await ingest_file(filename, file_bytes)
                    
                    # 2. If successful, send the completion message
                    reply = f"✅ Success! You can now ask questions about the content of '{filename}'."
                
                except Exception as e:
                    # 3. If an error occurs, send a failure message
                    print(f"Error during file ingestion: {e}")
                    reply = (
                        f"❌ Sorry, there was an error processing '{filename}'. "
                        "Please try uploading it again."
                    )
                
                # Send the final reply (either success or failure)
                await send_whatsapp_message(from_number, reply)
                return {"status": "file processing finished"}

        # Ignore all other message types (including status updates)
        return {"status": "ignored"}

    except Exception as e:
        print(f"Failed to process WhatsApp webhook: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook payload")
    
@app.get("/webhook")
def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token")
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return int(hub_challenge)
    return {"status": "verification failed"}

@app.get("/health")
def health():
    return {"status": "ok"}