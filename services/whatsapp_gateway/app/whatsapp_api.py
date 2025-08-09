import os
import httpx

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
GRAPH_API_BASE = "https://graph.facebook.com/v18.0"

async def download_media(media_id: str) -> bytes:
    # Step 1: Get media URL
    url = f"{GRAPH_API_BASE}/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        media_url = resp.json()["url"]
        # Step 2: Download media
        media_resp = await client.get(media_url, headers=headers)
        media_resp.raise_for_status()
        return media_resp.content

async def send_whatsapp_message(phone_number_id: str, to: str, text: str):
    url = f"{GRAPH_API_BASE}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()