import os
import httpx
from dotenv import load_dotenv
from pathlib import Path
import struct
from typing import Optional, Tuple

load_dotenv()

WHATSAPP_PROVIDER = os.getenv("WHATSAPP_PROVIDER", "twilio").strip().lower()
MAX_OUTBOUND_TEXT_LEN = int(os.getenv("WHATSAPP_MAX_TEXT_LENGTH", "1500"))
META_STICKER_DIMENSION = 512
META_STATIC_STICKER_MAX_BYTES = 100 * 1024
META_ANIMATED_STICKER_MAX_BYTES = 500 * 1024


def _resolve_thinking_asset_path() -> Path:
    """Resolve thinking asset path robustly across different working directories."""
    configured = os.getenv("THINKING_ASSET_PATH", "RagsApp-logo/thinking.gif")
    configured_path = Path(configured).expanduser()

    # 1) Absolute configured path
    if configured_path.is_absolute() and configured_path.exists():
        return configured_path

    # 2) Relative to current process working directory
    if configured_path.exists():
        return configured_path.resolve()

    # 3) Relative to repository root (derived from this file location)
    repo_root = Path(__file__).resolve().parents[3]
    repo_relative = repo_root / configured
    if repo_relative.exists():
        return repo_relative

    # 4) Safe default under repo root
    default_repo_asset = repo_root / "RagsApp-logo" / "thinking.webp"
    if default_repo_asset.exists():
        return default_repo_asset

    # Return configured path for clearer logs even when missing
    return configured_path


def _select_meta_sticker_asset(asset_path: Path) -> Optional[Path]:
    """Meta stickers must be WebP. If a GIF is configured, try sibling .webp."""
    suffix = asset_path.suffix.lower()
    if suffix == ".webp":
        return asset_path

    if suffix == ".gif":
        sibling_webp = asset_path.with_suffix(".webp")
        if sibling_webp.exists():
            print(
                "Meta requires WebP sticker assets; "
                f"using sibling WebP '{sibling_webp}' for configured GIF '{asset_path}'."
            )
            return sibling_webp
        print(
            "Configured thinking asset is GIF, but Meta sticker API requires WebP and no sibling .webp was found."
        )
        return None

    print(f"Unsupported Meta sticker asset type '{suffix}'. Expected .webp (or .gif with sibling .webp).")
    return None


def _parse_webp_vp8x_info(file_bytes: bytes) -> Tuple[Optional[int], Optional[int], bool]:
    """Parse WebP VP8X chunks for canvas dimensions and animation flag.

    Returns (width, height, is_animated). If dimensions cannot be read, width/height are None.
    """
    if len(file_bytes) < 12 or file_bytes[0:4] != b"RIFF" or file_bytes[8:12] != b"WEBP":
        return None, None, False

    pos = 12
    width = None
    height = None
    is_animated = False

    while pos + 8 <= len(file_bytes):
        chunk_type = file_bytes[pos:pos + 4]
        chunk_size = struct.unpack("<I", file_bytes[pos + 4:pos + 8])[0]
        data_start = pos + 8
        data_end = data_start + chunk_size
        if data_end > len(file_bytes):
            break

        if chunk_type == b"VP8X" and chunk_size >= 10:
            flags = file_bytes[data_start]
            is_animated = bool(flags & 0b00000010)
            width_minus_1 = int.from_bytes(file_bytes[data_start + 4:data_start + 7], "little")
            height_minus_1 = int.from_bytes(file_bytes[data_start + 7:data_start + 10], "little")
            width = width_minus_1 + 1
            height = height_minus_1 + 1
        elif chunk_type == b"ANIM":
            is_animated = True

        pos = data_end + (chunk_size % 2)

    return width, height, is_animated


def _validate_meta_sticker_asset(file_bytes: bytes, path: Path) -> Tuple[bool, str]:
    """Validate Meta sticker constraints for WebP before attempting send."""
    if path.suffix.lower() != ".webp":
        return False, f"Meta sticker asset must be .webp, got '{path.suffix}'"

    width, height, is_animated = _parse_webp_vp8x_info(file_bytes)
    size_bytes = len(file_bytes)
    max_size = META_ANIMATED_STICKER_MAX_BYTES if is_animated else META_STATIC_STICKER_MAX_BYTES

    if size_bytes > max_size:
        kind = "animated" if is_animated else "static"
        return False, (
            f"{kind.capitalize()} sticker '{path.name}' is {size_bytes} bytes, exceeds limit {max_size} bytes"
        )

    # Meta sticker delivery expects 512x512 canvas.
    if width is not None and height is not None and (width != META_STICKER_DIMENSION or height != META_STICKER_DIMENSION):
        return False, (
            f"Sticker '{path.name}' has {width}x{height}, expected {META_STICKER_DIMENSION}x{META_STICKER_DIMENSION}"
        )

    return True, "ok"


async def _send_meta_typing_indicator(message_id: str) -> bool:
    """Send official Meta typing indicator + read receipt for the incoming message."""
    try:
        access_token, phone_number_id, graph_version = _get_meta_config()
        url = f"https://graph.facebook.com/{graph_version}/{phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
            "typing_indicator": {"type": "text"},
        }
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload, timeout=30.0)
            if resp.status_code not in (200, 201):
                print(f"Meta typing indicator failed ({resp.status_code}): {resp.text}")
                return False
            print(f"Meta typing indicator sent for message_id={message_id}")
            return True
    except Exception as e:
        print(f"Meta typing indicator exception: {e}")
        return False

def _get_twilio_config():
    """Get Twilio config at call time, after env vars are loaded."""
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    phone = os.getenv("TWILIO_PHONE_NUMBER")
    if not sid or not token or not phone:
        raise RuntimeError(
            f"Twilio config incomplete: SID={'set' if sid else 'MISSING'}, "
            f"TOKEN={'set' if token else 'MISSING'}, PHONE={'set' if phone else 'MISSING'}"
        )
    return sid, token, phone

def _normalize_whatsapp_number(value: str) -> str:
    if not value:
        return value
    normalized = value.strip().replace(" ", "")
    if not normalized.startswith("whatsapp:"):
        normalized = f"whatsapp:{normalized}"
    return normalized


def _normalize_meta_number(value: str) -> str:
    if not value:
        return value
    normalized = value.strip().replace(" ", "")
    if normalized.startswith("whatsapp:"):
        normalized = normalized.split(":", 1)[1]
    if normalized.startswith("+"):
        normalized = normalized[1:]
    return normalized


def _get_meta_config():
    access_token = os.getenv("WHATSAPP_META_ACCESS_TOKEN")
    phone_number_id = os.getenv("WHATSAPP_META_PHONE_NUMBER_ID")
    graph_version = os.getenv("WHATSAPP_META_GRAPH_VERSION", "v25.0")
    if not access_token or not phone_number_id:
        raise RuntimeError(
            "Meta config incomplete: "
            f"ACCESS_TOKEN={'set' if access_token else 'MISSING'}, "
            f"PHONE_NUMBER_ID={'set' if phone_number_id else 'MISSING'}"
        )
    return access_token, phone_number_id, graph_version

async def download_media(media_url: str) -> bytes:
    """Download media from Twilio or Meta depending on configured provider."""
    if WHATSAPP_PROVIDER == "meta":
        access_token, _, graph_version = _get_meta_config()
        meta_id = str(media_url).strip()
        if not meta_id:
            raise RuntimeError("Meta media id is missing")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            metadata_resp = await client.get(
                f"https://graph.facebook.com/{graph_version}/{meta_id}",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=30.0,
            )
            metadata_resp.raise_for_status()
            media_meta = metadata_resp.json()
            download_url = media_meta.get("url")
            if not download_url:
                raise RuntimeError(f"Meta media URL missing for media id {meta_id}")

            media_resp = await client.get(
                download_url,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=60.0,
            )
            media_resp.raise_for_status()
            return media_resp.content

    sid, token, _ = _get_twilio_config()
    async with httpx.AsyncClient(follow_redirects=True) as client:
        media_resp = await client.get(
            media_url,
            auth=(sid, token),
            timeout=60.0
        )
        media_resp.raise_for_status()
        return media_resp.content

async def send_whatsapp_message(to: str, text: str, from_number_override: str = None):
    """Send a WhatsApp message via Twilio or Meta API."""
    if text is None:
        text = ""
    text = str(text)
    if len(text) > MAX_OUTBOUND_TEXT_LEN:
        print(f"Outbound text too long ({len(text)}), truncating to {MAX_OUTBOUND_TEXT_LEN} characters.")
        text = text[:MAX_OUTBOUND_TEXT_LEN]

    if WHATSAPP_PROVIDER == "meta":
        access_token, phone_number_id, graph_version = _get_meta_config()
        to_number = _normalize_meta_number(to)
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "text",
            "text": {"body": text},
        }
        url = f"https://graph.facebook.com/{graph_version}/{phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload, timeout=30.0)
            if resp.status_code not in (200, 201):
                print(f"Meta send failed ({resp.status_code}): {resp.text}")
                return False
            print(f"Meta WhatsApp message sent to {to_number}: '{text[:80]}...'")
            return True

    sid, token, phone = _get_twilio_config()
    from_number = _normalize_whatsapp_number(from_number_override or phone)
    to_number = _normalize_whatsapp_number(to)
    
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    
    payload = {
        "To": to_number,
        "From": from_number,
        "Body": text
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            url,
            headers=headers,
            data=payload,
            auth=(sid, token),
            timeout=30.0
        )
        
        if resp.status_code not in (200, 201):
            print(f"Twilio send failed ({resp.status_code}): {resp.text}")
            return False
        print(f"✅ WhatsApp message sent to {to}: '{text[:80]}...'")
        return True


async def send_thinking_indicator(to: str, from_number_override: str = None, meta_message_id: str = None) -> bool:
    """Send animated thinking indicator when possible, fallback to text by caller."""
    if WHATSAPP_PROVIDER == "meta":
        try:
            access_token, phone_number_id, graph_version = _get_meta_config()
            configured_asset = _resolve_thinking_asset_path()
            if not configured_asset.exists():
                print(f"Thinking asset not found at '{configured_asset}', skipping media indicator.")
                return await _send_meta_typing_indicator(meta_message_id) if meta_message_id else False

            full_path = _select_meta_sticker_asset(configured_asset)
            if not full_path:
                return await _send_meta_typing_indicator(meta_message_id) if meta_message_id else False

            print(f"Using thinking asset from '{full_path}'.")

            with full_path.open("rb") as f:
                file_bytes = f.read()

            valid_sticker, reason = _validate_meta_sticker_asset(file_bytes, full_path)
            if not valid_sticker:
                print(f"Meta sticker preflight failed: {reason}")
                return await _send_meta_typing_indicator(meta_message_id) if meta_message_id else False

            upload_url = f"https://graph.facebook.com/{graph_version}/{phone_number_id}/media"
            send_url = f"https://graph.facebook.com/{graph_version}/{phone_number_id}/messages"
            headers = {"Authorization": f"Bearer {access_token}"}

            async with httpx.AsyncClient() as client:
                upload_resp = await client.post(
                    upload_url,
                    headers=headers,
                    data={"messaging_product": "whatsapp", "type": "image/webp"},
                    files={"file": (full_path.name, file_bytes, "image/webp")},
                    timeout=30.0,
                )
                if upload_resp.status_code not in (200, 201):
                    print(f"Meta media upload failed ({upload_resp.status_code}): {upload_resp.text}")
                    return False
                media_id = (upload_resp.json() or {}).get("id")
                if not media_id:
                    return False

                payload = {
                    "messaging_product": "whatsapp",
                    "to": _normalize_meta_number(to),
                    "type": "sticker",
                    "sticker": {"id": media_id},
                }
                send_resp = await client.post(send_url, headers={**headers, "Content-Type": "application/json"}, json=payload, timeout=30.0)
                if send_resp.status_code not in (200, 201):
                    print(f"Meta thinking sticker send failed ({send_resp.status_code}): {send_resp.text}")
                    return await _send_meta_typing_indicator(meta_message_id) if meta_message_id else False
                print(f"Meta thinking sticker accepted for delivery. media_id={media_id}")
                return True
        except Exception as e:
            print(f"Meta thinking indicator failed: {e}")
            return await _send_meta_typing_indicator(meta_message_id) if meta_message_id else False

    try:
        media_url = os.getenv("TWILIO_THINKING_MEDIA_URL", "").strip()
        if not media_url:
            return False

        sid, token, phone = _get_twilio_config()
        from_number = _normalize_whatsapp_number(from_number_override or phone)
        to_number = _normalize_whatsapp_number(to)
        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        payload = {
            "To": to_number,
            "From": from_number,
            "Body": "",
            "MediaUrl": media_url,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=payload,
                auth=(sid, token),
                timeout=30.0,
            )
            if resp.status_code not in (200, 201):
                print(f"Twilio thinking media send failed ({resp.status_code}): {resp.text}")
                return False
            return True
    except Exception as e:
        print(f"Twilio thinking indicator failed: {e}")
        return False


def _guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/png"


def _build_twilio_media_url(image_path: Path) -> str:
    if str(image_path).startswith("http://") or str(image_path).startswith("https://"):
        return str(image_path)

    media_base = os.getenv("WHATSAPP_MEDIA_BASE_URL", "").strip().rstrip("/")
    if not media_base:
        return ""

    # Expected when memes are served at <base>/memes/...
    output_root = Path(os.getenv("MEME_OUTPUT_DIR", "memes")).resolve()
    try:
        relative = image_path.resolve().relative_to(output_root.parent)
    except Exception:
        relative = Path(image_path.name)

    return f"{media_base}/{str(relative).replace(os.sep, '/')}"


async def send_whatsapp_image(
    to: str,
    image_path: str,
    caption: str = "",
    from_number_override: str = None,
) -> bool:
    path = Path(image_path)
    if not path.exists():
        print(f"Image file not found: {image_path}")
        return False

    if WHATSAPP_PROVIDER == "meta":
        try:
            access_token, phone_number_id, graph_version = _get_meta_config()
            upload_url = f"https://graph.facebook.com/{graph_version}/{phone_number_id}/media"
            send_url = f"https://graph.facebook.com/{graph_version}/{phone_number_id}/messages"
            mime_type = _guess_mime_type(path)
            headers = {"Authorization": f"Bearer {access_token}"}
            data = path.read_bytes()

            async with httpx.AsyncClient() as client:
                upload_resp = await client.post(
                    upload_url,
                    headers=headers,
                    data={"messaging_product": "whatsapp", "type": mime_type},
                    files={"file": (path.name, data, mime_type)},
                    timeout=40.0,
                )
                if upload_resp.status_code not in (200, 201):
                    print(f"Meta image upload failed ({upload_resp.status_code}): {upload_resp.text}")
                    return False

                media_id = (upload_resp.json() or {}).get("id")
                if not media_id:
                    print("Meta image upload returned no media id")
                    return False

                payload = {
                    "messaging_product": "whatsapp",
                    "to": _normalize_meta_number(to),
                    "type": "image",
                    "image": {
                        "id": media_id,
                    },
                }
                if caption.strip():
                    payload["image"]["caption"] = caption[:1024]

                send_resp = await client.post(
                    send_url,
                    headers={**headers, "Content-Type": "application/json"},
                    json=payload,
                    timeout=40.0,
                )
                if send_resp.status_code not in (200, 201):
                    print(f"Meta image send failed ({send_resp.status_code}): {send_resp.text}")
                    return False

                return True
        except Exception as e:
            print(f"Meta send_whatsapp_image failed: {e}")
            return False

    try:
        media_url = _build_twilio_media_url(path)
        if not media_url:
            print(
                "Twilio image send skipped: WHATSAPP_MEDIA_BASE_URL is not set for public media hosting."
            )
            return False

        sid, token, phone = _get_twilio_config()
        from_number = _normalize_whatsapp_number(from_number_override or phone)
        to_number = _normalize_whatsapp_number(to)
        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        payload = {
            "To": to_number,
            "From": from_number,
            "Body": (caption or "")[:1024],
            "MediaUrl": media_url,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=payload,
                auth=(sid, token),
                timeout=40.0,
            )
            if resp.status_code not in (200, 201):
                print(f"Twilio image send failed ({resp.status_code}): {resp.text}")
                return False
            return True
    except Exception as e:
        print(f"Twilio send_whatsapp_image failed: {e}")
        return False