import base64
import logging
from io import BytesIO
from typing import AsyncGenerator, Optional

from PIL import Image
import filetype

import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

VISION_PROMPT_PATH = (
    Path(__file__).parent.parent.parent.parent / "shared" / "prompts" / "vision_img.yaml"
)

def load_vision_prompt() -> str:
    try:
        with open(VISION_PROMPT_PATH, "r") as f:
            data = yaml.safe_load(f)
        return data["vision_img"]["template"]
    except Exception as e:
        logger.error(f"Failed to load vision prompt: {e}")
        return "Describe the image in detail."

class ImageParser:
    """
    Async image parser for RagsApp. Converts images to base64, sends to VLM with vision prompt,
    and yields the LLM's response.
    """

    MIME_TYPE_MAPPING = {
        "bmp": "image/bmp",
        "gif": "image/gif",
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "png": "image/png",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        "webp": "image/webp",
    }

    def __init__(self, aclient_openai):
        self.aclient_openai = aclient_openai
        self.vision_prompt = load_vision_prompt()

    def _get_image_media_type(self, data: bytes, filename: Optional[str] = None) -> str:
        kind = filetype.guess(data)
        if kind and kind.mime.startswith("image/"):
            return kind.mime
        if filename:
            ext = filename.split(".")[-1].lower()
            return self.MIME_TYPE_MAPPING.get(ext, "application/octet-stream")
        return "application/octet-stream"

    async def ingest(
        self, data: bytes, filename: Optional[str] = None, model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Ingest an image file, send to OpenAI Vision model, yield the description.
        """
        media_type = self._get_image_media_type(data, filename)
        image_data = base64.b64encode(data).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        },
                    },
                ],
            }
        ]
        model = model or "gpt-4o-mini"
        try:
            response = await self.aclient_openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0.2,
            )
            content = response.choices[0].message.content
            yield content
        except Exception as e:
            logger.error(f"Error during image ingestion: {e}")
            yield "Error processing image."