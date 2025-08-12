import os
import re
import base64
import yaml
from io import BytesIO
from typing import AsyncGenerator, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

import olefile
import logging
from .base_parser import AsyncParser
from openai import AsyncOpenAI

load_dotenv()

OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

class DOCParser(AsyncParser[bytes]):
    """
    A parser for DOC (legacy Microsoft Word) data, including text and images.
    """

    def __init__(self, aclient_openai: Optional[AsyncOpenAI], server_type: str, processor_ref: Optional[Any] = None):
        self.olefile = olefile
        self.aclient_openai = aclient_openai
        self.server_type = server_type
        self.processor_ref = processor_ref
        self.vision_prompt_text = self._load_vision_prompt()
    
    def _load_vision_prompt(self) -> str:
        """Loads the vision prompt from the specified YAML file."""
        try:
            prompt_file_path = Path("./prompts") / "vision_img.yaml"
            with open(prompt_file_path, 'r') as f:
                prompt_data = yaml.safe_load(f)
            if prompt_data and "vision_img" in prompt_data and "template" in prompt_data["vision_img"]:
                template_content = prompt_data["vision_img"]["template"]
                print("Successfully loaded vision prompt template.")
                return template_content
            else:
                print(f"Vision prompt template not found or invalid in {prompt_file_path}.")
                return "Describe the image in detail."
        except Exception as e:
            print(f"Error loading vision prompt: {e}")
            return "Describe the image in detail."
    
    async def _get_image_description(self, image_bytes: bytes, content_type: str) -> str:
        """Generates a description for an image using its specific content type."""
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        media_type = content_type

        if self.server_type == "ARMY":
            if not self.processor_ref:
                print("Processor reference not available for NVIDIA VLM call. Skipping image description.")
                return ""
            
            print(f"Using NVIDIA VLM for image ({media_type}).")
            messages = [{"role": "user", "content": self.vision_prompt_text, "image": image_data}]
            
            description = await self.processor_ref._call_nvidia_api(
                payload_messages=messages, is_vision_call=True, max_tokens=1024
            )
            return f"\n[Image Description]: {description.strip()}\n" if description else ""

        if not self.aclient_openai:
            print("OpenAI client not available, skipping image description.")
            return ""
        
        try:
            print(f"Using OpenAI Vision for image ({media_type}).")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.vision_prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                ],
            }]
            response = await self.aclient_openai.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=messages,
                max_tokens=1024,
            )
            description = response.choices[0].message.content
            return f"\n[Image Description]: {description.strip()}\n" if description else ""
        except Exception as e:
            print(f"Error getting image description from OpenAI: {e}")
            return ""
    
    def _get_content_type_from_bytes(self, image_bytes: bytes) -> Optional[str]:
        """Identifies the image MIME type by checking the file's magic numbers."""
        if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'image/png'
        if image_bytes.startswith(b'\xFF\xD8\xFF'):
            return 'image/jpeg'
        if image_bytes.startswith(b'GIF87a') or image_bytes.startswith(b'GIF89a'):
            return 'image/gif'
        if image_bytes.startswith(b'BM'):
            return 'image/bmp'
        if image_bytes.startswith(b'\xD7\xCD\xC6\x9A'):
            return 'image/wmf'
        # Add other magic numbers if needed
        return None

    async def ingest(self, data: bytes, **kwargs) -> AsyncGenerator[str, None]:
        """Ingest DOC data, yielding all text first, followed by descriptions of any found images."""
        if not isinstance(data, bytes):
            raise TypeError("DOC data must be in bytes format.")

        file_obj = BytesIO(data)
        ole = None
        try:
            ole = self.olefile.OleFileIO(file_obj)
            
            # --- Text Extraction ---
            if ole.exists("WordDocument"):
                word_stream = ole.openstream("WordDocument").read()
                text = word_stream.decode("utf-8", errors="ignore").replace("\x00", "")
                paragraphs = self._clean_text(text)
                for paragraph in paragraphs:
                    if paragraph.strip():
                        yield paragraph.strip()
            
            # --- Image Extraction ---
            print("Scanning DOC file for image streams.")
            image_count = 0
            # <-- MODIFIED: Iterate through all streams to find images by content, not just name
            for stream_path in ole.listdir(streams=True):
                try:
                    # Read the stream data first
                    stream_bytes = ole.openstream(stream_path).read()
                    if not stream_bytes:
                        continue
                    
                    # Identify the content type from the bytes
                    content_type = self._get_content_type_from_bytes(stream_bytes)
                    
                    # If it's a recognized image type, process it
                    if content_type:
                        image_count += 1
                        print(f"Found image in stream {'/'.join(stream_path)} ({content_type}). Generating description.")
                        description = await self._get_image_description(stream_bytes, content_type)
                        if description:
                            yield description
                except Exception as img_e:
                    print(f"Could not read or process stream {'/'.join(stream_path)}: {img_e}")
            
            if image_count > 0:
                 print(f"Finished processing {image_count} image(s).")


        except Exception as e:
            print(f"Error processing DOC file: {str(e)}", exc_info=True)
            raise ValueError(f"Error processing DOC file: {str(e)}") from e
        finally:
            if ole:
                ole.close()
            file_obj.close()

    def _clean_text(self, text: str) -> list[str]:
        """Clean and split the extracted text into paragraphs."""
        # Remove non-printable control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Split into paragraphs
        paragraphs = re.split(r"(\r\n|\n|\r){2,}", text)
        return [p.strip() for p in paragraphs if p and p.strip()]