import re
import base64
import yaml
from io import BytesIO
from typing import AsyncGenerator, Optional, Any
from pathlib import Path
import olefile
from .base_parser import AsyncParser
from openai import AsyncOpenAI

OPENAI_CHAT_MODEL = "gpt-4o-mini"

class DOCParser(AsyncParser[bytes]):
    """
    A parser for DOC (legacy Microsoft Word) data, including text and images.
    """

    def __init__(self, aclient_openai: Optional[AsyncOpenAI], processor_ref: Optional[Any] = None):
        self.olefile = olefile
        self.aclient_openai = aclient_openai
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

        if not self.aclient_openai:
            print("OpenAI client not available, skipping image description.")
            return ""
        
        try:
            print(f"Using GPT-4o for smart image extraction ({media_type}).")
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
                max_tokens=4096,
                temperature=0.3,
            )
            description = response.choices[0].message.content
            return f"\n{description.strip()}\n" if description else ""
        except Exception as e:
            print(f"Error getting image extraction from OpenAI: {e}")
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
            
            print("Scanning DOC file for image streams.")
            # Collect all valid images first, then process in parallel
            image_tasks = []
            image_stream_names = []
            for stream_path in ole.listdir(streams=True):
                try:
                    stream_bytes = ole.openstream(stream_path).read()
                    if not stream_bytes:
                        continue
                    content_type = self._get_content_type_from_bytes(stream_bytes)
                    if content_type:
                        image_tasks.append(self._get_image_description(stream_bytes, content_type))
                        image_stream_names.append('/'.join(stream_path))
                except Exception as img_e:
                    print(f"❌ Could not read stream {'/'.join(stream_path)}: {img_e}")
            
            if image_tasks:
                print(f"📸 Found {len(image_tasks)} images. Processing in parallel...")
                import asyncio
                image_results = await asyncio.gather(*image_tasks, return_exceptions=True)
                for i, result in enumerate(image_results):
                    if isinstance(result, Exception):
                        print(f"❌ Image processing failed for {image_stream_names[i]}: {result}")
                    elif result:
                        yield result
                print(f"✅ Finished processing {len(image_tasks)} image(s) from DOC file.")
            else:
                print("No images found in DOC file.")


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