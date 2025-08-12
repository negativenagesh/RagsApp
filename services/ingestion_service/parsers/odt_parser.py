import os
import logging
from io import BytesIO
from typing import AsyncGenerator, Optional, Any
import base64
import yaml
from pathlib import Path
from dotenv import load_dotenv

from .base_parser import AsyncParser
from openai import AsyncOpenAI

# Set up logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Check for odfpy installation
try:
    from odf import opendocument,text,table,draw,element
    ODFPY_INSTALLED = True
except ImportError:
    ODFPY_INSTALLED = False

OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class ODTParser(AsyncParser[bytes]):
    """A parser for ODT data, including text, tables, and image descriptions."""

    def __init__(self, aclient_openai: Optional[AsyncOpenAI], server_type: str, processor_ref: Optional[Any] = None):
        if not ODFPY_INSTALLED:
            msg = "ODT parsing requires 'odfpy'. Please install it (`pip install odfpy`)."
            logger.error(msg)
            raise ImportError(msg)
            
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
                logger.info("Successfully loaded vision prompt template.")
                return template_content
            else:
                logger.warning(f"Vision prompt template not found or invalid in {prompt_file_path}.")
                return "Describe the image in detail."
        except Exception as e:
            logger.error(f"Error loading vision prompt: {e}")
            return "Describe the image in detail."
    
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
        return None

    async def _get_image_description(self, image_bytes: bytes, content_type: str) -> str:
        """Generates a description for an image using its specific content type."""
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        media_type = content_type

        # if self.server_type == "ARMY":
        #     if not self.processor_ref:
        #         logger.warning("Processor reference not available for NVIDIA VLM call. Skipping image description.")
        #         return ""
            
        #     logger.info(f"Using NVIDIA VLM for image ({media_type}).")
        #     messages = [{"role": "user", "content": self.vision_prompt_text, "image": image_data}]
            
        #     description = await self.processor_ref._call_nvidia_api(
        #         payload_messages=messages, is_vision_call=True, max_tokens=1024
        #     )
        #     return f"\n[Image Description]: {description.strip()}\n" if description else ""

        if not self.aclient_openai:
            logger.warning("OpenAI client not available, skipping image description.")
            return ""
        
        try:
            logger.info(f"Using OpenAI Vision for image ({media_type}).")
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
            logger.error(f"Error getting image description from OpenAI: {e}")
            return ""

    def _extract_text_from_element(self, el: element.Element) -> str:
        """Recursively extracts text from an ODF element."""
        texts = []
        for child in el.childNodes:
            if child.nodeType == child.TEXT_NODE:
                texts.append(child.data)
            elif child.NodeType == child.ELEMENT_NODE:
                texts.append(self._extract_text_from_element(child))
        return "".join(texts)

    def _convert_table_to_markdown(self, table_element: table.Table) -> str:
        """Converts an odfpy Table object to a Markdown formatted string."""
        markdown_rows = []
        rows = table.getElementsByType(table_element.TableRow)
        if not rows:
            return ""
        
        header_cells = rows[0].getElementsByType(table_element.TableCell)
        header_texts = [self._extract_text_from_element(cell).strip() for cell in header_cells]
        if not any(header_texts):
            return ""
        
        markdown_rows.append("| " + " | ".join(header_texts) + " |")
        markdown_rows.append("| " + " | ".join(["---"] * len(header_texts)) + " |")

        for row in rows[1:]:
            body_cells = row.getElementsByType(table_element.TableCell)
            body_texts = [self._extract_text_from_element(cell).strip() for cell in body_cells]
            if len(body_texts) == len(header_texts):
                markdown_rows.append("| " + " | ".join(body_texts) + " |")
        
        return "\n".join(markdown_rows)

    async def ingest(self, data: bytes, **kwargs) -> AsyncGenerator[str, None]:
        """Ingest ODT data, yielding paragraphs, tables, and image descriptions in document order."""
        if not isinstance(data, bytes):
            raise TypeError("ODT data must be in bytes format.")

        odt_stream = BytesIO(data)
        try:
            doc = opendocument.load(odt_stream)
            logger.info("Loaded ODT file. Iterating through content blocks.")
            
            body = doc.text
            for element in body.childNodes:
                 # Skip non-element nodes like text nodes containing only whitespace
                if element.nodeType != element.ELEMENT_NODE:
                    continue
                element_qname = element.qname
                # Process Paragraphs and Headings
                if element.tagName in ["text:p", "text:h"]:
                    text_content = self._extract_text_from_element(element).strip()
                    if text_content:
                        yield text_content
                    
                    # Check for images within the element
                    frames = element.getElementsByType(draw.Frame)
                    for frame in frames:
                        images = frame.getElementsByType(draw.Image)
                        for image in images:
                            href = image.getAttribute("href")
                            if href:
                                try:
                                    image_bytes = doc.getPart(href)
                                    content_type = self._get_content_type_from_bytes(image_bytes)
                                    if content_type:
                                        logger.info(f"Found image '{href}'. Generating description.")
                                        description = await self._get_image_description(image_bytes, content_type)
                                        if description:
                                            yield description
                                except Exception as img_e:
                                    logger.warning(f"Could not process image part '{href}': {img_e}")

                # Process Tables
                elif element.tagName == "table:table":
                    markdown_table = self._convert_table_to_markdown(element)
                    if markdown_table:
                        yield markdown_table

        except Exception as e:
            logger.error(f"Failed to read ODT stream: {e}", exc_info=True)
            raise ValueError(f"Error processing ODT file: {str(e)}") from e
        finally:
            odt_stream.close()