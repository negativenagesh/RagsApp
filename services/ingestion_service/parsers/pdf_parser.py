# base/parsers/pdf_parser.py
"""
PDF Parser module for extracting text, tables, and image descriptions from PDF files.
Uses pdfplumber for text/table extraction and PyMuPDF (fitz) for image extraction.
"""
import asyncio
import base64
import yaml
from io import BytesIO
from pathlib import Path
from typing import AsyncGenerator, Optional, Any, Tuple

import fitz
import pdfplumber
from openai import AsyncOpenAI


class PDFParser:
    """A parser for extracting tables and text from PDF files using pdfplumber."""

    def __init__(self, aclient_openai: Optional[AsyncOpenAI], config: dict, processor_ref: Optional[Any] = None):
        
        self.aclient_openai = aclient_openai
        self.processor_ref = processor_ref
        self.vision_prompt_text = self._load_vision_prompt()
    
    def _load_vision_prompt(self) -> str:
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
    
    async def _get_image_description(self, image_bytes: bytes) -> str:
        
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        media_type = "image/png"  # PyMuPDF pixmaps default to png

        if not self.aclient_openai:
            print("OpenAI client not available, skipping image description.")
            return ""

        OPENAI_CHAT_MODEL = "gpt-4o-mini"

        print("Using OpenAI VLM for image description.")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.vision_prompt_text},
                    {
                        "type": "image_url",
                        "image_url": { "url": f"data:{media_type};base64,{image_data}" },
                    },
                ],
            }
        ]

        response = await self.aclient_openai.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=1.0,
        )
        description = response.choices[0].message.content or ""

        print(f"*** image description: {description}")
        return f"\n[Image Description]: {description.strip()}\n" if description else ""


    async def ingest(self, data: bytes) -> AsyncGenerator[Tuple[str, int], None]:
        """
        Ingest PDF data and yield a stream of content parts (text, tables, image descriptions)
        along with their corresponding page number.
        """
        if not isinstance(data, bytes):
            raise TypeError("PDF data must be in bytes format.")

        pdf_stream = BytesIO(data)
        try:
            with pdfplumber.open(pdf_stream) as pdf_plumber:
                with fitz.open(stream=data, filetype="pdf") as pdf_fitz:
                    if len(pdf_plumber.pages) != len(pdf_fitz):
                        print("Warning: Page count mismatch between pdfplumber and PyMuPDF.")
                    
                    for page_num, p_page in enumerate(pdf_plumber.pages, 1):
                        # 1. Extract text from the page. Each block is a potential paragraph.
                        page_text = p_page.extract_text()
                        if page_text:
                            yield (page_text.strip(), page_num)
                        
                        # 2. Extract tables from the same page as separate blocks.
                        tables = p_page.extract_tables()
                        for table in tables:
                            table_markdown = self._convert_table_to_markdown(table)
                            yield (table_markdown, page_num)
                        
                        # 3. Extract images using PyMuPDF and get descriptions as separate blocks.
                        if page_num <= len(pdf_fitz):
                            fitz_page = pdf_fitz.load_page(page_num - 1)
                            image_list = fitz_page.get_images(full=True)
                            if image_list:
                                print(f"Found {len(image_list)} images on page {page_num}.")
                                for img_info in image_list:
                                    xref = img_info[0]
                                    base_image = pdf_fitz.extract_image(xref)
                                    image_bytes = base_image["image"]
                                    
                                    img_w = int(base_image.get("width", 0) or 0)
                                    img_h = int(base_image.get("height", 0) or 0)
                                    if img_w < 100 or img_h < 100:
                                        print(f"Skipping image extraction for page {page_num} due to small image size.")
                                        continue

                                    description = await self._get_image_description(image_bytes)
                                    if description:
                                        yield (description, page_num)
                        else:
                            print(f"Skipping image extraction for page {page_num} due to page count mismatch.")
                    
        except Exception as e:
            print(f"Failed to process PDF: {e}")
            raise ValueError(f"Error processing PDF: {str(e)}") from e

    def _convert_table_to_markdown(self, table: list) -> str:
        """Convert a table (list of rows) to Markdown format."""
        if not table or not table[0]:
            return ""
        
        header = [str(cell) if cell is not None else "" for cell in table[0]]
        markdown = "| " + " | ".join(header) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"
        for row in table[1:]:
            if row:
                processed_row = [str(cell) if cell is not None else "" for cell in row]
                if len(processed_row) == len(header):
                    markdown += "| " + " | ".join(processed_row) + " |\n"
        return markdown
