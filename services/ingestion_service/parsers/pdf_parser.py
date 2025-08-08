import base64
import yaml
import fitz
import pdfplumber
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, AsyncGenerator

# Use the shared prompts directory as in your processor.py
SHARED_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "shared" / "prompts"
OPENAI_CHAT_MODEL = "gpt-4o-mini"  # Or import from config

class PDFParser:
    """A parser for extracting tables, text, and images from PDF files using pdfplumber and PyMuPDF."""

    def __init__(self, aclient_openai: Optional[Any], processor_ref: Optional[Any] = None):
        self.aclient_openai = aclient_openai
        self.processor_ref = processor_ref
        self.vision_prompt_text = self._load_vision_prompt()

    def _load_vision_prompt(self) -> str:
        try:
            prompt_file_path = SHARED_PROMPTS_DIR / "vision_img.yaml"
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
        if not self.processor_ref:
            print("Processor reference not available for OpenAI call. Skipping image description.")
            return ""

        print("Using OpenAI VLM for image description.")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.vision_prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_data}"},
                    },
                ],
            }
        ]

        description = await self.processor_ref._call_openai_api(
            model_name=OPENAI_CHAT_MODEL,
            payload_messages=messages,
            is_vision_call=True,
            max_tokens=1024,
            temperature=1.0,
        )

        print(f"*** image description: {description}")
        return f"\n[Image Description]: {description.strip()}\n" if description else ""

    async def ingest(self, data: bytes) -> AsyncGenerator[str, None]:
        """
        Ingest PDF data and yield a stream of content (text and markdown tables).
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
                        # 1. Extract text from the page
                        page_text = p_page.extract_text()
                        if page_text:
                            yield page_text.strip()

                        # 2. Extract tables from the same page
                        tables = p_page.extract_tables()
                        for table in tables:
                            table_markdown = self._convert_table_to_markdown(table)
                            yield table_markdown

                        # 3. Extract images using PyMuPDF and get descriptions
                        if page_num <= len(pdf_fitz):
                            fitz_page = pdf_fitz.load_page(page_num - 1)
                            image_list = fitz_page.get_images(full=True)
                            if image_list:
                                print(f"Found {len(image_list)} images on page {page_num}.")
                                for img_info in image_list:
                                    xref = img_info[0]
                                    base_image = pdf_fitz.extract_image(xref)
                                    image_bytes = base_image["image"]

                                    description = await self._get_image_description(image_bytes)
                                    if description:
                                        yield description
                        else:
                            print(f"Skipping image extraction for page {page_num} due to page count mismatch.")

        except Exception as e:
            print(f"Failed to process PDF: {e}")
            raise ValueError(f"Error processing PDF file: {str(e)}") from e

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