# base/parsers/mistral_ocr_parser.py
"""
Mistral OCR Parser module for cloud-based OCR using Mistral AI's OCR API.
Provides high-quality OCR extraction via the mistral-ocr-latest model.
"""
import os
import asyncio
import tempfile
from typing import AsyncGenerator, Optional, List

from .ocr_utils import clean_ocr_repetitions

# Optional dependency check
try:
    from mistralai import Mistral
    MISTRAL_SDK_AVAILABLE = True
except ImportError:
    MISTRAL_SDK_AVAILABLE = False


class MistralOCRParser:
    """OCR via Mistral OCR API (mistral-ocr-latest)."""
    
    def __init__(self, api_key: Optional[str]):
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not set.")
        if not MISTRAL_SDK_AVAILABLE:
            raise ImportError("mistralai SDK is not installed.")
        self.client = Mistral(api_key=api_key)

    async def ingest(self, data: bytes) -> AsyncGenerator[str, None]:
        if not isinstance(data, bytes):
            raise TypeError("PDF data must be in bytes format.")
        
        # Run blocking SDK calls in a thread
        def _run_ocr(tmp_path: str) -> List[str]:
            with open(tmp_path, "rb") as f:
                uploaded_pdf = self.client.files.upload(
                    file={"file_name": os.path.basename(tmp_path), "content": f},
                    purpose="ocr",
                )
            signed_url = self.client.files.get_signed_url(file_id=uploaded_pdf.id)
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "document_url", "document_url": signed_url.url},
                include_image_base64=False,
            )
            pages = []
            for p in getattr(ocr_response, "pages", []) or []:
                text = getattr(p, "markdown", None) or getattr(p, "text", "") or ""
                pages.append(text)
            return pages

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            try:
                pages_text = await asyncio.to_thread(_run_ocr, tmp.name)
            except Exception as e:
                print(f"Mistral OCR failed: {e}")
                raise
                
        # Now yield each page with its page number (starting from 1)
        for page_num, page_text in enumerate(pages_text, 1):
            # Clean OCR repetitions before yielding
            cleaned_text = clean_ocr_repetitions(page_text)
            yield (cleaned_text, page_num)

