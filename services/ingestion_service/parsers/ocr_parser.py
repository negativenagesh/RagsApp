import asyncio
from io import BytesIO
from typing import AsyncGenerator

try:
    from pdf2image import convert_from_bytes
    PYPDF2IMAGE_INSTALLED = True
except ImportError:
    PYPDF2IMAGE_INSTALLED = False

try:
    import pytesseract
    PYTESSERACT_INSTALLED = True
except ImportError:
    PYTESSERACT_INSTALLED = False

class OCRParser:
    """A parser for OCR-based text extraction from PDF files."""

    def __init__(self):
        if not PYPDF2IMAGE_INSTALLED or not PYTESSERACT_INSTALLED:
            msg = (
                "OCR parsing requires 'pdf2image' and 'pytesseract'. "
                "Please install them and ensure Tesseract-OCR is in your system's PATH."
            )
            print(f"ERROR: {msg}")
            raise ImportError(msg)

    async def ingest(self, data: bytes) -> AsyncGenerator[str, None]:
        """Ingest PDF data, perform OCR, and yield text from each page."""
        if not isinstance(data, bytes):
            raise TypeError("PDF data must be in bytes format.")

        pdf_stream = BytesIO(data)
        print("Starting OCR text extraction. This may take a while...")
        try:
            images = convert_from_bytes(pdf_stream.read())
            print(f"Converted {len(images)} pages to images for OCR.")
            for i, image in enumerate(images):
                page_num = i + 1
                try:
                    # Run OCR in a thread to avoid blocking the event loop
                    page_text = await asyncio.to_thread(pytesseract.image_to_string, image)
                    if not page_text or not page_text.strip():
                        print(f"OCR found no text on page {page_num}.")
                    yield page_text
                except pytesseract.TesseractNotFoundError:
                    print("Tesseract executable not found. Please install Tesseract-OCR and ensure it's in your system's PATH.")
                    raise
                except Exception as ocr_err:
                    print(f"Error during OCR on page {page_num}: {ocr_err}")
                    yield ""
        except Exception as e:
            print(f"Failed to convert PDF to images for OCR: {e}")
            raise ValueError(f"Error processing PDF for OCR: {str(e)}") from e
        finally:
            pdf_stream.close()