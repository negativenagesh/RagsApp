import logging
from io import BytesIO
from typing import AsyncGenerator

from .base_parser import AsyncParser

logger = logging.getLogger(__name__)

class TextParser(AsyncParser[bytes]):
    """A parser for raw text data from .txt files."""

    def __init__(self):
        """Initializes the TextParser."""
        pass

    async def ingest(self, data: bytes, **kwargs) -> AsyncGenerator[str, None]:
        """Ingest text data and yield the decoded string content."""
        if not isinstance(data, bytes):
            raise TypeError("Text data must be in bytes format.")

        try:
            # Decode bytes to string, ignoring errors for robustness
            text_content = data.decode("utf-8", errors="ignore")
            if text_content and text_content.strip():
                yield text_content
            else:
                logger.warning("Provided text data is empty or contains only whitespace.")
                # Yield nothing if the content is empty
                return
        except Exception as e:
            logger.error(f"Failed to decode or process text data: {e}", exc_info=True)
            raise ValueError(f"Error processing text file: {str(e)}") from e
