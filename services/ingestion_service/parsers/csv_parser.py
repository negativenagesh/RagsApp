import logging
import csv
from io import StringIO
from typing import AsyncGenerator

from .base_parser import AsyncParser

logger = logging.getLogger(__name__)

class CSVParser(AsyncParser[bytes]):
    """A parser for CSV data that automatically detects delimiters."""

    def __init__(self):
        """Initializes the CSVParser."""
        pass

    def _get_delimiter(self, sample_data: str) -> str:
        """Detects the delimiter from a sample of the data."""
        try:
            sniffer = csv.Sniffer()
            #Sniff with a common set of delimiters
            dialect = sniffer.sniff(sample_data, delimiters=',;\t|')
            return dialect.delimiter
        except csv.Error:
            logger.warning("CSV Sniffer could not detect delimiter, defaulting to ','.")
            return ','

    async def ingest(self, data: bytes, **kwargs) -> AsyncGenerator[str, None]:
        """Ingest CSV data and yield each row as a comma-separated string."""
        if not isinstance(data, bytes):
            raise TypeError("CSV data must be in bytes format.")

        try:
            try:
                decoded_data = data.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning("UTF-8 decoding failed, trying 'latin-1' as a fallback.")
                decoded_data = data.decode("latin-1")

            # Use a sample of the data to detect the delimiter
            sample_for_sniffing = decoded_data[:4096]  # Use first 4KB for sniffing
            if not sample_for_sniffing.strip():
                logger.warning("CSV file appears to be empty. No data to process.")
                return
                
            delimiter = self._get_delimiter(sample_for_sniffing)
            logger.info(f"Detected CSV delimiter: '{delimiter}'")

            csv_file = StringIO(decoded_data)
            csv_reader = csv.reader(csv_file, delimiter=delimiter)

            header = next(csv_reader, None)
            if header:
                logger.info(f"CSV header found: {header}")
                yield ", ".join(header)

            for row in csv_reader:
                #Filter out completely empty rows
                if any(field.strip() for field in row):
                    yield ", ".join(str(field) for field in row)

        except Exception as e:
            logger.error(f"Failed to read or process CSV data: {e}", exc_info=True)
            raise ValueError(f"Error processing CSV file: {str(e)}") from e