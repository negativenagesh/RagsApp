"""Compatibility shim.

The canonical ingestion implementation now lives in app/index8jan.py.
This module keeps legacy imports working while delegating to the canonical entrypoint.
"""

from .index8jan import process_and_ingest_file

__all__ = ["process_and_ingest_file"]
