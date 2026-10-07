"""Celery app + the async ingestion task.

Ingestion (parsing + OCR + chunking + embedding) is slow and shouldn't block
an HTTP request, so `/documents/upload` enqueues this task and returns
immediately with a `doc_id`; the client polls `/documents/{doc_id}/status`.
"""
from __future__ import annotations

import logging
from pathlib import Path

from celery import Celery

from src.chunking.metadata import build_chunk_metadata
from src.chunking.structure_splitter import split_into_clauses
from src.config import get_settings
from src.indexing.hybrid_index import index_chunks
from src.ingestion.router import route_document

logger = logging.getLogger(__name__)
settings = get_settings()

celery_app = Celery("legalmind", broker=settings.redis_url, backend=settings.redis_url)

# In-memory status store for the scaffold; swap for a Redis hash or a real
# document-metadata DB table once documents.py needs to survive a restart.
DOCUMENT_STATUS: dict[str, str] = {}


@celery_app.task(name="legalmind.ingest_document")
def ingest_document_task(file_path: str, doc_id: str, doc_type: str | None, jurisdiction: str | None) -> str:
    DOCUMENT_STATUS[doc_id] = "parsing"
    try:
        parsed_pages = route_document(file_path, doc_id=doc_id)

        DOCUMENT_STATUS[doc_id] = "chunking"
        chunks = []
        for page in parsed_pages:
            spans = split_into_clauses(page.text)
            for span in spans:
                chunks.append(build_chunk_metadata(
                    page, span, doc_type=doc_type, jurisdiction=jurisdiction,
                ))

        DOCUMENT_STATUS[doc_id] = "indexing"
        index_chunks(chunks)

        DOCUMENT_STATUS[doc_id] = "ready"
    except Exception:
        logger.exception("Ingestion failed for doc_id=%s", doc_id)
        DOCUMENT_STATUS[doc_id] = "failed"
        raise
    return doc_id
