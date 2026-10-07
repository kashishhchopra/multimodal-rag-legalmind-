"""A tiny persistent registry of uploaded documents + their processing status.

Kept deliberately simple (a single JSON file guarded by a lock) so the project
has no extra database dependency beyond the vector store. The registry holds
*document-level* metadata and live processing status; per-chunk and per-visual
metadata lives in ChromaDB (see retrieval/vector_store.py).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

from ..config import get_settings
from ..schemas import Document, ProcessingStage

_lock = threading.RLock()


def _registry_path() -> Path:
    return get_settings().data_dir / "registry.json"


def _load_raw() -> Dict[str, dict]:
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Corrupt/partial file — start clean rather than crash the server.
        return {}


def _save_raw(data: Dict[str, dict]) -> None:
    path = _registry_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic on POSIX


def save(doc: Document) -> None:
    with _lock:
        data = _load_raw()
        data[doc.document_id] = json.loads(doc.model_dump_json())
        _save_raw(data)


def get(document_id: str) -> Optional[Document]:
    with _lock:
        data = _load_raw()
        raw = data.get(document_id)
        return Document(**raw) if raw else None


def list_all() -> List[Document]:
    with _lock:
        data = _load_raw()
    docs = [Document(**raw) for raw in data.values()]
    # newest first
    docs.sort(key=lambda d: d.upload_time, reverse=True)
    return docs


def delete(document_id: str) -> None:
    with _lock:
        data = _load_raw()
        data.pop(document_id, None)
        _save_raw(data)


def update_stage(
    document_id: str,
    stage: ProcessingStage,
    progress: float,
    message: str = "",
    error: Optional[str] = None,
) -> None:
    """Update just the live-status fields — called repeatedly during ingestion."""
    with _lock:
        doc = get(document_id)
        if not doc:
            return
        doc.stage = stage
        doc.progress = round(progress, 3)
        doc.stage_message = message
        if error is not None:
            doc.error = error
        save(doc)
