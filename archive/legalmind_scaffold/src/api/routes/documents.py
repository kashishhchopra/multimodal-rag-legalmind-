"""Document upload & ingestion-status endpoints."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from src.config import DATA_DIR
from src.api.tasks import DOCUMENT_STATUS, ingest_document_task

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile,
    doc_type: str | None = None,
    jurisdiction: str | None = None,
) -> dict[str, str]:
    doc_id = str(uuid.uuid4())
    dest_dir = DATA_DIR / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{doc_id}_{file.filename}"

    contents = await file.read()
    dest_path.write_bytes(contents)

    DOCUMENT_STATUS[doc_id] = "queued"
    ingest_document_task.delay(str(dest_path), doc_id, doc_type, jurisdiction)

    return {"doc_id": doc_id, "status": "queued"}


@router.get("/{doc_id}/status")
def get_status(doc_id: str) -> dict[str, str]:
    status = DOCUMENT_STATUS.get(doc_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown doc_id")
    return {"doc_id": doc_id, "status": status}
