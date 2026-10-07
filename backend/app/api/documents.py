"""Document + query API routes (spec §14)."""
from __future__ import annotations

import datetime as dt
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response

from ..config import get_settings
from ..retrieval import vector_store
from ..schemas import (
    Document,
    ProcessingStage,
    QueryRequest,
    ClauseOut,
    ClauseReport,
    QueryResponse,
    SearchRequest,
    TextChunk,
    VisualElement,
)
from ..services import query_service
from ..services.grounding import crop_region, highlight_region
from ..services.ingest_service import process_document
from ..schemas import BoundingBox
from ..utils import registry
from ..utils.errors import (
    DocumentNotFoundError,
    DocumentNotReadyError,
    EmptyDocumentError,
    InvalidFileError,
    DocumentTooLargeError,
    SourceNotFoundError,
    UnsupportedFileTypeError,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])
search_router = APIRouter(prefix="/api", tags=["search"])

# Serialize ingestion so the heavy embedding models aren't loaded concurrently.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------
@router.post("/upload", response_model=Document)
async def upload_document(file: UploadFile = File(...)) -> Document:
    settings = get_settings()

    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise UnsupportedFileTypeError("Only PDF files are supported in this version.")

    data = await file.read()
    if not data:
        raise EmptyDocumentError("The uploaded file is empty.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise DocumentTooLargeError(
            f"File is {size_mb:.1f} MB; the limit is {settings.max_upload_mb} MB."
        )
    if not data[:5].startswith(b"%PDF"):
        raise InvalidFileError("This does not look like a valid PDF file.")

    document_id = uuid.uuid4().hex[:16]
    out_dir = settings.data_dir / document_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "source.pdf"
    pdf_path.write_bytes(data)

    doc = Document(
        document_id=document_id,
        filename=filename,
        upload_time=dt.datetime.utcnow().isoformat() + "Z",
        stage=ProcessingStage.QUEUED,
        progress=0.0,
        stage_message="Queued",
    )
    registry.save(doc)

    # Kick off background processing (real stages update the registry).
    _executor.submit(process_document, document_id, str(pdf_path), filename)
    return doc


# --------------------------------------------------------------------------
# List / get / status / delete
# --------------------------------------------------------------------------
@router.get("", response_model=List[Document])
async def list_documents() -> List[Document]:
    return registry.list_all()


@router.get("/{document_id}", response_model=Document)
async def get_document(document_id: str) -> Document:
    doc = registry.get(document_id)
    if not doc:
        raise DocumentNotFoundError(f"No document with id '{document_id}'.")
    return doc


@router.get("/{document_id}/status", response_model=Document)
async def get_status(document_id: str) -> Document:
    doc = registry.get(document_id)
    if not doc:
        raise DocumentNotFoundError(f"No document with id '{document_id}'.")
    return doc


@router.delete("/{document_id}")
async def delete_document(document_id: str) -> dict:
    doc = registry.get(document_id)
    if not doc:
        raise DocumentNotFoundError(f"No document with id '{document_id}'.")
    vector_store.delete_document(document_id)
    registry.delete(document_id)
    # best-effort cleanup of files
    out_dir = get_settings().data_dir / document_id
    try:
        for p in out_dir.glob("*"):
            p.unlink()
        out_dir.rmdir()
    except OSError:
        pass
    return {"deleted": document_id}


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------
@router.post("/{document_id}/query", response_model=QueryResponse)
async def query_document(document_id: str, req: QueryRequest) -> QueryResponse:
    doc = registry.get(document_id)
    if not doc:
        raise DocumentNotFoundError(f"No document with id '{document_id}'.")
    if doc.stage != ProcessingStage.READY:
        raise DocumentNotReadyError(
            f"Document is not ready yet (current stage: {doc.stage.value})."
        )
    return query_service.answer_question(
        document_id, req.question, req.top_k_text, req.top_k_visual
    )


@search_router.post("/search", response_model=QueryResponse)
async def search_corpus(req: SearchRequest) -> QueryResponse:
    """Cross-document question answering over every READY document, with metadata filters."""
    return query_service.answer_question(
        None, req.question, req.top_k, None, filters=req.filters or None
    )


@router.get("/{document_id}/clauses", response_model=ClauseReport)
async def document_clauses(document_id: str, min_margin: float = 0.0) -> ClauseReport:
    """Clause-type extraction (CUAD classifier) with rubric-based risk flags."""
    from ..legal import clauses as clause_mod

    doc = registry.get(document_id)
    if not doc:
        raise DocumentNotFoundError(f"No document with id '{document_id}'.")
    if doc.stage != ProcessingStage.READY:
        raise DocumentNotReadyError(f"Document is not ready yet (current stage: {doc.stage.value}).")
    corpus = vector_store.text_corpus(document_id)
    chunks = [
        {"id": i, "text": t, "page_number": m.get("page_number", 0)}
        for i, t, m in zip(corpus["ids"], corpus["documents"], corpus["metadatas"])
    ]
    found = clause_mod.extract_clauses(chunks, min_margin=min_margin)
    out = [
        ClauseOut(
            category=c.category, text=c.text, page_number=c.page_number, chunk_id=c.chunk_id,
            confidence_margin=c.margin, risk_level=c.risk.level, risk_reason=c.risk.reason,
            suggested_edit=c.risk.suggestion, red_flags=c.risk.flags,
        )
        for c in found
    ]
    summary = {lvl: sum(1 for c in out if c.risk_level == lvl) for lvl in ("high", "medium", "low")}
    return ClauseReport(
        document_id=document_id, clauses=out, summary=summary,
        warning=None if clause_mod.model_available() else "clause_model_missing: run scripts/train_clause_classifier.py",
    )


@search_router.get("/legal/sections")
async def map_sections(q: str) -> dict:
    """Old<->new criminal-law section lookup, e.g. ?q=Section 420 IPC"""
    from ..legal.section_mapper import expand_query

    expanded, notes = expand_query(q)
    return {"query": q, "expanded_query": expanded, "mappings": notes}


# --------------------------------------------------------------------------
# Sources (text chunks) and visuals (spec §14)
# --------------------------------------------------------------------------
@router.get("/{document_id}/sources/{source_id}", response_model=TextChunk)
async def get_source(document_id: str, source_id: str) -> TextChunk:
    item = vector_store.get_by_id(vector_store.TEXT_COLLECTION, source_id)
    if not item:
        raise SourceNotFoundError(f"No text source with id '{source_id}'.")
    meta = item["metadata"]
    text = item["document"]
    return TextChunk(
        chunk_id=source_id,
        document_id=document_id,
        page_number=int(meta.get("page_number", 0)),
        text=text,
        char_count=len(text),
    )


@router.get("/{document_id}/visuals/{visual_id}", response_model=VisualElement)
async def get_visual(document_id: str, visual_id: str) -> VisualElement:
    item = vector_store.get_by_id(vector_store.VISUAL_COLLECTION, visual_id)
    if not item:
        raise SourceNotFoundError(f"No visual with id '{visual_id}'.")
    meta = item["metadata"]
    from ..retrieval.retriever import _image_url  # local import to avoid cycle

    bbox = None
    if all(k in meta for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")):
        bbox = BoundingBox(
            x0=meta["bbox_x0"], y0=meta["bbox_y0"], x1=meta["bbox_x1"], y1=meta["bbox_y1"]
        )
    return VisualElement(
        visual_id=visual_id,
        document_id=document_id,
        page_number=int(meta.get("page_number", 0)),
        image_path=meta.get("image_path", ""),
        image_url=_image_url(document_id, meta.get("image_path", "")),
        image_type=meta.get("image_type", "image"),
        bounding_box=bbox,
        ocr_text=meta.get("ocr_text", ""),
        caption=meta.get("caption", ""),
        figure_label=meta.get("figure_label", ""),
    )


@router.get("/{document_id}/visuals/{visual_id}/grounding")
async def get_visual_grounding(
    document_id: str, visual_id: str, mode: str = "highlight"
) -> Response:
    """Return a PNG with the best OCR region highlighted (or cropped).

    The region is chosen at query time and stored in region metadata; here we
    just re-derive the top region for this visual for a standalone view.
    """
    item = vector_store.get_by_id(vector_store.VISUAL_COLLECTION, visual_id)
    if not item:
        raise SourceNotFoundError(f"No visual with id '{visual_id}'.")
    image_path = item["metadata"].get("image_path", "")
    if not image_path or not Path(image_path).exists():
        raise SourceNotFoundError("Visual image file is missing.")

    # Find any region belonging to this visual (first one) for a default highlight.
    regions = vector_store.get_by_id(vector_store.REGION_COLLECTION, f"{visual_id}_r0")
    bbox = None
    if regions:
        m = regions["metadata"]
        if all(k in m for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")):
            bbox = BoundingBox(x0=m["bbox_x0"], y0=m["bbox_y0"], x1=m["bbox_x1"], y1=m["bbox_y1"])

    if mode == "crop" and bbox is not None:
        png = crop_region(image_path, bbox)
    else:
        png = highlight_region(image_path, bbox)
    return Response(content=png, media_type="image/png")
