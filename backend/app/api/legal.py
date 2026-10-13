"""Legal-intelligence endpoints: summary, timeline, redaction, memo export, comparison, feedback."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .. import auth
from ..config import get_settings
from ..legal import compare as compare_mod
from ..legal import corpus_catalog
from ..legal import doctype as doctype_mod
from ..legal import feedback as feedback_mod
from ..legal import memo as memo_mod
from ..legal import telemetry
from ..legal.redact import redact
from ..legal.summary import summarise
from ..legal.timeline import extract_timeline
from ..retrieval import lexical, vector_store
from ..schemas import ProcessingStage
from ..utils import registry
from ..utils.errors import DocumentNotFoundError, DocumentNotReadyError

router = APIRouter(prefix="/api", tags=["legal"])


def _chunks(document_id: str) -> List[dict]:
    doc = registry.get(document_id)
    if not doc:
        raise DocumentNotFoundError(f"No document with id '{document_id}'.")
    if doc.stage != ProcessingStage.READY:
        raise DocumentNotReadyError(f"Document is not ready yet (current stage: {doc.stage.value}).")
    corpus = vector_store.text_corpus(document_id)
    rows = [{"id": i, "text": t, "page_number": m.get("page_number", 0), "section": m.get("section")}
            for i, t, m in zip(corpus["ids"], corpus["documents"], corpus["metadatas"])]
    rows.sort(key=lambda r: (r["page_number"], r["id"]))
    return rows


@router.get("/documents/{document_id}/summary")
async def document_summary(document_id: str) -> dict:
    return {"document_id": document_id, **summarise(_chunks(document_id))}


@router.get("/documents/{document_id}/type")
async def document_type(document_id: str) -> dict:
    chunks = _chunks(document_id)
    head = " ".join(c["text"] for c in chunks[:2])
    return {"document_id": document_id, "prediction": doctype_mod.classify(head)}


@router.get("/documents/{document_id}/timeline")
async def document_timeline(document_id: str) -> dict:
    ev = extract_timeline(_chunks(document_id))
    return {"document_id": document_id, "events": [e.__dict__ for e in ev]}


class RedactRequest(BaseModel):
    text: str = Field(..., max_length=100_000)


@router.post("/redact")
async def redact_text(req: RedactRequest) -> dict:
    text, counts = redact(req.text)
    return {"text": text, "redactions": counts, "note": "Identifiers only; personal names and victim identity are not detected."}


class MemoRequest(BaseModel):
    title: str = "LegalMind research memo"
    question: str
    answer: str
    citations: List[dict] = Field(default_factory=list)
    verification: Optional[dict] = None
    format: str = Field(default="pdf", pattern="^(pdf|docx|md)$")
    redact_pii: bool = True


@router.post("/memo")
async def export_memo(req: MemoRequest) -> Response:
    q, a = req.question, req.answer
    cites = req.citations
    if req.redact_pii:
        q, a = redact(q)[0], redact(a)[0]
        cites = [{**c, "snippet": redact(c.get("snippet", ""))[0]} for c in cites]
    build, mime = memo_mod.FORMATS[req.format]
    body = build(req.title, q, a, cites, req.verification)
    return Response(content=body, media_type=mime, headers={"Content-Disposition": f'attachment; filename="memo.{req.format}"'})


class CompareRequest(BaseModel):
    document_a: str
    document_b: str
    min_margin: float = 0.0


@router.post("/compare")
async def compare_documents(req: CompareRequest) -> dict:
    return compare_mod.compare(_chunks(req.document_a), _chunks(req.document_b), req.min_margin)


class FeedbackRequest(BaseModel):
    document_id: str = ""
    question: str
    answer: str
    rating: int = Field(..., ge=-1, le=1)
    reason: str = ""
    status: str = ""   # the answer's warning/verification status, so thumbs-down can be sliced by failure type


@router.post("/feedback")
async def add_feedback(req: FeedbackRequest) -> dict:
    fid = feedback_mod.add(get_settings().data_dir, req.document_id, req.question, req.answer, req.rating, req.reason, req.status)
    return {"id": fid}


@router.get("/feedback/stats")
async def feedback_stats() -> dict:
    return feedback_mod.stats(get_settings().data_dir)


@router.get("/admin/stats")
async def admin_stats(_: dict | None = Depends(auth.require_admin)) -> dict:
    """Corpus, ingestion-state, feedback and latency overview for the admin dashboard."""
    docs = registry.list_all()
    stages: dict = {}
    for d in docs:
        stages[d.stage.value] = stages.get(d.stage.value, 0) + 1
    try:
        chunks = vector_store.text_count()
    except Exception:  # noqa: BLE001
        chunks = None
    data_dir = get_settings().data_dir
    return {
        "documents": len(docs), "documents_by_stage": stages, "indexed_text_chunks": chunks,
        "feedback": feedback_mod.stats(data_dir), "queries": telemetry.summary(data_dir),
        "auth_enabled": auth.enabled(), "corpus": lexical.breakdown(),
    }


@router.get("/corpus/documents")
async def corpus_documents(q: str = "", year: Optional[int] = None, kind: Optional[str] = None, offset: int = 0, limit: int = 25) -> dict:
    """Browse the indexed judgments and statutes (title/citation search, year and type filters, paging)."""
    return corpus_catalog.search(q, year, kind if kind in ("judgment", "statute") else None, max(0, offset), min(max(1, limit), 100))
