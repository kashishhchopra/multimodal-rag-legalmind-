"""Ingestion orchestration: parse -> embed -> index, with real progress stages.

Runs in a background thread (started by the upload endpoint). Every stage
transition is written to the document registry so the frontend progress bar
reflects *actual* backend work, never a fake timer (spec §8).
"""
from __future__ import annotations

import logging
import traceback
import uuid
from pathlib import Path
from typing import List

from PIL import Image

from ..config import get_settings
from ..models import embeddings
from ..retrieval import vector_store
from ..schemas import Document, ProcessingStage, VisualType
from ..utils import registry
from ..ingestion.pdf_processor import ParsedDocument, ParsedVisual, parse_pdf

logger = logging.getLogger("mmrag.ingest")


def _visual_metadata(document_id: str, visual: ParsedVisual) -> dict:
    meta = {
        "document_id": document_id,
        "page_number": visual.page_number,
        "image_path": visual.image_path,
        "image_type": visual.image_type.value,
        "figure_label": visual.figure_label or "",
        "caption": visual.caption or "",
        "ocr_text": (visual.ocr_text or "")[:800],
        "width": visual.width,
        "height": visual.height,
    }
    if visual.bbox:
        meta.update(
            bbox_x0=visual.bbox.x0, bbox_y0=visual.bbox.y0,
            bbox_x1=visual.bbox.x1, bbox_y1=visual.bbox.y1,
        )
    return meta


def _visual_caption_text(visual: ParsedVisual) -> str:
    """A short textual description of a visual (used as its Chroma 'document')."""
    parts = [visual.figure_label, visual.caption, visual.image_type.value, visual.ocr_text]
    return " ".join(p for p in parts if p).strip() or f"{visual.image_type.value} image"


def process_document(document_id: str, pdf_path: str, filename: str) -> None:
    """The full pipeline for one document. Never raises — records failure in registry."""
    settings = get_settings()
    try:
        registry.update_stage(document_id, ProcessingStage.READING, 0.05, "Reading document")

        out_dir = settings.data_dir / document_id

        def _progress(stage: str, frac: float, message: str) -> None:
            try:
                registry.update_stage(document_id, ProcessingStage(stage), frac, message)
            except ValueError:
                registry.update_stage(document_id, ProcessingStage.READING, frac, message)

        parsed: ParsedDocument = parse_pdf(
            pdf_path, out_dir, progress=_progress, max_pages=settings.max_pages
        )

        # ---------------- EMBED ----------------
        registry.update_stage(document_id, ProcessingStage.EMBEDDING, 0.72, "Creating embeddings")

        # Text chunks
        chunk_ids: List[str] = []
        chunk_docs: List[str] = []
        chunk_meta: List[dict] = []
        for ch in parsed.chunks:
            cid = f"{document_id}_c{uuid.uuid4().hex[:10]}"
            chunk_ids.append(cid)
            chunk_docs.append(ch.text)
            chunk_meta.append(
                {"document_id": document_id, "page_number": ch.page_number, "chunk_id": cid}
            )
        text_vecs = embeddings.embed_text_passages(chunk_docs) if chunk_docs else None

        # Visuals (CLIP image embeddings) + regions (BGE text embeddings)
        visual_ids: List[str] = []
        visual_docs: List[str] = []
        visual_meta: List[dict] = []
        visual_images: List[Image.Image] = []

        region_ids: List[str] = []
        region_docs: List[str] = []
        region_meta: List[dict] = []

        for v in parsed.visuals:
            vid = f"{document_id}_v{uuid.uuid4().hex[:10]}"
            try:
                img = Image.open(v.image_path).convert("RGB")
            except Exception as exc:
                logger.warning("Skipping unreadable visual %s: %s", v.image_path, exc)
                continue
            visual_ids.append(vid)
            visual_images.append(img)
            visual_docs.append(_visual_caption_text(v))
            visual_meta.append(_visual_metadata(document_id, v))

            for ri, region in enumerate(v.regions):
                rid = f"{vid}_r{ri}"
                region_ids.append(rid)
                region_docs.append(region.text)
                region_meta.append(
                    {
                        "document_id": document_id,
                        "visual_id": vid,
                        "page_number": v.page_number,
                        "bbox_x0": region.bbox.x0, "bbox_y0": region.bbox.y0,
                        "bbox_x1": region.bbox.x1, "bbox_y1": region.bbox.y1,
                    }
                )

        visual_vecs = embeddings.embed_images(visual_images) if visual_images else None
        region_vecs = embeddings.embed_text_passages(region_docs) if region_docs else None

        # ---------------- INDEX ----------------
        registry.update_stage(document_id, ProcessingStage.INDEXING, 0.9, "Indexing document")
        if chunk_ids and text_vecs is not None:
            vector_store.add_text_chunks(chunk_ids, text_vecs, chunk_docs, chunk_meta)
        if visual_ids and visual_vecs is not None:
            vector_store.add_visuals(visual_ids, visual_vecs, visual_docs, visual_meta)
        if region_ids and region_vecs is not None:
            vector_store.add_regions(region_ids, region_vecs, region_docs, region_meta)

        # ---------------- DONE ----------------
        doc = registry.get(document_id) or Document(
            document_id=document_id, filename=filename, upload_time="", page_count=0
        )
        doc.page_count = parsed.page_count
        doc.text_chunk_count = len(chunk_ids)
        doc.image_count = len(visual_ids)
        doc.figure_count = parsed.figure_count
        doc.region_count = len(region_ids)
        doc.stage = ProcessingStage.READY
        doc.progress = 1.0
        doc.stage_message = "Ready"
        doc.error = None
        registry.save(doc)
        logger.info(
            "Document %s ready: %d chunks, %d visuals, %d regions",
            document_id, len(chunk_ids), len(visual_ids), len(region_ids),
        )

    except Exception as exc:  # noqa: BLE001 — must never crash the worker thread
        logger.error("Ingestion failed for %s: %s\n%s", document_id, exc, traceback.format_exc())
        registry.update_stage(
            document_id, ProcessingStage.FAILED, 0.0, "Processing failed", error=str(exc)
        )
