"""Hybrid multimodal retrieval (spec sections 4 & 10).

For a single question we run TWO independent retrieval paths:

  1. TEXT path   : question -> BGE query embedding -> search `text_chunks`.
  2. VISUAL path : question -> CLIP *text* embedding -> search `visual_elements`
                   (cross-modal: a text query directly matches image vectors).

We also search `visual_regions` so that, for each retrieved visual, we can pick
the OCR line-region most relevant to the question — that box is what the UI
highlights (honest, box-level visual grounding — no fabricated detections).

Scores from the two paths are combined into a single heuristic confidence:

    combined = ALPHA * mean(top text sims) + BETA * mean(top visual sims)

ALPHA/BETA default to 0.6/0.4 (text is usually the more precise signal; visual
similarity from CLIP is coarser). Both are configurable in .env.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import get_settings
from ..legal.section_mapper import expand_query
from ..models import embeddings
from ..schemas import (
    BoundingBox,
    Citation,
    RetrievedText,
    RetrievedVisual,
    SourceKind,
    VisualType,
)
from . import vector_store
from . import lexical
from .hybrid import BM25Index, rrf_fuse


def _image_url(document_id: str, image_path: str) -> str:
    """Build the public /static URL for a stored image from its disk path."""
    basename = os.path.basename(image_path) if image_path else ""
    return f"/static/{document_id}/{basename}" if basename else ""


def _bbox_from_meta(meta: Dict, prefix: str = "bbox_") -> Optional[BoundingBox]:
    keys = [f"{prefix}x0", f"{prefix}y0", f"{prefix}x1", f"{prefix}y1"]
    if all(k in meta and meta[k] is not None for k in keys):
        return BoundingBox(
            x0=float(meta[keys[0]]),
            y0=float(meta[keys[1]]),
            x1=float(meta[keys[2]]),
            y1=float(meta[keys[3]]),
        )
    return None


@dataclass
class GroundingRegion:
    visual_id: str
    text: str
    bbox: Optional[BoundingBox]
    score: float


@dataclass
class RetrievalResult:
    text_evidence: List[RetrievedText] = field(default_factory=list)
    visual_evidence: List[RetrievedVisual] = field(default_factory=list)
    grounding: Dict[str, GroundingRegion] = field(default_factory=dict)  # visual_id -> region
    citations: List[Citation] = field(default_factory=list)
    combined_confidence: float = 0.0


_bm25_cache: Dict[tuple, tuple] = {}  # (scope key) -> (corpus size, BM25Index)


def _bm25_for(document_id: Optional[str], filters: Optional[Dict]) -> BM25Index:
    """BM25 index for the requested scope, rebuilt whenever the collection size changes."""
    key = (document_id, tuple(sorted((filters or {}).items())))
    size = vector_store.text_count()
    cached = _bm25_cache.get(key)
    if cached and cached[0] == size:
        return cached[1]
    corpus = vector_store.text_corpus(document_id, filters)
    index = BM25Index(corpus["ids"], corpus["documents"])
    _bm25_cache[key] = (size, index)
    return index


class MultimodalRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _text_hits(
        self, question: str, k: int, document_id: Optional[str], filters: Optional[Dict]
    ) -> List[Dict]:
        """Dense search, optionally fused with BM25 via RRF. Hit `score` stays a cosine similarity."""
        s = self.settings
        query = expand_query(question)[0] if s.expand_legal_sections else question
        qvec = embeddings.embed_text_query(query)
        if not s.hybrid_enabled:
            return vector_store.query_text(qvec, k, document_id, filters)

        pool = max(k, s.hybrid_candidates)
        dense = vector_store.query_text(qvec, pool, document_id, filters)
        if lexical.count() > 0 and lexical.can_filter(filters):
            lex_hits = lexical.search(query, pool, document_id, filters)  # on-disk FTS5: scales to the full corpus
        else:
            lex_hits = _bm25_for(document_id, filters).search(query, pool)  # small/legacy stores: in memory
        fused = rrf_fuse([[h["id"] for h in dense], [i for i, _ in lex_hits]], s.rrf_k)[:k]

        by_id = {h["id"]: h for h in dense}
        dense_rank = {h["id"]: r for r, h in enumerate(dense, 1)}
        lex_rank = {i: r for r, (i, _) in enumerate(lex_hits, 1)}
        missing = [i for i, _ in fused if i not in by_id]
        if missing:  # BM25-only hits: fetch text/metadata and score them densely so scores stay comparable
            got = vector_store.text_corpus_by_ids(missing)
            embs = vector_store.text_embeddings(missing)
            for _id in missing:
                if _id in got:
                    sim = float(np.dot(qvec, embs[_id])) if _id in embs else 0.0
                    by_id[_id] = {**got[_id], "score": max(0.0, min(1.0, sim))}
        hits = []
        for _id, _ in fused:
            if _id in by_id:
                hits.append({**by_id[_id], "dense_rank": dense_rank.get(_id), "lexical_rank": lex_rank.get(_id)})
        return hits

    def retrieve(
        self,
        document_id: Optional[str],
        question: str,
        top_k_text: Optional[int] = None,
        top_k_visual: Optional[int] = None,
        filters: Optional[Dict] = None,
    ) -> RetrievalResult:
        """document_id=None searches the whole corpus (cross-document), narrowed by `filters`."""
        k_text = top_k_text or self.settings.top_k_text
        k_visual = top_k_visual or self.settings.top_k_visual

        # ---- TEXT path ----
        text_hits = self._text_hits(question, k_text, document_id, filters)
        text_evidence = [
            RetrievedText(
                chunk_id=h["id"],
                page_number=int(h["metadata"].get("page_number", 0)),
                text=h["document"],
                score=round(h["score"], 4),
                document_id=h["metadata"].get("document_id", ""),
                lexical_rank=h.get("lexical_rank"),
                dense_rank=h.get("dense_rank"),
            )
            for h in text_hits
        ]

        # ---- VISUAL path (cross-modal CLIP) ----
        visual_vec = embeddings.embed_query_for_visual(question)
        visual_hits = vector_store.query_visuals(visual_vec, k_visual, document_id)
        visual_evidence: List[RetrievedVisual] = []
        for h in visual_hits:
            meta = h["metadata"]
            visual_evidence.append(
                RetrievedVisual(
                    visual_id=h["id"],
                    page_number=int(meta.get("page_number", 0)),
                    image_url=_image_url(meta.get("document_id", document_id or ""), meta.get("image_path", "")),
                    image_type=VisualType(meta.get("image_type", "image")),
                    figure_label=meta.get("figure_label", ""),
                    ocr_preview=(meta.get("ocr_text", "") or "")[:160],
                    score=round(h["score"], 4),
                    bounding_box=_bbox_from_meta(meta),
                )
            )

        # ---- REGION path (for visual grounding) ----
        grounding: Dict[str, GroundingRegion] = {}
        if visual_evidence:
            region_vec = embeddings.embed_text_query(question)
            region_hits = vector_store.query_regions(region_vec, k_visual * 4, document_id)
            retrieved_visual_ids = {v.visual_id for v in visual_evidence}
            for h in region_hits:
                meta = h["metadata"]
                vid = meta.get("visual_id", "")
                if vid in retrieved_visual_ids and vid not in grounding:
                    grounding[vid] = GroundingRegion(
                        visual_id=vid,
                        text=h["document"],
                        bbox=_bbox_from_meta(meta),
                        score=round(h["score"], 4),
                    )

        # ---- Combined confidence ----
        alpha, beta = self.settings.score_alpha, self.settings.score_beta
        text_mean = sum(t.score for t in text_evidence) / len(text_evidence) if text_evidence else 0.0
        vis_mean = sum(v.score for v in visual_evidence) / len(visual_evidence) if visual_evidence else 0.0
        combined = alpha * text_mean + beta * vis_mean

        # ---- Citations (no invented sources: only what we actually retrieved) ----
        citations: List[Citation] = []
        for t in text_evidence:
            citations.append(
                Citation(
                    kind=SourceKind.TEXT,
                    source_id=t.chunk_id,
                    page_number=t.page_number,
                    label=f"Page {t.page_number}",
                    score=t.score,
                    snippet=t.text[:180],
                )
            )
        for v in visual_evidence:
            label = v.figure_label or f"{v.image_type.value.title()} (page {v.page_number})"
            citations.append(
                Citation(
                    kind=SourceKind.VISUAL,
                    source_id=v.visual_id,
                    page_number=v.page_number,
                    label=label,
                    score=v.score,
                    snippet=v.ocr_preview,
                    image_url=v.image_url,
                    figure_label=v.figure_label,
                )
            )
            g = grounding.get(v.visual_id)
            if g and g.text:
                citations.append(
                    Citation(
                        kind=SourceKind.REGION,
                        source_id=f"{v.visual_id}:region",
                        page_number=v.page_number,
                        label=f"Visual region — “{g.text[:40]}”",
                        score=g.score,
                        snippet=g.text,
                        image_url=v.image_url,
                        figure_label=v.figure_label,
                    )
                )

        return RetrievalResult(
            text_evidence=text_evidence,
            visual_evidence=visual_evidence,
            grounding=grounding,
            citations=citations,
            combined_confidence=round(combined, 4),
        )
