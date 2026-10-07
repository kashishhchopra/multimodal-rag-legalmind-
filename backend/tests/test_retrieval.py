"""Tests for hybrid multimodal retrieval and citation assembly."""
from __future__ import annotations

from app.retrieval.retriever import MultimodalRetriever
from app.schemas import SourceKind


def test_retrieval_returns_text_and_visual(ingested_document: str):
    r = MultimodalRetriever().retrieve(
        ingested_document, "Explain the switching circuit and what R1 does"
    )
    assert len(r.text_evidence) >= 1
    assert len(r.visual_evidence) >= 1
    # scores are in [0, 1]
    assert all(0.0 <= t.score <= 1.0 for t in r.text_evidence)
    assert all(0.0 <= v.score <= 1.0 for v in r.visual_evidence)


def test_retrieval_text_finds_r1_description(ingested_document: str):
    r = MultimodalRetriever().retrieve(ingested_document, "current limiting resistor R1")
    joined = " ".join(t.text.lower() for t in r.text_evidence)
    assert "r1" in joined


def test_citations_are_grounded_not_invented(ingested_document: str):
    """Every citation must correspond to a real retrieved piece of evidence."""
    r = MultimodalRetriever().retrieve(ingested_document, "Explain Figure 12 and Q1")
    text_ids = {t.chunk_id for t in r.text_evidence}
    visual_ids = {v.visual_id for v in r.visual_evidence}
    for c in r.citations:
        if c.kind == SourceKind.TEXT:
            assert c.source_id in text_ids
        elif c.kind == SourceKind.VISUAL:
            assert c.source_id in visual_ids
        elif c.kind == SourceKind.REGION:
            assert c.source_id.split(":")[0] in visual_ids


def test_combined_confidence_within_bounds(ingested_document: str):
    r = MultimodalRetriever().retrieve(ingested_document, "transistor switching")
    assert 0.0 <= r.combined_confidence <= 1.0


def test_visual_evidence_has_image_url(ingested_document: str):
    r = MultimodalRetriever().retrieve(ingested_document, "circuit diagram")
    for v in r.visual_evidence:
        assert v.image_url.startswith("/static/")
