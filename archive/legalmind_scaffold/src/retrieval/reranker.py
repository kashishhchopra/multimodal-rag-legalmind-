"""Cross-encoder re-ranking of the fused hybrid candidates down to the
top-k that actually goes into the LLM prompt.

A cross-encoder scores (query, chunk) jointly rather than independently
embedding each, which is why it's more precise than the dense/BM25 fusion
alone — worth the extra latency only on the shortlist, not the whole corpus.
"""
from __future__ import annotations

from functools import lru_cache

from src.config import get_settings
from .hybrid_retriever import RetrievedChunk


@lru_cache
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    settings = get_settings()
    if not candidates:
        return []

    model = _cross_encoder(settings.reranker_model)
    pairs = [(query, c.chunk.text) for c in candidates]
    scores = model.predict(pairs)

    reranked = [
        RetrievedChunk(chunk=c.chunk, score=float(score))
        for c, score in sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    ]
    return reranked[: settings.top_k_rerank]
