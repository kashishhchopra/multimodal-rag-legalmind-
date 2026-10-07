"""Layer-3 read path: dense + BM25 retrieval fused with reciprocal-rank
fusion, with optional metadata pre-filtering.

Legal queries hinge on exact terms ("Section 12(b)", a defined party name, a
statute number) that pure semantic search misses — this is why dense-only
retrieval is not enough for this domain (see docs/ARCHITECTURE.md 5.4).
"""
from __future__ import annotations

from dataclasses import dataclass

from qdrant_client.http import models as qmodels

from src.config import get_settings
from src.schemas import Chunk, ChunkMetadata

from src.indexing.embeddings import embed_texts
from src.indexing.qdrant_client import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, get_qdrant_client


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def _metadata_filter(doc_ids: list[str] | None, jurisdiction: str | None, doc_type: str | None) -> qmodels.Filter | None:
    conditions = []
    if doc_ids:
        conditions.append(qmodels.FieldCondition(key="metadata.doc_id", match=qmodels.MatchAny(any=doc_ids)))
    if jurisdiction:
        conditions.append(qmodels.FieldCondition(key="metadata.jurisdiction", match=qmodels.MatchValue(value=jurisdiction)))
    if doc_type:
        conditions.append(qmodels.FieldCondition(key="metadata.doc_type", match=qmodels.MatchValue(value=doc_type)))
    return qmodels.Filter(must=conditions) if conditions else None


def hybrid_search(
    query: str,
    *,
    doc_ids: list[str] | None = None,
    jurisdiction: str | None = None,
    doc_type: str | None = None,
) -> list[RetrievedChunk]:
    settings = get_settings()
    client = get_qdrant_client()
    query_filter = _metadata_filter(doc_ids, jurisdiction, doc_type)

    dense_vec = embed_texts([query])[0]
    sparse_query = _bm25_query_vector(query)

    dense_hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=dense_vec,
        using=DENSE_VECTOR_NAME,
        limit=settings.top_k_dense,
        query_filter=query_filter,
    ).points

    sparse_hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=sparse_query,
        using=SPARSE_VECTOR_NAME,
        limit=settings.top_k_sparse,
        query_filter=query_filter,
    ).points

    fused = _reciprocal_rank_fusion([dense_hits, sparse_hits])
    return fused


def _bm25_query_vector(query: str) -> qmodels.SparseVector:
    """Rough token-presence sparse vector for the query side. Term IDs must
    align with how index-time sparse vectors were built (see
    src/indexing/hybrid_index.py) — in production this should share a single
    corpus-wide vocabulary rather than being rebuilt ad hoc per query."""
    tokens = query.lower().split()
    # Placeholder: Qdrant can also do native sparse encoding server-side;
    # this stub keeps the interface stable while that's wired up.
    return qmodels.SparseVector(indices=list(range(len(tokens))), values=[1.0] * len(tokens))


def _reciprocal_rank_fusion(result_lists: list[list], k: int = 60) -> list[RetrievedChunk]:
    settings = get_settings()
    scores: dict[str, float] = {}
    points_by_id: dict[str, object] = {}

    for results in result_lists:
        for rank, point in enumerate(results):
            points_by_id[point.id] = point
            scores[point.id] = scores.get(point.id, 0.0) + 1.0 / (k + rank + 1)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[: settings.top_k_dense]

    retrieved: list[RetrievedChunk] = []
    for point_id in ranked_ids:
        point = points_by_id[point_id]
        payload = point.payload or {}
        metadata = ChunkMetadata(**payload.get("metadata", {}))
        chunk = Chunk(chunk_id=str(point_id), text=payload.get("text", ""), metadata=metadata)
        retrieved.append(RetrievedChunk(chunk=chunk, score=scores[point_id]))

    return retrieved
