"""Thin wrapper around the Qdrant client: connection + collection schema.

The collection stores a dense vector for semantic search and a sparse vector
(BM25-style term weights) in the same point, so hybrid retrieval (see
src/retrieval/hybrid_retriever.py) can query both in one round trip. The
full payload carries `ChunkMetadata` so retrieval results can be filtered by
jurisdiction/doc type and so the citation object can be built directly from
the payload without a second lookup.
"""
from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import get_settings

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(dense_dim: int) -> None:
    settings = get_settings()
    client = get_qdrant_client()

    if client.collection_exists(settings.qdrant_collection):
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            DENSE_VECTOR_NAME: qmodels.VectorParams(size=dense_dim, distance=qmodels.Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(),
        },
    )

    # Payload indexes for fast metadata pre-filtering (jurisdiction, doc
    # type, doc_id) ahead of the dense/sparse search.
    for field_name in ("doc_id", "doc_type", "jurisdiction"):
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name=f"metadata.{field_name}",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
