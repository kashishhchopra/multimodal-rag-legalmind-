"""Layer-2 write path: embed chunks (dense + sparse) and upsert into Qdrant
with full metadata payload.
"""
from __future__ import annotations

from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi

from src.config import get_settings
from src.schemas import Chunk

from .embeddings import embed_texts
from .qdrant_client import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, ensure_collection, get_qdrant_client


def _bm25_sparse_vector(tokens: list[str], corpus_tokens: list[list[str]]) -> qmodels.SparseVector:
    """Build a sparse vector of BM25-style term weights for one document,
    using the batch's own token vocabulary. In production this vocabulary
    should be the corpus-wide vocabulary, refreshed periodically, not just
    the current upsert batch."""
    bm25 = BM25Okapi(corpus_tokens)
    vocab = sorted(set(t for doc in corpus_tokens for t in doc))
    vocab_index = {term: i for i, term in enumerate(vocab)}

    scores = bm25.get_scores(tokens)  # not directly per-term, so approximate below
    term_counts: dict[str, int] = {}
    for t in tokens:
        term_counts[t] = term_counts.get(t, 0) + 1

    indices = [vocab_index[t] for t in term_counts if t in vocab_index]
    values = [float(c) for t, c in term_counts.items() if t in vocab_index]
    return qmodels.SparseVector(indices=indices, values=values)


def index_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        return

    settings = get_settings()
    texts = [c.text for c in chunks]
    dense_vectors = embed_texts(texts)
    ensure_collection(dense_dim=len(dense_vectors[0]))

    corpus_tokens = [t.lower().split() for t in texts]

    points = []
    for chunk, dense_vec, tokens in zip(chunks, dense_vectors, corpus_tokens):
        points.append(qmodels.PointStruct(
            id=chunk.chunk_id,
            vector={
                DENSE_VECTOR_NAME: dense_vec,
                SPARSE_VECTOR_NAME: _bm25_sparse_vector(tokens, corpus_tokens),
            },
            payload={
                "text": chunk.text,
                "metadata": chunk.metadata.model_dump(),
            },
        ))

    client = get_qdrant_client()
    client.upsert(collection_name=settings.qdrant_collection, points=points)
