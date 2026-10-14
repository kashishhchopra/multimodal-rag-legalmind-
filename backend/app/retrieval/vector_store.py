"""ChromaDB vector store — three persistent collections.

    text_chunks      : BGE (384-d) embeddings of text/table chunks
    visual_elements  : CLIP (512-d) embeddings of figures/diagrams/images
    visual_regions   : BGE (384-d) embeddings of OCR line-regions inside visuals
                       (used for box-level visual grounding)

Chroma stores the metadata that lets us reconstruct citations and locate images
on disk. All collections use cosine space; since our embeddings are normalised,
we report similarity = 1 - cosine_distance.

Metadata values must be scalar (str/int/float/bool), so bounding boxes are
flattened to bbox_x0..bbox_y1 fields.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

import numpy as np

from ..config import get_settings
from ..utils.errors import VectorStoreError
from . import lexical

logger = logging.getLogger("mmrag.vectorstore")

# Silence a noisy (and harmless) telemetry bug in chromadb 0.5.x / posthog.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

TEXT_COLLECTION = "text_chunks"
VISUAL_COLLECTION = "visual_elements"
REGION_COLLECTION = "visual_regions"

_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                path = str(get_settings().vector_dir)
                logger.info("Opening ChromaDB at %s", path)
                _client = chromadb.PersistentClient(
                    path=path,
                    settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
                )
    return _client


def _collection(name: str):
    try:
        return _get_client().get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
    except Exception as exc:
        raise VectorStoreError(f"Could not open collection '{name}': {exc}") from exc


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------
def add_text_chunks(
    ids: List[str],
    embeddings: np.ndarray,
    documents: List[str],
    metadatas: List[Dict],
) -> None:
    if not ids:
        return
    try:
        _collection(TEXT_COLLECTION).add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            documents=documents,
            metadatas=metadatas,
        )
    except Exception as exc:
        raise VectorStoreError(f"Failed to index text chunks: {exc}") from exc
    try:  # keep the on-disk lexical index in step (see lexical.py)
        lexical.add(ids, documents, metadatas)
    except Exception as exc:  # pragma: no cover - lexical is an accelerator; the in-memory path still works
        logger.warning("lexical index write failed: %s", exc)


def add_visuals(
    ids: List[str],
    embeddings: np.ndarray,
    documents: List[str],
    metadatas: List[Dict],
) -> None:
    if not ids:
        return
    try:
        _collection(VISUAL_COLLECTION).add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            documents=documents,
            metadatas=metadatas,
        )
    except Exception as exc:
        raise VectorStoreError(f"Failed to index visuals: {exc}") from exc


def add_regions(
    ids: List[str],
    embeddings: np.ndarray,
    documents: List[str],
    metadatas: List[Dict],
) -> None:
    if not ids:
        return
    try:
        _collection(REGION_COLLECTION).add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            documents=documents,
            metadatas=metadatas,
        )
    except Exception as exc:
        raise VectorStoreError(f"Failed to index regions: {exc}") from exc


# --------------------------------------------------------------------------
# Querying
# --------------------------------------------------------------------------
def build_where(document_id: Optional[str] = None, filters: Optional[Dict] = None) -> Optional[Dict]:
    """Chroma `where` clause from an optional document scope plus exact-match metadata filters
    (e.g. {"court": "Supreme Court", "year": 2023, "doc_type": "judgment"}). None = whole corpus."""
    conds = [{"document_id": document_id}] if document_id else []
    conds += [{k: v} for k, v in (filters or {}).items() if v is not None]
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _hits(ids, distances_or_sims, metadatas, documents, is_distance: bool) -> List[Dict]:
    out: List[Dict] = []
    for i, _id in enumerate(ids):
        sim = 1.0 - float(distances_or_sims[i]) if is_distance else float(distances_or_sims[i])
        out.append({"id": _id, "score": max(0.0, min(1.0, sim)), "metadata": metadatas[i] or {}, "document": documents[i] or ""})
    return out


def _exact_query(collection_name: str, embedding: np.ndarray, k: int, where: Optional[Dict]) -> List[Dict]:
    """Exact cosine search among the vectors matching `where` (used for a single document: a few hundred vectors at most).
    The approximate HNSW search with a very selective filter on a large index can fail ('ef or M is too small') or miss
    results, and an exact scan of one document is cheap and always complete."""
    got = _collection(collection_name).get(where=where, include=["embeddings", "documents", "metadatas"])
    embs = got["embeddings"]
    if got["ids"] is None or len(got["ids"]) == 0 or embs is None:
        return []
    sims = np.asarray(embs, dtype=np.float32) @ np.asarray(embedding, dtype=np.float32)
    top = np.argsort(-sims)[: max(1, k)]
    return _hits([got["ids"][i] for i in top], [sims[i] for i in top], [got["metadatas"][i] for i in top], [got["documents"][i] for i in top], False)


def _query(
    collection_name: str,
    embedding: np.ndarray,
    k: int,
    document_id: Optional[str],
    filters: Optional[Dict] = None,
) -> List[Dict]:
    where = build_where(document_id, filters)
    try:
        if document_id:
            return _exact_query(collection_name, embedding, k, where)
        n = max(1, k)
        while True:
            try:
                kwargs = {"where": where} if where else {}
                res = _collection(collection_name).query(query_embeddings=[embedding.tolist()], n_results=n, **kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                if n > 1 and ("contigious" in str(exc) or "ef or M" in str(exc)):
                    n = max(1, n // 2)  # a selective filter on a big index: ask for fewer results rather than fail
                    continue
                raise
    except Exception as exc:
        raise VectorStoreError(f"Query on '{collection_name}' failed: {exc}") from exc
    if not res or not res.get("ids") or not res["ids"][0]:
        return []
    return _hits(res["ids"][0], res["distances"][0], res["metadatas"][0], res["documents"][0], True)


def query_text(
    embedding: np.ndarray, k: int, document_id: Optional[str], filters: Optional[Dict] = None
) -> List[Dict]:
    return _query(TEXT_COLLECTION, embedding, k, document_id, filters)


def text_corpus(document_id: Optional[str] = None, filters: Optional[Dict] = None) -> Dict[str, List]:
    """All text chunks in scope (ids, documents, metadatas) for building the BM25 index."""
    try:
        kwargs = {"where": w} if (w := build_where(document_id, filters)) else {}
        res = _collection(TEXT_COLLECTION).get(include=["documents", "metadatas"], **kwargs)
    except Exception as exc:
        raise VectorStoreError(f"Could not read text corpus: {exc}") from exc
    return {"ids": res["ids"], "documents": res["documents"], "metadatas": res["metadatas"]}


def text_corpus_by_ids(ids: List[str]) -> Dict[str, Dict]:
    try:
        res = _collection(TEXT_COLLECTION).get(ids=ids, include=["documents", "metadatas"])
    except Exception as exc:
        raise VectorStoreError(f"Could not fetch chunks: {exc}") from exc
    return {i: {"id": i, "document": d or "", "metadata": m or {}} for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])}


def text_embeddings(ids: List[str]) -> Dict[str, np.ndarray]:
    """Stored text embeddings for specific chunk ids (to score BM25-only hits densely)."""
    if not ids:
        return {}
    try:
        res = _collection(TEXT_COLLECTION).get(ids=ids, include=["embeddings"])
    except Exception as exc:
        raise VectorStoreError(f"Could not fetch embeddings: {exc}") from exc
    embs = res["embeddings"]
    return {i: np.asarray(e, dtype=np.float32) for i, e in zip(res["ids"], embs if embs is not None else [])}


def text_count() -> int:
    return _collection(TEXT_COLLECTION).count()


def query_visuals(embedding: np.ndarray, k: int, document_id: str) -> List[Dict]:
    return _query(VISUAL_COLLECTION, embedding, k, document_id)


def query_regions(embedding: np.ndarray, k: int, document_id: str) -> List[Dict]:
    return _query(REGION_COLLECTION, embedding, k, document_id)


# --------------------------------------------------------------------------
# Fetch-by-id (for the /sources and /visuals endpoints)
# --------------------------------------------------------------------------
def get_by_id(collection_name: str, item_id: str) -> Optional[Dict]:
    try:
        res = _collection(collection_name).get(ids=[item_id])
    except Exception as exc:
        raise VectorStoreError(f"Fetch from '{collection_name}' failed: {exc}") from exc
    if not res or not res.get("ids"):
        return None
    return {
        "id": res["ids"][0],
        "metadata": (res["metadatas"] or [{}])[0] or {},
        "document": (res["documents"] or [""])[0] or "",
    }


def delete_document(document_id: str) -> None:
    """Remove all vectors belonging to a document from every collection."""
    for name in (TEXT_COLLECTION, VISUAL_COLLECTION, REGION_COLLECTION):
        try:
            _collection(name).delete(where={"document_id": document_id})
        except Exception as exc:  # pragma: no cover
            logger.warning("Delete from %s failed: %s", name, exc)
    try:
        lexical.delete_document(document_id)
    except Exception as exc:  # pragma: no cover
        logger.warning("lexical delete failed: %s", exc)
