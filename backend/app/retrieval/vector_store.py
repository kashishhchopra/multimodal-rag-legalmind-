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
def _query(collection_name: str, embedding: np.ndarray, k: int, document_id: str) -> List[Dict]:
    try:
        res = _collection(collection_name).query(
            query_embeddings=[embedding.tolist()],
            n_results=max(1, k),
            where={"document_id": document_id},
        )
    except Exception as exc:
        raise VectorStoreError(f"Query on '{collection_name}' failed: {exc}") from exc

    hits: List[Dict] = []
    if not res or not res.get("ids") or not res["ids"][0]:
        return hits
    ids = res["ids"][0]
    distances = res["distances"][0]
    metadatas = res["metadatas"][0]
    documents = res["documents"][0]
    for i, _id in enumerate(ids):
        # cosine distance -> similarity in [0,1] (clamped)
        sim = 1.0 - float(distances[i])
        sim = max(0.0, min(1.0, sim))
        hits.append(
            {
                "id": _id,
                "score": sim,
                "metadata": metadatas[i] or {},
                "document": documents[i] or "",
            }
        )
    return hits


def query_text(embedding: np.ndarray, k: int, document_id: str) -> List[Dict]:
    return _query(TEXT_COLLECTION, embedding, k, document_id)


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
