"""Embedding models — the multimodal heart of retrieval.

Two Sentence-Transformers models are loaded lazily (only on first use, so the
API server starts fast and tests that don't need embeddings stay light):

  * TEXT model  (BGE, 384-d): text <-> text semantic search over chunks.
  * IMAGE model (CLIP, 512-d): a *shared* space where an image and a text
    query can be compared directly. This is what enables cross-modal
    "describe the circuit diagram" -> retrieve the right *image* retrieval.

Design notes
------------
* All vectors are L2-normalised, so an inner-product search == cosine similarity.
* BGE recommends prefixing *queries* (not passages) with an instruction; we do
  that in `embed_text_query` only.
* CLIP text and image encoders share one embedding space, so we use the same
  model for both `embed_image` and `embed_query_for_visual`.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from typing import List, Sequence

import numpy as np

from ..config import get_settings
from ..utils.errors import EmbeddingError

logger = logging.getLogger("mmrag.embeddings")

# BGE query instruction (improves query-passage matching).
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------------------
# OFFLINE / TEST fallback embeddings.
# Enabled with MMRAG_FAKE_EMBEDDINGS=1. These are deterministic, dependency-free
# vectors used ONLY for fast tests and for booting the app without downloading
# models. They are NOT semantically meaningful — real retrieval quality requires
# the actual BGE + CLIP models. Never enable this flag in a real deployment.
# ---------------------------------------------------------------------------
_FAKE_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _fake_enabled() -> bool:
    return os.getenv("MMRAG_FAKE_EMBEDDINGS", "0").lower() in ("1", "true", "yes")


def _hash_text_vec(text: str) -> np.ndarray:
    vec = np.zeros(_FAKE_DIM, dtype=np.float32)
    text = (text or "").lower()
    tokens = _TOKEN_RE.findall(text)
    # word tokens + char trigrams give a rough lexical signature
    grams = list(tokens)
    for tok in tokens:
        grams.extend(tok[i:i + 3] for i in range(max(0, len(tok) - 2)))
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)
        vec[h % _FAKE_DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _fake_image_vec(image) -> np.ndarray:
    from PIL import Image

    img = image.convert("L").resize((16, 16))
    arr = np.asarray(img, dtype=np.float32).flatten()
    norm = np.linalg.norm(arr)
    return (arr / norm if norm > 0 else arr).astype(np.float32)


_text_model = None
_image_model = None
_text_lock = threading.Lock()
_image_lock = threading.Lock()


def _get_text_model():
    global _text_model
    if _text_model is None:
        with _text_lock:
            if _text_model is None:
                from sentence_transformers import SentenceTransformer

                name = get_settings().text_embedding_model
                logger.info("Loading text embedding model: %s", name)
                _text_model = SentenceTransformer(name)
    return _text_model


def _get_image_model():
    global _image_model
    if _image_model is None:
        with _image_lock:
            if _image_model is None:
                from sentence_transformers import SentenceTransformer

                name = get_settings().image_embedding_model
                logger.info("Loading image embedding model: %s", name)
                _image_model = SentenceTransformer(name)
    return _image_model


def text_dim() -> int:
    if _fake_enabled():
        return _FAKE_DIM
    return int(_get_text_model().get_sentence_embedding_dimension())


def image_dim() -> int:
    if _fake_enabled():
        return _FAKE_DIM
    return int(_get_image_model().get_sentence_embedding_dimension())


# --------------------------------------------------------------------------
# Text (BGE) — for the text collection
# --------------------------------------------------------------------------
def embed_text_passages(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, text_dim()), dtype=np.float32)
    if _fake_enabled():
        return np.vstack([_hash_text_vec(t) for t in texts]).astype(np.float32)
    try:
        model = _get_text_model()
        vecs = model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)
    except Exception as exc:
        raise EmbeddingError(f"Text embedding failed: {exc}") from exc


def embed_text_query(query: str) -> np.ndarray:
    if _fake_enabled():
        return _hash_text_vec(query)
    try:
        model = _get_text_model()
        vec = model.encode(
            [_BGE_QUERY_PREFIX + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vec.astype(np.float32)
    except Exception as exc:
        raise EmbeddingError(f"Text query embedding failed: {exc}") from exc


# --------------------------------------------------------------------------
# Visual (CLIP) — for the visual collection (shared image/text space)
# --------------------------------------------------------------------------
def embed_images(images: Sequence["object"]) -> np.ndarray:
    """Embed a list of PIL images into the shared CLIP space."""
    if not images:
        return np.zeros((0, image_dim()), dtype=np.float32)
    if _fake_enabled():
        return np.vstack([_fake_image_vec(im) for im in images]).astype(np.float32)
    try:
        model = _get_image_model()
        vecs = model.encode(
            list(images),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)
    except Exception as exc:
        raise EmbeddingError(f"Image embedding failed: {exc}") from exc


def embed_query_for_visual(query: str) -> np.ndarray:
    """Embed a *text* query into the SAME CLIP space as the images."""
    if _fake_enabled():
        # No cross-modal semantics offline; return a stable vector so the visual
        # retrieval path still executes end-to-end (ranking is not meaningful).
        return _hash_text_vec(query)
    try:
        model = _get_image_model()
        vec = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vec.astype(np.float32)
    except Exception as exc:
        raise EmbeddingError(f"Visual query embedding failed: {exc}") from exc
