"""Dense embedding generation, provider-swappable via config so a BGE
ablation can be run against OpenAI embeddings for the eval report.
"""
from __future__ import annotations

from functools import lru_cache

from src.config import get_settings


def embed_texts(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    if settings.embedding_provider == "bge":
        return _embed_with_bge(texts, model_name=settings.embedding_model or "BAAI/bge-large-en-v1.5")
    return _embed_with_openai(texts, model=settings.embedding_model)


def _embed_with_openai(texts: list[str], *, model: str) -> list[list[float]]:
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


@lru_cache
def _bge_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _embed_with_bge(texts: list[str], *, model_name: str) -> list[list[float]]:
    model = _bge_model(model_name)
    return model.encode(texts, normalize_embeddings=True).tolist()
