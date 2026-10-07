"""Central, typed configuration for LegalMind.

Every other module reads settings from here rather than calling
``os.environ`` directly, so the whole app has one place to see (and swap)
provider choices — model IDs, the vector store URL, feature flags.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM ---
    llm_provider: str = "anthropic"          # anthropic | openai
    llm_model: str = "claude-sonnet-5"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # --- Embeddings ---
    embedding_provider: str = "openai"       # openai | bge
    embedding_model: str = "text-embedding-3-large"

    # --- Vector store ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "legalmind_chunks"

    # --- Retrieval ---
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 6
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Chunking ---
    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # --- OCR / vision ---
    ocr_engine: str = "tesseract"            # tesseract | paddleocr
    vision_model: str = "claude-sonnet-5"
    min_text_chars_for_native: int = 40      # below this, a page is treated as a scan

    # --- Task queue ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Observability ---
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "legalmind"

    # --- App ---
    app_env: str = "dev"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
