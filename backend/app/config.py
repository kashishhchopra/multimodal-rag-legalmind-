"""Central configuration, loaded from environment variables / .env file.

Uses pydantic-settings so every value is validated and typed. Nothing here
should ever contain a secret literal — secrets come from the environment.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ directory (two levels up from this file: app/config.py -> app -> backend)
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM ----
    llm_provider: str = Field(default="gemini")
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="gemini-3.6-flash")

    # ---- Embeddings ----
    # Defaults to the legal-domain fine-tuned BGE model (see
    # scripts/finetune_legal_embeddings.py) if it has been trained and saved
    # locally; otherwise falls back to the general-purpose base model so the
    # app still runs before training has been run.
    text_embedding_model: str = Field(
        default_factory=lambda: (
            str(BACKEND_DIR / "app" / "models" / "legal-bge-finetuned")
            if (BACKEND_DIR / "app" / "models" / "legal-bge-finetuned" / "config.json").exists()
            else "BAAI/bge-small-en-v1.5"
        )
    )
    image_embedding_model: str = Field(default="clip-ViT-B-32")

    # ---- Paths (relative values are resolved against backend/) ----
    vector_db_path: str = Field(default="./vector_store")
    data_path: str = Field(default="./data")

    # ---- Retrieval ----
    top_k_text: int = Field(default=5)
    top_k_visual: int = Field(default=3)
    score_alpha: float = Field(default=0.6)
    score_beta: float = Field(default=0.4)

    # ---- Hybrid retrieval (BM25 + dense, reciprocal rank fusion) ----
    hybrid_enabled: bool = Field(default=True)
    rrf_k: int = Field(default=60)
    hybrid_candidates: int = Field(default=30)  # per-ranker pool fed into the fusion
    expand_legal_sections: bool = Field(default=True)  # Section 420 IPC -> also BNS 318(4)

    # ---- Hallucination guard ----
    guard_enabled: bool = Field(default=True)
    guard_min_top_score: float = Field(default=0.30)   # best chunk cosine below this -> refuse before calling the LLM
    guard_min_claim_support: float = Field(default=0.5)  # content-word overlap needed for a claim to count as supported

    # ---- Server ----
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    cors_origins: str = Field(default="http://localhost:5173,http://127.0.0.1:5173")

    # ---- Limits ----
    max_upload_mb: int = Field(default=50)
    max_pages: int = Field(default=200)

    @field_validator("llm_provider")
    @classmethod
    def _normalise_provider(cls, v: str) -> str:
        return v.strip().lower()

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def data_dir(self) -> Path:
        p = Path(self.data_path)
        if not p.is_absolute():
            p = BACKEND_DIR / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def vector_dir(self) -> Path:
        p = Path(self.vector_db_path)
        if not p.is_absolute():
            p = BACKEND_DIR / p
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the .env file is only parsed once per process."""
    return Settings()
