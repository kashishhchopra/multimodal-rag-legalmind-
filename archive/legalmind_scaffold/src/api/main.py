"""Layer 5: FastAPI application entry point.

Wires together session/document management, the async ingestion queue, and
the feature routes. Kept thin — actual logic lives in the layer-specific
modules (ingestion, retrieval, generation, features); routes only translate
HTTP <-> those calls.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import documents, features, query

app = FastAPI(
    title="LegalMind API",
    description="Grounded, multimodal RAG over legal documents.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(features.router, prefix="/features", tags=["features"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
