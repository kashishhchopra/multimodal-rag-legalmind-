"""FastAPI application entrypoint.

Run from the backend/ directory:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.documents import router as documents_router, search_router
from .api.legal import router as legal_router
from .config import get_settings
from .utils.errors import AppError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mmrag")

settings = get_settings()

app = FastAPI(
    title="LegalMind — Multimodal Legal Document AI",
    description="Retrieve and reason over BOTH text and visual evidence (clauses, tables, "
    "seals, signatures) in legal documents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve extracted images / page renders at /static/<document_id>/<file>.png
app.mount("/static", StaticFiles(directory=str(settings.data_dir)), name="static")

app.include_router(documents_router)
app.include_router(search_router)
app.include_router(legal_router)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    """Turn typed application errors into clean JSON the frontend can display."""
    return JSONResponse(status_code=exc.http_status, content=exc.to_dict())


@app.get("/api/health", tags=["health"])
async def health() -> dict:
    from .ingestion.ocr import tesseract_available

    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_key_configured": bool(settings.llm_api_key),
        "text_model": settings.text_embedding_model,
        "image_model": settings.image_embedding_model,
        "ocr_available": tesseract_available(),
    }


@app.get("/", tags=["health"])
async def root() -> dict:
    return {"name": "Multimodal RAG API", "docs": "/docs", "health": "/api/health"}
