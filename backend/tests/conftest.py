"""Shared pytest fixtures.

Tests run with MMRAG_FAKE_EMBEDDINGS=1 so they are fast, deterministic and need
no model downloads or API keys. Data + vector store are redirected to a temp dir
so tests never touch a real deployment. These env vars MUST be set before any
app module is imported, hence at the very top of this file.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# ---- Environment must be configured before importing app code ----
_TMP = Path(tempfile.mkdtemp(prefix="mmrag_test_"))
os.environ["MMRAG_FAKE_EMBEDDINGS"] = "1"
os.environ["DATA_PATH"] = str(_TMP / "data")
os.environ["VECTOR_DB_PATH"] = str(_TMP / "vector_store")
os.environ["LLM_PROVIDER"] = "gemini"  # never depend on the developer's real .env
os.environ["LLM_API_KEY"] = ""  # force the evidence-only fallback in query tests
os.environ["CORS_ORIGINS"] = "http://localhost:5173"

import datetime  # noqa: E402
import shutil  # noqa: E402

import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.sample_pdf import build_sample_pdf  # noqa: E402
from app.schemas import Document  # noqa: E402
from app.services.ingest_service import process_document  # noqa: E402
from app.utils import registry  # noqa: E402


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    """A small technical PDF (text + schematic + table) used across tests."""
    out = get_settings().data_dir / "sample_fixture.pdf"
    build_sample_pdf(out)
    return out


@pytest.fixture(scope="session")
def ingested_document(sample_pdf) -> str:
    """Ingest the sample PDF once and return its document_id (READY)."""
    doc_id = "testdoc"
    out_dir = get_settings().data_dir / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "source.pdf"
    shutil.copy(sample_pdf, pdf_path)
    registry.save(
        Document(
            document_id=doc_id,
            filename="sample_fixture.pdf",
            upload_time=datetime.datetime.utcnow().isoformat() + "Z",
        )
    )
    process_document(doc_id, str(pdf_path), "sample_fixture.pdf")
    return doc_id


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)
