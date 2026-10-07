"""Integration tests for the FastAPI layer using an in-process TestClient."""
from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

client = TestClient(app)


def _wait_ready(document_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/documents/{document_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        if data["stage"] in ("ready", "failed"):
            return data
        time.sleep(0.5)
    raise AssertionError("Document did not finish processing in time")


@pytest.fixture(scope="module")
def uploaded_doc(sample_pdf: Path) -> str:
    with open(sample_pdf, "rb") as fh:
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("electronics_manual.pdf", fh, "application/pdf")},
        )
    assert resp.status_code == 200
    doc_id = resp.json()["document_id"]
    data = _wait_ready(doc_id)
    assert data["stage"] == "ready"
    assert data["page_count"] == 3
    assert data["image_count"] >= 2
    return doc_id


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_rejects_non_pdf():
    r = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_file_type"


def test_upload_rejects_fake_pdf():
    r = client.post(
        "/api/documents/upload",
        files={"file": ("bad.pdf", io.BytesIO(b"not really a pdf"), "application/pdf")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_file"


def test_missing_document_404():
    assert client.get("/api/documents/nope").status_code == 404


def test_list_documents(uploaded_doc: str):
    r = client.get("/api/documents")
    assert r.status_code == 200
    assert any(d["document_id"] == uploaded_doc for d in r.json())


def test_query_returns_evidence_and_citations(uploaded_doc: str):
    r = client.post(
        f"/api/documents/{uploaded_doc}/query",
        json={"question": "Explain Figure 12 and what R1 and Q1 do"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["text_evidence"], "expected text evidence"
    assert body["visual_evidence"], "expected visual evidence"
    assert body["citations"], "expected citations"
    # no LLM key in tests -> graceful fallback
    assert body["warning"] == "missing_api_key"


def test_query_before_ready_conflict(sample_pdf: Path):
    with open(sample_pdf, "rb") as fh:
        up = client.post(
            "/api/documents/upload",
            files={"file": ("m.pdf", fh, "application/pdf")},
        )
    doc_id = up.json()["document_id"]
    # immediately query — likely still processing
    r = client.post(f"/api/documents/{doc_id}/query", json={"question": "hi"})
    assert r.status_code in (200, 409)  # 409 if not ready yet, 200 if it finished fast
    _wait_ready(doc_id)


def test_sources_and_visuals_endpoints(uploaded_doc: str):
    q = client.post(
        f"/api/documents/{uploaded_doc}/query",
        json={"question": "what does R1 do"},
    ).json()
    source_id = q["text_evidence"][0]["chunk_id"]
    visual_id = q["visual_evidence"][0]["visual_id"]

    s = client.get(f"/api/documents/{uploaded_doc}/sources/{source_id}")
    assert s.status_code == 200
    assert s.json()["chunk_id"] == source_id
    assert s.json()["text"]

    v = client.get(f"/api/documents/{uploaded_doc}/visuals/{visual_id}")
    assert v.status_code == 200
    assert v.json()["image_url"].startswith("/static/")

    g = client.get(f"/api/documents/{uploaded_doc}/visuals/{visual_id}/grounding")
    assert g.status_code == 200
    assert g.headers["content-type"] == "image/png"
    assert len(g.content) > 100


def test_static_image_served(uploaded_doc: str):
    q = client.post(
        f"/api/documents/{uploaded_doc}/query", json={"question": "circuit"}
    ).json()
    url = q["visual_evidence"][0]["image_url"]
    r = client.get(url)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
