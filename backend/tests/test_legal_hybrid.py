"""Tests for the legal section mapper, BM25/RRF fusion and cross-document hybrid retrieval."""
from __future__ import annotations

from app.legal.section_mapper import expand_query, find_references, new_to_old, old_to_new
from app.retrieval.hybrid import BM25Index, rrf_fuse, tokenize
from app.retrieval.retriever import MultimodalRetriever


# ---- section mapper ----
def test_ipc_420_maps_to_bns_318():
    m = old_to_new("IPC", "420")
    assert m and m.new_code == "BNS" and m.new_section.startswith("318")


def test_find_multiple_references_and_aliases():
    refs = find_references("Sections 34 and 120B of the Indian Penal Code; also s. 438 CrPC")
    assert ("ipc", "34") in refs and ("ipc", "120B") in refs and ("crpc", "438") in refs


def test_expand_query_adds_counterpart_and_leaves_unknown_alone():
    q, notes = expand_query("punishment under Section 302 IPC")
    assert "BNS 103" in q and notes
    assert expand_query("Section 99999 IPC") == ("Section 99999 IPC", [])
    assert expand_query("no section here") == ("no section here", [])


def test_reverse_lookup_returns_all_old_sections():
    olds = {m.old_section for m in new_to_old("BNS", "318")}
    assert {"415", "420"} <= olds


# ---- BM25 / RRF ----
def test_tokenizer_keeps_section_numbers():
    assert "318(4)" in tokenize("cheating under section 318(4) of the BNS")


def test_bm25_prefers_exact_term():
    idx = BM25Index(["a", "b", "c"], ["indemnity clause survives termination", "governing law is Delhi", "payment within 30 days"])
    assert idx.search("indemnity", 3)[0][0] == "a"
    assert idx.search("zzzz", 3) == []


def test_rrf_rewards_agreement():
    fused = rrf_fuse([["x", "y", "z"], ["y", "x", "w"]])
    assert [i for i, _ in fused[:2]] == ["y", "x"] or [i for i, _ in fused[:2]] == ["x", "y"]
    assert {i for i, _ in fused} == {"x", "y", "z", "w"}
    assert fused[0][1] >= fused[-1][1]


# ---- hybrid retrieval ----
def test_cross_document_search_without_document_id(ingested_document: str):
    r = MultimodalRetriever().retrieve(None, "current limiting resistor R1")
    assert r.text_evidence
    assert all(t.document_id for t in r.text_evidence)


def test_hybrid_exposes_ranks_and_exact_term_hit(ingested_document: str):
    r = MultimodalRetriever().retrieve(ingested_document, "R1")
    assert any(t.lexical_rank is not None for t in r.text_evidence)
    assert "r1" in " ".join(t.text.lower() for t in r.text_evidence)


def test_metadata_filter_excludes_non_matching(ingested_document: str):
    r = MultimodalRetriever().retrieve(None, "resistor", filters={"document_id": "does-not-exist"})
    assert r.text_evidence == []


# ---- judgment ingestion ----
def _fake_judgment_pdf() -> bytes:
    import fitz

    doc = fitz.open()
    body = (
        "[2024] 9 S.C.R. 770 : 2024 INSC 762\nAsha Devi\nv.\nThe State of Delhi\n(Criminal Appeal No. 1 of 2020)\n"
        "27 September 2024\nIssue for Consideration\nWhether conviction under ss. 302/34 IPC can stand.\n"
        "Headnotes\nPenal Code, 1860 - ss. 302/34 - common intention.\nJudgment\n"
        "1. Leave granted. The appellant challenges conviction relying on (2019) 5 SCC 100 and AIR 2001 SC 55.\n"
        "2. We find the prosecution evidence reliable and the conviction under Section 302 IPC is affirmed.\n"
    ) * 4
    for _ in range(2):
        doc.new_page().insert_textbox(fitz.Rect(40, 40, 550, 800), body, fontsize=9)
    return doc.tobytes()


def test_judgment_metadata_sections_and_dedupe_hash():
    from app.ingestion.judgments import chunk_judgment, extract_metadata, parse_pdf

    data = _fake_judgment_pdf()
    parsed = parse_pdf(data, "2024_9_770_773_EN.pdf")
    assert parsed.kind == "native" and parse_pdf(data, "copy.pdf").sha256 == parsed.sha256
    meta = extract_metadata(parsed, "2024INSC762")
    assert meta.citation == "2024 INSC 762" and meta.year == 2024
    assert "IPC 302" in meta.statutes_cited and "IPC 34" in meta.statutes_cited
    assert "(2019) 5 SCC 100" in meta.cases_cited
    chunks = chunk_judgment(parsed)
    assert chunks and {c.section for c in chunks} >= {"issue", "headnotes", "judgment"}
    assert all(len(c.text) <= 1900 for c in chunks)


# ---- risk rubric / clauses / API ----
def test_risk_rubric_raises_level_on_red_flag():
    from app.legal.risk import assess

    assert assess("Governing Law", "Laws of India govern.").level == "low"
    r = assess("Governing Law", "Party A may terminate in its sole discretion without notice.")
    assert r.level == "medium" and "sole-discretion wording" in r.flags
    assert assess("Uncapped Liability", "x").level == "high"


def test_clause_extraction_flags_known_clause_types():
    from app.legal import clauses

    if not clauses.model_available():
        import pytest

        pytest.skip("clause model not trained")
    chunks = [
        {"id": "a", "page_number": 1, "text": "This Agreement shall be governed by and construed in accordance with the laws of the State of New York."},
        {"id": "b", "page_number": 2, "text": "Neither party shall assign this Agreement without the prior written consent of the other party."},
    ]
    cats = {c.chunk_id: c.category for c in clauses.extract_clauses(chunks, exclude=set())}
    assert cats.get("a") == "Governing Law" and cats.get("b") == "Anti-Assignment"


def test_section_lookup_and_search_endpoints(ingested_document):
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    r = client.get("/api/legal/sections", params={"q": "Section 420 IPC"})
    assert r.status_code == 200 and "BNS 318(4)" in r.json()["expanded_query"]
    r = client.post("/api/search", json={"question": "current limiting resistor R1", "top_k": 3})
    assert r.status_code == 200 and r.json()["text_evidence"]
    r = client.get(f"/api/documents/{ingested_document}/clauses")
    assert r.status_code == 200 and "disclaimer" in r.json()
