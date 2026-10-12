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


# ---- statutes ----
def test_statute_parser_finds_monotonic_sections(tmp_path):
    import fitz

    from app.ingestion.statutes import parse_sections

    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(40, 40, 550, 800), "1. Short title. This Act may be called the Test Act.\n(2) It extends to India.\n"
                        "2. Definitions. In this Act, unless the context otherwise requires, cheating means deceiving.\n"
                        "Section 99. noise that is not a heading\n3. Punishment. Whoever cheats shall be punished.\n", fontsize=10)
    pdf = tmp_path / "t.pdf"
    doc.save(pdf)
    chunks = parse_sections(str(pdf), "TEST")
    assert [c.section for c in chunks] == ["1", "2", "3"]
    assert chunks[1].chunk_id == "TEST_s2_0" and "cheating" in chunks[1].text


# ---- verification / guard ----
def test_verify_flags_unsupported_numbers_and_pages():
    from app.legal.verify import retrieval_gate, verify_answer

    ev = [{"id": "c1", "page": 2, "text": "The agreement terminates after thirty days written notice by either party."},
          {"id": "c2", "page": 3, "text": "The governing law is the law of India and courts at Delhi have jurisdiction."}]
    good = "The agreement terminates after thirty days written notice by either party (Page 2)."
    assert verify_answer(good, ev).status == "ok"
    bad = "The agreement carries a penalty of 50,000 rupees on early termination by either party."
    v = verify_answer(bad, ev)
    assert v.status == "flagged" and v.claims[0].missing_numbers
    wrong_page = "The governing law is the law of India and courts at Delhi have jurisdiction (Page 9)."
    assert verify_answer(wrong_page, ev).claims[0].bad_page_refs == [9]
    assert verify_answer("Disclaimer: This is an AI-generated summary of the document.", ev).status == "no_claims"
    assert retrieval_gate([0.1, 0.2], 0.3) is False and retrieval_gate([0.5], 0.3) is True


# ---- timeline / redact / summary / memo / compare / feedback ----
def test_timeline_normalises_dates_and_sorts():
    from app.legal.timeline import extract_timeline

    ev = extract_timeline([{"id": "a", "page_number": 1, "text": "The FIR was lodged on 5 March 2019. The trial court convicted him on 12.08.2021. Notice dated September 3, 2018 was served."}])
    assert [e.date for e in ev] == ["2018-09-03", "2019-03-05", "2021-08-12"]


def test_redact_identifiers_only():
    from app.legal.redact import redact

    text, counts = redact("Call 9876543210 or mail a.b@example.com; PAN ABCDE1234F; Aadhaar 2345 6789 0123; account number 123456789012.")
    assert counts == {"AADHAAR": 1, "PAN": 1, "EMAIL": 1, "PHONE": 1, "ACCOUNT": 1}
    assert "9876543210" not in text and "ABCDE1234F" not in text


def test_summary_is_extractive_with_sections():
    from app.legal.summary import summarise

    chunks = [
        {"id": "1", "page_number": 1, "section": "issue", "text": "Whether the conviction under Section 302 IPC can be sustained on circumstantial evidence alone."},
        {"id": "2", "page_number": 2, "section": "judgment", "text": "The complaint was lodged by the informant against the appellant on that day. We hold that the chain of circumstances is complete. Accordingly the appeal is dismissed."},
    ]
    s = summarise(chunks)
    assert s["issue"] and "IPC 302" in s["statutes_cited"]
    assert any("appeal is dismissed" in h["text"] for h in s["holding"])
    full = " ".join(c["text"] for c in chunks)
    assert all(h["text"] in full for h in s["holding"] + s["facts"])  # nothing invented


def test_memo_formats_and_feedback_roundtrip(tmp_path):
    from app.legal import feedback, memo

    for fmt, magic in (("pdf", b"%PDF"), ("docx", b"PK"), ("md", b"# ")):
        out = memo.FORMATS[fmt][0]("T", "Q?", "A.", [{"label": "Page 1", "page_number": 1, "snippet": "s"}], {"status": "ok", "support_ratio": 1.0})
        assert out.startswith(magic)
    feedback.add(tmp_path, "d", "q", "a", -1, "wrong", "unsupported_claims")
    feedback.add(tmp_path, "d", "q", "a", 1, "", "")
    assert feedback.stats(tmp_path) == {"up": 1, "down": 1, "down_by_answer_status": {"unsupported_claims": 1}}


def test_compare_aligns_by_clause_category():
    from app.legal import clauses, compare

    if not clauses.model_available():
        import pytest

        pytest.skip("clause model not trained")
    a = [{"id": "a1", "page_number": 1, "text": "This Agreement shall be governed by and construed in accordance with the laws of the State of New York."}]
    b = [{"id": "b1", "page_number": 1, "text": "This Agreement shall be governed by and construed in accordance with the laws of India."},
         {"id": "b2", "page_number": 2, "text": "Neither party shall assign this Agreement without the prior written consent of the other party."}]
    out = compare.compare(a, b)
    cats = {r["category"]: r["status"] for r in out["rows"]}
    assert cats.get("Governing Law") in ("similar", "different") and cats.get("Anti-Assignment") == "only_in_b"


# ---- conversation ----
def test_followup_rewrite_with_and_without_llm():
    from app.services.conversation import is_hindi, rewrite_followup, style_instruction, to_english

    hist = [{"role": "user", "content": "What is the notice period for termination?"}, {"role": "assistant", "content": "Thirty days."}]

    class Fake:
        def generate(self, system, user, images, max_output_tokens=None):
            return "What is the penalty for terminating the agreement early?\nextra line"

    assert rewrite_followup("And what is the penalty for it?", hist, Fake()) == "What is the penalty for terminating the agreement early?"
    assert rewrite_followup("And what is the penalty for it?", hist, None).startswith("What is the notice period")
    assert rewrite_followup("Define indemnity clause obligations.", hist, Fake()) == "Define indemnity clause obligations."  # standalone: untouched
    assert is_hindi("धारा 420 के तहत सजा क्या है") and not is_hindi("Section 420")
    assert to_english("धारा 420 के तहत सजा क्या है", type("P", (), {"generate": lambda *a, **k: "What is the punishment under Section 420?"})()).startswith("What is the punishment")
    assert "plain English" in style_instruction(True, "en") and "Hindi" in style_instruction(False, "hi")


# ---- scanned-PDF OCR ----
def test_ocr_recovers_text_from_image_only_pdf():
    import shutil

    import pytest

    if not shutil.which("tesseract"):
        pytest.skip("tesseract not installed")
    import fitz

    from app.ingestion.scan_ocr import ocr_pdf

    src = fitz.open()
    src.new_page().insert_textbox(fitz.Rect(40, 60, 550, 700), "The appeal is allowed. The conviction under Section 302 of the Indian Penal Code is set aside.", fontsize=16)
    pix = src[0].get_pixmap(dpi=200)
    scan = fitz.open()
    scan.new_page(width=pix.width, height=pix.height).insert_image(fitz.Rect(0, 0, pix.width, pix.height), pixmap=pix)  # image only, no text layer
    assert not scan[0].get_text().strip()
    pages = ocr_pdf(scan.tobytes())
    assert "conviction" in pages[0].text.lower() and "302" in pages[0].text and pages[0].confidence > 60


# ---- auth / telemetry / health ----
def test_token_roundtrip_and_tamper(monkeypatch):
    import pytest
    from fastapi import HTTPException

    from app import auth

    monkeypatch.setenv("AUTH_SECRET", "x" * 32)
    tok = auth.make_token("alice", "user")
    assert auth.read_token(tok)["sub"] == "alice"
    with pytest.raises(HTTPException):
        auth.read_token(tok[:-2] + "aa")
    with pytest.raises(HTTPException):
        auth.read_token(auth.make_token("a", "user", now=0))  # expired long ago


def test_auth_flow_ownership_and_admin(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("AUTH_ENABLED", "1")
    monkeypatch.setenv("AUTH_SECRET", "y" * 32)
    c = TestClient(app)
    assert c.get("/api/documents").status_code == 401
    assert c.get("/api/health").status_code == 200
    import uuid

    admin, user = f"admin_{uuid.uuid4().hex[:6]}", f"user_{uuid.uuid4().hex[:6]}"
    pw = "correct horse battery"
    # the first user ever registered becomes admin; in a shared test DB that may already exist, so only assert roles are valid
    r1 = c.post("/api/auth/register", json={"username": admin, "password": pw}).json()
    r2 = c.post("/api/auth/register", json={"username": user, "password": pw}).json()
    assert r1["role"] in ("admin", "user") and r2["role"] == "user"
    assert c.post("/api/auth/login", json={"username": user, "password": "wrong password!!"}).status_code == 401
    tok = c.post("/api/auth/login", json={"username": user, "password": pw}).json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    assert c.get("/api/documents", headers=h).status_code == 200
    assert c.get("/api/admin/stats", headers=h).status_code == 403
    assert c.post("/api/auth/register", json={"username": "ab", "password": "short"}).status_code == 422


def test_health_never_echoes_malformed_provider(monkeypatch):
    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.main import app

    monkeypatch.setenv("LLM_PROVIDER", "sk-this-looks-like-a-secret-1234567890")
    get_settings.cache_clear()
    try:
        # main.py captured `settings` at import; the health route reads that object, so patch it directly too
        import app.main as m

        old = m.settings.llm_provider
        m.settings.llm_provider = "sk-this-looks-like-a-secret-1234567890"
        body = TestClient(app).get("/api/health").json()
        m.settings.llm_provider = old
    finally:
        get_settings.cache_clear()
    assert "sk-this" not in str(body) and body["llm_provider"].startswith("invalid")


def test_telemetry_summary(tmp_path):
    from app.legal import telemetry

    for ms, st in ((100, "ok"), (300, "ok"), (900, "low_retrieval_confidence")):
        telemetry.record(tmp_path, "document", ms, st, 400, 80, 3)
    s = telemetry.summary(tmp_path)
    assert s["queries"] == 3 and s["latency_p50_ms"] == 300 and s["by_status"]["low_retrieval_confidence"] == 1


def test_doctype_router_hint():
    import pytest

    from app.legal import doctype

    if not doctype.MODEL_PATH.exists():
        pytest.skip("router not trained")
    j = doctype.classify("IN THE SUPREME COURT OF INDIA CIVIL APPELLATE JURISDICTION Criminal Appeal No. 12 of 2020 The appeal is allowed and the conviction set aside. Heard learned counsel.")
    c = doctype.classify("This Services Agreement is entered into between the Company and the Contractor. Either party may terminate this Agreement on thirty days written notice. Governing law shall be the laws of New York.")
    assert j["type"] == "judgment" and c["type"] == "contract" and "summary" in j["suggested_tools"]
