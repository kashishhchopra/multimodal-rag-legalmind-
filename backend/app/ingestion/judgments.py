"""Bulk-ingestion building blocks for court judgments (Supreme Court AWS dump and similar).

Pipeline stages (roadmap 5.2): classify (native/scanned) -> clean -> metadata + citations ->
structure-aware chunks. Pure functions, safe to run in worker processes. Embedding/indexing
lives in scripts/ingest_judgments.py so this module imports no ML libraries.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import fitz  # PyMuPDF

from ..legal.section_mapper import find_references

# A page with fewer extractable characters than this is treated as a scan.
NATIVE_MIN_CHARS_PER_PAGE = 200

_INSC = re.compile(r"\b(\d{4})\s*INSC\s*(\d+)\b")
_CASE_CITES = [
    re.compile(r"\bAIR\s+\d{4}\s+SC\s+\d+\b"),
    re.compile(r"\(\d{4}\)\s*\d+\s*SCC\s*\d+\b"),
    re.compile(r"\b\d{4}\s+INSC\s+\d+\b"),
    re.compile(r"\[\d{4}\]\s*\d+\s*S\.C\.R\.\s*\d+\b"),
]
_DATE = re.compile(r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b")
_TITLE_V = re.compile(r"^\s*(?:v\.?|vs\.?|versus)\s*$", re.IGNORECASE)
# Headings that start a new logical section in SC reporter-style PDFs.
_SECTION_HEADS = [
    ("issue", re.compile(r"^\s*Issue\(?s?\)?\s+for\s+Consideration\s*$", re.I)),
    ("headnotes", re.compile(r"^\s*Headnotes?\W*$", re.I)),
    ("held", re.compile(r"^\s*Held\s*:?\s*$", re.I)),
    ("cases_cited", re.compile(r"^\s*Case\s+Law\s+Cited\s*$", re.I)),
    ("list_of_acts", re.compile(r"^\s*List\s+of\s+Acts\s*$", re.I)),
    ("appearances", re.compile(r"^\s*Appearances\s+for\s+Parties\s*$", re.I)),
    ("judgment", re.compile(r"^\s*(?:Judgment|Order|Reportable|Non-Reportable)\s*$", re.I)),
    ("facts", re.compile(r"^\s*(?:Facts|Factual\s+Matrix|Brief\s+Facts)\b.*$", re.I)),
    ("analysis", re.compile(r"^\s*(?:Analysis|Discussion|Reasoning|Conclusion|Submissions?)\b.*$", re.I)),
]
_HYPHEN = re.compile(r"(\w)-\n(\w)")
_PAGE_NO = re.compile(r"^\s*\d{1,4}\s*$")


@dataclass
class ParsedJudgment:
    doc_id: str
    source_name: str
    n_pages: int
    kind: str  # native | scanned | mixed
    pages: List[str]
    sha256: str


@dataclass
class JudgmentChunk:
    chunk_id: str
    doc_id: str
    page_number: int
    section: str
    text: str


@dataclass
class JudgmentMeta:
    doc_id: str
    title: str = ""
    citation: str = ""
    decision_date: str = ""
    year: int = 0
    court: str = "Supreme Court of India"
    doc_type: str = "judgment"
    language: str = "en"
    statutes_cited: List[str] = field(default_factory=list)
    cases_cited: List[str] = field(default_factory=list)


def parse_pdf(data: bytes, source_name: str) -> ParsedJudgment:
    doc = fitz.open(stream=data, filetype="pdf")
    pages = [p.get_text() for p in doc]
    native = sum(1 for t in pages if len(t.strip()) >= NATIVE_MIN_CHARS_PER_PAGE)
    kind = "native" if native == len(pages) else "scanned" if native == 0 else "mixed"
    sha = hashlib.sha256(data).hexdigest()
    return ParsedJudgment(doc_id=re.sub(r"\.pdf$", "", source_name), source_name=source_name, n_pages=len(pages), kind=kind, pages=pages, sha256=sha)


def clean_page(text: str) -> str:
    """De-hyphenate line breaks, drop bare page numbers, collapse whitespace; keeps paragraph breaks."""
    text = _HYPHEN.sub(r"\1\2", text.replace("­", ""))
    lines = [ln.rstrip() for ln in text.splitlines() if not _PAGE_NO.match(ln)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract_metadata(parsed: ParsedJudgment, nc_display: Optional[str] = None) -> JudgmentMeta:
    head = clean_page(parsed.pages[0]) if parsed.pages else ""
    full = "\n".join(parsed.pages)
    m = JudgmentMeta(doc_id=parsed.doc_id)
    ins = _INSC.search(nc_display or "") or _INSC.search(head)
    if ins:
        m.citation, m.year = f"{ins.group(1)} INSC {ins.group(2)}", int(ins.group(1))
    d = _DATE.search(head)
    if d:
        m.decision_date = d.group(1)
        m.year = m.year or int(d.group(1)[-4:])
    if not m.year:
        ym = re.match(r"(\d{4})_", parsed.source_name)
        m.year = int(ym.group(1)) if ym else 0
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]
    for i, ln in enumerate(lines[:14]):  # "Petitioner \n v. \n Respondent" block
        if _TITLE_V.match(ln) and 0 < i < len(lines) - 1:
            m.title = f"{lines[i-1].rstrip(',')} v. {lines[i+1].rstrip(',')}"[:300]
            break
    m.statutes_cited = sorted({f"{c.upper()} {s}" for c, s in find_references(full)})
    cites = {c.group(0) for rx in _CASE_CITES for c in rx.finditer(full)}
    m.cases_cited = sorted(cites - {m.citation})
    return m


def chunk_judgment(parsed: ParsedJudgment, max_chars: int = 1600, overlap: int = 200) -> List[JudgmentChunk]:
    """Split on section headings and numbered paragraphs; overlap is applied only inside a section."""
    # (page, section, paragraph) stream, built line by line so headings fused to body text are still caught
    paras: List[tuple[int, str, str]] = []
    section = "front_matter"
    cur: List[str] = []
    cur_page = 1

    def close() -> None:
        nonlocal cur
        block = re.sub(r"\s+", " ", " ".join(cur)).strip()
        if block:
            paras.append((cur_page, section, block))
        cur = []

    for pno, raw in enumerate(parsed.pages, 1):
        for line in clean_page(raw).splitlines():
            stripped = line.strip()
            head = next((n for n, rx in _SECTION_HEADS if len(stripped) < 80 and rx.match(stripped)), None)
            if head:
                close()
                section = head
            elif not stripped or re.match(r"\d{1,3}\.\s", stripped):
                close()  # blank line / numbered paragraph starts a new paragraph
                if stripped:
                    cur, cur_page = [stripped], pno
            else:
                if not cur:
                    cur_page = pno
                cur.append(stripped)
    close()
    chunks: List[JudgmentChunk] = []
    buf, buf_page, buf_section = "", 1, "front_matter"

    def flush(carry: bool) -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(JudgmentChunk(f"{parsed.doc_id}_c{len(chunks):04d}", parsed.doc_id, buf_page, buf_section, buf.strip()))
        tail = buf[-overlap:] if carry and overlap else ""
        buf = tail.split(" ", 1)[1] if " " in tail else tail  # start the overlap on a word boundary

    for pno, sec, text in paras:
        if buf and sec != buf_section:
            flush(carry=False)  # never overlap across sections
        if not buf:
            buf_page, buf_section = pno, sec
        while len(text) > max_chars:  # oversized paragraph: cut on a sentence boundary
            cut = text.rfind(". ", 0, max_chars) + 1 or max_chars
            if len(buf) + cut > max_chars:
                flush(carry=True)
            buf = (buf + " " + text[:cut]).strip()
            text = text[cut:].strip()
        if len(buf) + len(text) + 1 > max_chars:
            flush(carry=True)
            buf_page, buf_section = pno, sec
        buf = (buf + " " + text).strip()
    flush(carry=False)
    return [c for c in chunks if len(c.text) >= 40]


def quality_report(parsed: ParsedJudgment, chunks: List[JudgmentChunk], meta: JudgmentMeta) -> Dict:
    return {
        "doc_id": parsed.doc_id, "pages": parsed.n_pages, "kind": parsed.kind, "chunks": len(chunks),
        "chars": sum(len(p) for p in parsed.pages),
        "has_citation": bool(meta.citation), "has_title": bool(meta.title), "has_date": bool(meta.decision_date),
        "n_statutes": len(meta.statutes_cited), "n_cases_cited": len(meta.cases_cited),
    }
