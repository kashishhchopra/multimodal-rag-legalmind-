"""Browsable catalogue of the large indexed corpus (judgments and statutes).

The vector index knows chunks, not document titles, so the catalogue reads the ingestion output (docs.jsonl,
statutes.jsonl, chunks.jsonl under CORPUS_DIR, default <repo>/data_lake/corpus). If those files are absent (for example in a
container without data_lake) the catalogue is empty and the app behaves as before. Corpus documents behave like uploaded ones for
questions, clauses, summary and timeline, but they are read-only: they cannot be deleted through the API.
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from ..config import BACKEND_DIR
from ..schemas import Document, ProcessingStage

STATUTE_NAMES = {
    "BNS": "Bharatiya Nyaya Sanhita, 2023", "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2023", "BSA": "Bharatiya Sakshya Adhiniyam, 2023",
    "COI": "Constitution of India (as on 1 April 2019)", "CONTRACT": "Indian Contract Act, 1872", "SRA": "Specific Relief Act, 1963",
    "LIMITATION": "Limitation Act, 1963", "STAMP": "Indian Stamp Act, 1899", "ITACT": "Information Technology Act, 2000",
    "DPDP": "Digital Personal Data Protection Act, 2023",
}
_lock = threading.Lock()


def _dir() -> Path:
    return Path(os.getenv("CORPUS_DIR", str(BACKEND_DIR.parent / "data_lake" / "corpus")))


@lru_cache(maxsize=1)
def _load() -> Dict[str, dict]:
    """document_id -> catalogue entry. Loaded once (about 2.6k small records plus per-document chunk counts)."""
    d, out = _dir(), {}
    if (d / "docs.jsonl").exists():
        for rec in map(json.loads, (d / "docs.jsonl").open()):
            out[rec["doc_id"]] = {"document_id": rec["doc_id"], "kind": "judgment", "title": rec.get("title") or rec["doc_id"],
                                  "citation": rec.get("citation", ""), "year": rec.get("year", 0), "court": rec.get("court", ""),
                                  "date": rec.get("decision_date", ""), "pages": rec.get("pages", 0), "chunks": 0}
        if (d / "chunks.jsonl").exists():
            counts: Counter = Counter()
            for line in (d / "chunks.jsonl").open():
                i = line.find('"doc_id": "')
                if i >= 0:
                    counts[line[i + 11: line.find('"', i + 11)]] += 1
            for k, v in counts.items():
                if k in out:
                    out[k]["chunks"] = v
    if (d / "statutes.jsonl").exists():
        per: Counter = Counter()
        pages: Dict[str, int] = {}
        years: Dict[str, int] = {}
        for rec in map(json.loads, (d / "statutes.jsonl").open()):
            per[rec["act"]] += 1
            pages[rec["act"]] = max(pages.get(rec["act"], 0), rec.get("page_number", 0))
            years[rec["act"]] = rec.get("year", 0)
        for act, n in per.items():
            did = f"statute_{act}"
            out[did] = {"document_id": did, "kind": "statute", "title": STATUTE_NAMES.get(act, act), "citation": act, "year": years[act],
                        "court": "", "date": "", "pages": pages[act], "chunks": n}
    return out


def is_corpus(document_id: str) -> bool:
    return document_id in _load()


def get(document_id: str) -> Optional[Document]:
    e = _load().get(document_id)
    if not e:
        return None
    label = e["title"] + (f" ({e['citation']})" if e["citation"] and e["kind"] == "judgment" else "")
    return Document(document_id=e["document_id"], filename=label, upload_time=f"{e['year'] or 1970}-01-01T00:00:00Z", page_count=e["pages"],
                    text_chunk_count=e["chunks"], stage=ProcessingStage.READY, progress=1.0, stage_message="Corpus document (read-only)")


def search(q: str = "", year: Optional[int] = None, kind: Optional[str] = None, offset: int = 0, limit: int = 25) -> dict:
    items: List[dict] = list(_load().values())
    if kind:
        items = [e for e in items if e["kind"] == kind]
    if year:
        items = [e for e in items if e["year"] == year]
    if q.strip():
        terms = q.lower().split()
        items = [e for e in items if all(t in f"{e['title']} {e['citation']} {e['document_id']} {e['date']}".lower() for t in terms)]
    items.sort(key=lambda e: (e["kind"] != "statute", -(e["year"] or 0), e["title"]))
    years = Counter(e["year"] for e in _load().values() if e["kind"] == "judgment")
    return {"total": len(items), "items": items[offset: offset + limit], "years": dict(sorted(years.items(), reverse=True)),
            "totals": dict(Counter(e["kind"] for e in _load().values()))}
