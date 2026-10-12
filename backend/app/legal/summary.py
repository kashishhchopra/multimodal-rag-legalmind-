"""Extractive structured judgment summary (facts / issue / holding / authorities) from indexed chunks.

Works from the `section` metadata written by the judgment ingester (issue, headnotes, facts, judgment, analysis) and
falls back to cue-word heuristics for uploaded documents without sections. Purely extractive: every sentence returned
exists verbatim in the source, with its page, so nothing is invented. An abstractive model trained on IN-Abs is the planned upgrade.
"""
from __future__ import annotations

import re
from typing import Dict, List

from ..ingestion.judgments import _CASE_CITES
from .section_mapper import find_references

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"“])")
_HOLDING_CUES = re.compile(
    r"\b(we hold|it is held|we are of the (?:considered )?opinion|we find|accordingly|in the result|appeal is (?:allowed|dismissed)|"
    r"petition is (?:allowed|dismissed)|set aside|upheld|affirmed|quashed|conviction|acquit|ordered|directed)\b", re.I)
_FACT_CUES = re.compile(r"\b(filed|lodged|alleged|appellant|respondent|petitioner|complaint|FIR|trial court|High Court|dated|convicted|sentenced)\b", re.I)


def _sentences(chunks: List[dict], section: str | None = None) -> List[dict]:
    out = []
    for c in chunks:
        if section and (c.get("section") != section):
            continue
        for s in _SENT.split(re.sub(r"\s+", " ", c["text"])):
            if len(s.split()) >= 4:
                out.append({"text": s.strip(), "page": int(c.get("page_number", 0)), "chunk_id": c["id"]})
    return out


def _top(sents: List[dict], cue: re.Pattern, n: int, tail: bool = False) -> List[dict]:
    hits = [s for s in sents if cue.search(s["text"])]
    return (hits[-n:] if tail else hits[:n])


def summarise(chunks: List[dict], max_per_section: int = 4) -> Dict:
    """chunks: [{id, text, page_number, section?}] in document order."""
    has_sections = any(c.get("section") for c in chunks)
    issue = _sentences(chunks, "issue")[:max_per_section] if has_sections else []
    head = _sentences(chunks, "headnotes")[:max_per_section] if has_sections else []
    body = [c for c in chunks if c.get("section") in ("judgment", "analysis", "facts")] if has_sections else chunks
    body_s = _sentences(body)
    facts = _top(_sentences(chunks, "facts") if has_sections else body_s, _FACT_CUES, max_per_section)
    holding = _top(body_s, _HOLDING_CUES, max_per_section, tail=True)  # holdings concentrate at the end
    full = " ".join(c["text"] for c in chunks)
    statutes = sorted({f"{c.upper()} {s}" for c, s in find_references(full)})
    cases = sorted({m.group(0) for rx in _CASE_CITES for m in rx.finditer(full)})
    return {
        "issue": issue, "headnotes": head, "facts": facts, "holding": holding,
        "statutes_cited": statutes[:60], "cases_cited": cases[:60],
        "method": "extractive", "note": "Sentences are quoted verbatim with page numbers; verify against the source.",
    }
