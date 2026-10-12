"""Clause-by-clause contract comparison: align the two documents by clause category, then diff the aligned text."""
from __future__ import annotations

import difflib
from typing import Dict, List

from .clauses import extract_clauses
from .risk import LEVELS


def _by_category(chunks: List[dict], min_margin: float) -> Dict[str, dict]:
    best: Dict[str, dict] = {}
    for c in extract_clauses(chunks, min_margin=min_margin):
        if c.category not in best or c.margin > best[c.category]["margin"]:
            best[c.category] = {"text": c.text, "page": c.page_number, "margin": c.margin, "risk": c.risk.level}
    return best


def compare(chunks_a: List[dict], chunks_b: List[dict], min_margin: float = 0.0) -> Dict:
    a, b = _by_category(chunks_a, min_margin), _by_category(chunks_b, min_margin)
    rows = []
    for cat in sorted(set(a) | set(b)):
        ca, cb = a.get(cat), b.get(cat)
        if ca and cb:
            sim = difflib.SequenceMatcher(None, ca["text"].split(), cb["text"].split()).ratio()
            status = "identical" if sim > 0.98 else "similar" if sim > 0.7 else "different"
            diff = [t for t in difflib.unified_diff(ca["text"].split(". "), cb["text"].split(". "), lineterm="", n=0) if t[:1] in "+-" and t[:3] not in ("+++", "---")]
            riskier = LEVELS.index(cb["risk"]) - LEVELS.index(ca["risk"])
        else:
            sim, status, diff, riskier = 0.0, "only_in_a" if ca else "only_in_b", [], 0
        rows.append({"category": cat, "status": status, "similarity": round(sim, 3), "a": ca, "b": cb, "diff": diff[:20],
                     "risk_shift": "b_riskier" if riskier > 0 else "a_riskier" if riskier < 0 else "same"})
    order = {"only_in_a": 0, "only_in_b": 0, "different": 1, "similar": 2, "identical": 3}
    rows.sort(key=lambda r: order[r["status"]])
    return {"rows": rows, "summary": {s: sum(1 for r in rows if r["status"] == s) for s in order}}
