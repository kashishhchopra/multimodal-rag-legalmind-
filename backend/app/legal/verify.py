"""Citation verification and hallucination guard.

Two cheap, explainable checks, no extra model download:
  1. Retrieval gate (before the LLM): refuse when the best retrieved chunk is too weak a match.
  2. Claim support (after the LLM): every answer sentence is matched to the retrieved evidence by content-word
     overlap; numbers, section numbers and dates in a claim must appear in the evidence, otherwise the claim is
     flagged "unsupported". Cited "(Page N)" labels are checked against the pages actually retrieved.

This is a lexical grounding check, not entailment: a paraphrase that shares few words can be flagged wrongly and a
claim that reuses evidence words with the opposite meaning can pass. Use it as a triage signal; an NLI verifier
(ContractNLI-style) is the planned upgrade.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..retrieval.hybrid import tokenize

_DISCLAIMER = re.compile(r"disclaimer\s*:", re.I)
_HEDGE = re.compile(r"\b(not clear|not stated|not specified|insufficient|cannot determine|no evidence|not (?:mentioned|provided|available))\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"“•\-\d])|\n+")
_NUMERIC = re.compile(r"\b\d[\d,./-]*\d\b|\b\d\b|\b\d+[A-Z]\b")
_PAGE_REF = re.compile(r"\(?\bPages?\s+(\d+(?:\s*[,&-]\s*\d+)*)\)?", re.I)


@dataclass
class ClaimCheck:
    claim: str
    support: float                # fraction of content words found in the best chunk (0..1)
    best_source: Optional[str]    # chunk_id of the best-supporting chunk
    missing_numbers: List[str] = field(default_factory=list)
    bad_page_refs: List[int] = field(default_factory=list)
    supported: bool = True


@dataclass
class Verification:
    support_ratio: float
    claims: List[ClaimCheck]
    unsupported: List[str]
    status: str                   # "ok" | "flagged" | "no_claims"


def split_claims(answer: str) -> List[str]:
    out = []
    for s in _SENT.split(answer):
        s = s.strip(" -•*\t")
        if len(s.split()) < 5 or _DISCLAIMER.search(s) or _HEDGE.search(s):
            continue
        out.append(s)
    return out


def _numbers(text: str) -> List[str]:
    return [n.strip(",.").lower() for n in _NUMERIC.findall(text)]


def verify_answer(
    answer: str,
    evidence: Sequence[dict],            # [{"id": str, "text": str, "page": int}]
    min_support: float = 0.5,
) -> Verification:
    claims = split_claims(answer)
    if not claims or not evidence:
        return Verification(1.0, [], [], "no_claims")
    ev_tokens = [(e["id"], set(tokenize(e["text"])), e["text"].lower(), e.get("page")) for e in evidence]
    all_text = " ".join(t for _, _, t, _ in ev_tokens)
    pages = {p for *_, p in ev_tokens if p is not None}
    checks: List[ClaimCheck] = []
    for c in claims:
        toks = set(tokenize(c)) - {"shall", "may", "also", "will", "would", "per", "under", "any", "such"}
        words = {t for t in toks if not t[0].isdigit()}
        best_id, best = None, 0.0
        for cid, ts, _, _ in ev_tokens:
            frac = len(words & ts) / len(words) if words else 1.0
            if frac > best:
                best, best_id = frac, cid
        missing = [n for n in _numbers(re.sub(r"\(?\bPages?\s+[\d,\s&-]+\)?", "", c, flags=re.I)) if n not in all_text]
        bad_pages = sorted({int(p) for m in _PAGE_REF.findall(c) for p in re.findall(r"\d+", m) if pages and int(p) not in pages})
        ok = best >= min_support and not missing and not bad_pages
        checks.append(ClaimCheck(c, round(best, 3), best_id, missing, bad_pages, ok))
    unsupported = [k.claim for k in checks if not k.supported]
    ratio = 1 - len(unsupported) / len(checks)
    return Verification(round(ratio, 3), checks, unsupported, "ok" if not unsupported else "flagged")


def retrieval_gate(top_scores: Sequence[float], min_top: float) -> bool:
    """True when retrieval is strong enough to answer; False means refuse instead of guessing."""
    return bool(top_scores) and max(top_scores) >= min_top
