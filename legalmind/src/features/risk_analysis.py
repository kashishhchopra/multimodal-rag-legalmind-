"""WOW feature: risk / red-flag analysis.

Scans a document's clauses for unusual or one-sided terms — auto-renewal
traps, unlimited liability, one-sided indemnification, missing termination
rights, unusual governing law — and returns a red/amber/green rating per
clause with a one-line explanation. Deliberately simple output shape so the
UI can render it as a colour-coded list with no further processing.
"""
from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel

from src.generation.answerer import call_llm
from src.generation.prompts import RISK_ANALYSIS_PROMPT
from src.retrieval.hybrid_retriever import RetrievedChunk, hybrid_search

# Clause categories most likely to hide one-sided terms — used to fan out
# the retrieval query rather than relying on a single generic query.
RISK_CATEGORY_QUERIES = [
    "auto-renewal and automatic extension of term",
    "limitation of liability and indemnification",
    "termination rights and termination for convenience",
    "governing law and dispute resolution / arbitration",
    "exclusivity and non-compete restrictions",
    "confidentiality and data ownership",
]


class RiskLevel(str, Enum):
    RED = "RED"
    AMBER = "AMBER"
    GREEN = "GREEN"


class RiskFinding(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
    section_path: str | None
    level: RiskLevel
    explanation: str


def analyze_risks(doc_id: str) -> list[RiskFinding]:
    findings: list[RiskFinding] = []
    seen_chunk_ids: set[str] = set()

    for category_query in RISK_CATEGORY_QUERIES:
        hits = hybrid_search(category_query, doc_ids=[doc_id])
        for hit in hits[:2]:  # top matches per category, avoid duplicate work
            if hit.chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(hit.chunk.chunk_id)
            findings.append(_classify_clause(hit))

    return sorted(findings, key=lambda f: {"RED": 0, "AMBER": 1, "GREEN": 2}[f.level.value])


def _classify_clause(hit: RetrievedChunk) -> RiskFinding:
    prompt = RISK_ANALYSIS_PROMPT.format(
        section_path=hit.chunk.metadata.section_path or "unlabelled",
        page=hit.chunk.metadata.page,
        clause_text=hit.chunk.text,
    )
    response = call_llm(prompt).strip()

    level = _extract_level(response)
    explanation = _strip_level_prefix(response)

    return RiskFinding(
        chunk_id=hit.chunk.chunk_id,
        doc_id=hit.chunk.metadata.doc_id,
        page=hit.chunk.metadata.page,
        section_path=hit.chunk.metadata.section_path,
        level=level,
        explanation=explanation,
    )


def _extract_level(response: str) -> RiskLevel:
    match = re.search(r"\b(RED|AMBER|GREEN)\b", response, re.IGNORECASE)
    if match:
        return RiskLevel(match.group(1).upper())
    return RiskLevel.AMBER  # unclassified defaults to a flag for human review, not silently GREEN


def _strip_level_prefix(response: str) -> str:
    return re.sub(r"^\W*\b(RED|AMBER|GREEN)\b\W*", "", response, flags=re.IGNORECASE).strip()
