"""WOW feature: cross-document conflict detection.

Approach (per docs/ARCHITECTURE.md 06): retrieve the same clause type (e.g.
"termination notice", "governing law") from each document in the set via
metadata-filtered hybrid search, then run a pairwise LLM comparison over the
extracted clauses to surface contradictions — different notice periods,
conflicting governing law, contradictory obligations.
"""
from __future__ import annotations

from itertools import combinations

from pydantic import BaseModel

from src.generation.answerer import call_llm
from src.generation.prompts import CONFLICT_DETECTION_PROMPT
from src.retrieval.hybrid_retriever import RetrievedChunk, hybrid_search


class ConflictFinding(BaseModel):
    chunk_id_a: str
    chunk_id_b: str
    doc_id_a: str
    doc_id_b: str
    explanation: str


def _fetch_clauses_by_type(clause_type: str, doc_ids: list[str]) -> list[RetrievedChunk]:
    """Pull the top matching clause for `clause_type` from each document,
    so the comparison is apples-to-apples across the set rather than
    comparing whatever happens to rank highest overall."""
    results: list[RetrievedChunk] = []
    for doc_id in doc_ids:
        hits = hybrid_search(clause_type, doc_ids=[doc_id])
        if hits:
            results.append(hits[0])
    return results


def detect_conflicts(clause_type: str, doc_ids: list[str]) -> list[ConflictFinding]:
    clauses = _fetch_clauses_by_type(clause_type, doc_ids)
    findings: list[ConflictFinding] = []

    for a, b in combinations(clauses, 2):
        clauses_block = (
            f"[chunk_id={a.chunk.chunk_id}, doc={a.chunk.metadata.doc_id}]\n{a.chunk.text}\n\n"
            f"[chunk_id={b.chunk.chunk_id}, doc={b.chunk.metadata.doc_id}]\n{b.chunk.text}"
        )
        prompt = CONFLICT_DETECTION_PROMPT.format(clause_type=clause_type, clauses_block=clauses_block)
        response = call_llm(prompt)

        if _looks_like_no_conflict(response):
            continue

        findings.append(ConflictFinding(
            chunk_id_a=a.chunk.chunk_id,
            chunk_id_b=b.chunk.chunk_id,
            doc_id_a=a.chunk.metadata.doc_id,
            doc_id_b=b.chunk.metadata.doc_id,
            explanation=response.strip(),
        ))

    return findings


def _looks_like_no_conflict(response: str) -> bool:
    lowered = response.strip().lower()
    return lowered.startswith("no conflict") or "no conflict" in lowered[:60]
