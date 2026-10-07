"""WOW feature: clause comparison against a playbook (a "standard" or
market-normal clause library).

A redline-style diff plus a plain summary of what changed and why it
matters — the same shape a real legal-tech review tool produces.
"""
from __future__ import annotations

import difflib

from pydantic import BaseModel

from src.generation.answerer import call_llm
from src.generation.prompts import PLAYBOOK_COMPARE_PROMPT
from src.retrieval.hybrid_retriever import RetrievedChunk, hybrid_search

# In a full build this would be a curated JSON/YAML library, one entry per
# clause type, versioned like the document corpus. Kept inline here as a
# starter set — replace/extend via your own playbook source.
DEFAULT_PLAYBOOK: dict[str, str] = {
    "termination for convenience": (
        "Either party may terminate this Agreement for convenience upon "
        "thirty (30) days' prior written notice to the other party."
    ),
    "limitation of liability": (
        "Neither party's aggregate liability under this Agreement shall "
        "exceed the fees paid in the twelve (12) months preceding the claim, "
        "except for breaches of confidentiality or indemnification obligations."
    ),
    "governing law": (
        "This Agreement shall be governed by the laws of the State of "
        "Delaware, without regard to conflict-of-law principles."
    ),
}


class PlaybookDeviation(BaseModel):
    chunk_id: str
    clause_type: str
    diff: str
    explanation: str


def compare_to_playbook(
    clause_type: str,
    doc_id: str,
    *,
    playbook: dict[str, str] | None = None,
) -> PlaybookDeviation | None:
    playbook = playbook or DEFAULT_PLAYBOOK
    standard_clause = playbook.get(clause_type.lower())
    if standard_clause is None:
        raise KeyError(f"No playbook entry for clause type {clause_type!r}")

    hits = hybrid_search(clause_type, doc_ids=[doc_id])
    if not hits:
        return None
    hit: RetrievedChunk = hits[0]

    diff = _redline_diff(standard_clause, hit.chunk.text)

    prompt = PLAYBOOK_COMPARE_PROMPT.format(
        standard_clause=standard_clause,
        section_path=hit.chunk.metadata.section_path or "unlabelled",
        page=hit.chunk.metadata.page,
        document_clause=hit.chunk.text,
    )
    explanation = call_llm(prompt).strip()

    return PlaybookDeviation(
        chunk_id=hit.chunk.chunk_id,
        clause_type=clause_type,
        diff=diff,
        explanation=explanation,
    )


def _redline_diff(standard: str, document: str) -> str:
    """Word-level unified diff, readable as a redline in the UI."""
    diff = difflib.unified_diff(
        standard.split(), document.split(),
        fromfile="playbook", tofile="document", lineterm="",
    )
    return " ".join(diff)
