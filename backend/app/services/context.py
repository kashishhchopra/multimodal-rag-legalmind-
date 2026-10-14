"""Multi-query retrieval fusion and answer-context assembly.

Why: sending only the top 5 passages for every question made answers thin or "not found" whenever the answer sat a few
passages away. Now:
  * fuse_hits: the question and up to 3 LLM-written alternative queries are each searched (dense + lexical) and the rankings
    are merged with reciprocal rank fusion, so differently-worded passages are still found;
  * build_context: for a question about ONE document that fits the size budget, broad or weakly-matched questions get the
    WHOLE document in reading order (nothing can be missed); otherwise the best passages plus their neighbouring passages
    (a clause continues in the next chunk) in reading order. Cross-document search uses the fused hits only.
Nothing here contains canned answers; the LLM writes every answer from the supplied text.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import get_settings
from ..retrieval import vector_store
from ..retrieval.hybrid import rrf_fuse
from ..schemas import RetrievedText

WEAK_TOP_SCORE = 0.60   # best dense similarity below this: the question did not match the text well, so show more of the document
BROAD_TYPES = ("overview", "list", "analytical")


def _natkey(s: str) -> List:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def fuse_hits(retriever, queries: Sequence[str], k: int, document_id: Optional[str], filters: Optional[Dict]) -> List[Dict]:
    """Hits (dicts with id/document/metadata/score) merged across queries, best first. score = best cosine seen."""
    rankings, by_id = [], {}
    for q in queries:
        hits = retriever._text_hits(q, k, document_id, filters)
        rankings.append([h["id"] for h in hits])
        for h in hits:
            if h["id"] not in by_id or h["score"] > by_id[h["id"]]["score"]:
                by_id[h["id"]] = h
    return [by_id[i] for i, _ in rrf_fuse(rankings, get_settings().rrf_k) if i in by_id]


def _to_text(row: Dict, score: float = 0.0) -> RetrievedText:
    return RetrievedText(chunk_id=row["id"], page_number=row["page"], text=row["text"], score=round(score, 4), document_id=row.get("doc", ""))


def _hit_to_text(h: Dict) -> RetrievedText:
    return RetrievedText(chunk_id=h["id"], page_number=int(h["metadata"].get("page_number", 0)), text=h["document"], score=round(h["score"], 4),
                         document_id=h["metadata"].get("document_id", ""), lexical_rank=h.get("lexical_rank"), dense_rank=h.get("dense_rank"))


def document_rows(document_id: str) -> List[Dict]:
    """All passages of one document in reading order."""
    corpus = vector_store.text_corpus(document_id)
    rows = [{"id": i, "text": t, "page": int((m or {}).get("page_number", 0)), "doc": (m or {}).get("document_id", document_id)}
            for i, t, m in zip(corpus["ids"], corpus["documents"], corpus["metadatas"])]
    rows.sort(key=lambda r: (r["page"], _natkey(r["id"])))
    return rows


def fits_whole(rows: List[Dict]) -> bool:
    return bool(rows) and sum(len(r["text"]) for r in rows) <= get_settings().context_full_doc_chars


def build_context(document_id: Optional[str], hits: List[Dict], question_type: str, rows: Optional[List[Dict]] = None) -> Tuple[List[RetrievedText], str]:
    """Returns (passages for the LLM in reading order, mode)."""
    s = get_settings()
    top_score = max((h["score"] for h in hits), default=0.0)
    if document_id is None:  # cross-document: the fused hits themselves, best first
        return [_hit_to_text(h) for h in hits[:10]], "retrieved"

    rows = rows if rows is not None else document_rows(document_id)
    if not rows:
        return [_hit_to_text(h) for h in hits[:s.context_top_k]], "retrieved"
    total = sum(len(r["text"]) for r in rows)
    broad = question_type in BROAD_TYPES
    # a document that fits the budget is always sent whole: nothing relevant can be missed, whatever the question
    if total <= s.context_full_doc_chars:
        score = {h["id"]: h["score"] for h in hits}
        return [_to_text(r, score.get(r["id"], 0.0)) for r in rows], "full_document"

    # large document or a pointed question: best passages + one neighbour either side, in reading order, within budget
    pos = {r["id"]: i for i, r in enumerate(rows)}
    keep: Dict[int, float] = {}
    for h in hits[:s.context_top_k]:
        i = pos.get(h["id"])
        if i is None:
            continue
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(rows):
                keep[j] = max(keep.get(j, 0.0), h["score"] if j == i else 0.0)
    out, used = [], 0
    for j in sorted(keep):
        used += len(rows[j]["text"])
        if used > s.context_retrieved_chars:
            break
        out.append(_to_text(rows[j], keep[j]))
    return out, "retrieved"


def top_evidence(hits: List[Dict], n: int = 8) -> List[RetrievedText]:
    """The most relevant passages (for the UI's evidence list and citations)."""
    return [_hit_to_text(h) for h in hits[:n]]
