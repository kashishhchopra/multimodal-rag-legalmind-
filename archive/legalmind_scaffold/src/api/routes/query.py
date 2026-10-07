"""Layer-3/4 entry point over HTTP: the grounded Q&A endpoint.

This is the CORE feature — everything else in features/ builds on the same
hybrid_search -> rerank -> answer_question path.
"""
from __future__ import annotations

from pydantic import BaseModel

from fastapi import APIRouter

from src.generation.answerer import answer_question
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.query_rewrite import rewrite_query
from src.retrieval.reranker import rerank
from src.schemas import AnswerResult

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None
    jurisdiction: str | None = None
    doc_type: str | None = None


@router.post("", response_model=AnswerResult)
def run_query(request: QueryRequest) -> AnswerResult:
    variants = rewrite_query(request.question)

    all_candidates = []
    for variant in variants:
        all_candidates.extend(hybrid_search(
            variant,
            doc_ids=request.doc_ids,
            jurisdiction=request.jurisdiction,
            doc_type=request.doc_type,
        ))

    # De-dup by chunk_id before reranking (multi-query expansion can surface
    # the same chunk more than once).
    unique_by_id = {c.chunk.chunk_id: c for c in all_candidates}
    reranked = rerank(request.question, list(unique_by_id.values()))

    return answer_question(request.question, reranked)
