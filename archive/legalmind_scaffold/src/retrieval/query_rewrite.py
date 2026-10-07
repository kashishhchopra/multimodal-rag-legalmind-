"""Query rewriting / multi-query expansion to improve recall ahead of hybrid
search — e.g. expanding "termination notice" to also search "notice period",
"early termination", "right to terminate".
"""
from __future__ import annotations

from src.generation.prompts import QUERY_REWRITE_PROMPT


def rewrite_query(query: str, *, llm_call=None) -> list[str]:
    """Return the original query plus up to 3 paraphrases/expansions.

    `llm_call` is injected (rather than imported directly) so this module
    stays easy to unit test without a live LLM; see
    src/generation/answerer.py for the default provider-agnostic wrapper.
    """
    if llm_call is None:
        from src.generation.answerer import call_llm as llm_call

    response = llm_call(QUERY_REWRITE_PROMPT.format(query=query))
    variants = [line.strip("- ").strip() for line in response.splitlines() if line.strip()]
    return [query] + variants[:3]
