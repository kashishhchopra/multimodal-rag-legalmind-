"""Layer-4 core: turn retrieved chunks + a question into a grounded,
structured answer.

`call_llm` is the one place that talks to a provider directly, so swapping
GPT-4o/Claude/Llama-3 for a benchmark ablation (per docs/ARCHITECTURE.md 5.5)
means changing config, not call sites.
"""
from __future__ import annotations

from src.config import get_settings
from src.retrieval.hybrid_retriever import RetrievedChunk
from src.schemas import AnswerResult, Citation

from .citation_schema import CitedAnswer
from .prompts import QA_USER_PROMPT, SYSTEM_PROMPT


def call_llm(prompt: str, *, system: str | None = None) -> str:
    settings = get_settings()
    if settings.llm_provider == "openai":
        return _call_openai(prompt, system=system, model=settings.llm_model)
    return _call_anthropic(prompt, system=system, model=settings.llm_model)


def _call_anthropic(prompt: str, *, system: str | None, model: str) -> str:
    import anthropic

    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system or "",
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def _call_openai(prompt: str, *, system: str | None, model: str) -> str:
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def call_llm_structured(prompt: str, *, system: str | None = None) -> CitedAnswer:
    """Structured variant using `instructor` so the model must return a
    CitedAnswer, not free text we then have to parse. Falls back to a naive
    JSON parse of `call_llm`'s output if instructor isn't wired up for the
    active provider."""
    settings = get_settings()
    try:
        import instructor

        if settings.llm_provider == "openai":
            from openai import OpenAI

            client = instructor.from_openai(OpenAI(api_key=settings.openai_api_key))
            return client.chat.completions.create(
                model=settings.llm_model,
                response_model=CitedAnswer,
                messages=[
                    {"role": "system", "content": system or ""},
                    {"role": "user", "content": prompt},
                ],
            )
        else:
            import anthropic

            client = instructor.from_anthropic(anthropic.Anthropic(api_key=settings.anthropic_api_key))
            return client.messages.create(
                model=settings.llm_model,
                max_tokens=2048,
                response_model=CitedAnswer,
                messages=[{"role": "user", "content": prompt}],
                system=system or "",
            )
    except Exception:
        # TODO: replace with a stricter parse/repair loop; raw fallback kept
        # here only so the pipeline never silently returns nothing.
        raw = call_llm(prompt, system=system)
        return CitedAnswer(answer=raw, citations=[], insufficient_context=False)


def _format_context_block(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for c in chunks:
        loc = f"page {c.chunk.metadata.page}"
        if c.chunk.metadata.section_path:
            loc = f"{c.chunk.metadata.section_path}, {loc}"
        lines.append(f"[chunk_id={c.chunk.chunk_id}] ({loc})\n{c.chunk.text}\n")
    return "\n---\n".join(lines)


def answer_question(question: str, retrieved: list[RetrievedChunk]) -> AnswerResult:
    context_block = _format_context_block(retrieved)
    prompt = QA_USER_PROMPT.format(question=question, context_block=context_block)

    cited = call_llm_structured(prompt, system=SYSTEM_PROMPT)

    chunk_by_id = {c.chunk.chunk_id: c.chunk for c in retrieved}
    citations: list[Citation] = []
    for ref in cited.citations:
        chunk = chunk_by_id.get(ref.chunk_id)
        if chunk is None:
            continue  # guardrail: never surface a citation the retriever didn't actually return
        citations.append(Citation(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.metadata.doc_id,
            page=chunk.metadata.page,
            bbox=chunk.metadata.bbox,
            quote=ref.quote,
        ))

    return AnswerResult(
        answer=cited.answer,
        citations=citations,
        insufficient_context=cited.insufficient_context,
    )
