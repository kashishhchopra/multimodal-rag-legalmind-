"""Query orchestration: retrieve -> assemble prompt -> generate grounded answer.

If the LLM key is missing or the provider fails, we still return the retrieved
text + visual evidence and a clear warning, so the app degrades gracefully and
the *retrieval* half of the demo works without any API key (spec §16).
"""
from __future__ import annotations

import logging
import re
import time
from typing import List

from PIL import Image

from ..config import get_settings
from ..retrieval import vector_store
from ..retrieval.retriever import MultimodalRetriever
from ..legal import telemetry
from ..legal.verify import retrieval_gate, verify_answer
from ..schemas import Citation, ClaimVerification, QueryResponse, SourceKind, VerificationReport
from ..utils.errors import LLMError, MissingAPIKeyError
from . import conversation
from . import llm as llm_module
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .context import build_context, document_rows, fits_whole, fuse_hits, top_evidence
from .question_judge import heuristic_profile, plan_question

logger = logging.getLogger("mmrag.query")

_retriever = MultimodalRetriever()


def _load_visual_images(visual_ids: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for vid in visual_ids:
        item = vector_store.get_by_id(vector_store.VISUAL_COLLECTION, vid)
        if not item:
            continue
        path = item["metadata"].get("image_path", "")
        if not path:
            continue
        try:
            images.append(Image.open(path).convert("RGB"))
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not load visual %s for LLM: %s", vid, exc)
    return images


def answer_question(
    document_id,
    question: str,
    top_k_text=None,
    top_k_visual=None,
    filters=None,
    history=None,
    plain_english: bool = False,
    language: str = "en",
) -> QueryResponse:
    """document_id=None runs a cross-document search over the whole corpus. Latency/status are logged (no text stored)."""
    t0 = time.perf_counter()
    resp = _answer_question(document_id, question, top_k_text, top_k_visual, filters, history, plain_english, language)
    try:
        telemetry.record(get_settings().data_dir, "corpus" if document_id is None else "document", (time.perf_counter() - t0) * 1000,
                         resp.warning or (resp.verification.status if resp.verification else "ok"), len(question) + sum(len(t.text) for t in resp.text_evidence),
                         len(resp.answer), len(resp.text_evidence))
    except Exception:  # noqa: BLE001 - telemetry must never break a query
        logger.warning("telemetry write failed", exc_info=True)
    return resp


def _answer_question(
    document_id, question, top_k_text, top_k_visual, filters, history, plain_english, language
) -> QueryResponse:
    settings = get_settings()
    try:  # one provider for planning, rewriting, translation and the answer; none -> heuristics + evidence-only
        provider = llm_module.get_provider()
    except (MissingAPIKeyError, LLMError):
        provider = None
    hist = [h if isinstance(h, dict) else h.model_dump() for h in (history or [])]
    retrieval_q = conversation.rewrite_followup(question, hist, provider)
    retrieval_q = conversation.to_english(retrieval_q, provider)
    if conversation.is_hindi(question):
        language = "hi"

    # LLM quota is scarce on free tiers (Gemini's is ~20 requests/day for some models), so spend ONE call per question when
    # possible. A document that fits the context budget is sent whole: no multi-query search is needed and the answer style
    # comes from the wording. Only large documents / corpus-wide search pay for a planning call (type + alternative queries).
    rows = document_rows(document_id) if document_id else None
    if rows is not None and fits_whole(rows):
        profile, extra = heuristic_profile(retrieval_q), []
    elif provider is not None:
        profile, extra = plan_question(retrieval_q, provider)
    else:
        profile, extra = heuristic_profile(retrieval_q), []
    queries = list(dict.fromkeys([retrieval_q, *extra]))[:4]

    result = _retriever.retrieve(document_id, retrieval_q, top_k_text, top_k_visual, filters)  # visuals, grounding, base text
    hits = fuse_hits(_retriever, queries, settings.context_top_k, document_id, filters)
    context, mode = build_context(document_id, hits, profile.question_type.value, rows)

    # The evidence list/citations shown to the user = the most relevant passages across all queries.
    shown = top_evidence(hits, 8) or result.text_evidence
    citations = [c for c in result.citations if c.kind != SourceKind.TEXT] + [
        Citation(kind=SourceKind.TEXT, source_id=t.chunk_id, page_number=t.page_number, label=f"Page {t.page_number}", score=t.score, snippet=t.text[:180])
        for t in shown
    ]
    response = QueryResponse(
        question=question, answer="", citations=citations, text_evidence=shown, visual_evidence=result.visual_evidence,
        combined_confidence=result.combined_confidence, provider=settings.llm_provider,
        retrieval_question=retrieval_q if retrieval_q != question else "", context_mode=mode, context_chunks=len(context),
    )

    if not context and not result.visual_evidence:
        response.answer = (
            "I couldn't find any text in the index for that question. Try rephrasing, or check that the document finished processing."
        )
        response.warning = "no_evidence"
        return response

    # The retrieval gate only protects cross-document search; for a question about ONE document the LLM is shown that
    # document's text (whole, or the best parts) and decides for itself whether it answers the question.
    if settings.guard_enabled and document_id is None and shown and not retrieval_gate([t.score for t in shown], settings.guard_min_top_score):
        logger.info("guard refused cross-document query: top score %.3f < %.2f", max(t.score for t in shown), settings.guard_min_top_score)
        response.answer = (
            "I could not find sufficiently relevant passages in the corpus to answer this reliably. "
            "Try rephrasing, naming the section, clause or party, or narrowing the filters."
        )
        response.warning = "low_retrieval_confidence"
        response.verification = VerificationReport(status="refused", support_ratio=0.0)
        return response

    if provider is None:
        response.answer = _evidence_only_answer(result, shown)
        response.warning = "missing_api_key"
        return response
    try:
        note = ("the complete document, in reading order" if mode == "full_document"
                else f"the {len(context)} passages most relevant to the question, with their neighbouring passages, in reading order")
        labels = _source_labels(context) if document_id is None else None  # corpus-wide: say which document each passage is from
        user_prompt = build_user_prompt(
            retrieval_q if language != "hi" else question, context, result.visual_evidence,
            (profile.length_instruction + " " + conversation.style_instruction(plain_english, language)).strip(), note, labels,
        )
        images = _load_visual_images([v.visual_id for v in result.visual_evidence])
        response.answer = _clean_answer(provider.generate(SYSTEM_PROMPT, user_prompt, images, max_output_tokens=profile.max_output_tokens))
        if settings.guard_enabled:
            v = verify_answer(
                response.answer,
                [{"id": t.chunk_id, "text": t.text, "page": t.page_number} for t in context],
                settings.guard_min_claim_support,
            )
            response.verification = VerificationReport(
                status=v.status, support_ratio=v.support_ratio, unsupported=v.unsupported,
                claims=[ClaimVerification(**c.__dict__) for c in v.claims],
            )
            if v.status == "flagged" and v.support_ratio < 0.6 and not response.warning:  # a few paraphrase flags are normal
                response.warning = "unsupported_claims"
    except MissingAPIKeyError:
        response.answer = _evidence_only_answer(result, shown)
        response.warning = "missing_api_key"
    except LLMError as exc:
        logger.error("LLM failed: %s", exc)
        response.answer = _evidence_only_answer(result, shown)
        quota = any(t in str(exc).lower() for t in ("429", "quota", "resource_exhausted", "rate limit", "insufficient"))
        response.warning = "llm_quota" if quota else "llm_failed"

    return response


def _source_labels(context) -> dict:
    """document_id -> short human label (case citation / title / act name) for cross-document answers."""
    from ..legal import corpus_catalog

    out = {}
    for t in context:
        d = t.document_id
        if d and d not in out:
            e = corpus_catalog._load().get(d)
            out[d] = (e["citation"] if e and e["kind"] == "judgment" and e["citation"] else e["title"] if e else d)[:60]
    return out


_LABEL = re.compile(r"\[(?:T|V)\d+\]\s*[;,]?\s*")
_PREFACE = re.compile(r"^(?:based on|according to) the (?:retrieved |supplied |provided )?(?:text )?evidence\s*[:,]\s*", re.I)


def _clean_answer(text: str) -> str:
    """Remove internal evidence labels ([T3]) the model sometimes echoes, and a filler first clause."""
    text = _LABEL.sub("", text)
    text = re.sub(r"\(\s*[;,]\s*", "(", text)      # "(, Page 3" left after a leading label
    text = re.sub(r"\s*[;,]\s*\)", ")", text)      # "Page 3; )" left after a trailing label
    text = re.sub(r"\(\s*\)", "", text)            # citations that only held a label
    text = _PREFACE.sub("", text.strip())
    return text[:1].upper() + text[1:] if text else text


def _evidence_only_answer(result, shown=None) -> str:
    """Fallback shown when no LLM is available: summarise the retrieved evidence."""
    lines = [
        "⚠️ No LLM answer was generated (the API key is missing or the provider "
        "failed), but here is the multimodal evidence the retriever found:",
        "",
    ]
    texts = shown or result.text_evidence
    if texts:
        lines.append("Relevant text:")
        for t in texts[:3]:
            lines.append(f"  • (Page {t.page_number}) {t.text[:200]}")
    if result.visual_evidence:
        lines.append("")
        lines.append("Relevant visuals:")
        for v in result.visual_evidence:
            label = v.figure_label or f"{v.image_type.value} on page {v.page_number}"
            lines.append(f"  • {label} (see image panel)")
    lines.append("")
    lines.append("Set LLM_API_KEY in backend/.env to enable full grounded answers.")
    return "\n".join(lines)
