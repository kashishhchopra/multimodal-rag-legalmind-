"""Query orchestration: retrieve -> assemble prompt -> generate grounded answer.

If the LLM key is missing or the provider fails, we still return the retrieved
text + visual evidence and a clear warning, so the app degrades gracefully and
the *retrieval* half of the demo works without any API key (spec §16).
"""
from __future__ import annotations

import logging
from typing import List

from PIL import Image

from ..config import get_settings
from ..retrieval import vector_store
from ..retrieval.retriever import MultimodalRetriever
from ..legal.verify import retrieval_gate, verify_answer
from ..schemas import ClaimVerification, QueryResponse, VerificationReport
from ..utils.errors import LLMError, MissingAPIKeyError
from . import conversation
from . import llm as llm_module
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .question_judge import DEFAULT_PROFILE, judge_question

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
    """document_id=None runs a cross-document search over the whole corpus."""
    settings = get_settings()
    try:  # an LLM is optional for rewriting/translation: heuristics apply without one
        helper = llm_module.get_provider() if (history or conversation.is_hindi(question)) else None
    except (MissingAPIKeyError, LLMError):
        helper = None
    retrieval_q = conversation.rewrite_followup(question, [h if isinstance(h, dict) else h.model_dump() for h in (history or [])], helper)
    retrieval_q = conversation.to_english(retrieval_q, helper)
    if conversation.is_hindi(question):
        language = "hi"
    result = _retriever.retrieve(document_id, retrieval_q, top_k_text, top_k_visual, filters)

    response = QueryResponse(
        question=question,
        answer="",
        citations=result.citations,
        text_evidence=result.text_evidence,
        visual_evidence=result.visual_evidence,
        combined_confidence=result.combined_confidence,
        provider=settings.llm_provider,
        retrieval_question=retrieval_q if retrieval_q != question else "",
    )

    # No evidence at all → be honest.
    if not result.text_evidence and not result.visual_evidence:
        response.answer = (
            "I couldn't find any relevant text or visual evidence in this document "
            "for that question. Try rephrasing, or ask about content you know is in the file."
        )
        response.warning = "no_evidence"
        return response

    if settings.guard_enabled and result.text_evidence and not retrieval_gate(
        [t.score for t in result.text_evidence], settings.guard_min_top_score
    ):
        # Weak retrieval: say so instead of letting the LLM improvise from irrelevant text.
        logger.info("guard refused: top score %.3f < %.2f", max(t.score for t in result.text_evidence), settings.guard_min_top_score)
        response.answer = (
            "I could not find sufficiently relevant evidence in the indexed text to answer this reliably. "
            "Try rephrasing, naming the section/clause or party, or narrowing the filters."
        )
        response.warning = "low_retrieval_confidence"
        response.verification = VerificationReport(status="refused", support_ratio=0.0)
        return response

    try:
        provider = llm_module.get_provider()

        # Judge the question's shape (overview/factual/list/analytical/yes-no)
        # so the real answer is calibrated to what was actually asked instead
        # of every question getting the same generic treatment. A failure
        # here must never break answer generation, so it's isolated from the
        # main try/except below and always yields a usable profile.
        try:
            profile = judge_question(question, provider)
        except Exception as exc:  # pragma: no cover - defensive, judge_question already catches internally
            logger.warning("Unexpected error judging question, using default profile: %s", exc)
            profile = DEFAULT_PROFILE

        user_prompt = build_user_prompt(
            retrieval_q if language != "hi" else question, result.text_evidence, result.visual_evidence,
            (profile.length_instruction + " " + conversation.style_instruction(plain_english, language)).strip(),
        )
        images = _load_visual_images([v.visual_id for v in result.visual_evidence])
        response.answer = provider.generate(
            SYSTEM_PROMPT, user_prompt, images, max_output_tokens=profile.max_output_tokens
        )
        if settings.guard_enabled:
            v = verify_answer(
                response.answer,
                [{"id": t.chunk_id, "text": t.text, "page": t.page_number} for t in result.text_evidence],
                settings.guard_min_claim_support,
            )
            response.verification = VerificationReport(
                status=v.status, support_ratio=v.support_ratio, unsupported=v.unsupported,
                claims=[ClaimVerification(**c.__dict__) for c in v.claims],
            )
            if v.status == "flagged" and not response.warning:
                response.warning = "unsupported_claims"
    except MissingAPIKeyError:
        response.answer = _evidence_only_answer(result)
        response.warning = "missing_api_key"
    except LLMError as exc:
        logger.error("LLM failed: %s", exc)
        response.answer = _evidence_only_answer(result)
        response.warning = "llm_failed"

    return response


def _evidence_only_answer(result) -> str:
    """Fallback shown when no LLM is available: summarise the retrieved evidence."""
    lines = [
        "⚠️ No LLM answer was generated (the API key is missing or the provider "
        "failed), but here is the multimodal evidence the retriever found:",
        "",
    ]
    if result.text_evidence:
        lines.append("Relevant text:")
        for t in result.text_evidence[:3]:
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
