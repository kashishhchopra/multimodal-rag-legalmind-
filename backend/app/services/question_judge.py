"""Judge the *type* of a user's question so the answer's length and shape can
be calibrated to match it, instead of every question getting the same
undifferentiated treatment.

This is deliberately a runtime judgment, not a lookup: `judge_question` asks
the same LLM provider that will answer the question to classify it — purely
from the question text in front of it — into one of a handful of general
shapes (overview / factual / list / analytical / yes-no). Nothing here is a
pre-written question or a stored answer; the mapping below only says *how
long/shaped an answer of each type should be*, not *what the answer is*.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMProvider

logger = logging.getLogger("mmrag.question_judge")


class QuestionType(str, Enum):
    OVERVIEW = "overview"       # "what is this about", "summarise this document"
    FACTUAL = "factual"         # "what is the notice period", "who are the parties"
    LIST = "list"               # "list all clauses about X", "what are the risks"
    ANALYTICAL = "analytical"   # "does this conflict with...", "compare X and Y"
    YES_NO = "yes_no"           # "can I terminate early", "is there a penalty"


@dataclass(frozen=True)
class QuestionProfile:
    question_type: QuestionType
    length_instruction: str
    max_output_tokens: int


QUESTION_PROFILES: dict[QuestionType, QuestionProfile] = {
    QuestionType.OVERVIEW: QuestionProfile(
        question_type=QuestionType.OVERVIEW,
        length_instruction=(
            "This is a broad question. Give a clear, complete overview in a few short paragraphs or bullets: what "
            "the document is, who is involved (parties, court, bench), the key issues or terms, and the outcome or "
            "main obligations. Cover the whole document, citing pages."
        ),
        max_output_tokens=1100,
    ),
    QuestionType.FACTUAL: QuestionProfile(
        question_type=QuestionType.FACTUAL,
        length_instruction=(
            "This is a specific question. State the direct answer first with its citation, then add the relevant "
            "supporting detail from the document (context, conditions, exceptions, related provisions) so the answer "
            "is complete rather than a bare fact."
        ),
        max_output_tokens=700,
    ),
    QuestionType.LIST: QuestionProfile(
        question_type=QuestionType.LIST,
        length_instruction=(
            "This question asks for an enumeration. Give a COMPLETE bulleted list of every matching item in the supplied "
            "text (do not stop at a few), one point per bullet with a one-line explanation and its citation."
        ),
        max_output_tokens=1400,
    ),
    QuestionType.ANALYTICAL: QuestionProfile(
        question_type=QuestionType.ANALYTICAL,
        length_instruction=(
            "This question asks for analysis or reasoning. Give a thorough, well-structured explanation: the relevant "
            "text, how it applies, and the conclusion, citing the evidence each point relies on."
        ),
        max_output_tokens=1400,
    ),
    QuestionType.YES_NO: QuestionProfile(
        question_type=QuestionType.YES_NO,
        length_instruction=(
            "This is a yes/no question. Start with 'Yes', 'No', 'Partly' or 'Not stated in the document', then explain "
            "with the supporting text and its citation."
        ),
        max_output_tokens=700,
    ),
}

# Used whenever classification fails or returns something unrecognized — a
# safe, moderate middle ground rather than defaulting to either extreme.
DEFAULT_PROFILE = QUESTION_PROFILES[QuestionType.ANALYTICAL]

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict question-type classifier. Given a user's question about a "
    "legal document, respond with EXACTLY ONE WORD from this list, and nothing else "
    "— no punctuation, no explanation:\n"
    "overview | factual | list | analytical | yes_no\n\n"
    "- overview: asks broadly what the document/case is about, or to summarise it.\n"
    "- factual: asks for one specific fact, term, date, name, or number.\n"
    "- list: asks to enumerate multiple items (\"list all...\", \"what are the...\").\n"
    "- analytical: asks for comparison, conflict-checking, reasoning, or \"why\"/\"how\".\n"
    "- yes_no: a question that is naturally answered yes or no.\n"
)


def judge_question(question: str, provider: "LLMProvider") -> QuestionProfile:
    """Classify `question` using `provider`, returning the matching
    QuestionProfile. Never raises — any failure or unrecognized output falls
    back to DEFAULT_PROFILE so a broken judge call never breaks the actual
    answer generation that follows it."""
    try:
        raw = provider.generate(_JUDGE_SYSTEM_PROMPT, question, images=[], max_output_tokens=10)
    except Exception as exc:
        logger.warning("Question judging failed, falling back to default profile: %s", exc)
        return DEFAULT_PROFILE

    label = (raw or "").strip().lower()
    for question_type in QuestionType:
        if question_type.value in label:
            return QUESTION_PROFILES[question_type]

    logger.info("Question judge returned unrecognized label %r, using default profile", raw)
    return DEFAULT_PROFILE


_PLAN_SYSTEM_PROMPT = (
    "You prepare a question about a legal document for retrieval. Reply with ONLY a JSON object, no prose:\n"
    '{"type": "overview|factual|list|analytical|yes_no", "queries": ["...", "..."]}\n'
    "type: overview = asks what the document/case is about or to summarise it; factual = one specific fact; "
    "list = enumerate several items; analytical = reasoning, comparison, why/how; yes_no = answered yes or no.\n"
    "queries: 2 or 3 DIFFERENT search queries that would find the answer's passages in the document (use key terms, "
    "synonyms, section/clause names, likely wording in the text; do not answer the question)."
)


def plan_question(question: str, provider: "LLMProvider") -> tuple[QuestionProfile, list[str]]:
    """One LLM call returning the question's type and extra search queries (multi-query retrieval). Never raises:
    on any failure it returns the default profile and no extra queries, so retrieval simply uses the question itself."""
    import json
    import re

    try:
        raw = provider.generate(_PLAN_SYSTEM_PROMPT, question, images=[], max_output_tokens=300)
        m = re.search(r"\{.*\}", raw or "", re.S)
        data = json.loads(m.group(0)) if m else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Question planning failed, using defaults: %s", exc)
        return DEFAULT_PROFILE, []
    profile = DEFAULT_PROFILE
    label = str(data.get("type", "")).strip().lower()
    for qt in QuestionType:
        if qt.value == label:
            profile = QUESTION_PROFILES[qt]
    queries = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and 3 < len(q.strip()) < 300][:3]
    return profile, queries


_OVERVIEW_RX = None


def heuristic_profile(question: str) -> QuestionProfile:
    """Choose the answer style from the question's wording alone (no LLM call). Used when the whole document is sent to
    the model anyway, so a separate planning call would only spend quota. Wording rules decide format, never content."""
    import re

    q = question.lower().strip()
    if re.search(r"\b(summari[sz]e|summary|overview|in brief|key points|what is (this|the) (document|case|judgment|agreement|act)|"
                 r"what('s| is) .* about|tell me about|explain this (document|case|judgment|agreement))\b", q):
        return QUESTION_PROFILES[QuestionType.OVERVIEW]
    if re.search(r"\b(list|enumerate|all the|what are (the|all)|which (clauses|provisions|sections|cases|statutes|authorities|parties))\b", q):
        return QUESTION_PROFILES[QuestionType.LIST]
    if re.match(r"(is|are|does|do|did|can|could|will|would|was|were|has|have|should|may)\b", q):
        return QUESTION_PROFILES[QuestionType.YES_NO]
    if re.search(r"\b(why|how|compare|comparison|reason|reasoning|analy[sz]e|difference|conflict|implication|impact|explain)\b", q):
        return QUESTION_PROFILES[QuestionType.ANALYTICAL]
    return QUESTION_PROFILES[QuestionType.FACTUAL]
