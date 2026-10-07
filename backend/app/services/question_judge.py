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
            "This is a broad overview question. Answer in 2-4 sentences at a high "
            "level (what the document is, who's involved, what it's about). Do not "
            "use a bullet list."
        ),
        max_output_tokens=220,
    ),
    QuestionType.FACTUAL: QuestionProfile(
        question_type=QuestionType.FACTUAL,
        length_instruction=(
            "This is a narrow factual question. Answer in 1-2 sentences: state the "
            "direct fact first, then its citation. No preamble, no restating the question."
        ),
        max_output_tokens=150,
    ),
    QuestionType.LIST: QuestionProfile(
        question_type=QuestionType.LIST,
        length_instruction=(
            "This question asks for an enumeration. Answer as a short bulleted list, "
            "one point per bullet, each with its own citation."
        ),
        max_output_tokens=400,
    ),
    QuestionType.ANALYTICAL: QuestionProfile(
        question_type=QuestionType.ANALYTICAL,
        length_instruction=(
            "This question asks for analysis or comparison. Answer in a short "
            "reasoned explanation (up to about 6 sentences), citing the evidence "
            "each point relies on."
        ),
        max_output_tokens=350,
    ),
    QuestionType.YES_NO: QuestionProfile(
        question_type=QuestionType.YES_NO,
        length_instruction=(
            "This is a yes/no question. Start with 'Yes', 'No', or 'Unclear from the "
            "evidence', then give 1-2 sentences of justification with a citation."
        ),
        max_output_tokens=150,
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
