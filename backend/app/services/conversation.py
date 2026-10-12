"""Follow-up question rewriting and language/plain-English handling.

* rewrite_followup: turns "what about the penalty?" into a standalone question using the recent chat turns. Uses the
  LLM when available; without one it falls back to a conservative heuristic (prepend the previous user question to a
  short, pronoun-led follow-up) so retrieval still has context.
* Hindi mode is translate-then-retrieve: a Hindi question is translated to English for retrieval, and the answer is
  requested in Hindi. Translation here is done by the configured LLM; IndicTrans2 is the planned open replacement and
  has not been evaluated against it yet (roadmap 6.6).
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger("mmrag.conversation")

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_FOLLOWUP = re.compile(r"^(and|what about|how about|why|also|then|what if|does it|is it|can it|who|when|where)\b|\b(it|this|that|they|them|those|these|the same)\b", re.I)

PLAIN_ENGLISH = (
    "Write the answer in plain English for a non-lawyer: short sentences, no Latin phrases, explain any legal term in "
    "brackets the first time it appears. Keep every citation."
)
HINDI = (
    "Write the final answer in Hindi (Devanagari). Keep statute names, section numbers, case names and citations exactly as "
    "in the evidence (in English/Latin script)."
)


def is_hindi(text: str) -> bool:
    return len(_DEVANAGARI.findall(text)) >= max(3, len(text) // 5)


def rewrite_followup(question: str, history: List[dict], provider=None) -> str:
    """history: [{"role": "user"|"assistant", "content": str}], oldest first."""
    prior_users = [h["content"] for h in history if h.get("role") == "user"]
    if not prior_users or not _FOLLOWUP.search(question):
        return question
    if provider is not None:
        try:
            turns = "\n".join(f"{h['role'].upper()}: {h['content'][:400]}" for h in history[-6:])
            prompt = (
                "Rewrite the final user question as ONE standalone question that can be understood without the "
                "conversation. Keep every legal term, section number and party name. Output only the question.\n\n"
                f"{turns}\nUSER: {question}\n\nStandalone question:"
            )
            out = provider.generate("You rewrite follow-up questions for retrieval.", prompt, [], max_output_tokens=120).strip()
            if 5 < len(out) < 400:
                return out.splitlines()[0]
        except Exception as exc:  # noqa: BLE001 - never fail the query because rewriting failed
            logger.warning("follow-up rewrite failed, using heuristic: %s", exc)
    return f"{prior_users[-1]} {question}"


def to_english(question: str, provider) -> str:
    """Translate a Hindi question to English for retrieval; returns the input unchanged on any failure."""
    if provider is None or not is_hindi(question):
        return question
    try:
        out = provider.generate("You are a precise Hindi-to-English legal translator.",
                                f"Translate to English. Keep names and section numbers. Output only the translation.\n\n{question}",
                                [], max_output_tokens=200).strip()
        return out or question
    except Exception as exc:  # noqa: BLE001
        logger.warning("translation failed: %s", exc)
        return question


def style_instruction(plain_english: bool, language: str) -> str:
    parts: List[str] = []
    if plain_english:
        parts.append(PLAIN_ENGLISH)
    if language == "hi":
        parts.append(HINDI)
    return " ".join(parts)
