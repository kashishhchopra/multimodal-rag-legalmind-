"""CORE feature: plain-English translation toggle.

Cheap to build relative to the other features, and it's what makes a live
demo land with judges who are not lawyers — the same grounded clause, shown
in the language a non-lawyer actually reads.
"""
from __future__ import annotations

from src.generation.answerer import call_llm
from src.generation.prompts import PLAIN_ENGLISH_PROMPT


def translate_to_plain_english(clause_text: str) -> str:
    prompt = PLAIN_ENGLISH_PROMPT.format(clause_text=clause_text)
    return call_llm(prompt).strip()
