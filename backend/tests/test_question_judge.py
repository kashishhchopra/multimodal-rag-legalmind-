"""Unit tests for the question-type judge — no LLM/API key required, since
the provider is stubbed to return a scripted label."""
from __future__ import annotations

import pytest

from app.services.question_judge import (
    DEFAULT_PROFILE,
    QuestionType,
    judge_question,
)


class _StubProvider:
    """Minimal LLMProvider stand-in: returns whatever label was scripted,
    or raises if `raise_error` is set."""

    def __init__(self, label: str = "", raise_error: bool = False):
        self._label = label
        self._raise_error = raise_error

    def generate(self, system_prompt, user_prompt, images, max_output_tokens=None):
        if self._raise_error:
            raise RuntimeError("simulated provider failure")
        return self._label


@pytest.mark.parametrize(
    "label,expected_type",
    [
        ("overview", QuestionType.OVERVIEW),
        ("Factual", QuestionType.FACTUAL),
        ("  list  ", QuestionType.LIST),
        ("ANALYTICAL", QuestionType.ANALYTICAL),
        ("yes_no", QuestionType.YES_NO),
    ],
)
def test_judge_question_maps_each_label(label, expected_type):
    provider = _StubProvider(label=label)
    profile = judge_question("irrelevant question text", provider)
    assert profile.question_type == expected_type


def test_judge_question_falls_back_to_default_on_garbled_output():
    provider = _StubProvider(label="I'm not sure, maybe several things?")
    profile = judge_question("some question", provider)
    assert profile == DEFAULT_PROFILE


def test_judge_question_falls_back_to_default_on_empty_output():
    provider = _StubProvider(label="")
    profile = judge_question("some question", provider)
    assert profile == DEFAULT_PROFILE


def test_judge_question_falls_back_to_default_on_provider_exception():
    provider = _StubProvider(raise_error=True)
    profile = judge_question("some question", provider)
    assert profile == DEFAULT_PROFILE


def test_every_profile_has_positive_max_output_tokens():
    from app.services.question_judge import QUESTION_PROFILES

    for question_type, profile in QUESTION_PROFILES.items():
        assert profile.question_type == question_type
        assert profile.max_output_tokens > 0
        assert profile.length_instruction
