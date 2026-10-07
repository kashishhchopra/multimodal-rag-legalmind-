"""Structured-output schema for grounded answers.

The LLM is asked to return exactly this shape (via function-calling /
Instructor), never free text — that's what makes reliable citation and
highlighting possible instead of trying to regex a citation out of prose.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CitationRef(BaseModel):
    chunk_id: str = Field(description="The exact chunk_id this claim relies on")
    quote: str = Field(description="The verbatim short quote from the chunk supporting the claim")


class CitedAnswer(BaseModel):
    answer: str = Field(description="The answer to the user's question, grounded only in the provided context")
    citations: list[CitationRef] = Field(default_factory=list)
    insufficient_context: bool = Field(
        default=False,
        description="True if the provided context does not contain enough information to answer confidently",
    )
