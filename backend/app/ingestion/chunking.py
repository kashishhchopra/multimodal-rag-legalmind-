"""Text chunking.

Splits page text into overlapping, sentence-aware chunks so that:
  * each chunk is small enough to embed meaningfully, and
  * we keep the page number for every chunk (needed for citations).

A simple, dependency-free splitter is intentional — it is easy to explain in a
viva and behaves predictably. Overlap preserves context across chunk borders.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    page_number: int


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def chunk_page_text(
    text: str,
    page_number: int,
    max_chars: int = 800,
    overlap_chars: int = 150,
) -> List[Chunk]:
    """Chunk one page's text into overlapping windows on sentence boundaries."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(Chunk(text=current.strip(), page_number=page_number))
            # start next chunk with a tail overlap for context continuity
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = (tail + " " + sentence).strip()
        else:
            current = (current + " " + sentence).strip()
    if current.strip():
        chunks.append(Chunk(text=current.strip(), page_number=page_number))
    return chunks
