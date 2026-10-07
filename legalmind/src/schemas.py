"""Shared data contracts used across every layer.

Keeping these in one module (rather than redefining ad hoc dicts in each
package) is what lets the bounding-box / page metadata attached at ingestion
time travel unchanged through chunking, indexing, retrieval, and generation —
see "grounding is carried in metadata, not re-derived" in docs/ARCHITECTURE.md.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ParsePath(str, Enum):
    """Which ingestion route produced a page's text — logged for evaluation."""

    NATIVE = "native"       # extractable text layer, parsed deterministically
    OCR = "ocr"             # scanned page, ran through Tesseract/PaddleOCR
    VISION = "vision"       # OCR failed/low-confidence, ran through a vision LLM


class BoundingBox(BaseModel):
    """Pixel or point coordinates of a region on a page, top-left origin."""

    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float


class TableBlock(BaseModel):
    page: int
    bbox: BoundingBox
    rows: list[list[str]]
    caption: str | None = None


class SignatureBlock(BaseModel):
    page: int
    bbox: BoundingBox
    kind: Literal["signature", "stamp", "initials"] = "signature"
    confidence: float = 0.0


class ParsedPage(BaseModel):
    """Output of layer 1 (ingestion) for a single page."""

    doc_id: str
    page_number: int
    text: str
    parse_path: ParsePath
    tables: list[TableBlock] = Field(default_factory=list)
    signatures: list[SignatureBlock] = Field(default_factory=list)
    page_width: float = 0.0
    page_height: float = 0.0
    # per-line/word boxes, used to localise a chunk's bbox within the page
    word_boxes: list[tuple[str, BoundingBox]] = Field(default_factory=list)


class ChunkMetadata(BaseModel):
    """Attached to every chunk at layer 2 (chunking) and carried unchanged
    through indexing and retrieval into the final citation."""

    doc_id: str
    doc_type: str | None = None            # e.g. "NDA", "lease", "judgment"
    jurisdiction: str | None = None
    section_path: str | None = None        # e.g. "12 > 12(b) > Termination for Cause"
    page: int
    bbox: BoundingBox | None = None
    parse_path: ParsePath = ParsePath.NATIVE


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class Citation(BaseModel):
    """What the LLM cites and what the UI needs to draw a highlight."""

    chunk_id: str
    doc_id: str
    page: int
    bbox: BoundingBox | None = None
    quote: str


class AnswerResult(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_context: bool = False
