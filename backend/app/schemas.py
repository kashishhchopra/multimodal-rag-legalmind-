"""Pydantic data models — the contract between backend and frontend.

Mirrors the data model described in the project spec (section 9):
Document, TextChunk, VisualElement, Region, plus the query/answer shapes.
Embeddings themselves are NOT carried in these API models (they live in the
vector store); we expose only the metadata needed by the UI.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------
class ProcessingStage(str, Enum):
    QUEUED = "queued"
    READING = "reading"
    EXTRACTING_TEXT = "extracting_text"
    DETECTING_FIGURES = "detecting_figures"
    EXTRACTING_IMAGES = "extracting_images"
    OCR = "ocr"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class VisualType(str, Enum):
    IMAGE = "image"
    FIGURE = "figure"
    DIAGRAM = "diagram"
    CHART = "chart"
    TABLE = "table"
    SCHEMATIC = "schematic"


class SourceKind(str, Enum):
    TEXT = "text"
    VISUAL = "visual"
    REGION = "region"


# --------------------------------------------------------------------------
# Core data model
# --------------------------------------------------------------------------
class BoundingBox(BaseModel):
    """Coordinates in PDF points, origin top-left, relative to the page."""
    x0: float
    y0: float
    x1: float
    y1: float


class TextChunk(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    text: str
    char_count: int = 0


class VisualElement(BaseModel):
    visual_id: str
    document_id: str
    page_number: int
    image_path: str          # relative path served via /static
    image_url: str = ""      # convenience URL for the frontend
    image_type: VisualType = VisualType.IMAGE
    bounding_box: Optional[BoundingBox] = None
    ocr_text: str = ""
    caption: str = ""        # nearby caption text if detected (e.g. "Figure 12")
    figure_label: str = ""   # e.g. "Figure 12" / "Table 3"


class Document(BaseModel):
    document_id: str
    filename: str
    upload_time: str
    page_count: int = 0
    text_chunk_count: int = 0
    figure_count: int = 0
    image_count: int = 0
    region_count: int = 0
    stage: ProcessingStage = ProcessingStage.QUEUED
    progress: float = 0.0            # 0..1
    stage_message: str = ""
    error: Optional[str] = None


# --------------------------------------------------------------------------
# Query / answer
# --------------------------------------------------------------------------
class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=4000)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k_text: Optional[int] = None
    top_k_visual: Optional[int] = None
    history: List[ChatTurn] = Field(default_factory=list, max_length=20)   # earlier turns, oldest first
    plain_english: bool = False
    language: str = Field(default="en", pattern="^(en|hi)$")


class Citation(BaseModel):
    kind: SourceKind
    source_id: str
    page_number: int
    label: str               # human-readable, e.g. "Page 45" or "Figure 12"
    score: float = 0.0
    snippet: str = ""        # text preview or OCR preview
    image_url: str = ""      # for visual/region citations
    figure_label: str = ""


class RetrievedText(BaseModel):
    chunk_id: str
    page_number: int
    text: str
    score: float
    document_id: str = ""      # set on cross-document search so results can be attributed
    lexical_rank: Optional[int] = None  # 1-based BM25 rank when hybrid retrieval is on
    dense_rank: Optional[int] = None


class RetrievedVisual(BaseModel):
    visual_id: str
    page_number: int
    image_url: str
    image_type: VisualType
    figure_label: str = ""
    ocr_preview: str = ""
    score: float
    bounding_box: Optional[BoundingBox] = None


class ClaimVerification(BaseModel):
    claim: str
    support: float
    best_source: Optional[str] = None
    missing_numbers: list = Field(default_factory=list)
    bad_page_refs: list = Field(default_factory=list)
    supported: bool = True


class VerificationReport(BaseModel):
    status: str                    # ok | flagged | no_claims | refused
    support_ratio: float = 1.0
    claims: List[ClaimVerification] = Field(default_factory=list)
    unsupported: List[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation] = []
    text_evidence: List[RetrievedText] = []
    visual_evidence: List[RetrievedVisual] = []
    combined_confidence: float = 0.0     # heuristic 0..1, see retrieval.py
    provider: str = ""
    warning: str = ""                    # e.g. "LLM key missing — evidence only"
    verification: Optional[VerificationReport] = None
    retrieval_question: str = ""         # the standalone/translated question actually used for retrieval
    context_mode: str = ""               # "full_document" | "retrieved": how much of the document the LLM was shown
    context_chunks: int = 0              # number of passages given to the LLM


class SearchRequest(BaseModel):
    """Cross-document search over the whole corpus, optionally filtered by metadata."""
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    filters: dict = Field(default_factory=dict)  # e.g. {"court": "Supreme Court", "year": 2023}


class ClauseOut(BaseModel):
    category: str
    text: str
    page_number: int
    chunk_id: str
    confidence_margin: float
    risk_level: str
    risk_reason: str
    suggested_edit: str
    red_flags: list = Field(default_factory=list)


class ClauseReport(BaseModel):
    document_id: str
    clauses: list  # list[ClauseOut]
    summary: dict
    disclaimer: str = "Automated analysis for research support only; not legal advice. Review with a qualified lawyer."
    warning: Optional[str] = None
