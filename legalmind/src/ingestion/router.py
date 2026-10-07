"""Layer 1 entry point: classify each page as native-text or scan/photo, then
dispatch to the right parser.

Key design decision (see docs/ARCHITECTURE.md 5.1): deterministic extractors
for native text are cheap and give exact coordinates; a vision model is used
only for pages that fail the text-layer check. We log which path every page
took (`ParsedPage.parse_path`) so the eval report can break metrics out by
native vs. OCR vs. vision, and measure the "OCR-doc answer parity" target.
"""
from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from src.config import get_settings
from src.schemas import ParsedPage, ParsePath

from .native_parser import parse_native_page
from .ocr_parser import ocr_page
from .vision_parser import vision_parse_page
from .table_extractor import extract_tables
from .signature_detector import detect_signature_blocks

logger = logging.getLogger(__name__)


def classify_page(page: "fitz.Page") -> ParsePath:
    """A page is "native" if it has a real text layer with enough content;
    otherwise it's treated as a scan and routed to OCR/vision."""
    settings = get_settings()
    text = page.get_text("text").strip()
    if len(text) >= settings.min_text_chars_for_native:
        return ParsePath.NATIVE
    return ParsePath.OCR


def route_document(file_path: str | Path, doc_id: str) -> list[ParsedPage]:
    """Parse every page of a document, routing each independently.

    A single PDF may mix native and scanned pages (e.g. a signed signature
    page photographed and appended) — classification happens per page, not
    per document.
    """
    file_path = Path(file_path)
    parsed_pages: list[ParsedPage] = []

    doc = fitz.open(file_path)
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            path = classify_page(page)

            if path == ParsePath.NATIVE:
                parsed = parse_native_page(page, doc_id=doc_id, page_number=page_index + 1)
            else:
                # Try OCR first; a caller-side quality check (confidence /
                # garbled-text heuristic) can escalate to the vision model.
                parsed = ocr_page(page, doc_id=doc_id, page_number=page_index + 1)
                if _looks_low_quality(parsed):
                    logger.info(
                        "doc=%s page=%d: OCR output looked low-quality, escalating to vision model",
                        doc_id, page_index + 1,
                    )
                    parsed = vision_parse_page(page, doc_id=doc_id, page_number=page_index + 1)

            parsed.tables = extract_tables(file_path, page_number=page_index + 1)
            parsed.signatures = detect_signature_blocks(page, page_number=page_index + 1)
            parsed_pages.append(parsed)
    finally:
        doc.close()

    return parsed_pages


def _looks_low_quality(parsed: ParsedPage) -> bool:
    """Cheap heuristic: very short output or high symbol-to-alpha ratio
    suggests OCR garbled the page and it's worth the vision-model retry."""
    text = parsed.text.strip()
    if len(text) < 20:
        return True
    alpha = sum(c.isalpha() for c in text)
    return alpha / max(len(text), 1) < 0.5
