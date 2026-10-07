"""Deterministic extraction for pages that already have a text layer.

Uses PyMuPDF to pull text plus exact word-level bounding boxes, which is how
we get citation highlighting "for free" later without ever calling an LLM to
locate text on the page.
"""
from __future__ import annotations

from src.schemas import BoundingBox, ParsedPage, ParsePath


def parse_native_page(page: "fitz.Page", *, doc_id: str, page_number: int) -> ParsedPage:
    rect = page.rect
    text = page.get_text("text")

    word_boxes: list[tuple[str, BoundingBox]] = []
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        word_boxes.append((
            word,
            BoundingBox(
                x0=x0, y0=y0, x1=x1, y1=y1,
                page_width=rect.width, page_height=rect.height,
            ),
        ))

    return ParsedPage(
        doc_id=doc_id,
        page_number=page_number,
        text=text,
        parse_path=ParsePath.NATIVE,
        page_width=rect.width,
        page_height=rect.height,
        word_boxes=word_boxes,
    )
