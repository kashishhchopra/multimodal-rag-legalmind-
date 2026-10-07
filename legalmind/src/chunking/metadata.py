"""Attach page/bbox/section metadata to each clause span, and locate the
bounding box for a chunk's text within a page's word boxes.

This is the module where ingestion-time grounding (word boxes from
`native_parser`/`ocr_parser`) becomes chunk-time grounding
(`ChunkMetadata.bbox`) — the hand-off described in the "grounding is carried
in metadata" design note.
"""
from __future__ import annotations

import uuid

from src.schemas import BoundingBox, Chunk, ChunkMetadata, ParsedPage
from .structure_splitter import ClauseSpan


def locate_bbox_for_text(parsed_page: ParsedPage, clause_text: str) -> BoundingBox | None:
    """Find the union bounding box of the words in `clause_text` within the
    page's word boxes. Falls back to None (whole-page highlight) if the page
    has no word boxes (e.g. a vision-parsed page)."""
    if not parsed_page.word_boxes:
        return None

    target_words = clause_text.split()[:8]  # match on the first few words; cheap and usually unique
    if not target_words:
        return None

    matched_boxes: list[BoundingBox] = []
    words_iter = iter(parsed_page.word_boxes)
    for word, box in words_iter:
        if word == target_words[0]:
            matched_boxes = [box]
            remaining = target_words[1:]
            for next_word, next_box in words_iter:
                if not remaining:
                    break
                if next_word == remaining[0]:
                    matched_boxes.append(next_box)
                    remaining.pop(0)
            break

    if not matched_boxes:
        return None

    return BoundingBox(
        x0=min(b.x0 for b in matched_boxes),
        y0=min(b.y0 for b in matched_boxes),
        x1=max(b.x1 for b in matched_boxes),
        y1=max(b.y1 for b in matched_boxes),
        page_width=parsed_page.page_width,
        page_height=parsed_page.page_height,
    )


def build_chunk_metadata(
    parsed_page: ParsedPage,
    span: ClauseSpan,
    *,
    doc_type: str | None = None,
    jurisdiction: str | None = None,
) -> Chunk:
    bbox = locate_bbox_for_text(parsed_page, span.text)
    metadata = ChunkMetadata(
        doc_id=parsed_page.doc_id,
        doc_type=doc_type,
        jurisdiction=jurisdiction,
        section_path=span.section_path,
        page=parsed_page.page_number,
        bbox=bbox,
        parse_path=parsed_page.parse_path,
    )
    return Chunk(chunk_id=str(uuid.uuid4()), text=span.text, metadata=metadata)
