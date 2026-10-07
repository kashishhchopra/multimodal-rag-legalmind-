"""Vision-LLM fallback for pages where OCR fails or looks garbled — hard
scans, skewed photographs, handwriting-adjacent stamps.

More accurate than Tesseract on degraded input but slower and costlier, so it
is deliberately the last resort in the routing chain (see router.py).
Bounding boxes recovered here are coarser (page divided into a grid the model
locates text within) since a vision model doesn't return exact pixel word
boxes the way OCR engines do.
"""
from __future__ import annotations

import base64
import io
import json

from PIL import Image

from src.config import get_settings
from src.schemas import BoundingBox, ParsedPage, ParsePath

RENDER_DPI = 300

VISION_PROMPT = """\
You are transcribing a scanned legal document page. Return the full plain-text
transcription of everything on the page, preserving clause numbering and line
breaks. Do not summarise or omit anything, including footers and page numbers.
"""


def vision_parse_page(page: "fitz.Page", *, doc_id: str, page_number: int) -> ParsedPage:
    settings = get_settings()
    pix = page.get_pixmap(dpi=RENDER_DPI)
    image = Image.open(io.BytesIO(pix.tobytes("png")))

    text = _transcribe_with_vision_model(image, model=settings.vision_model)

    return ParsedPage(
        doc_id=doc_id,
        page_number=page_number,
        text=text,
        parse_path=ParsePath.VISION,
        page_width=image.width,
        page_height=image.height,
        # No per-word boxes from the vision path; highlighting for these
        # pages falls back to whole-page (or whole-block) bounding boxes.
        word_boxes=[],
    )


def _transcribe_with_vision_model(image: Image.Image, *, model: str) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    # TODO: call the configured vision-capable LLM (Claude / GPT-4o / Qwen2-VL)
    # with `VISION_PROMPT` and the base64 image, provider-selected via
    # `src.generation.answerer`'s LLM client wrapper so the model stays
    # swappable. Left unimplemented in the scaffold — wire up your provider
    # of choice here.
    raise NotImplementedError(
        f"Vision transcription not yet wired up for model={model!r}; "
        "see TODO in src/ingestion/vision_parser.py"
    )
