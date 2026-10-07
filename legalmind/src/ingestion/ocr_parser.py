"""OCR path for scanned pages: rasterize with PyMuPDF, run Tesseract (or
PaddleOCR) to recover text plus per-word bounding boxes.

This is the cheaper of the two scan-handling routes; `router.py` escalates to
`vision_parser.py` only when the output here looks garbled.
"""
from __future__ import annotations

import io

from PIL import Image

from src.config import get_settings
from src.schemas import BoundingBox, ParsedPage, ParsePath

RENDER_DPI = 300


def ocr_page(page: "fitz.Page", *, doc_id: str, page_number: int) -> ParsedPage:
    settings = get_settings()
    pix = page.get_pixmap(dpi=RENDER_DPI)
    image = Image.open(io.BytesIO(pix.tobytes("png")))

    if settings.ocr_engine == "paddleocr":
        text, word_boxes = _run_paddleocr(image)
    else:
        text, word_boxes = _run_tesseract(image)

    return ParsedPage(
        doc_id=doc_id,
        page_number=page_number,
        text=text,
        parse_path=ParsePath.OCR,
        page_width=image.width,
        page_height=image.height,
        word_boxes=word_boxes,
    )


def _run_tesseract(image: Image.Image) -> tuple[str, list[tuple[str, BoundingBox]]]:
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    words: list[tuple[str, BoundingBox]] = []
    lines: list[str] = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
        words.append((
            word,
            BoundingBox(
                x0=x, y0=y, x1=x + w, y1=y + h,
                page_width=image.width, page_height=image.height,
            ),
        ))
        lines.append(word)
    return " ".join(lines), words


def _run_paddleocr(image: Image.Image) -> tuple[str, list[tuple[str, BoundingBox]]]:
    # TODO: wire up PaddleOCR as an alternative engine (better for
    # non-Latin scripts / low-quality photographs than Tesseract).
    raise NotImplementedError("PaddleOCR backend not yet implemented")
