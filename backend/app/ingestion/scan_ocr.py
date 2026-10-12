"""OCR for image-only (scanned) judgment PDFs, with per-page confidence.

Renders each page at 200 dpi and runs Tesseract. Returns text per page plus the mean word confidence so low-quality
pages can be flagged instead of silently indexed. Languages: "eng" by default; pass "eng+hin" for bilingual pages
(Tesseract's `hin` data is installed here). Slow: roughly 2-6 s per page on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import fitz
import pytesseract
from PIL import Image

LOW_CONFIDENCE = 60.0  # mean word confidence (0-100) below which a page is flagged for review


@dataclass
class OcrPage:
    page_number: int
    text: str
    confidence: float
    low_confidence: bool


def ocr_pdf(data: bytes, lang: str = "eng", dpi: int = 200, max_pages: int = 0) -> List[OcrPage]:
    doc = fitz.open(stream=data, filetype="pdf")
    out: List[OcrPage] = []
    for i, page in enumerate(doc, 1):
        if max_pages and i > max_pages:
            break
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        d = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
        words, confs, line_key, lines = [], [], None, []
        for t, c, b, p, ln in zip(d["text"], d["conf"], d["block_num"], d["par_num"], d["line_num"]):
            if not t.strip():
                continue
            key = (b, p, ln)
            if key != line_key and lines:
                words.append(" ".join(lines)); lines = []
            line_key = key
            lines.append(t)
            if float(c) >= 0:
                confs.append(float(c))
        if lines:
            words.append(" ".join(lines))
        conf = sum(confs) / len(confs) if confs else 0.0
        out.append(OcrPage(i, "\n".join(words), round(conf, 1), conf < LOW_CONFIDENCE))
    return out
