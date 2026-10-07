"""OCR helpers built on Tesseract (via pytesseract), with a diagram-aware mode.

Tesseract is an *external binary* (`brew install tesseract` on macOS). We probe
for it once and degrade gracefully: if it (or OpenCV) is missing, OCR simply
returns empty results instead of crashing ingestion (spec §16).

Two modes:
  * Plain OCR  — used for text-dense visuals (tables, figures, charts). Tesseract
    handles these well out of the box.
  * Diagram OCR — line-art schematics defeat Tesseract because long wires/shapes
    dominate page segmentation. We isolate glyph-sized connected components
    (dropping the long lines and large shapes) before OCR, which recovers most
    component labels (e.g. R1, LED). Labels that overlap a shape (e.g. a base
    label inside a transistor circle) may still be missed — a documented limit.
"""
from __future__ import annotations

import logging
import shutil
from functools import lru_cache
from typing import List, Tuple

from PIL import Image

logger = logging.getLogger("mmrag.ocr")

# (text, (x0, y0, x1, y1)) in pixel coordinates of the input image
WordBox = Tuple[str, Tuple[int, int, int, int]]


@lru_cache
def tesseract_available() -> bool:
    if shutil.which("tesseract"):
        return True
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@lru_cache
def _cv2_available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except Exception:
        return False


def _isolate_glyphs(pil_img: Image.Image) -> Image.Image:
    """Keep only glyph-sized connected components; drop long lines / big shapes.

    Returns a clean black-on-white image better suited to OCR of sparse diagrams.
    Falls back to the original image if OpenCV is unavailable.
    """
    if not _cv2_available():
        return pil_img
    try:
        import cv2
        import numpy as np

        arr = np.array(pil_img.convert("L"))
        th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(th, 8)
        H, W = th.shape
        keep = np.zeros_like(th)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            fill = area / (w * h + 1e-6)
            # glyph heuristics: reasonable height, not a long line, decent fill,
            # and not spanning a large fraction of the image (that's a wire/shape).
            if (
                15 <= h <= 140
                and 6 <= w <= 300
                and (w / (h + 1e-6)) < 7
                and fill > 0.12
                and w < 0.5 * W
                and h < 0.5 * H
            ):
                keep[labels == i] = 255
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        out = cv2.bitwise_not(keep)
        return Image.fromarray(out)
    except Exception as exc:  # pragma: no cover
        logger.warning("Glyph isolation failed, using raw image: %s", exc)
        return pil_img


def _prep(image: Image.Image, is_diagram: bool) -> Image.Image:
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return _isolate_glyphs(image) if is_diagram else image


def run_ocr(image: Image.Image, is_diagram: bool = False) -> str:
    """Return a plain OCR string for an image, or "" on any failure."""
    if not tesseract_available():
        return ""
    try:
        import pytesseract

        img = _prep(image, is_diagram)
        config = "--psm 11" if is_diagram else "--psm 3"
        text = pytesseract.image_to_string(img, config=config)
        return " ".join(text.split()).strip()
    except Exception as exc:  # pragma: no cover
        logger.warning("OCR failed on an image: %s", exc)
        return ""


def ocr_line_regions(image: Image.Image, is_diagram: bool = False) -> List[WordBox]:
    """Return line-grouped (text, bbox) regions for visual grounding.

    bbox coordinates are pixels in the *input image's* space, so they can be
    drawn directly onto the stored figure PNG.
    """
    if not tesseract_available():
        return []
    try:
        import pytesseract
        from pytesseract import Output

        img = _prep(image, is_diagram)
        config = "--psm 11" if is_diagram else "--psm 3"
        data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
    except Exception as exc:  # pragma: no cover
        logger.warning("OCR region extraction failed: %s", exc)
        return []

    lines: dict[tuple, dict] = {}
    n = len(data.get("text", []))
    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if conf >= 0 and conf < 30:  # drop very low-confidence noise
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        e = lines.setdefault(key, {"words": [], "x0": x, "y0": y, "x1": x + w, "y1": y + h})
        e["words"].append(word)
        e["x0"] = min(e["x0"], x)
        e["y0"] = min(e["y0"], y)
        e["x1"] = max(e["x1"], x + w)
        e["y1"] = max(e["y1"], y + h)

    regions: List[WordBox] = []
    for e in lines.values():
        text = " ".join(e["words"]).strip()
        if len(text) < 2:
            continue
        regions.append((text, (e["x0"], e["y0"], e["x1"], e["y1"])))
    return regions
