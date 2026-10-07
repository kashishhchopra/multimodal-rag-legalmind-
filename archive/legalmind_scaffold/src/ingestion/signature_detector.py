"""Heuristic detection of signature blocks and stamps on a page.

Used to (a) flag pages that carry an executed signature so the UI can badge
a document as "signed", and (b) exclude signature-block noise from clause
chunking. A full solution would fine-tune a small detector; this scaffold
uses an OpenCV contour heuristic (dense, non-text ink blobs in the lower
third of the page) as a placeholder.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from src.schemas import BoundingBox, SignatureBlock


def detect_signature_blocks(page: "fitz.Page", *, page_number: int) -> list[SignatureBlock]:
    import cv2

    pix = page.get_pixmap(dpi=200)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    arr = np.array(image)

    # Look only in the bottom third of the page, where signature blocks
    # conventionally sit.
    h, w = arr.shape
    band_top = int(h * 0.66)
    band = arr[band_top:h, :]

    _, thresh = cv2.threshold(band, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks: list[SignatureBlock] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        # Heuristic thresholds: irregular, moderately sized ink blobs — wide
        # enough to be a signature stroke, not a full paragraph of text.
        if area < 800 or cw < 40 or ch < 10 or ch > 150:
            continue
        blocks.append(SignatureBlock(
            page=page_number,
            bbox=BoundingBox(
                x0=x, y0=band_top + y, x1=x + cw, y1=band_top + y + ch,
                page_width=w, page_height=h,
            ),
            kind="signature",
            confidence=0.4,  # heuristic only — replace with a trained detector
        ))
    return blocks
