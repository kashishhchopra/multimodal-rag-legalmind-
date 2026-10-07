"""Visual grounding helpers (spec §5).

We do *honest*, box-level grounding:
  * regions come from real OCR word/line boxes inside a figure, and
  * we either draw a highlight rectangle over that box on the figure, or crop it.

We never fabricate object detections. If no region is available, the caller
simply shows the whole figure (documented limitation).
"""
from __future__ import annotations

import io
from typing import Optional, Tuple

from PIL import Image, ImageDraw

from ..schemas import BoundingBox

_HIGHLIGHT = (255, 87, 34)  # deep orange, high contrast on schematics


def _clamp_box(bbox: BoundingBox, w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(int(bbox.x0), w - 1))
    y0 = max(0, min(int(bbox.y0), h - 1))
    x1 = max(x0 + 1, min(int(bbox.x1), w))
    y1 = max(y0 + 1, min(int(bbox.y1), h))
    return x0, y0, x1, y1


def highlight_region(image_path: str, bbox: Optional[BoundingBox]) -> bytes:
    """Return PNG bytes of the figure with the region boxed (or the plain figure)."""
    img = Image.open(image_path).convert("RGB")
    if bbox is not None:
        x0, y0, x1, y1 = _clamp_box(bbox, img.width, img.height)
        draw = ImageDraw.Draw(img, "RGBA")
        # translucent fill + solid border
        draw.rectangle([x0, y0, x1, y1], fill=(*_HIGHLIGHT, 60))
        for width in range(3):
            draw.rectangle([x0 - width, y0 - width, x1 + width, y1 + width], outline=_HIGHLIGHT)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def crop_region(image_path: str, bbox: BoundingBox, pad: int = 12) -> bytes:
    """Return PNG bytes of just the region (with a little padding)."""
    img = Image.open(image_path).convert("RGB")
    x0, y0, x1, y1 = _clamp_box(bbox, img.width, img.height)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(img.width, x1 + pad)
    y1 = min(img.height, y1 + pad)
    crop = img.crop((x0, y0, x1, y1))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()
