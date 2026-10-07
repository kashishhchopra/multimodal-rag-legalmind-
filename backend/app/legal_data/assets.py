"""Programmatically-drawn visual assets used inside the generated legal PDFs.

Real legal documents carry seals, stamps and signatures alongside text and
tables — exactly the kind of visual evidence this project's multimodal
pipeline (raster-image extraction + OCR + CLIP embeddings) is built to index.
We draw these with Pillow so the corpus needs no external image downloads and
ships fully offline / license-free.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_seal(out_path: Path, org_name: str, seal_text: str = "OFFICIAL SEAL") -> Path:
    """A circular court/company seal — used as an embedded raster image."""
    size = 320
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)
    cx = cy = size // 2
    for r, w in ((size // 2 - 4, 5), (size // 2 - 16, 2)):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(20, 40, 110, 255), width=w)

    # star ring
    n_stars, r_star = 14, size // 2 - 30
    for i in range(n_stars):
        ang = 2 * math.pi * i / n_stars
        sx, sy = cx + r_star * math.cos(ang), cy + r_star * math.sin(ang)
        d.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=(20, 40, 110, 255))

    # curved-ish org name (approximated as centered wrapped text) + center emblem
    d.ellipse([cx - 46, cy - 46, cx + 46, cy + 46], outline=(20, 40, 110, 255), width=3)
    d.polygon(
        [(cx, cy - 34), (cx + 20, cy + 18), (cx - 20, cy + 18)],
        outline=(20, 40, 110, 255),
        width=3,
    )
    f_small = _font(15)
    words = org_name.upper().split()
    line1, line2 = " ".join(words[: len(words) // 2]) or org_name, " ".join(words[len(words) // 2 :])
    d.text((cx, cy - 90), line1, font=f_small, fill=(20, 40, 110, 255), anchor="mm")
    d.text((cx, cy + 95), line2, font=f_small, fill=(20, 40, 110, 255), anchor="mm")
    f_mid = _font(13)
    d.text((cx, cy + 62), seal_text, font=f_mid, fill=(150, 20, 20, 255), anchor="mm")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def make_signature(out_path: Path, name: str, seed: int = 0) -> Path:
    """A scribble-style signature image with a printed name caption beneath it."""
    w, h = 420, 150
    img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)
    rng_x = 20
    y0 = h * 0.55
    points = []
    n = 60
    for i in range(n):
        x = rng_x + i * (w - 2 * rng_x) / n
        y = y0 + 28 * math.sin(i * 0.7 + seed) + 10 * math.sin(i * 1.9 + seed * 2)
        points.append((x, y))
    d.line(points, fill=(15, 15, 90, 255), width=3, joint="curve")
    d.line([(20, h - 20), (w - 20, h - 20)], fill=(0, 0, 0, 180), width=1)
    d.text((w / 2, h - 8), name, font=_font(13), fill=(0, 0, 0, 255), anchor="mm")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def make_court_emblem(out_path: Path, title: str = "SUPREME COURT") -> Path:
    """A simple scales-of-justice emblem used at the top of judgment documents."""
    size = 220
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)
    cx = size // 2
    top = 30
    d.line([(cx, top), (cx, size - 40)], fill=(30, 30, 30, 255), width=5)
    d.line([(cx - 60, top + 10), (cx + 60, top + 10)], fill=(30, 30, 30, 255), width=5)
    for dx in (-60, 60):
        bx = cx + dx
        d.line([(bx, top + 10), (bx - 22, top + 45)], fill=(30, 30, 30, 255), width=3)
        d.line([(bx, top + 10), (bx + 22, top + 45)], fill=(30, 30, 30, 255), width=3)
        d.arc([bx - 26, top + 30, bx + 26, top + 60], 0, 180, fill=(30, 30, 30, 255), width=3)
    d.polygon([(cx - 45, size - 40), (cx + 45, size - 40), (cx + 30, size - 20), (cx - 30, size - 20)],
              outline=(30, 30, 30, 255), width=3)
    d.text((cx, size - 8), title, font=_font(12), fill=(30, 30, 30, 255), anchor="mm")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
