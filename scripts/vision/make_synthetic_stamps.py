"""Procedural stamps, seals and signatures composited on real page backgrounds, with YOLO labels.

Fixes the "only 6 template seals" limitation: every sample has random shape, text, colour, ink
opacity, rotation, blur, partial overlap with body text, and scan noise.

    python3 scripts/vision/make_synthetic_stamps.py --backgrounds backend/data --out data_lake/synthetic_stamps -n 2000

Backgrounds: any PDFs under --backgrounds are rendered with PyMuPDF (first pages); if none are
found, plain ruled "paper" pages are generated. Pass real scanned SC judgment PDFs for best
domain match. Output is a YOLO dataset: images/{train,val}, labels/{train,val}, data.yaml.
Classes: 0 = stamp/seal, 1 = signature.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CLASSES = ["stamp", "signature"]
INKS = [(30, 60, 160), (170, 30, 40), (20, 20, 20), (90, 40, 130), (20, 110, 60)]
WORDS = ["SUPREME COURT", "HIGH COURT", "NOTARY PUBLIC", "REGISTRAR", "GOVT. OF INDIA", "APPROVED", "RECEIVED",
         "CERTIFIED TRUE COPY", "SUB-REGISTRAR", "OATH COMMISSIONER", "STAMP DUTY PAID", "ADVOCATE", "FILED", "DELHI", "MUMBAI"]
_FONT_CANDIDATES = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def _font(size: int) -> ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def make_stamp(rng: random.Random, size: int) -> Image.Image:
    """RGBA stamp: circular/oval/rect/double-ring variants with ring text and centre text."""
    ink, alpha = rng.choice(INKS), rng.randint(140, 235)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = max(2, size // rng.randint(28, 45))
    shape = rng.choice(["circle", "circle", "oval", "rect", "double"])
    m = w * 2
    box = [m, m, size - m, size - m] if shape != "oval" else [m, size // 5, size - m, size - size // 5]
    col = (*ink, alpha)
    if shape in ("circle", "oval", "double"):
        d.ellipse(box, outline=col, width=w)
        if shape == "double":
            d.ellipse([b + (w * 3 if i < 2 else -w * 3) for i, b in enumerate(box)], outline=col, width=max(1, w // 2))
    else:
        d.rounded_rectangle(box, radius=size // 12, outline=col, width=w)
    f = _font(max(10, size // 9))
    text = rng.choice(WORDS)
    tw = d.textlength(text, font=f)
    d.text(((size - tw) / 2, size / 2 - f.size / 2), text, font=f, fill=col)
    if shape in ("circle", "double"):  # ring text approximated by evenly spaced glyphs along the arc
        r, ring = size / 2 - w * 5, rng.choice(WORDS)
        for i, ch in enumerate(ring):
            ang = math.radians(200 + 140 * i / max(1, len(ring) - 1))
            cx, cy = size / 2 + r * math.cos(ang), size / 2 + r * math.sin(ang)
            g = Image.new("RGBA", (f.size * 2, f.size * 2), (0, 0, 0, 0))
            ImageDraw.Draw(g).text((f.size / 2, f.size / 2), ch, font=_font(max(8, size // 12)), fill=col)
            g = g.rotate(-math.degrees(ang) - 90 + 180, resample=Image.BICUBIC)
            img.alpha_composite(g, (int(cx - f.size), int(cy - f.size)))
    # speckle: emulate uneven ink transfer
    a = np.array(img)
    mask = np.random.default_rng(rng.randint(0, 2**31)).random(a.shape[:2]) < rng.uniform(0.05, 0.25)
    a[..., 3] = np.where(mask, a[..., 3] // 3, a[..., 3])
    return Image.fromarray(a)


def make_signature(rng: random.Random, w: int, h: int) -> Image.Image:
    """RGBA handwritten-looking signature: a few smooth looping strokes plus an underline flourish."""
    ink = rng.choice([(15, 15, 90), (10, 10, 10), (30, 30, 130)])
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    width = max(2, h // rng.randint(14, 24))
    x, y = w * 0.05, h * 0.55
    pts = []
    for i in range(rng.randint(5, 9)):
        x += w * rng.uniform(0.08, 0.15)
        pts.append((x, h * 0.5 + math.sin(i * rng.uniform(1.2, 2.4)) * h * rng.uniform(0.2, 0.45)))
    # Catmull-Rom-ish smoothing through control points
    path = []
    for i in range(len(pts) - 1):
        p0, p1 = pts[max(0, i - 1)], pts[i]
        p2, p3 = pts[i + 1], pts[min(len(pts) - 1, i + 2)]
        for t in np.linspace(0, 1, 14):
            path.append(tuple(0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t * t
                                     + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t ** 3) for k in (0, 1)))
    d.line(path, fill=(*ink, 235), width=width, joint="curve")
    y0 = h * rng.uniform(0.75, 0.9)
    d.line([(w * 0.1, y0), (w * rng.uniform(0.6, 0.95), y0 - h * rng.uniform(0, 0.12))], fill=(*ink, 220), width=max(1, width - 1))
    return img


def blank_page(rng: random.Random, w: int = 900, h: int = 1250) -> Image.Image:
    img = Image.new("RGB", (w, h), (rng.randint(238, 252),) * 3)
    d = ImageDraw.Draw(img)
    f = _font(15)
    for y in range(90, h - 100, 26):
        d.text((80, y), " ".join(rng.choice(WORDS).lower() for _ in range(rng.randint(5, 9))), font=f, fill=(55, 55, 55))
    return img


def load_backgrounds(root: Path, limit: int) -> list[Image.Image]:
    import fitz  # PyMuPDF

    pages: list[Image.Image] = []
    for pdf in sorted(root.rglob("*.pdf")):
        try:
            doc = fitz.open(pdf)
            for pno in range(min(2, doc.page_count)):
                pix = doc[pno].get_pixmap(dpi=100)
                pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        except Exception:  # noqa: BLE001 - skip unreadable PDFs
            continue
        if len(pages) >= limit:
            break
    return pages


def paste(page: Image.Image, obj: Image.Image, rng: random.Random, cls: int):
    angle = rng.uniform(-35, 35) if cls == 0 else rng.uniform(-12, 12)
    obj = obj.rotate(angle, expand=True, resample=Image.BICUBIC)
    if rng.random() < 0.6:
        obj = obj.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.4)))
    bbox = obj.getchannel("A").point(lambda v: 255 if v > 30 else 0).getbbox()
    if not bbox:
        return None
    x = rng.randint(0, max(0, page.width - obj.width))
    y = rng.randint(int(page.height * 0.45), max(int(page.height * 0.45), page.height - obj.height))  # stamps cluster low
    page.paste(obj, (x, y), obj)
    x0, y0, x1, y1 = x + bbox[0], y + bbox[1], x + bbox[2], y + bbox[3]
    return x0, y0, x1, y1


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.convert("L") if rng.random() < 0.5 else img
    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-1.5, 1.5), fillcolor=(245,) * (1 if img.mode == "L" else 3))
    a = np.asarray(img).astype(np.float32)
    a += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(2, 9), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backgrounds", type=Path, default=Path("backend/data"))
    ap.add_argument("--out", type=Path, default=Path("data_lake/synthetic_stamps"))
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    bgs = load_backgrounds(args.backgrounds, 200) if args.backgrounds.exists() else []
    print(f"{len(bgs)} real page backgrounds (+ generated blanks)")
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        split = "val" if rng.random() < args.val_frac else "train"
        page = (rng.choice(bgs).copy() if bgs and rng.random() < 0.7 else blank_page(rng)).convert("RGB")
        labels = []
        for _ in range(rng.choice([0, 1, 1, 2, 3])):
            cls = 0 if rng.random() < 0.6 else 1
            side = int(min(page.size) * rng.uniform(0.1, 0.28))
            obj = make_stamp(rng, side) if cls == 0 else make_signature(rng, int(side * 1.7), int(side * 0.7))
            box = paste(page, obj, rng, cls)
            if box:
                x0, y0, x1, y1 = box
                W, H = page.size
                labels.append(f"{cls} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        degrade(page, rng).save(args.out / "images" / split / f"{i:05d}.jpg", quality=rng.randint(60, 92))
        (args.out / "labels" / split / f"{i:05d}.txt").write_text("\n".join(labels))
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {k}: {v}\n" for k, v in enumerate(CLASSES))
    )
    print(f"wrote {args.n} images to {args.out}")


if __name__ == "__main__":
    main()
