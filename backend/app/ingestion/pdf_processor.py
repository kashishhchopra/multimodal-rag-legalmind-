"""PDF parsing: the real extraction engine of the ingestion pipeline.

Responsibilities
----------------
1. Extract page text (PyMuPDF) and render each page to a PNG (for highlighting).
2. Extract embedded raster images with their placement bounding boxes.
3. Detect *vector* diagram/schematic regions (circuit schematics are frequently
   drawn as vector graphics, so `get_images()` alone would miss them).
4. Detect nearby caption labels ("Figure 12", "Table 3") and classify a rough
   visual type from the caption + OCR text.
5. Run OCR on every visual and derive word/line *regions* (bounding boxes inside
   the figure) that make honest, box-level visual grounding possible.
6. Extract tables via pdfplumber as text chunks tagged as tables.

Everything here is best-effort and defensive: a failure on one image or one page
is logged and skipped rather than aborting the whole document.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import fitz  # PyMuPDF
from PIL import Image

from ..schemas import BoundingBox, VisualType
from .chunking import Chunk, chunk_page_text
from .ocr import ocr_line_regions, run_ocr, tesseract_available

logger = logging.getLogger("mmrag.pdf")

# Regexes for caption / figure-label detection
_FIGURE_RE = re.compile(r"\b(fig(?:ure)?\.?\s*\d+[a-z]?)\b", re.IGNORECASE)
_TABLE_RE = re.compile(r"\b(table\s*\d+[a-z]?)\b", re.IGNORECASE)

# Page render resolution (2x = 144 DPI) — sharp enough for display + OCR.
_RENDER_ZOOM = 2.0
# Vector diagrams are cropped tighter and often have small labels, so render
# them at higher resolution to give OCR (and the viewer) a sharper image.
_DIAGRAM_ZOOM = 3.5

# Token that looks like a circuit/component reference designator (R1, Q3, C12, U2, D4...)
_COMPONENT_RE = re.compile(r"^[A-Z]{1,3}\d{1,3}[A-Z]?$")


@dataclass
class ParsedRegion:
    text: str
    bbox: BoundingBox           # relative to the visual image (pixels)


@dataclass
class ParsedVisual:
    page_number: int
    image_path: str             # absolute path on disk
    image_type: VisualType
    bbox: Optional[BoundingBox]  # placement on the PDF page (points)
    ocr_text: str = ""
    caption: str = ""
    figure_label: str = ""
    width: int = 0
    height: int = 0
    regions: List[ParsedRegion] = field(default_factory=list)


@dataclass
class ParsedDocument:
    page_count: int
    chunks: List[Chunk]
    visuals: List[ParsedVisual]
    page_render_paths: List[str]   # index 0 == page 1
    table_count: int = 0
    figure_count: int = 0
    image_count: int = 0


ProgressCB = Callable[[str, float, str], None]


def _noop(stage: str, progress: float, message: str) -> None:  # default callback
    pass


def _classify_visual(caption: str, ocr_text: str) -> VisualType:
    """Best-effort visual-type classification from caption + OCR keywords."""
    hay = f"{caption} {ocr_text}".lower()
    if "table" in caption.lower():
        return VisualType.TABLE
    if any(k in hay for k in ("circuit", "schematic", "resistor", "transistor", "capacitor", "vcc", "gnd")):
        return VisualType.SCHEMATIC
    if any(k in hay for k in ("chart", "axis", "y-axis", "x-axis", "graph", "plot", "%")):
        return VisualType.CHART
    if any(k in hay for k in ("diagram", "block", "flow", "architecture", "step")):
        return VisualType.DIAGRAM
    if caption:
        return VisualType.FIGURE
    return VisualType.IMAGE


def _find_caption(page: "fitz.Page", rect: "fitz.Rect") -> tuple[str, str]:
    """Look for a 'Figure N' / 'Table N' caption near an image rect.

    Returns (figure_label, caption_line). Searches text blocks whose vertical
    band is just below or just above the image.
    """
    try:
        blocks = page.get_text("blocks")  # (x0,y0,x1,y1, text, block_no, type)
    except Exception:
        return "", ""
    best_label, best_caption = "", ""
    for b in blocks:
        bx0, by0, bx1, by1, btext = b[0], b[1], b[2], b[3], b[4]
        if not btext or not btext.strip():
            continue
        # horizontal overlap with the image and vertically close (below or above).
        # Small negative tolerance absorbs padding / rounding on the region rect.
        horizontal_overlap = min(bx1, rect.x1) - max(bx0, rect.x0)
        below = -20 <= (by0 - rect.y1) <= 90
        above = -20 <= (rect.y0 - by1) <= 50
        if horizontal_overlap > 0 and (below or above):
            m = _FIGURE_RE.search(btext) or _TABLE_RE.search(btext)
            if m:
                best_label = m.group(1).strip()
                best_caption = " ".join(btext.split())[:200]
                if below:  # captions are usually below → prefer these
                    return best_label, best_caption
    return best_label, best_caption


def _regions_from_ocr(pil_img: Image.Image, is_diagram: bool = False) -> List[ParsedRegion]:
    """Derive line-level regions (text + bbox) for box-level visual grounding.

    Delegates to the OCR module, which applies diagram-aware preprocessing when
    `is_diagram` is True. Returns [] if OCR is unavailable.
    """
    regions: List[ParsedRegion] = []
    for text, (x0, y0, x1, y1) in ocr_line_regions(pil_img, is_diagram=is_diagram):
        regions.append(
            ParsedRegion(text=text, bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1))
        )
    return regions


def _extract_raster_visuals(
    doc: "fitz.Document",
    page: "fitz.Page",
    page_index: int,
    out_dir: Path,
) -> List[ParsedVisual]:
    """Extract embedded raster images on a page with placement bboxes + captions."""
    visuals: List[ParsedVisual] = []
    seen_rects: list[fitz.Rect] = []
    try:
        xrefs = page.get_images(full=True)
    except Exception as exc:
        logger.warning("get_images failed on page %d: %s", page_index + 1, exc)
        return visuals

    for img_index, info in enumerate(xrefs):
        xref = info[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        rect = rects[0] if rects else page.rect

        # Skip near-duplicate placements (same image tiled/duplicated)
        if any(abs(rect.x0 - r.x0) < 2 and abs(rect.y0 - r.y0) < 2 for r in seen_rects):
            continue
        seen_rects.append(rect)

        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.n >= 5:  # CMYK or with alpha → convert to RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            # Skip tiny decorative images (logos, bullets, rules)
            if pix.width < 40 or pix.height < 40:
                continue
            img_path = out_dir / f"p{page_index + 1}_img{img_index}.png"
            pix.save(str(img_path))
            pil = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.warning("Raster extract failed p%d img%d: %s", page_index + 1, img_index, exc)
            continue

        figure_label, caption = _find_caption(page, rect)
        ocr_text = run_ocr(pil)
        visuals.append(
            ParsedVisual(
                page_number=page_index + 1,
                image_path=str(img_path),
                image_type=_classify_visual(caption, ocr_text),
                bbox=BoundingBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1),
                ocr_text=ocr_text,
                caption=caption,
                figure_label=figure_label,
                width=pil.width,
                height=pil.height,
                regions=_regions_from_ocr(pil),
            )
        )
    return visuals


def _extract_vector_diagram(
    page: "fitz.Page",
    page_index: int,
    out_dir: Path,
) -> Optional[ParsedVisual]:
    """Detect a vector-drawn diagram/schematic region and crop it.

    Many schematics and block diagrams are vector paths, not embedded rasters.
    We cluster the page's drawing paths; if they are numerous and cover a
    meaningful area, we crop that region from a high-res page render and treat
    it as a visual. This is a heuristic (documented as a limitation).
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return None
    if len(drawings) < 6:  # not enough vector content to be a diagram
        return None

    # Union bbox of all drawing paths
    x0 = min((d["rect"].x0 for d in drawings), default=None)
    y0 = min((d["rect"].y0 for d in drawings), default=None)
    x1 = max((d["rect"].x1 for d in drawings), default=None)
    y1 = max((d["rect"].y1 for d in drawings), default=None)
    if x0 is None:
        return None

    rect = fitz.Rect(x0, y0, x1, y1)
    page_w, page_h = page.rect.width, page.rect.height
    page_area = page_w * page_h
    width_frac = (rect.x1 - rect.x0) / page_w if page_w else 0
    height_frac = (rect.y1 - rect.y0) / page_h if page_h else 0
    # Require a 2D region (excludes header rules / underlines which are thin strips)
    if page_area <= 0 or rect.get_area() / page_area < 0.02:
        return None
    if width_frac < 0.08 or height_frac < 0.03:
        return None

    # Pad slightly and clamp to page
    rect = rect + (-6, -6, 6, 6)
    rect = rect & page.rect
    try:
        mat = fitz.Matrix(_DIAGRAM_ZOOM, _DIAGRAM_ZOOM)
        pix = page.get_pixmap(matrix=mat, clip=rect)
        img_path = out_dir / f"p{page_index + 1}_vector.png"
        pix.save(str(img_path))
        pil = Image.open(img_path).convert("RGB")
    except Exception as exc:
        logger.warning("Vector diagram crop failed p%d: %s", page_index + 1, exc)
        return None

    figure_label, caption = _find_caption(page, rect)
    ocr_text = run_ocr(pil, is_diagram=True)
    vtype = _classify_visual(caption, ocr_text)
    if vtype in (VisualType.IMAGE, VisualType.FIGURE):
        vtype = VisualType.DIAGRAM  # vector content is a diagram by default
    return ParsedVisual(
        page_number=page_index + 1,
        image_path=str(img_path),
        image_type=vtype,
        bbox=BoundingBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1),
        ocr_text=ocr_text,
        caption=caption,
        figure_label=figure_label,
        width=pil.width,
        height=pil.height,
        regions=_regions_from_ocr(pil, is_diagram=True),
    )


def _extract_tables(pdf_path: str, page_index: int) -> List[str]:
    """Extract tables on a page as tab-separated text blocks (pdfplumber)."""
    tables_text: List[str] = []
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            if page_index >= len(pdf.pages):
                return []
            page = pdf.pages[page_index]
            for table in page.extract_tables() or []:
                rows = [
                    " | ".join((cell or "").strip() for cell in row)
                    for row in table
                    if any(cell for cell in row)
                ]
                if len(rows) >= 2:  # header + at least one data row
                    tables_text.append("\n".join(rows))
    except Exception as exc:
        logger.warning("Table extraction failed p%d: %s", page_index + 1, exc)
    return tables_text


def parse_pdf(
    pdf_path: str,
    out_dir: Path,
    progress: ProgressCB = _noop,
    max_pages: int = 200,
) -> ParsedDocument:
    """Parse a PDF into chunks + visuals + page renders. Raises on unreadable PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    page_count = min(doc.page_count, max_pages)
    if page_count == 0:
        doc.close()
        raise ValueError("PDF has no pages")

    chunks: List[Chunk] = []
    visuals: List[ParsedVisual] = []
    page_render_paths: List[str] = []
    table_count = 0

    total_text = 0
    for pi in range(page_count):
        page = doc.load_page(pi)
        frac = (pi + 1) / page_count

        # --- text ---
        progress("extracting_text", 0.15 + 0.35 * frac, f"Extracting text (page {pi + 1}/{page_count})")
        text = page.get_text("text") or ""
        total_text += len(text.strip())
        chunks.extend(chunk_page_text(text, page_number=pi + 1))

        # --- page render (for highlighting) ---
        try:
            mat = fitz.Matrix(_RENDER_ZOOM, _RENDER_ZOOM)
            pix = page.get_pixmap(matrix=mat)
            render_path = out_dir / f"page_{pi + 1}.png"
            pix.save(str(render_path))
            page_render_paths.append(str(render_path))
        except Exception as exc:
            logger.warning("Page render failed p%d: %s", pi + 1, exc)
            page_render_paths.append("")

        # --- tables ---
        for tbl in _extract_tables(pdf_path, pi):
            table_count += 1
            chunks.append(Chunk(text=f"[TABLE p{pi + 1}]\n{tbl}", page_number=pi + 1))

        # --- raster images ---
        progress("detecting_figures", 0.50 + 0.20 * frac, f"Detecting figures (page {pi + 1}/{page_count})")
        visuals.extend(_extract_raster_visuals(doc, page, pi, out_dir))

        # --- vector diagram region ---
        vec = _extract_vector_diagram(page, pi, out_dir)
        if vec is not None:
            visuals.append(vec)

    doc.close()

    if total_text == 0 and not visuals:
        raise ValueError("Document appears to be empty (no text and no images found)")

    figure_count = sum(1 for v in visuals if v.figure_label or v.image_type in (
        VisualType.FIGURE, VisualType.DIAGRAM, VisualType.SCHEMATIC, VisualType.CHART))
    image_count = len(visuals)

    return ParsedDocument(
        page_count=page_count,
        chunks=chunks,
        visuals=visuals,
        page_render_paths=page_render_paths,
        table_count=table_count,
        figure_count=figure_count,
        image_count=image_count,
    )
