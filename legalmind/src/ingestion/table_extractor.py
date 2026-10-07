"""Table extraction for payment schedules, fee tables, and exhibits that
plain text extraction flattens into unreadable rows.

pdfplumber is used first (fast, works well on native PDFs with ruled tables);
Camelot's lattice/stream modes are the fallback for borderless tables typical
of scanned exhibits.
"""
from __future__ import annotations

from pathlib import Path

from src.schemas import BoundingBox, TableBlock


def extract_tables(file_path: str | Path, *, page_number: int) -> list[TableBlock]:
    tables = _extract_with_pdfplumber(file_path, page_number)
    if tables:
        return tables
    return _extract_with_camelot(file_path, page_number)


def _extract_with_pdfplumber(file_path: str | Path, page_number: int) -> list[TableBlock]:
    import pdfplumber

    blocks: list[TableBlock] = []
    with pdfplumber.open(file_path) as pdf:
        if page_number - 1 >= len(pdf.pages):
            return blocks
        page = pdf.pages[page_number - 1]
        for table in page.find_tables():
            rows = table.extract()
            x0, top, x1, bottom = table.bbox
            blocks.append(TableBlock(
                page=page_number,
                bbox=BoundingBox(
                    x0=x0, y0=top, x1=x1, y1=bottom,
                    page_width=page.width, page_height=page.height,
                ),
                rows=[[cell or "" for cell in row] for row in rows],
            ))
    return blocks


def _extract_with_camelot(file_path: str | Path, page_number: int) -> list[TableBlock]:
    # TODO: run camelot.read_pdf(file_path, pages=str(page_number),
    # flavor="lattice") and fall back to flavor="stream"; map each table's
    # `._bbox` into a BoundingBox. Left as a stub since Camelot needs
    # Ghostscript and is heavier than the pdfplumber happy path.
    return []
