"""Tests for PDF ingestion: text, images/figures, tables, OCR, regions."""
from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.ingestion.chunking import chunk_page_text
from app.ingestion.ocr import tesseract_available
from app.ingestion.pdf_processor import parse_pdf
from app.schemas import VisualType


def test_chunking_splits_and_preserves_page():
    text = "First sentence here. Second one follows. " * 40
    chunks = chunk_page_text(text, page_number=7, max_chars=200, overlap_chars=40)
    assert len(chunks) > 1
    assert all(c.page_number == 7 for c in chunks)
    assert all(len(c.text) <= 400 for c in chunks)  # max + overlap slack


def test_chunking_empty_text_returns_nothing():
    assert chunk_page_text("", page_number=1) == []
    assert chunk_page_text("   \n  ", page_number=1) == []


def test_parse_pdf_extracts_text(sample_pdf: Path):
    parsed = parse_pdf(str(sample_pdf), get_settings().data_dir / "_t_text")
    assert parsed.page_count == 3
    assert len(parsed.chunks) >= 3
    all_text = " ".join(c.text for c in parsed.chunks).lower()
    assert "current-limiting" in all_text or "current" in all_text
    assert "q1" in all_text


def test_parse_pdf_detects_visuals(sample_pdf: Path):
    parsed = parse_pdf(str(sample_pdf), get_settings().data_dir / "_t_vis")
    assert len(parsed.visuals) >= 2
    types = {v.image_type for v in parsed.visuals}
    # a schematic/diagram and a table should both be found
    assert VisualType.SCHEMATIC in types or VisualType.DIAGRAM in types
    assert VisualType.TABLE in types


def test_parse_pdf_detects_figure_label(sample_pdf: Path):
    parsed = parse_pdf(str(sample_pdf), get_settings().data_dir / "_t_lbl")
    labels = {v.figure_label for v in parsed.visuals if v.figure_label}
    assert any("Figure 12" in lbl for lbl in labels)


def test_parse_pdf_writes_image_files(sample_pdf: Path):
    out = get_settings().data_dir / "_t_files"
    parsed = parse_pdf(str(sample_pdf), out)
    for v in parsed.visuals:
        assert Path(v.image_path).exists()
    # page renders exist too
    assert any(p and Path(p).exists() for p in parsed.page_render_paths)


def test_parse_empty_pdf_raises(tmp_path: Path):
    import fitz

    empty = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()  # a single blank page, no text, no images
    doc.save(str(empty))
    doc.close()
    try:
        parse_pdf(str(empty), tmp_path / "_out")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_ocr_regions_when_available(sample_pdf: Path):
    """If tesseract is present, the table visual should yield grounding regions."""
    parsed = parse_pdf(str(sample_pdf), get_settings().data_dir / "_t_ocr")
    table = next((v for v in parsed.visuals if v.image_type == VisualType.TABLE), None)
    assert table is not None
    if tesseract_available():
        assert len(table.regions) >= 1
        for r in table.regions:
            assert r.bbox.x1 > r.bbox.x0 and r.bbox.y1 > r.bbox.y0
