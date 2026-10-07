import fitz
import pytest

from src.ingestion.native_parser import parse_native_page
from src.ingestion.router import classify_page
from src.schemas import ParsePath


def _make_pdf_with_text(text: str) -> "fitz.Document":
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    return doc


def test_classify_page_native_when_text_present():
    doc = _make_pdf_with_text("This is a clause with enough characters to count as native text.")
    assert classify_page(doc[0]) == ParsePath.NATIVE
    doc.close()


def test_classify_page_ocr_when_no_text_layer():
    doc = fitz.open()
    doc.new_page()  # blank page, no text layer
    assert classify_page(doc[0]) == ParsePath.OCR
    doc.close()


def test_native_parser_returns_word_boxes():
    doc = _make_pdf_with_text("Section 1. Termination.")
    parsed = parse_native_page(doc[0], doc_id="test-doc", page_number=1)

    assert parsed.parse_path == ParsePath.NATIVE
    assert "Termination" in parsed.text
    assert len(parsed.word_boxes) > 0
    doc.close()
