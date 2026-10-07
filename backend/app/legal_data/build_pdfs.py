"""Render the synthetic legal content (content.py) into real, multimodal PDFs.

Every PDF produced here mixes:
  * body text (clauses / facts / statute text) — indexed as text chunks,
  * a ruled table (key terms / case summary) — extracted by pdfplumber,
  * embedded raster images (seal, signature, court emblem) with captions —
    extracted, OCR'd and CLIP-embedded like any other figure.

This is what makes the generated corpus genuinely *multimodal*, not a text
dump: the pipeline in `app/ingestion/pdf_processor.py` treats these documents
exactly like a real scanned/exported legal PDF.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import wrap
from typing import List, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from .assets import make_court_emblem, make_seal, make_signature
from .content import CaseJudgment, Contract, StatuteSection

PAGE_W, PAGE_H = A4
MARGIN = 22 * mm
LINE_H = 5.0 * mm


class PageWriter:
    """Thin helper for flowing wrapped text across pages with headings/tables/images."""

    def __init__(self, c: canvas.Canvas, header: str):
        self.c = c
        self.header = header
        self.y = PAGE_H - MARGIN
        self._new_page(first=True)

    def _new_page(self, first: bool = False) -> None:
        if not first:
            self.c.showPage()
        self.c.setFont("Helvetica-Oblique", 8)
        self.c.setFillGray(0.4)
        self.c.drawString(MARGIN, PAGE_H - 12 * mm, self.header)
        self.c.setFillGray(0)
        self.y = PAGE_H - 20 * mm

    def _ensure_space(self, needed: float) -> None:
        if self.y - needed < MARGIN:
            self._new_page()

    def heading(self, text: str, size: int = 13) -> None:
        self._ensure_space(9 * mm)
        self.c.setFont("Helvetica-Bold", size)
        self.c.drawString(MARGIN, self.y, text)
        self.y -= 8 * mm

    def subheading(self, text: str, size: int = 11) -> None:
        self._ensure_space(7 * mm)
        self.c.setFont("Helvetica-Bold", size)
        self.c.drawString(MARGIN, self.y, text)
        self.y -= 6 * mm

    def paragraph(self, text: str, size: int = 10, width_chars: int = 100) -> None:
        self.c.setFont("Helvetica", size)
        for raw_line in text.split("\n"):
            for line in (wrap(raw_line, width_chars) or [""]):
                self._ensure_space(LINE_H)
                self.c.drawString(MARGIN, self.y, line)
                self.y -= LINE_H
        self.y -= 2 * mm

    def caption(self, text: str) -> None:
        self._ensure_space(LINE_H)
        self.c.setFont("Helvetica-Oblique", 9)
        self.c.drawString(MARGIN, self.y, text)
        self.y -= LINE_H + 2 * mm

    def table(self, title: str, rows: List[Tuple[str, str]]) -> None:
        self._ensure_space(10 * mm + len(rows) * 7 * mm)
        self.c.setFont("Helvetica-Bold", 10.5)
        self.c.drawString(MARGIN, self.y, title)
        self.y -= 7 * mm
        col1_w, col2_w = 55 * mm, PAGE_W - 2 * MARGIN - 55 * mm
        row_h = 7 * mm
        top = self.y
        n = len(rows)
        self.c.setLineWidth(0.7)
        for i in range(n + 1):
            yy = top - i * row_h
            self.c.line(MARGIN, yy, MARGIN + col1_w + col2_w, yy)
        for x in (MARGIN, MARGIN + col1_w, MARGIN + col1_w + col2_w):
            self.c.line(x, top, x, top - n * row_h)
        for i, (k, v) in enumerate(rows):
            self.c.setFont("Helvetica-Bold", 9)
            self.c.drawString(MARGIN + 2 * mm, top - (i + 1) * row_h + 2.5 * mm, k[:40])
            self.c.setFont("Helvetica", 9)
            self.c.drawString(MARGIN + col1_w + 2 * mm, top - (i + 1) * row_h + 2.5 * mm, str(v)[:60])
        self.y = top - n * row_h - 6 * mm

    def image(self, path: Path, w_mm: float, h_mm: float, caption_text: str = "") -> None:
        self._ensure_space(h_mm * mm + 10 * mm)
        self.c.drawImage(str(path), MARGIN, self.y - h_mm * mm, width=w_mm * mm,
                          height=h_mm * mm, mask="auto")
        self.y -= h_mm * mm + 3 * mm
        if caption_text:
            self.caption(caption_text)

    def page_break(self) -> None:
        self._new_page()


DISCLAIMER = (
    "SYNTHETIC EDUCATIONAL DOCUMENT — This document was generated for the LegalMind "
    "Multimodal RAG final-year project. Names, parties, monetary figures, dates and case "
    "citations are fictitious. Statutory text is paraphrased for demonstration and is NOT an "
    "authoritative legal text. Nothing in this system or corpus constitutes legal advice."
)


def _asset_dir(out_dir: Path) -> Path:
    d = out_dir / "_assets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def render_contract(c: canvas.Canvas, contract: Contract, assets_dir: Path, doc_index: int) -> None:
    pw = PageWriter(c, header=f"{contract.title} — Legal Knowledge Base")
    pw.heading(contract.doc_type.upper())
    pw.subheading(contract.title)
    pw.paragraph(
        f"This {contract.doc_type} (\"Agreement\") is made and entered into as of "
        f"{contract.effective_date}, by and between "
        + " and ".join(f"{name} (\"{role}\")" for role, name in contract.parties) + "."
    )

    pw.table("Key Terms", contract.key_terms)

    seal_path = assets_dir / f"seal_{doc_index % 6}.png"
    if not seal_path.exists():
        make_seal(seal_path, org_name=contract.parties[0][1])
    pw.image(seal_path, 30, 30, caption_text=f"Company Seal — {contract.parties[0][1]}")

    pw.page_break()
    pw.heading("CLAUSES")
    for i, (head, body) in enumerate(contract.clauses, 1):
        pw.subheading(f"{i}. {head}")
        pw.paragraph(body)

    pw.page_break()
    pw.subheading("IN WITNESS WHEREOF")
    pw.paragraph(
        "The Parties have executed this Agreement as of the Effective Date set out above, "
        "in the presence of the witnesses whose signatures appear below."
    )
    for i, (role, name) in enumerate(contract.parties):
        sig_path = assets_dir / f"sig_{doc_index}_{i}.png"
        make_signature(sig_path, name, seed=doc_index * 3 + i)
        pw.image(sig_path, 55, 20, caption_text=f"Signature — {name} ({role})")
    pw.paragraph(DISCLAIMER, size=7.5)


def render_case(c: canvas.Canvas, case: CaseJudgment, assets_dir: Path, doc_index: int) -> None:
    pw = PageWriter(c, header=f"{case.citation} — Legal Knowledge Base")
    emblem_path = assets_dir / f"emblem_{doc_index % 6}.png"
    if not emblem_path.exists():
        make_court_emblem(emblem_path, title=case.court.upper())
    pw.image(emblem_path, 28, 28, caption_text=f"Seal of the {case.court}")
    pw.heading(f"IN THE {case.court.upper()}")
    pw.subheading(case.case_name)
    pw.paragraph(f"Citation: {case.citation}")

    pw.table("Case Summary", [
        ("Court", case.court), ("Date of Judgment", case.date),
        ("Coram", case.judge), ("Outcome", case.outcome),
        ("Citation", case.citation),
    ])

    pw.subheading("FACTS")
    pw.paragraph(case.facts)
    pw.subheading("ISSUES FOR DETERMINATION")
    for i, issue in enumerate(case.issues, 1):
        pw.paragraph(f"{i}. {issue}")
    pw.subheading("HOLDING AND REASONING")
    pw.paragraph(case.holding)
    pw.subheading("ORDER")
    pw.paragraph(f"For the foregoing reasons, it is ordered that: {case.outcome}. "
                  f"No order as to costs save as stated above.")

    sig_path = assets_dir / f"judge_sig_{doc_index}.png"
    make_signature(sig_path, case.judge.replace("Hon'ble Justice ", ""), seed=doc_index + 7)
    pw.image(sig_path, 55, 20, caption_text=f"Signature — {case.judge}")
    pw.paragraph(DISCLAIMER, size=7.5)


def render_statute_group(c: canvas.Canvas, act_name: str, sections: List[StatuteSection],
                          assets_dir: Path, doc_index: int) -> None:
    pw = PageWriter(c, header=f"{act_name} — Legal Knowledge Base")
    emblem_path = assets_dir / f"emblem_statute_{doc_index % 3}.png"
    if not emblem_path.exists():
        make_court_emblem(emblem_path, title="STATUTE BOOK")
    pw.image(emblem_path, 26, 26, caption_text=f"{act_name} — Official Compilation (synthetic)")
    pw.heading(act_name.upper())
    pw.table("Sections in this Chapter", [(f"Section {s.section_no}", s.heading) for s in sections])
    for s in sections:
        pw.subheading(f"Section {s.section_no}. {s.heading}")
        pw.paragraph(s.body)
    pw.paragraph(DISCLAIMER, size=7.5)


def build_legal_knowledge_base(out_path: Path, n_contracts_per_type: int = 3, n_cases: int = 24) -> Path:
    """Build ONE large multi-page multimodal legal PDF covering contracts, case law and statutes.

    Also writes a `<name>_manifest.json` next to the PDF recording the start page of every
    logical document inside it, so the eval dataset / demo questions can cite real pages.
    """
    import json

    from .content import generate_cases, generate_contracts, generate_statute_sections

    out_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = _asset_dir(out_path.parent)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    manifest: List[dict] = []

    # Cover page
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 60 * mm, "LEGAL KNOWLEDGE BASE")
    c.setFont("Helvetica", 13)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 72 * mm, "Contracts • Case Law • Statutes")
    c.setFont("Helvetica-Oblique", 9)
    for i, line in enumerate(wrap(DISCLAIMER, 90)):
        c.drawCentredString(PAGE_W / 2, PAGE_H - 100 * mm - i * 5 * mm, line)
    c.showPage()

    idx = 0
    contracts = generate_contracts(n_per_type=n_contracts_per_type)
    for contract in contracts:
        start_page = c.getPageNumber()
        render_contract(c, contract, assets_dir, idx)
        manifest.append({"kind": "contract", "title": contract.title, "doc_type": contract.doc_type,
                          "start_page": start_page})
        c.showPage()
        idx += 1

    cases = generate_cases(n=n_cases)
    for case in cases:
        start_page = c.getPageNumber()
        render_case(c, case, assets_dir, idx)
        manifest.append({"kind": "case", "title": case.case_name, "citation": case.citation,
                          "start_page": start_page})
        c.showPage()
        idx += 1

    sections = generate_statute_sections()
    by_act: dict[str, List[StatuteSection]] = {}
    for s in sections:
        by_act.setdefault(s.act_name, []).append(s)
    for act_name, secs in by_act.items():
        start_page = c.getPageNumber()
        render_statute_group(c, act_name, secs, assets_dir, idx)
        manifest.append({"kind": "statute", "title": act_name, "start_page": start_page,
                          "sections": [s.section_no for s in secs]})
        c.showPage()
        idx += 1

    c.save()
    out_path.with_name(out_path.stem + "_manifest.json").write_text(
        json.dumps({"n_contracts_per_type": n_contracts_per_type, "n_cases": n_cases,
                    "documents": manifest}, indent=2),
        encoding="utf-8",
    )
    return out_path


def build_standalone_samples(out_dir: Path) -> List[Path]:
    """A few single-document PDFs (one NDA, one judgment, one statute chapter) for
    demoing the "upload your own document" flow with a smaller, focused file."""
    from .content import generate_cases, generate_contracts, generate_statute_sections

    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = _asset_dir(out_dir)
    paths = []

    contract = generate_contracts(n_per_type=1)[0]
    p = out_dir / "sample_nda_agreement.pdf"
    c = canvas.Canvas(str(p), pagesize=A4)
    render_contract(c, contract, assets_dir, 100)
    c.save()
    paths.append(p)

    case = generate_cases(n=1)[0]
    p = out_dir / "sample_case_judgment.pdf"
    c = canvas.Canvas(str(p), pagesize=A4)
    render_case(c, case, assets_dir, 101)
    c.save()
    paths.append(p)

    sections = generate_statute_sections()[:4]
    p = out_dir / "sample_statute_chapter.pdf"
    c = canvas.Canvas(str(p), pagesize=A4)
    render_statute_group(c, "General Contract Code", sections, assets_dir, 102)
    c.save()
    paths.append(p)

    return paths


if __name__ == "__main__":
    import sys

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent.parent / "data" / "legal_knowledge_base.pdf"
    )
    build_legal_knowledge_base(dest)
    print(f"Wrote legal knowledge base PDF -> {dest}")
    samples = build_standalone_samples(dest.parent / "legal_samples")
    print("Wrote standalone samples:", *samples, sep="\n  - ")
