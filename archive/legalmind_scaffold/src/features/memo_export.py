"""STRETCH feature: summary-memo generation & export.

Turns the tool from a chatbot into something that produces a deliverable —
a structured brief (parties, key dates, obligations, risks, open questions)
exportable as PDF/DOCX.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from src.generation.answerer import call_llm
from src.generation.prompts import MEMO_EXPORT_PROMPT
from src.retrieval.hybrid_retriever import hybrid_search


class MemoSection(BaseModel):
    title: str
    content: str


class Memo(BaseModel):
    doc_id: str
    sections: list[MemoSection]


MEMO_SECTIONS = ["Parties", "Key Dates", "Obligations", "Risks", "Open Questions"]


def generate_memo(doc_id: str) -> Memo:
    sections: list[MemoSection] = []
    for section_title in MEMO_SECTIONS:
        hits = hybrid_search(section_title, doc_ids=[doc_id])
        context_block = "\n---\n".join(
            f"[chunk_id={h.chunk.chunk_id}] (page {h.chunk.metadata.page})\n{h.chunk.text}"
            for h in hits
        )
        prompt = MEMO_EXPORT_PROMPT.format(context_block=context_block or "(no relevant context found)")
        content = call_llm(prompt).strip()
        sections.append(MemoSection(title=section_title, content=content))

    return Memo(doc_id=doc_id, sections=sections)


def export_memo_to_pdf(memo: Memo, out_path: str | Path) -> Path:
    # TODO: render via reportlab/weasyprint for a formatted PDF; a plain-text
    # fallback is provided so the export endpoint has something to return
    # while that's wired up.
    out_path = Path(out_path)
    lines = [f"LegalMind Summary Memo — {memo.doc_id}", ""]
    for section in memo.sections:
        lines += [section.title, "-" * len(section.title), section.content, ""]
    out_path.write_text("\n".join(lines))
    return out_path
