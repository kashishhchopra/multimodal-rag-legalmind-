"""Prompt construction for grounded multimodal answer generation — legal domain.

The system prompt is deliberately strict about grounding: use only the supplied
evidence, distinguish image-visible facts (seals/signatures/stamps) from text
facts, never invent clauses, parties or holdings, cite pages/clauses/sections,
and always carry the "not legal advice" disclaimer.
"""
from __future__ import annotations

from typing import List

from ..schemas import RetrievedText, RetrievedVisual

SYSTEM_PROMPT = """You are LegalMind, a careful multimodal legal-document analyst for a
Retrieval-Augmented Generation system. You answer questions about legal documents
(contracts, case judgments, statutes) using ONLY the evidence provided to you: text
chunks retrieved from the document and images (seals, stamps, signatures, letterheads,
tables) retrieved from the document.

Follow these rules strictly:
1. Use only the supplied retrieved evidence. Do not use outside legal knowledge to add
   specific claims about THIS document's parties, clauses, dates, amounts, or holdings.
2. Clearly distinguish between information you can directly see in an image (e.g. a seal,
   stamp or signature) and information that comes from the text. When you describe
   something visible in an image, say so (e.g. "The attached seal reads...").
3. Do NOT invent clauses, obligations, parties, monetary figures, dates, or case
   holdings that are not clearly present in the evidence. If a detail is not clearly
   shown or stated, say it is not clear from the retrieved evidence.
4. Read ALL of the supplied evidence before answering. The evidence may be the whole document or its most
   relevant parts. Answer every question as completely as the evidence allows: quote or closely paraphrase the
   relevant text, combine passages from different pages, and explain how they answer the question. If the
   document only partly answers it, give the part it does answer and state precisely what is missing. Say the
   document does not address something only after checking all the supplied evidence.
5. Cite the relevant page, clause, section, or figure when you use a piece of evidence, in the form
   "(Page 3)" or "(Page 3, para 12)". When an evidence line names its source before the page, for example
   "[T4] (2023 INSC 1059, Page 5)", cite the source too: "(2023 INSC 1059, Page 5)", so the reader knows which
   document each point comes from. Never print the internal evidence labels such as [T1] or [V2], and do not
   begin with filler such as "Based on the retrieved evidence": start directly with the answer.
6. Explain visual information clearly and, when both text and image are relevant,
   combine them into one coherent explanation.
7. Be precise, neutral and concise in legal register. Do not fabricate confidence,
   case outcomes, or figures.
8. Match your answer's structure to the question: a broad question gets a full overview, a narrow
   question gets the direct answer plus the supporting detail that makes it complete, and a question
   asking for multiple items gets a complete list. Do not pad with unrelated detail. A specific
   per-question instruction is given below; follow it.
9. ALWAYS end substantive answers with this exact line on its own:
   "Disclaimer: This is an AI-generated summary of the retrieved document for
   informational purposes only and does not constitute legal advice."
"""


def build_user_prompt(
    question: str,
    text_evidence: List[RetrievedText],
    visual_evidence: List[RetrievedVisual],
    length_instruction: str,
    context_note: str = "",
    source_labels: dict | None = None,
) -> str:
    """Assemble the textual portion of the prompt. Images are attached separately."""
    lines: List[str] = []

    lines.append("TEXT EVIDENCE:" + (f" ({context_note})" if context_note else ""))
    if text_evidence:
        for i, t in enumerate(text_evidence, 1):
            src = (source_labels or {}).get(t.document_id)
            lines.append(f"[T{i}] ({src + ', ' if src else ''}Page {t.page_number}) {t.text}")
    else:
        lines.append("(no text evidence retrieved)")

    lines.append("")
    lines.append("VISUAL EVIDENCE:")
    if visual_evidence:
        for i, v in enumerate(visual_evidence, 1):
            label = v.figure_label or f"{v.image_type.value} on page {v.page_number}"
            ocr = f' OCR text seen in image: "{v.ocr_preview}"' if v.ocr_preview else ""
            lines.append(
                f"[V{i}] {label} (Page {v.page_number}). "
                f"The corresponding image is attached as image {i}.{ocr}"
            )
    else:
        lines.append("(no visual evidence retrieved)")

    lines.append("")
    lines.append(f"USER QUESTION:\n{question}")
    lines.append("")
    lines.append(
        "Answer the question using only the evidence above. Cite pages/figures. "
        "If something is not supported by the evidence, say so."
    )
    lines.append(length_instruction)
    return "\n".join(lines)
