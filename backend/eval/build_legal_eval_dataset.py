"""Build `legal_eval_dataset.json` from the manifest of the generated legal
knowledge base, so ground-truth `relevant_pages` are REAL page numbers in the
actual PDF that gets ingested (never hand-guessed).

Run once after (re)generating the knowledge base:
    python -m eval.build_legal_eval_dataset
"""
from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
MANIFEST_PATH = BACKEND_DIR / "data" / "legal_knowledge_base_manifest.json"
OUT_PATH = EVAL_DIR / "legal_eval_dataset.json"


def _sample(entries, kind, n):
    return [e for e in entries if e["kind"] == kind][:n]


def build() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    docs = manifest["documents"]

    items = []
    for i, c in enumerate(_sample(docs, "contract", 6), 1):
        p = c["start_page"]
        items.append({
            "id": f"contract_{i}",
            "question": f"Who are the parties in '{c['title']}' and what is the governing law?",
            "relevant_pages": [p],  # parties + key-terms table are on the contract's opening page
            "relevant_figures": [],
            "answer_should_mention": ["parties", "governing law"],
        })
        items.append({
            "id": f"contract_clause_{i}",
            "question": f"What does the Confidentiality clause say in '{c['title']}'?",
            "relevant_pages": [p + 1],  # clauses are rendered on the document's 2nd page
            "relevant_figures": [],
            "answer_should_mention": ["confidential", "disclose"],
        })

    for i, cs in enumerate(_sample(docs, "case", 8), 1):
        p = cs["start_page"]
        items.append({
            "id": f"case_{i}",
            "question": f"What was the outcome and holding in {cs['title']} ({cs['citation']})?",
            "relevant_pages": [p, p + 1],
            "relevant_figures": [],
            "answer_should_mention": ["holding", "outcome"],
        })

    for i, s in enumerate(_sample(docs, "statute", 8), 1):
        p = s["start_page"]
        sec = s["sections"][0]
        items.append({
            "id": f"statute_{i}",
            "question": f"What does Section {sec} of the {s['title']} provide?",
            "relevant_pages": [p, p + 1],
            "relevant_figures": [],
            "answer_should_mention": [s["title"], f"Section {sec}"],
        })

    dataset = {
        "document": "legal_knowledge_base.pdf",
        "document_id": "legal_knowledge_base",
        "notes": (
            "Auto-generated from the synthetic legal knowledge base's build manifest, so "
            "'relevant_pages' are guaranteed to be the REAL start pages of the cited "
            "document inside the actual generated PDF (see app/legal_data/build_pdfs.py). "
            "Run `python -m eval.evaluate_legal --k 5` to compute real retrieval metrics; "
            "never hand-edit metric values."
        ),
        "items": items,
    }
    OUT_PATH.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} eval items -> {OUT_PATH}")


if __name__ == "__main__":
    build()
