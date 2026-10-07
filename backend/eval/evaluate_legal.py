"""Retrieval evaluation for the legal knowledge base (mirrors eval/evaluate.py).

Computes Precision@K / Recall@K of the multimodal retriever against
`legal_eval_dataset.json`, whose ground-truth pages are generated directly
from the corpus build manifest (see build_legal_eval_dataset.py) — never
hand-guessed.

Usage (from backend/, after running scripts/build_legal_knowledge_base.py so
the document is already ingested):
    python -m eval.evaluate_legal --k 5
    MMRAG_FAKE_EMBEDDINGS=1 python -m eval.evaluate_legal --k 5   # offline plumbing check
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Set

from app.retrieval.retriever import MultimodalRetriever

EVAL_DIR = Path(__file__).resolve().parent
DOCUMENT_ID = "legal_knowledge_base"


def _precision_recall(retrieved: Set[int], relevant: Set[int]) -> tuple[float, float]:
    if not retrieved:
        return 0.0, 0.0 if relevant else 1.0
    tp = len(retrieved & relevant)
    precision = tp / len(retrieved)
    recall = tp / len(relevant) if relevant else 1.0
    return precision, recall


def evaluate(k: int = 5) -> None:
    dataset = json.loads((EVAL_DIR / "legal_eval_dataset.json").read_text())
    items = dataset["items"]
    retriever = MultimodalRetriever()

    fake = os.getenv("MMRAG_FAKE_EMBEDDINGS", "0") in ("1", "true", "yes")
    print("=" * 72)
    print("LEGAL KNOWLEDGE BASE — RETRIEVAL EVALUATION (Precision@K / Recall@K over pages)")
    print(f"K = {k}   |   embeddings = {'OFFLINE FALLBACK (not real models)' if fake else 'REAL (fine-tuned BGE + CLIP)'}")
    print("=" * 72)

    p_sum = r_sum = 0.0
    for item in items:
        result = retriever.retrieve(DOCUMENT_ID, item["question"], top_k_text=k, top_k_visual=k)
        retrieved_pages: Set[int] = {t.page_number for t in result.text_evidence}
        retrieved_pages |= {v.page_number for v in result.visual_evidence}
        relevant_pages = set(item["relevant_pages"])
        precision, recall = _precision_recall(retrieved_pages, relevant_pages)
        p_sum += precision
        r_sum += recall
        print(f"\n[{item['id']}] {item['question']}")
        print(f"    relevant pages : {sorted(relevant_pages)}")
        print(f"    retrieved pages: {sorted(retrieved_pages)}")
        print(f"    P@{k}={precision:.2f}  R@{k}={recall:.2f}")

    n = len(items)
    print("\n" + "-" * 72)
    print(f"MEAN Precision@{k}: {p_sum / n:.3f}")
    print(f"MEAN Recall@{k}   : {r_sum / n:.3f}")
    print("-" * 72)
    print("\nNOTE: run `python scripts/build_legal_knowledge_base.py` first so the")
    print(f"'{DOCUMENT_ID}' document exists in the vector store before evaluating.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    evaluate(args.k)
