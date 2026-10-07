"""Retrieval evaluation harness (spec §18).

Computes Precision@K and Recall@K of the multimodal retriever against a small,
hand-labelled dataset (eval_dataset.json). It measures whether the retriever
surfaces the *pages* and *figures* a human marked as relevant.

IMPORTANT
---------
* Numbers are computed from ACTUAL retrieval — nothing here is fabricated.
* Answer-quality metrics (groundedness, citation correctness, visual
  understanding) require the real LLM and human judgement; this script prints a
  scoring checklist for them rather than inventing scores.
* If MMRAG_FAKE_EMBEDDINGS=1, retrieval uses the offline lexical fallback, so the
  scores reflect plumbing, not the real BGE/CLIP models. Run with the real models
  (unset the flag, set an LLM key optional) for meaningful retrieval numbers.

Usage (from backend/):
    python -m eval.evaluate               # ingests the sample PDF and evaluates
    python -m eval.evaluate --k 3
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
from pathlib import Path
from typing import List, Set

from app.config import get_settings
from app.sample_pdf import build_sample_pdf
from app.retrieval.retriever import MultimodalRetriever
from app.schemas import Document
from app.services.ingest_service import process_document
from app.utils import registry

EVAL_DIR = Path(__file__).resolve().parent


def _prepare_sample_document() -> str:
    """Build + ingest the sample PDF, return its document_id."""
    doc_id = "eval_sample"
    data_dir = get_settings().data_dir
    pdf = data_dir / "eval_electronics_manual.pdf"
    build_sample_pdf(pdf)

    out_dir = data_dir / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(pdf, out_dir / "source.pdf")
    registry.save(
        Document(
            document_id=doc_id,
            filename="electronics_manual.pdf",
            upload_time=datetime.datetime.utcnow().isoformat() + "Z",
        )
    )
    process_document(doc_id, str(out_dir / "source.pdf"), "electronics_manual.pdf")
    return doc_id


def _precision_recall(retrieved: Set[int], relevant: Set[int]) -> tuple[float, float]:
    if not retrieved:
        return 0.0, 0.0 if relevant else 1.0
    tp = len(retrieved & relevant)
    precision = tp / len(retrieved)
    recall = tp / len(relevant) if relevant else 1.0
    return precision, recall


def evaluate(k: int = 5) -> None:
    dataset = json.loads((EVAL_DIR / "eval_dataset.json").read_text())
    items = dataset["items"]
    doc_id = _prepare_sample_document()
    retriever = MultimodalRetriever()

    fake = os.getenv("MMRAG_FAKE_EMBEDDINGS", "0") in ("1", "true", "yes")
    print("=" * 72)
    print("RETRIEVAL EVALUATION  (Precision@K / Recall@K over relevant PAGES)")
    print(f"K = {k}   |   embeddings = {'OFFLINE FALLBACK (not real models)' if fake else 'REAL (BGE + CLIP)'}")
    print("=" * 72)

    p_sum = r_sum = fig_hit = fig_total = 0.0
    for item in items:
        result = retriever.retrieve(doc_id, item["question"], top_k_text=k, top_k_visual=k)
        retrieved_pages: Set[int] = {t.page_number for t in result.text_evidence}
        retrieved_pages |= {v.page_number for v in result.visual_evidence}
        relevant_pages = set(item["relevant_pages"])
        precision, recall = _precision_recall(retrieved_pages, relevant_pages)
        p_sum += precision
        r_sum += recall

        # figure-level recall (did we retrieve the labelled figure?)
        retrieved_figs = {v.figure_label for v in result.visual_evidence if v.figure_label}
        for fig in item.get("relevant_figures", []):
            fig_total += 1
            if fig in retrieved_figs:
                fig_hit += 1

        print(f"\n[{item['id']}] {item['question']}")
        print(f"    relevant pages : {sorted(relevant_pages)}")
        print(f"    retrieved pages: {sorted(retrieved_pages)}")
        print(f"    P@{k}={precision:.2f}  R@{k}={recall:.2f}   "
              f"figures wanted={item.get('relevant_figures', [])} got={sorted(retrieved_figs)}")

    n = len(items)
    print("\n" + "-" * 72)
    print(f"MEAN Precision@{k}: {p_sum / n:.3f}")
    print(f"MEAN Recall@{k}   : {r_sum / n:.3f}")
    if fig_total:
        print(f"Figure retrieval recall: {fig_hit / fig_total:.3f}  ({int(fig_hit)}/{int(fig_total)})")
    print("-" * 72)

    print("\nANSWER-QUALITY SCORING CHECKLIST (score manually with the LLM enabled):")
    print("  For each question, rate 0-1 on:")
    print("   • Groundedness      — every claim traceable to retrieved evidence")
    print("   • Citation accuracy — cited pages/figures actually contain the fact")
    print("   • Visual grounding  — image-derived claims match what the figure shows")
    print("   • Relevance         — the answer addresses the question")
    print("  The dataset lists `answer_should_mention` keywords to speed up scoring.")
    print("  Do NOT report averaged answer scores unless you actually graded them.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    evaluate(args.k)
