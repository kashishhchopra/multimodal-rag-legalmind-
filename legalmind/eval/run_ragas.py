"""Evaluation harness: run the full query pipeline over `eval/testset.jsonl`
and score it with RAGAS (faithfulness, context precision) plus a custom
citation-accuracy check.

This is what distinguishes a research-grade capstone from a demo — see
docs/ARCHITECTURE.md 02 for the target thresholds this script measures
against.

Usage:
    python eval/run_ragas.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from datasets import Dataset

from src.generation.answerer import answer_question
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import rerank

TESTSET_PATH = Path(__file__).parent / "testset.jsonl"
REPORT_PATH = Path(__file__).parent / "report.md"


def load_testset() -> list[dict]:
    with open(TESTSET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_pipeline(question: str, doc_id: str) -> tuple[str, list[str], float]:
    start = time.perf_counter()
    candidates = hybrid_search(question, doc_ids=[doc_id])
    reranked = rerank(question, candidates)
    result = answer_question(question, reranked)
    latency = time.perf_counter() - start
    return result.answer, [c.chunk_id for c in result.citations], latency


def citation_accuracy(predicted_ids: list[str], gold_ids: list[str]) -> float:
    if not gold_ids:
        return 1.0 if not predicted_ids else 0.0
    hits = len(set(predicted_ids) & set(gold_ids))
    return hits / len(gold_ids)


def main() -> None:
    testset = load_testset()

    rows = []
    latencies = []
    citation_scores = []
    parse_path_scores: dict[str, list[float]] = {}

    for item in testset:
        answer, cited_ids, latency = run_pipeline(item["question"], item["doc_id"])
        latencies.append(latency)

        score = citation_accuracy(cited_ids, item.get("gold_chunk_ids", []))
        citation_scores.append(score)
        parse_path_scores.setdefault(item.get("parse_path", "unknown"), []).append(score)

        rows.append({
            "question": item["question"],
            "answer": answer,
            "ground_truth": item["ground_truth"],
            "contexts": [],  # TODO: populate with retrieved chunk texts for RAGAS
        })

    dataset = Dataset.from_list(rows)

    try:
        from ragas import evaluate
        from ragas.metrics import context_precision, faithfulness

        ragas_result = evaluate(dataset, metrics=[faithfulness, context_precision])
    except Exception as exc:  # RAGAS needs populated `contexts`/an LLM judge configured
        ragas_result = {"faithfulness": None, "context_precision": None, "error": str(exc)}

    p95_latency = sorted(latencies)[int(len(latencies) * 0.95) - 1] if latencies else 0.0
    avg_citation_accuracy = sum(citation_scores) / len(citation_scores) if citation_scores else 0.0

    write_report(ragas_result, p95_latency, avg_citation_accuracy, parse_path_scores)


def write_report(ragas_result, p95_latency, avg_citation_accuracy, parse_path_scores) -> None:
    lines = [
        "# LegalMind Evaluation Report",
        "",
        "## Headline metrics",
        f"- Faithfulness (RAGAS): {ragas_result.get('faithfulness', 'n/a') if isinstance(ragas_result, dict) else ragas_result['faithfulness']}",
        f"- Context precision (RAGAS): {ragas_result.get('context_precision', 'n/a') if isinstance(ragas_result, dict) else ragas_result['context_precision']}",
        f"- Citation accuracy: {avg_citation_accuracy:.2%}",
        f"- P95 latency: {p95_latency:.2f}s",
        "",
        "## Citation accuracy by ingestion path (OCR-doc parity check)",
    ]
    for path, scores in parse_path_scores.items():
        lines.append(f"- {path}: {sum(scores) / len(scores):.2%} (n={len(scores)})")

    lines += ["", "## Targets", "", "| Metric | Target | Result |", "|---|---|---|"]
    lines.append(f"| Faithfulness | ≥ 0.90 | {ragas_result.get('faithfulness', 'n/a') if isinstance(ragas_result, dict) else ragas_result['faithfulness']} |")
    lines.append(f"| Context precision | ≥ 0.80 | {ragas_result.get('context_precision', 'n/a') if isinstance(ragas_result, dict) else ragas_result['context_precision']} |")
    lines.append(f"| Citation accuracy | ≥ 85% | {avg_citation_accuracy:.2%} |")
    lines.append(f"| P95 latency | < 8s | {p95_latency:.2f}s |")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
