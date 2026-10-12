"""Retrieval ablation on BEIR/MTEB-format datasets (AILA case docs/statutes, LegalBench consumer QA, ...).

Fills the roadmap's ablation rows with REAL numbers:
  A0 bge-small base (dense)      A1 legal-bge-finetuned v1 (synthetic) (dense)
  A3 hybrid = dense + BM25 (RRF) A4 hybrid + zero-shot cross-encoder   (--reranker NAME)
  (A2 / A5 appear automatically once --model points at an embedder v2 / a fine-tuned reranker.)

    python -m eval.retrieval_bench --datasets mteb__AILA_casedocs mteb__AILA_statutes \
        --models BAAI/bge-small-en-v1.5 app/models/legal-bge-finetuned --out eval/results/ablation.json

Datasets come from `python scripts/datasets_cli.py fetch aila_casedocs` (data_lake/hf/<name>/).
Long documents are truncated to the model's max length (512 tokens for bge) — a known limitation
for judgment-length text; chunk-level evaluation is the next step.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from app.models.torch_fixes import assert_finite, de_mmap
from app.retrieval.hybrid import BM25Index, rrf_fuse

DEVICE = "cpu"  # MPS has stalled on long-document batches on macOS; override with --device
LAKE = Path(__file__).resolve().parents[2] / "data_lake" / "hf"


def load_beir(name: str, split: str = "test"):
    base = LAKE / name
    one = lambda p: pd.concat([pd.read_parquet(f) for f in sorted((base / p).glob("*.parquet"))])
    corpus = one("corpus/corpus")
    queries = one("queries/queries")
    qrels_df = one(f"default/{split}")
    qrels: Dict[str, Dict[str, float]] = {}
    for r in qrels_df.itertuples():
        if r.score > 0:
            qrels.setdefault(str(getattr(r, "_1")), {})[str(getattr(r, "_2"))] = float(r.score)
    docs = {str(r._1): ((r.title + ". ") if r.title else "") + r.text for r in corpus.itertuples()}
    qs = {str(r._1): r.text for r in queries.itertuples() if str(r._1) in qrels}
    return docs, qs, qrels


# ------------------------------------------------------------------ metrics
def recall_at_k(ranked: Sequence[str], rel: Dict[str, float], k: int) -> float:
    return len(set(ranked[:k]) & set(rel)) / len(rel)


def mrr_at_k(ranked: Sequence[str], rel: Dict[str, float], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k], 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], rel: Dict[str, float], k: int = 10) -> float:
    dcg = sum(rel.get(d, 0.0) / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1))
    ideal = sum(g / math.log2(i + 1) for i, g in enumerate(sorted(rel.values(), reverse=True)[:k], 1))
    return dcg / ideal if ideal else 0.0


def average_precision(ranked: Sequence[str], rel: Dict[str, float]) -> float:
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def evaluate(rankings: Dict[str, List[str]], qrels: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    qids = list(qrels)
    avg = lambda f: round(float(np.mean([f(rankings.get(q, []), qrels[q]) for q in qids])), 4)
    return {
        "R@5": avg(lambda r, rel: recall_at_k(r, rel, 5)),
        "R@10": avg(lambda r, rel: recall_at_k(r, rel, 10)),
        "MRR@10": avg(mrr_at_k),
        "nDCG@10": avg(ndcg_at_k),
        "MAP": avg(average_precision),
    }


# ------------------------------------------------------------------ rankers
def dense_rank(model_name: str, docs: Dict[str, str], qs: Dict[str, str], depth: int):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=DEVICE)
    ids = list(docs)
    t0 = time.time()
    d = model.encode([docs[i] for i in ids], batch_size=16, normalize_embeddings=True, show_progress_bar=False)
    # bge retrieval models expect this instruction on queries only
    prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_name.lower() or "legal-bge" in model_name else ""
    qids = list(qs)
    q = model.encode([prefix + qs[i] for i in qids], normalize_embeddings=True, show_progress_bar=False)
    sims = q @ d.T
    out = {qid: [ids[j] for j in np.argsort(-sims[n])[:depth]] for n, qid in enumerate(qids)}
    return out, time.time() - t0


def bm25_rank(docs: Dict[str, str], qs: Dict[str, str], depth: int):
    idx = BM25Index(list(docs), list(docs.values()))
    return {q: [i for i, _ in idx.search(t, depth)] for q, t in qs.items()}


def hybrid_rank(dense: Dict[str, List[str]], lexical: Dict[str, List[str]], depth: int, rrf_k: int = 60):
    return {q: [i for i, _ in rrf_fuse([dense[q], lexical.get(q, [])], rrf_k)[:depth]] for q in dense}


def rerank(model_name: str, ranked: Dict[str, List[str]], docs: Dict[str, str], qs: Dict[str, str], top: int):
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(model_name, max_length=512, device=DEVICE)
    de_mmap(ce)
    out = {}
    for q, cands in ranked.items():
        head = cands[:top]
        scores = assert_finite(ce.predict([(qs[q], docs[d]) for d in head], show_progress_bar=False))
        out[q] = [d for _, d in sorted(zip(scores, head), key=lambda x: -x[0])] + cands[top:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mteb__AILA_casedocs"])
    ap.add_argument("--models", nargs="+", default=["BAAI/bge-small-en-v1.5"])
    ap.add_argument("--reranker", default=None, help="cross-encoder name/path, e.g. BAAI/bge-reranker-base")
    ap.add_argument("--rerank-top", type=int, default=30)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="eval/results/ablation.json")
    args = ap.parse_args()
    global DEVICE
    DEVICE = args.device

    results = []
    for ds in args.datasets:
        docs, qs, qrels = load_beir(ds)
        print(f"\n== {ds}: {len(docs)} docs, {len(qs)} queries")
        lex = bm25_rank(docs, qs, args.depth)
        results.append({"dataset": ds, "config": "BM25 only", **evaluate(lex, qrels)})
        for m in args.models:
            dense, secs = dense_rank(m, docs, qs, args.depth)
            results.append({"dataset": ds, "config": f"dense: {m}", "encode_s": round(secs, 1), **evaluate(dense, qrels)})
            hyb = hybrid_rank(dense, lex, args.depth)
            results.append({"dataset": ds, "config": f"hybrid(RRF): {m}", **evaluate(hyb, qrels)})
            if args.reranker:
                rr = rerank(args.reranker, hyb, docs, qs, args.rerank_top)
                results.append({"dataset": ds, "config": f"hybrid+rerank({args.reranker}, top{args.rerank_top}): {m}", **evaluate(rr, qrels)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    cols = ["R@5", "R@10", "MRR@10", "nDCG@10", "MAP"]
    print("\n| dataset | config | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols))
    for r in results:
        print(f"| {r['dataset']} | {r['config']} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")


if __name__ == "__main__":
    main()
