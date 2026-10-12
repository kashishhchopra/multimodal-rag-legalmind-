"""Time-split retrieval evaluation on real Supreme Court judgments (no training leakage).

    .venv/bin/python -m eval.timesplit_eval --models BAAI/bge-small-en-v1.5 app/models/legal-bge-v2 \
        --reranker app/models/legal-reranker --test-year 2024

Queries: the "Issue for Consideration" text of judgments from --test-year onward (pairs.jsonl split == "test", source
"issue"). Pool: all reasoning/fact chunks of those same-year judgments plus a sample of earlier judgments as distractors.
Metric is document-level: a query is answered correctly at rank k if one of the top-k chunks comes from its own judgment.
Training used only judgments up to --train-max-year, so these queries and their documents were never seen in training.
Reports R@1/5/10, MRR@10 for dense, BM25, hybrid RRF and (optional) hybrid + cross-encoder reranking.
Caveat: issue statements share vocabulary with their judgments, so absolute numbers are optimistic; compare configurations.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from app.retrieval.hybrid import BM25Index, rrf_fuse

LAKE = Path(__file__).resolve().parents[2] / "data_lake"
QP = "Represent this sentence for searching relevant passages: "


def metrics(ranked_docs, gold):
    out = {}
    for k in (1, 5, 10):
        out[f"R@{k}"] = round(float(np.mean([g in r[:k] for r, g in zip(ranked_docs, gold)])), 4)
    rr = []
    for r, g in zip(ranked_docs, gold):
        rr.append(next((1 / (i + 1) for i, d in enumerate(r[:10]) if d == g), 0.0))
    out["MRR@10"] = round(float(np.mean(rr)), 4)
    return out


def dedupe_docs(chunk_ids, chunk_doc, depth=10):
    seen, out = set(), []
    for c in chunk_ids:
        d = chunk_doc[c]
        if d not in seen:
            seen.add(d); out.append(d)
        if len(out) >= depth:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["BAAI/bge-small-en-v1.5"])
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--test-year", type=int, default=2024)
    ap.add_argument("--max-queries", type=int, default=150)
    ap.add_argument("--distractor-docs", type=int, default=150)
    ap.add_argument("--rerank-top", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="eval/results/timesplit.json")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    from sentence_transformers import CrossEncoder, SentenceTransformer

    rng = random.Random(a.seed)
    pairs = [r for r in map(json.loads, (LAKE / "pairs" / "pairs.jsonl").open()) if r["split"] == "test" and r["source"] == "issue"]
    rng.shuffle(pairs)
    pairs = pairs[: a.max_queries]
    test_docs = {p["doc"] for p in pairs}
    docs = {d["doc_id"]: d for d in map(json.loads, (LAKE / "corpus" / "docs.jsonl").open())}
    older = [d for d, v in docs.items() if v["year"] < a.test_year and d not in test_docs]
    rng.shuffle(older)
    pool_docs = test_docs | set(older[: a.distractor_docs])
    chunks = [c for c in map(json.loads, (LAKE / "corpus" / "chunks.jsonl").open())
              if c["doc_id"] in pool_docs and c["section"] in ("judgment", "analysis", "facts")]
    ids, texts = [c["chunk_id"] for c in chunks], [c["text"] for c in chunks]
    chunk_doc = {c["chunk_id"]: c["doc_id"] for c in chunks}
    qs, gold = [p["query"] for p in pairs], [p["doc"] for p in pairs]
    print(f"{len(qs)} queries, {len(pool_docs)} docs, {len(texts)} chunks", flush=True)

    bm = BM25Index(ids, texts)
    lex = [[i for i, _ in bm.search(q, 100)] for q in qs]
    results = [{"config": "BM25", **metrics([dedupe_docs(r, chunk_doc) for r in lex], gold)}]
    for m in a.models:
        model = SentenceTransformer(m, device=a.device)
        model.max_seq_length = 256
        d = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        q = model.encode([QP + x for x in qs], normalize_embeddings=True, show_progress_bar=False)
        dense = [[ids[j] for j in np.argsort(-(q[i] @ d.T))[:100]] for i in range(len(qs))]
        results.append({"config": f"dense {m}", **metrics([dedupe_docs(r, chunk_doc) for r in dense], gold)})
        hyb = [[i for i, _ in rrf_fuse([dense[x], lex[x]])[:100]] for x in range(len(qs))]
        results.append({"config": f"hybrid {m}", **metrics([dedupe_docs(r, chunk_doc) for r in hyb], gold)})
        if a.reranker:
            ce = CrossEncoder(a.reranker, max_length=384, device=a.device)
            by_id = dict(zip(ids, texts))
            rer = []
            for x, cands in enumerate(hyb):
                head = cands[: a.rerank_top]
                sc = ce.predict([(qs[x][:600], by_id[c]) for c in head], show_progress_bar=False)
                rer.append([c for _, c in sorted(zip(sc, head), key=lambda t: -t[0])] + cands[a.rerank_top:])
            results.append({"config": f"hybrid+rerank({Path(a.reranker).name}) {m}", **metrics([dedupe_docs(r, chunk_doc) for r in rer], gold)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"test_year": a.test_year, "n_queries": len(qs), "n_chunks": len(texts), "results": results}, indent=2))
    print("\n| config | R@1 | R@5 | R@10 | MRR@10 |\n|---|---|---|---|---|")
    for r in results:
        print(f"| {r['config']} | {r['R@1']:.3f} | {r['R@5']:.3f} | {r['R@10']:.3f} | {r['MRR@10']:.3f} |")


if __name__ == "__main__":
    main()
