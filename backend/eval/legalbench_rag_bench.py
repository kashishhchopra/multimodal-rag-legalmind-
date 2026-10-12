"""LegalBench-RAG snippet-level retrieval benchmark (Pipitone & Alami 2024).

    .venv/bin/python -m eval.legalbench_rag_bench --benchmarks privacy_qa contractnli --max-queries 100 \
        --models BAAI/bge-small-en-v1.5

Data: `python scripts/datasets_cli.py fetch legalbench_rag`. Each test case has a query and ground-truth snippets
(file + character span). Documents are cut into fixed windows; for the top-k retrieved windows we compute the paper's
character-overlap metrics:  Precision@k = overlap chars / retrieved chars,  Recall@k = overlap chars / ground-truth chars.
To stay tractable on a laptop the corpus is restricted to the files that sampled queries point to plus --distractors other
files, so absolute numbers are NOT comparable with the paper's full-corpus results; compare configurations to each other.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from app.retrieval.hybrid import BM25Index, rrf_fuse

ROOT = Path(__file__).resolve().parents[2] / "data_lake" / "manual" / "legalbench_rag" / "extracted"
DEVICE = "cpu"
QPREFIX = "Represent this sentence for searching relevant passages: "


def windows(text: str, size: int, stride: int) -> List[Tuple[int, int]]:
    return [(s, min(len(text), s + size)) for s in range(0, max(1, len(text) - size + stride), stride)]


def overlap(a: Tuple[int, int], b: List[int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def score(hits: Dict[str, List[Tuple[str, int, int]]], tests, ks=(1, 2, 4, 8, 16, 32)):
    out = {}
    for k in ks:
        ps, rs = [], []
        for ti, t in enumerate(tests):
            got = hits[ti][:k]
            gt = t["snippets"]
            gt_chars = sum(s["span"][1] - s["span"][0] for s in gt)
            ov = sum(overlap((a, b), s["span"]) for f, a, b in got for s in gt if s["file_path"] == f)
            ret_chars = sum(b - a for _, a, b in got) or 1
            ps.append(min(1.0, ov / ret_chars))
            rs.append(min(1.0, ov / gt_chars))
        out[f"P@{k}"], out[f"R@{k}"] = round(float(np.mean(ps)), 4), round(float(np.mean(rs)), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["privacy_qa"])
    ap.add_argument("--models", nargs="+", default=["BAAI/bge-small-en-v1.5"])
    ap.add_argument("--max-queries", type=int, default=100)
    ap.add_argument("--distractors", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--stride", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="eval/results/legalbench_rag.json")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    rng = random.Random(args.seed)
    results = []
    for bench in args.benchmarks:
        tests = json.load(open(ROOT / "benchmarks" / f"{bench}.json"))["tests"]
        rng.shuffle(tests)
        tests = tests[: args.max_queries]
        need = sorted({s["file_path"] for t in tests for s in t["snippets"]})
        pool = sorted(p.relative_to(ROOT / "corpus").as_posix() for p in (ROOT / "corpus" / bench).rglob("*.txt"))
        extra = [p for p in pool if p not in need]
        rng.shuffle(extra)
        files = need + extra[: args.distractors]
        docs, spans, ids = {}, [], []
        for f in files:
            docs[f] = (ROOT / "corpus" / f).read_text(errors="ignore")
            for a, b in windows(docs[f], args.chunk, args.stride):
                spans.append((f, a, b)); ids.append(str(len(ids)))
        texts = [docs[f][a:b] for f, a, b in spans]
        print(f"== {bench}: {len(tests)} queries, {len(files)} files, {len(texts)} chunks", flush=True)
        qs = [t["query"] for t in tests]
        lex_idx = BM25Index(ids, texts)
        lex = [[i for i, _ in lex_idx.search(q, 64)] for q in qs]
        to_hits = lambda ranked: {ti: [spans[int(i)] for i in r] for ti, r in enumerate(ranked)}
        results.append({"benchmark": bench, "config": "BM25", **score(to_hits(lex), tests)})
        for m in args.models:
            model = SentenceTransformer(m, device=args.device)
            t0 = time.time()
            d = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
            q = model.encode([QPREFIX + x for x in qs], normalize_embeddings=True, show_progress_bar=False)
            dense = [[ids[j] for j in np.argsort(-(q[i] @ d.T))[:64]] for i in range(len(qs))]
            results.append({"benchmark": bench, "config": f"dense {m}", "encode_s": round(time.time() - t0, 1), **score(to_hits(dense), tests)})
            hyb = [[i for i, _ in rrf_fuse([dense[x], lex[x]])[:64]] for x in range(len(qs))]
            results.append({"benchmark": bench, "config": f"hybrid {m}", **score(to_hits(hyb), tests)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("\n| benchmark | config | P@1 | R@1 | P@4 | R@4 | P@16 | R@16 |\n|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['benchmark']} | {r['config']} | " + " | ".join(f"{r[k]:.3f}" for k in ("P@1", "R@1", "P@4", "R@4", "P@16", "R@16")) + " |")


if __name__ == "__main__":
    main()
