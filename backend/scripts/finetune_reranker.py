"""Fine-tune a cross-encoder reranker on (query, positive, hard-negative) pairs from build_training_pairs.py.

    .venv/bin/python scripts/finetune_reranker.py --pairs ../data_lake/pairs/pairs.jsonl \
        --base cross-encoder/ms-marco-MiniLM-L-6-v2 --out app/models/legal-reranker --max-train 5000

Binary relevance (positive=1, hard negatives=0). Reports val MRR@10 over candidate lists [pos + negs], the
zero-shot base BEFORE training and the tuned model AFTER, so row A4 (zero-shot) vs A5 (fine-tuned) is measured.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from app.models.torch_fixes import assert_finite, de_mmap
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader


def mrr(model, rows) -> float:
    out = []
    for r in rows:
        cands = [r["pos"]] + r["neg"]
        scores = assert_finite(model.predict([(r["query"], c) for c in cands], show_progress_bar=False))
        out.append(1.0 / (1 + int((scores > scores[0]).sum())))
    return round(float(np.mean(out)), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--out", type=Path, default=Path("app/models/legal-reranker"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=5000)
    ap.add_argument("--max-val", type=int, default=300)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rows = [r for r in map(json.loads, args.pairs.open()) if r.get("neg")]
    rng = random.Random(args.seed)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    rng.shuffle(train)
    if not val:  # document-disjoint hold-out when no val-year pairs exist yet
        docs = sorted({r["doc"] for r in train}); rng.shuffle(docs)
        held = set(docs[: max(1, len(docs) // 10)])
        val, train = [r for r in train if r["doc"] in held], [r for r in train if r["doc"] not in held]
        print("WARNING: no val-split pairs with hard negatives; using document-held-out slice")
    train, val = train[: args.max_train], val[: args.max_val]
    print(f"train {len(train)} val {len(val)}")
    if not train or not val:
        raise SystemExit("not enough pairs with hard negatives; ingest more judgments and rebuild pairs")

    model = CrossEncoder(args.base, num_labels=1, max_length=args.max_length, device=args.device)
    de_mmap(model)
    before = mrr(model, val)
    print("zero-shot MRR@10:", before)
    ex = []
    for r in train:
        ex.append(InputExample(texts=[r["query"], r["pos"]], label=1.0))
        for n in r["neg"][:3]:
            ex.append(InputExample(texts=[r["query"], n], label=0.0))
    model.fit(train_dataloader=DataLoader(ex, shuffle=True, batch_size=args.batch), epochs=args.epochs,
              warmup_steps=int(0.1 * len(ex) / args.batch), show_progress_bar=False)
    after = mrr(model, val)
    print("fine-tuned MRR@10:", after)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    (args.out / "training_report.json").write_text(json.dumps({"base": args.base, "n_train": len(train), "n_val": len(val), "zero_shot_mrr10": before, "finetuned_mrr10": after}, indent=2))


if __name__ == "__main__":
    main()
