"""Fine-tune an embedder (v2) on REAL legal pairs with hard negatives and Matryoshka loss.

    .venv/bin/python scripts/finetune_embedder_v2.py --pairs ../data_lake/pairs/pairs.jsonl \
        --base BAAI/bge-small-en-v1.5 --out app/models/legal-bge-v2 --epochs 1 --max-train 20000

Loss: MultipleNegativesRankingLoss (in-batch + mined hard negative) wrapped in MatryoshkaLoss so vectors can be
truncated later. Validation = in-pool retrieval on the val split (Recall@5/MRR@10), reported BEFORE and AFTER
training so the gain (or lack of one) is measured, not assumed. Only the train split is used for training.
CPU is slow: ~20k pairs is hours on a laptop; use Colab/Kaggle GPU (--device cuda) for the full set.
Experiment tracking: set MLFLOW_TRACKING_URI and install mlflow to log params/metrics (optional).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load(path: Path, split: str, limit: int, seed: int):
    rows = [r for r in map(json.loads, path.open()) if r["split"] == split]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def evaluator(rows):
    queries = {f"q{i}": QUERY_PREFIX + r["query"] for i, r in enumerate(rows)}
    corpus = {f"d{i}": r["pos"] for i, r in enumerate(rows)}
    rel = {f"q{i}": {f"d{i}"} for i in range(len(rows))}
    return InformationRetrievalEvaluator(queries, corpus, rel, name="val", mrr_at_k=[10], accuracy_at_k=[1, 5, 10], show_progress_bar=False)


def score(model, ev):
    res = ev(model)
    return {k.split("val_")[-1]: round(float(v), 4) for k, v in res.items() if "accuracy@5" in k or "mrr@10" in k or "ndcg@10" in k}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", type=Path, default=Path("app/models/legal-bge-v2"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--max-val", type=int, default=1000)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    train, val = load(args.pairs, "train", args.max_train, args.seed), load(args.pairs, "val", args.max_val, args.seed)
    if not val:  # no val-year judgments yet: fall back to a held-out slice of train docs (document-disjoint)
        docs = sorted({r["doc"] for r in train})
        random.Random(args.seed).shuffle(docs)
        held = set(docs[: max(1, len(docs) // 10)])
        val = [r for r in train if r["doc"] in held][: args.max_val]
        train = [r for r in train if r["doc"] not in held]
        print("WARNING: no val-split pairs; using a document-held-out slice of train for validation")
    print(f"train {len(train)}  val {len(val)}")

    model = SentenceTransformer(args.base, device=args.device)
    model.max_seq_length = args.max_seq
    ev = evaluator(val)
    before = score(model, ev)
    print("before:", before)

    cols = {"anchor": [], "positive": [], "negative": []}
    for r in train:
        cols["anchor"].append(QUERY_PREFIX + r["query"])
        cols["positive"].append(r["pos"])
        cols["negative"].append(random.Random(hash(r["query"]) & 0xFFFF).choice(r["neg"]) if r.get("neg") else random.choice(train)["pos"])
    ds = Dataset.from_dict(cols)

    dim = model.get_sentence_embedding_dimension()
    loss = losses.MatryoshkaLoss(model, losses.MultipleNegativesRankingLoss(model), matryoshka_dims=[dim, dim // 2, dim // 4])
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(args.out / "checkpoints"), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch,
        learning_rate=args.lr, warmup_ratio=0.1, batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=50,
        save_strategy="epoch", save_total_limit=1, report_to=["mlflow"] if _has_mlflow() else "none", seed=args.seed,
        use_cpu=args.device == "cpu",
    )
    SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss).train()
    after = score(model, ev)
    print("after:", after)
    model.save(str(args.out))
    (args.out / "training_report.json").write_text(json.dumps({"base": args.base, "n_train": len(train), "n_val": len(val), "before": before, "after": after, "epochs": args.epochs, "batch": args.batch, "lr": args.lr}, indent=2))
    print("saved", args.out)


def _has_mlflow() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
