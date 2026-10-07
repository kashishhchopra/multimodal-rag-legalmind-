"""Fine-tune the text-embedding model on the legal domain.

This is the "training" step of the project: we take the general-purpose
`BAAI/bge-small-en-v1.5` sentence embedding model and continue training it,
with `MultipleNegativesRankingLoss` (in-batch negatives), on ~900 synthetic
(question, relevant-passage) pairs drawn from the generated legal corpus
(contracts, case judgments, statute sections — see `app/legal_data/`).

Why this approach:
  * A from-scratch LLM or a generative model cannot be meaningfully trained on
    a laptop/CPU in a course-project timeframe with a small dataset. A
    *retrieval* embedding model can be — bi-encoder fine-tuning with
    contrastive loss on a few hundred/thousand pairs measurably improves
    domain retrieval quality in a few minutes on CPU, which is exactly what
    this system's retrieval quality depends on.
  * The fine-tuned model is saved locally and becomes the app's default text
    embedding model for legal documents (see `app/config.py`), while staying
    a drop-in `sentence-transformers` model — no changes needed elsewhere in
    the retrieval pipeline.

Usage (from backend/, with the venv active):
    python scripts/finetune_legal_embeddings.py
    python scripts/finetune_legal_embeddings.py --epochs 4 --batch-size 16

Output: app/models/legal-bge-finetuned/  (a full sentence-transformers model dir)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("finetune_legal")

BASE_MODEL = "BAAI/bge-small-en-v1.5"
OUT_DIR = BACKEND_DIR / "app" / "models" / "legal-bge-finetuned"
QA_PATH = BACKEND_DIR / "app" / "legal_data" / "legal_qa_pairs.json"


def _load_pairs() -> list[dict]:
    if not QA_PATH.exists():
        logger.info("QA pairs file missing, generating it now...")
        from app.legal_data.qa_pairs import generate_qa_pairs

        pairs = generate_qa_pairs()
        QA_PATH.write_text(json.dumps([p.__dict__ for p in pairs], indent=2), encoding="utf-8")
    return json.loads(QA_PATH.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-holdout", type=int, default=60, help="pairs held out for a before/after retrieval check")
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    pairs = _load_pairs()
    random.Random(42).shuffle(pairs)
    holdout, train_pairs = pairs[: args.eval_holdout], pairs[args.eval_holdout :]
    logger.info("Loaded %d QA pairs (%d train / %d holdout for eval)", len(pairs), len(train_pairs), len(holdout))

    # BGE convention: prefix the *query* side with the retrieval instruction.
    query_prefix = "Represent this sentence for searching relevant passages: "
    train_examples = [InputExample(texts=[query_prefix + p["question"], p["passage"]]) for p in train_pairs]

    logger.info("Loading base model: %s", BASE_MODEL)
    model = SentenceTransformer(BASE_MODEL)

    def recall_at_5(m: SentenceTransformer) -> float:
        """Quick sanity metric: for each holdout question, is its own passage the
        top-5 nearest passage among ALL holdout passages?"""
        import numpy as np

        qs = [query_prefix + p["question"] for p in holdout]
        ps = [p["passage"] for p in holdout]
        q_vecs = m.encode(qs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        p_vecs = m.encode(ps, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        sims = q_vecs @ p_vecs.T
        top5 = np.argsort(-sims, axis=1)[:, :5]
        hits = sum(1 for i in range(len(qs)) if i in top5[i])
        return hits / len(qs)

    logger.info("Computing BEFORE-finetuning holdout Recall@5 (base model)...")
    before = recall_at_5(model)
    logger.info("BEFORE fine-tuning: Recall@5 on %d held-out legal QA pairs = %.3f", len(holdout), before)

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    logger.info("Fine-tuning for %d epochs on %d pairs (batch size %d)...",
                args.epochs, len(train_examples), args.batch_size)
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(train_loader) * args.epochs),
        show_progress_bar=True,
    )

    logger.info("Computing AFTER-finetuning holdout Recall@5...")
    after = recall_at_5(model)
    logger.info("AFTER fine-tuning:  Recall@5 on %d held-out legal QA pairs = %.3f", len(holdout), after)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))

    report = {
        "base_model": BASE_MODEL,
        "train_pairs": len(train_examples),
        "holdout_pairs": len(holdout),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "recall_at_5_before": before,
        "recall_at_5_after": after,
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved fine-tuned model -> %s", out_dir)
    logger.info("Training report: %s", json.dumps(report, indent=2))
    print("\n" + "=" * 72)
    print(f"Recall@5 before fine-tuning: {before:.3f}")
    print(f"Recall@5 after  fine-tuning: {after:.3f}")
    print(f"Model saved to: {out_dir}")
    print("Set TEXT_EMBEDDING_MODEL=./app/models/legal-bge-finetuned in backend/.env to use it")
    print("(this is already the default once the folder exists — see app/config.py)")
    print("=" * 72)


if __name__ == "__main__":
    main()
