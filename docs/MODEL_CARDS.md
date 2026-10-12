# Model cards

Numbers below were produced in this repository on 2026-10-08/12; each links to the script and result file that made it.
Cards for models whose training has not run yet say so and carry no numbers.

## Clause classifier (CUAD): `backend/app/models/clause_classifier/cuad.joblib`
- **What:** TF-IDF (word 1-2 grams) + linear SVM, 41 CUAD clause categories + "None". Used by `GET /api/documents/{id}/clauses`.
- **Data:** CUAD (CC-BY-4.0), contract-level train/test split of `theatticusproject/cuad-qa`; "None" = random 600-character windows that overlap no annotated span.
- **Result (test split, 3,276 spans):** micro-F1 0.781, macro-F1 0.647. Weak labels: Price Restrictions (0.00), Unlimited/All-You-Can-Eat-License (0.00), Most Favored Nation (0.29), Affiliate License-Licensee (0.35). Per-label F1: `report.json` next to the model.
- **Limits:** US commercial contracts only; Indian clauses (stamp duty, arbitration seat) are not in training data. Margin is an SVM score, not a probability. Not for legal decisions.
- **Reproduce:** `backend/scripts/train_clause_classifier.py`.

## Provision classifier (LEDGAR): `.../ledgar.joblib`
- Same architecture, 100 provision types from LexGLUE/LEDGAR (SEC filings). Test split 10,000: micro-F1 0.875, macro-F1 0.824. Not used in the app yet.

## Risk rubric: `backend/app/legal/risk.py`
- Rule layer (category base risk + regex red flags), not a trained model. **Draft: needs review by a qualified lawyer.** References Indian Contract Act ss. 27 and 74 in two rules.

## Embedder v1 `backend/app/models/legal-bge-finetuned` (pre-existing)
- bge-small-en-v1.5 fine-tuned on synthetic legal documents. Reported 0.967 Recall@5 on the same synthetic distribution.
- **Real-data result (`backend/eval/results/ablation.json`):** does not beat the base model. AILA case docs dense nDCG@10 0.204 vs 0.235 base; LegalBench consumer-contracts QA 0.685 vs 0.734; AILA statutes 0.242 vs 0.230. Treat the synthetic score as optimistic.

## Embedder v2, reranker (`legal-bge-v2`, `legal-reranker`)
- Training scripts written (`finetune_embedder_v2.py`, `finetune_reranker.py`); results are written to `training_report.json` in each model folder and to `eval/results/timesplit.json` once `scripts/post_ingest_pipeline.sh` has run. **No numbers yet.**

## Stamp / signature detector
- Training script and 1,500-image synthetic set exist (`scripts/vision/`); **not trained yet**. Synthetic validation will be optimistic; a real annotated scan set is required for an honest number.
