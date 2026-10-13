#!/bin/sh
# Second large run: far more INDIAN pairs (adjacent-passage, 24 per judgment) and little US contract data, so the training set
# is actually dominated by the target domain. Waits for train_large.sh to finish (8 GB RAM). Log: data_lake/train_large2.log
set -e
cd "$(dirname "$0")/.."
PY="$PWD/backend/.venv/bin/python"
export TEXT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 VECTOR_DB_PATH="$PWD/data_lake/corpus_store" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
while pgrep -f "scripts/train_large.sh" >/dev/null; do sleep 30; done
M=backend/app/models
[ -f $M/legal-bge-v2/config.json ] && [ ! -d $M/legal-bge-v2-large1 ] && mv $M/legal-bge-v2 $M/legal-bge-v2-large1
[ -f $M/legal-reranker/config.json ] && [ ! -d $M/legal-reranker-large1 ] && mv $M/legal-reranker $M/legal-reranker-large1
echo "== pairs (Indian-dominated)"; (cd backend && $PY -W ignore scripts/build_training_pairs.py --train-max-year 2022 --val-year 2023 --ctx-per-doc 24 --max-ict 60000 --max-ict-val 600 --max-ledgar 3000 --max-cuad 3000)
echo "== embedder"; (cd backend && $PY -W ignore scripts/finetune_embedder_v2.py --pairs ../data_lake/pairs/pairs.jsonl --out app/models/legal-bge-v2 --max-train 40000 --max-val 600 --epochs 1 --batch 16 --max-seq 128 --grad-ckpt) || echo "embedder failed"
echo "== reranker"; (cd backend && $PY -W ignore scripts/finetune_reranker.py --pairs ../data_lake/pairs/pairs.jsonl --out app/models/legal-reranker --max-train 12000 --max-val 400 --epochs 1) || echo "reranker failed"
echo "== evals"
(cd backend && MODELS="BAAI/bge-small-en-v1.5"; for d in legal-bge-v2-small legal-bge-v2-large1 legal-bge-v2; do [ -f app/models/$d/config.json ] && MODELS="$MODELS app/models/$d"; done
 RR=""; [ -f app/models/legal-reranker/config.json ] && RR="--reranker app/models/legal-reranker"
 $PY -W ignore -m eval.timesplit_eval --models $MODELS $RR --out eval/results/timesplit_large2.json || echo "timesplit failed"
 $PY -W ignore -m eval.retrieval_bench --datasets mteb__AILA_casedocs mteb__AILA_statutes mteb__legalbench_consumer_contracts_qa --models $MODELS $RR --out eval/results/ablation_large2.json || echo "ablation failed")
echo "== done"
