#!/bin/sh
# Ingestion is complete (2,596 judgments in data_lake/corpus). Runs: statutes -> pairs -> reranker -> embedder v2 -> evaluations.
# Sequential on purpose (8 GB RAM). Logs to data_lake/pipeline.log. CPU training is slow: sizes are deliberately small;
# raise --max-train on a GPU. Re-runnable: each step overwrites its own outputs.
set -e
cd "$(dirname "$0")/.."
PY="$PWD/backend/.venv/bin/python"
export TEXT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 VECTOR_DB_PATH="$PWD/data_lake/corpus_store" PYTHONUNBUFFERED=1
echo "== statutes (parse only; indexed at the end by reindex_corpus.py)"; $PY -W ignore scripts/ingest_statutes.py
echo "== pairs";    (cd backend && $PY -W ignore scripts/build_training_pairs.py --train-max-year 2022 --val-year 2023)
echo "== reranker"; (cd backend && $PY -W ignore scripts/finetune_reranker.py --pairs ../data_lake/pairs/pairs.jsonl --max-train 2500 --max-val 200 --epochs 1) || echo "reranker step failed"
echo "== embedder v2"; (cd backend && $PY -W ignore scripts/finetune_embedder_v2.py --pairs ../data_lake/pairs/pairs.jsonl --max-train 4000 --max-val 400 --epochs 1 --batch 16) || echo "embedder step failed"
echo "== time-split eval"
(cd backend && MODELS="BAAI/bge-small-en-v1.5"; [ -d app/models/legal-bge-v2 ] && MODELS="$MODELS app/models/legal-bge-v2"; RR=""; [ -d app/models/legal-reranker ] && RR="--reranker app/models/legal-reranker"; $PY -W ignore -m eval.timesplit_eval --models $MODELS $RR) || echo "eval failed"
echo "== legalbench-rag"
(cd backend && $PY -W ignore -m eval.legalbench_rag_bench --benchmarks privacy_qa contractnli maud cuad --max-queries 100 --distractors 25) || echo "lbrag failed"
echo "== reindex corpus (resumable, slow)"; $PY -W ignore scripts/reindex_corpus.py || echo "reindex failed"
echo "== done"
