#!/bin/sh
# Re-runs the steps after the reranker (embedder v2, evaluations, reindex) once the earlier steps have succeeded.
set -e
cd "$(dirname "$0")/.."
PY="$PWD/backend/.venv/bin/python"
export TEXT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 VECTOR_DB_PATH="$PWD/data_lake/corpus_store" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "== embedder v2"; (cd backend && $PY -W ignore scripts/finetune_embedder_v2.py --pairs ../data_lake/pairs/pairs.jsonl --max-train 4000 --max-val 400 --epochs 1 --batch 16) || echo "embedder step failed"
echo "== time-split eval"
(cd backend && MODELS="BAAI/bge-small-en-v1.5"; [ -f app/models/legal-bge-v2/config.json ] && MODELS="$MODELS app/models/legal-bge-v2"; RR=""; [ -f app/models/legal-reranker/config.json ] && RR="--reranker app/models/legal-reranker"; $PY -W ignore -m eval.timesplit_eval --models $MODELS $RR) || echo "eval failed"
echo "== legalbench-rag"
(cd backend && $PY -W ignore -m eval.legalbench_rag_bench --benchmarks privacy_qa contractnli maud cuad --max-queries 100 --distractors 25) || echo "lbrag failed"
echo "== reindex corpus (resumable, slow)"; $PY -W ignore scripts/reindex_corpus.py || echo "reindex failed"
echo "== done"
