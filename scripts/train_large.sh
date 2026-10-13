#!/bin/sh
# Large-scale training + evaluation on the full corpus. Waits for reindex_corpus.py to finish (RAM is 8 GB: never run both).
# Steps: lexical index + demo migration -> big pair set -> embedder -> reranker -> evaluations. Logs: data_lake/train_large.log
# A trained model is only worth adopting if it beats the base model on HELD-OUT data (see ROADMAP_STATUS.md); this script
# reports numbers, it does not switch the app to any model.
set -e
cd "$(dirname "$0")/.."
PY="$PWD/backend/.venv/bin/python"
export TEXT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 VECTOR_DB_PATH="$PWD/data_lake/corpus_store" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "== wait for reindex"; while pgrep -f "scripts/reindex_corpus.py" >/dev/null; do sleep 30; done
echo "== lexical index"; $PY -W ignore scripts/build_lexical_index.py | tail -1
echo "== migrate demo docs"; $PY -W ignore scripts/migrate_demo_to_corpus.py || echo "migration failed"
M=backend/app/models
[ -f $M/legal-bge-v2/config.json ] && [ ! -d $M/legal-bge-v2-small ] && mv $M/legal-bge-v2 $M/legal-bge-v2-small
[ -f $M/legal-reranker/config.json ] && [ ! -d $M/legal-reranker-small ] && mv $M/legal-reranker $M/legal-reranker-small
echo "== big pairs"; (cd backend && $PY -W ignore scripts/build_training_pairs.py --train-max-year 2022 --val-year 2023 --max-ict 60000 --max-ict-val 600 --max-ledgar 20000 --max-cuad 10000)
echo "== embedder (large)"; (cd backend && $PY -W ignore scripts/finetune_embedder_v2.py --pairs ../data_lake/pairs/pairs.jsonl --out app/models/legal-bge-v2 --max-train 40000 --max-val 600 --epochs 1 --batch 16 --max-seq 128 --grad-ckpt) || echo "embedder failed"
echo "== reranker (large)"; (cd backend && $PY -W ignore scripts/finetune_reranker.py --pairs ../data_lake/pairs/pairs.jsonl --out app/models/legal-reranker --max-train 8000 --max-val 400 --epochs 1) || echo "reranker failed"
echo "== evals"
(cd backend && MODELS="BAAI/bge-small-en-v1.5"; for d in legal-bge-v2-small legal-bge-v2; do [ -f app/models/$d/config.json ] && MODELS="$MODELS app/models/$d"; done
 RR=""; [ -f app/models/legal-reranker/config.json ] && RR="--reranker app/models/legal-reranker"
 $PY -W ignore -m eval.timesplit_eval --models $MODELS $RR --out eval/results/timesplit_large.json || echo "timesplit failed"
 $PY -W ignore -m eval.retrieval_bench --datasets mteb__AILA_casedocs mteb__AILA_statutes mteb__legalbench_consumer_contracts_qa --models $MODELS $RR --out eval/results/ablation_large.json || echo "ablation failed"
 $PY -W ignore -m eval.legalbench_rag_bench --benchmarks privacy_qa contractnli --max-queries 100 --distractors 25 --out eval/results/legalbench_rag.json || echo "lbrag failed")
echo "== done"
