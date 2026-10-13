#!/bin/sh
# Start backend (port 8000) and frontend (port 5173) on the LARGE corpus: judgments + statutes + the migrated demo documents.
# Uses the base bge-small embedder (the v1 synthetic fine-tune measured worse on real data) and the corpus vector store.
# Environment variables here override backend/.env for this run only; your .env is not modified.
cd "$(dirname "$0")/.."
export VECTOR_DB_PATH="$PWD/data_lake/corpus_store" TEXT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 TOKENIZERS_PARALLELISM=false
(cd backend && exec .venv/bin/python -W ignore -m uvicorn app.main:app --host 127.0.0.1 --port 8000) &
(cd frontend && exec npm run dev -- --host 127.0.0.1) &
wait
