# archive/

`legalmind_scaffold/` is the earlier, unfinished second design (Qdrant + hybrid retriever + reranker sketches).
Decision (roadmap Sec. 12.1): **`backend/` + `frontend/` is the single codebase.** The scaffold's ideas were
re-implemented there with tests: BM25 + dense fusion (`backend/app/retrieval/hybrid.py`), structure-aware
judgment chunking (`backend/app/ingestion/judgments.py`), risk analysis (`backend/app/legal/risk.py`).
Not imported anywhere. Safe to delete once the report no longer references it; recoverable from git history.
