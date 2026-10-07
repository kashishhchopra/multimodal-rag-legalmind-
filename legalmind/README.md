# LegalMind

A multimodal, retrieval-augmented question-answering system for legal documents.
LegalMind ingests native PDFs, scanned PDFs, and photographed pages, then lets a
user interrogate the corpus in natural language. Every answer carries a
clause-level citation back to the exact page and bounding box it came from.

## Why this exists

Legal documents are long, dense, and frequently arrive as unclean scans. A wrong
reading has real cost. LegalMind's three differentiators:

1. **True multimodality** — reads scanned and image-based documents, not just clean text.
2. **Clause-level grounded citation** — every answer highlights the exact source region.
3. **Higher-order reasoning** — cross-document conflict detection, risk red-flagging,
   and plain-English translation, not just Q&A.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design.

## Non-goals

LegalMind is a decision-support tool, not a source of legal advice. It does not
autonomously file, sign, or send anything. It is scoped to a curated document
corpus, not the open web.

## Quickstart

```bash
cp .env.example .env          # fill in API keys
docker compose up -d          # starts Qdrant + Redis
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
streamlit run frontend/streamlit_app.py
```

Upload a document via the UI or `POST /documents/upload`, wait for ingestion to
finish (`GET /documents/{id}/status`), then ask a question via `POST /query`.

## Repository layout

```
legalmind/
├── docker-compose.yml     # app + qdrant + redis
├── docs/ARCHITECTURE.md   # full design doc
├── data/                  # sample corpus (native + scanned)
├── notebooks/             # EDA, ablations, eval runs
├── src/
│   ├── ingestion/         # parsers, OCR router, table & signature extractors
│   ├── chunking/          # structure-aware splitter + metadata
│   ├── indexing/          # embeddings, qdrant client, hybrid index
│   ├── retrieval/         # hybrid retriever, reranker, query rewrite
│   ├── generation/        # prompts, structured citation, guardrails
│   ├── features/          # conflict, risk, compare, translate, memo
│   ├── api/                # FastAPI routes, tasks (celery)
│   └── config.py
├── eval/
│   ├── testset.jsonl      # hand-labelled Q/A + gold citations
│   ├── run_ragas.py
│   └── report.md          # metrics + ablations
├── frontend/              # streamlit app
└── tests/                 # unit tests for parsers & retrieval
```

## Quality targets

| Metric | Target |
|---|---|
| Faithfulness (RAGAS) | ≥ 0.90 |
| Context precision | ≥ 0.80 |
| Citation accuracy | ≥ 85% |
| OCR-doc vs native accuracy gap | ≤ 10% |
| P95 query latency | < 8s |

Run `python eval/run_ragas.py` to reproduce metrics against `eval/testset.jsonl`.

## Status

Scaffold stage — module boundaries, interfaces, and stubs are in place per the
architecture doc. Fill in each `TODO` as you implement that layer.
