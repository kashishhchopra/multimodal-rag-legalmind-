# LegalMind 2.0: roadmap status

Last updated 2026-10-13. Maps to `LegalMind_BigData_Roadmap.pdf`. "Done" means implemented **and** run or tested here; "Written" means code exists but the full run has not happened.

## Findings so far (real data, CPU runs)

Retrieval on real legal data (`backend/eval/results/ablation.json`; BEIR-format mirrors, documents truncated to 512 tokens):

| Dataset | Config | R@5 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| AILA case docs | BM25 only | 0.189 | 0.400 | 0.270 |
| AILA case docs | dense bge-small base (A0) | 0.214 | 0.318 | 0.235 |
| AILA case docs | dense v1 synthetic fine-tune (A1) | 0.187 | 0.278 | 0.204 |
| AILA case docs | hybrid RRF, base | 0.207 | 0.348 | 0.271 |
| AILA statutes | dense base / dense v1 | 0.186 / 0.189 | 0.353 / 0.394 | 0.230 / 0.242 |
| LegalBench consumer-contracts QA | dense base (A0) | 0.856 | 0.673 | 0.734 |
| LegalBench consumer-contracts QA | dense v1 (A1) | 0.795 | 0.624 | 0.685 |
| LegalBench consumer-contracts QA | hybrid RRF, base (A3) | 0.856 | 0.690 | 0.748 |

1. **The v1 embedder (synthetic fine-tune, Recall@5 0.967 on synthetic data) does not beat the base model on real data.** It is worse on case documents and contracts and about level on statutes. This confirms the roadmap's suspicion that 0.967 was inflated. Report it as a limitation, not a win.
2. **Hybrid (BM25 + dense, RRF) helps on contracts and on case documents at k=10**, and gives mixed results on statutes. The corpora are small (82 to 186 documents), so differences of a few points are within noise.
3. Clause classifiers (TF-IDF + linear SVM baseline): LEDGAR micro-F1 0.875 / macro-F1 0.824 (100 labels); CUAD micro-F1 0.781 / macro-F1 0.647 (41 categories + "None"). Weak labels: Price Restrictions, Unlimited-License, Most Favored Nation.

## Section by section

| Roadmap item | Status | Where |
|---|---|---|
| 12.1 Rotate key, `.gitignore`, stray files, git init | **Partly done.** git, ignore rules, stray files removed. **Key not rotated: only you can do that.** | `.gitignore`, CI secret scan |
| 12.1 One codebase | Done: `legalmind/` archived | `archive/` |
| 12.1 DATASETS.md | Done, generated | `DATASETS.md`, `datasets/registry.yaml` |
| 12.1 CI | Done (pytest + secret scan) | `.github/workflows/ci.yml` |
| 3.x Dataset catalogue (all 55+ entries) | Registry covers every dataset; automated fetch for AWS S3 + Hugging Face; the rest print manual steps | `scripts/datasets_cli.py` |
| 5.2 Bulk ingestion (classify, clean, metadata, dedupe, chunk, index) | Done for native PDFs; scanned PDFs are queued, **OCR worker not written** | `backend/app/ingestion/judgments.py`, `scripts/ingest_judgments.py` |
| 5.1 BM25 + dense fusion, metadata filters, cross-document search | Done (in Chroma + in-memory BM25) | `backend/app/retrieval/hybrid.py`, `POST /api/search` |
| 5.1 Qdrant, reranker in the live path, citation graph, queue workers, MinIO/Postgres, tracing | **Not done** | |
| 6.1 Training pairs | Builder written (CUAD, LEDGAR, issue->reasoning, citation) | `backend/scripts/build_training_pairs.py` |
| 6.1 Embedder v2 training | Script written, **full run needs GPU** | `backend/scripts/finetune_embedder_v2.py` |
| 6.2 Reranker | Script written | `backend/scripts/finetune_reranker.py` |
| 6.3 QLoRA generator | **Not done** (roadmap defers it) | |
| 6.4 Clause classifier, risk scorer | Done (draft rubric, needs lawyer review) | `backend/scripts/train_clause_classifier.py`, `backend/app/legal/{clauses,risk}.py`, `GET /api/documents/{id}/clauses` |
| 3.2 Statutes: BNS, BNSS, BSA | Downloaded from MHA and parsed section by section (358 / 531 / 170 sections, matching the real Acts). **India Code Acts and the Constitution are not done** (legislative.gov.in URL 404s; India Code needs a terms check) | `scripts/ingest_statutes.py`, `backend/app/ingestion/statutes.py` |
| 6.4 IPC->BNS mapper | Done for 69 high-traffic sections across IPC/CrPC/Evidence Act; target sections checked against the official BNS/BNSS/BSA text, sub-clause numbers not checked; **not the full official table** | `backend/app/legal/section_mapper.py`, `GET /api/legal/sections` |
| 6.4 NER, rhetorical roles, summariser, doc router | Not done | |
| 6.5 Synthetic stamps | Done | `scripts/vision/make_synthetic_stamps.py` |
| 6.5 Detector training | Script written; run status below | `scripts/vision/train_detector.py` |
| 6.5 ColPali, DocLayNet model, CLIP fine-tune | Not done | |
| 6.6 Hindi / IndicTrans2 | Not done | |
| 8.x Retrieval ablation harness | Done (rows A0, A1, A3; A4 with `--reranker`) | `backend/eval/retrieval_bench.py` |
| 8.x RAGAS, time-split holdout, human eval, error analysis | Not done | |
| 7 Features: summariser, timeline, contract comparison, memo export, PII redaction, auth, Hindi, feedback | Not done | |

## Long jobs started in the background (resumable)

- `scripts/ingest_judgments.py --years 2019..2024 --limit 350 --index`: about 2,100 Supreme Court judgments into `data_lake/corpus_store` with the base bge-small embedder. Roughly 13 chunks/s on this laptop, so about 3 hours. Log: `data_lake/ingest.log`. Re-run the same command to resume. When it finishes: `python scripts/ingest_statutes.py --index`, then `scripts/build_training_pairs.py`.
- Not started because RAM is 8 GB and ingestion uses all cores: detector training (`data_lake/synthetic_stamps` is generated, 1,500 pages), embedder v2 and reranker fine-tuning.

## Needs you

- **Rotate the API key in `backend/.env`.** It is untracked, but the roadmap flags it as exposed.
- **Hugging Face login** for gated sets (`huggingface-cli login`): Aalap (OpenNyAI) returned 401.
- **Manual-access datasets**: India Code, Constitution, official BNS/BNSS/BSA mapping tables, Tobacco-800, LegalBench-RAG, IN-Abs, ILDC, HLDC, stamp sets (Roboflow/Kaggle). Run `python3 scripts/datasets_cli.py list` for the full list and links.
- **GPU** for embedder v2, reranker (at full size) and detector training.
