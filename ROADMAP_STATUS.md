# LegalMind 2.0: roadmap status

Last updated 2026-10-14. Maps to `LegalMind_BigData_Roadmap.pdf`. **Done** = implemented and run or tested here. **Written** = code exists, full run not done. 58 backend tests pass; the frontend typechecks.

## Measured results so far (real data, CPU)

Retrieval, `backend/eval/results/ablation.json` (BEIR-format mirrors, documents truncated to 512 tokens):

| Dataset | Config | R@5 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| AILA case docs | BM25 only | 0.189 | 0.400 | 0.270 |
| AILA case docs | dense base bge-small (A0) | 0.214 | 0.318 | 0.235 |
| AILA case docs | dense v1 synthetic fine-tune (A1) | 0.187 | 0.278 | 0.204 |
| AILA case docs | hybrid RRF, base (A3) | 0.207 | 0.348 | 0.271 |
| LegalBench consumer-contracts QA | dense base (A0) | 0.856 | 0.673 | 0.734 |
| LegalBench consumer-contracts QA | dense v1 (A1) | 0.795 | 0.624 | 0.685 |
| LegalBench consumer-contracts QA | hybrid RRF, base (A3) | 0.856 | 0.690 | 0.748 |

1. **The v1 embedder (synthetic fine-tune, Recall@5 0.967 on synthetic data) does not beat the base model on real data.** Report it as a limitation.
2. Hybrid BM25 + dense helps on contracts and on case documents at k=10; statutes are mixed. These test sets are small (82 to 186 documents), so differences of a few points are within noise.
3. Clause classifiers (TF-IDF + linear SVM): LEDGAR micro-F1 0.875 (macro 0.824); CUAD micro-F1 0.781 (macro 0.647). Weak: Price Restrictions, Unlimited-License, Most Favored Nation.
4. Cross-check of my hand-curated IPC/CrPC/Evidence Act mapping against the government correspondence tables found **two wrong rows** (IPC 191 and 325), now fixed. 45 of 49 curated IPC rows agreed outright.
5. LegalBench-RAG harness smoke test (privacy_qa, 40 queries, small corpus): dense P@4/R@4 0.083/0.204 vs BM25 0.016/0.059. The full run is queued in the pipeline.

## Trained-model results (CPU, small runs; 2026-10-13)

Time-split retrieval (`backend/eval/results/timesplit.json`): query = the "Issue for Consideration" text of 150 judgments from 2024 (never seen in training, which used judgments up to 2022); pool = 300 judgments, 9,612 chunks; correct = a top-k chunk comes from the query's own judgment.

| Config | R@1 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.960 | 1.000 | 1.000 | 0.977 |
| dense, base bge-small (A0) | 0.827 | 0.927 | 0.947 | 0.870 |
| hybrid RRF, base (A3) | 0.807 | 0.960 | 0.980 | 0.874 |
| dense, embedder v2 (A2) | 0.827 | 0.953 | 0.960 | 0.882 |
| hybrid RRF, v2 | 0.813 | 0.967 | 0.987 | 0.879 |
| hybrid + fine-tuned reranker, base | 0.667 | 0.940 | 0.980 | 0.784 |
| hybrid + fine-tuned reranker, v2 (A5) | 0.727 | 0.947 | 0.980 | 0.824 |

How to read this honestly:
- **Issue statements share vocabulary with their own judgments, so BM25 is near-perfect.** This test favours lexical search and is not a measure of real user queries. Do not quote these absolute numbers as general performance.
- **Embedder v2 is not a real improvement.** Validation MRR@10 was 0.330 before and 0.329 after training; on the time-split it is +0.012 MRR over base with 150 queries, which is within noise.
- **The reranker hurts here (R@1 0.667 vs 0.807).** On its own validation task (adjacent-passage retrieval, held-out 2023 judgments) it improved MRR@10 from 0.598 to 0.809, but that task differs from issue-to-judgment retrieval. **Do not put it in the live query path.**
- The training task is self-supervised (a sentence from one passage retrieves the neighbouring passage), because only 2024 judgments carry headnotes; this is a weak proxy for legal relevance. Real query-passage supervision (LLM-generated questions, IL-PCR) is still missing.
- Two earlier training-data designs were discarded after they gave a validation MRR of exactly 1.0: random negatives, then inverse-cloze. The cause of the second 1.0 turned out to be a NaN bug (see README troubleshooting), now guarded against.

## The app now runs on the large corpus (2026-10-14)

- Index: 122,674 chunks = 2,596 Supreme Court judgments (119,759 chunks) + 10 Acts (2,692) + the 4 demo PDFs (514, re-embedded with the base embedder). Start it with `scripts/run_local.sh` (uses the corpus index and the base bge-small; `backend/.env` is not modified).
- Lexical search is an on-disk SQLite FTS5 index (`backend/app/retrieval/lexical.py`), so the 120k-chunk corpus does not have to fit in RAM.
- Documents page has a **Corpus library** tab: browse/search/filter all judgments and statutes and open any one for questions, summary, timeline and clause analysis (read-only). Overview shows corpus counts.
- Known gaps: ranking quality on this corpus is not measured with real user queries; only judgments from 2019-2024 (about 350 per year, not every SC judgment); no High Court data; the demo uploads and corpus share one index.

## Large-scale training, first run (CPU)

Pair set: 32k pairs, but only 3.7k were Indian-law (adjacent-passage); the rest was US contract data (LEDGAR 20k, CUAD 8.4k). Result on the 2024 time-split: embedder dense MRR@10 0.858 vs base 0.870 (**worse**); hybrid + reranker R@1 0.527 vs 0.807 for hybrid alone (**worse**). The reranker's own validation MRR rose 0.634 -> 0.849, which shows it learned the self-supervised task, not retrieval relevance. A second run with an Indian-dominated pair set (about 24 passages per judgment) is in progress; its results will be added here. **No trained model is used by the app; the base model is.**

## By roadmap section

| Item | Status | Where |
|---|---|---|
| 12.1 Repo hygiene, git, CI, secret scan, one codebase | **Done** (API key rotation is yours) | `.gitignore`, `.github/workflows/ci.yml`, `archive/` |
| 3.x Dataset catalogue | Registry of 59 entries; automated fetch for AWS S3, Hugging Face (with filters) and plain URLs; DATASETS.md generated | `datasets/registry.yaml`, `scripts/datasets_cli.py` |
| Datasets downloaded | SC judgments 2019-2024, CUAD, LEDGAR + LexGLUE, ContractNLI, AILA (cases + statutes), 2 MTEB sets, IN-Abs (Zenodo), LegalBench-RAG, BNS/BNSS/BSA, Constitution (2019 English), Contract, Specific Relief, Limitation, Stamp, IT, DPDP Acts, 3 mapping tables | `data_lake/` (git-ignored) |
| 5.2 Bulk ingestion | Done for native PDFs; scanned PDFs queued (OCR worker not built). **Running:** ~2,100 judgments (350/year) | `scripts/ingest_judgments.py` |
| 5.1 BM25 + dense fusion, filters, cross-document search | Done | `backend/app/retrieval/hybrid.py`, `POST /api/search` |
| Statutes | BNS 358, BNSS 531, BSA 170 sections; Constitution 450 articles (Art. 370 missing); Contract 191, SRA 42, Limitation 113 (incl. schedule rows), Stamp 78, IT 94, DPDP 44 | `scripts/ingest_statutes.py` |
| IPC/CrPC/IEA to BNS/BNSS/BSA mapper | Done: 521 / 523 / 175 old sections from the correspondence tables plus curated rows. **Secondary compilation, not the MHA's own table; spot-check before relying** | `backend/app/legal/section_mapper.py`, `mapping_tables/` |
| 6.4 Clause classifier, risk rubric | Done (rubric is a draft needing legal review) | `/api/documents/{id}/clauses` |
| Citation verification + hallucination guard | Done: retrieval gate + lexical claim-support check (not NLI) | `backend/app/legal/verify.py` |
| Summariser (extractive), timeline, PII redaction (identifiers only), memo export PDF/DOCX/MD, contract comparison, feedback store | Done | `backend/app/legal/`, `backend/app/api/legal.py` |
| Follow-ups, plain-English mode, Hindi mode | Done via the configured LLM (translate-then-retrieve). **IndicTrans2 not used or compared** | `backend/app/services/conversation.py` |
| Frontend: search view, insights tabs, verification, feedback, memo buttons, mode toggles | Done | `frontend/src/components/` |
| 6.1/6.2 Training pairs, embedder v2, reranker | **Done at CPU scale** (results above; no real gain from the embedder, reranker not usable as-is) | `backend/scripts/`, `scripts/post_ingest_pipeline.sh`, `scripts/pipeline_part2.sh` |
| 6.5 Synthetic stamps, detector | 1,500 images generated; **training not run** | `scripts/vision/` |
| 8.x Retrieval ablation, time-split eval, LegalBench-RAG harness, answer eval | Ablation and time-split run (above); LegalBench-RAG run in progress; answer eval needs a working LLM key | `backend/eval/` |
| 3.x IL-TUR (includes IL-PCR, ILDC, NER, rhetorical roles, statute ID) | **Gated on Hugging Face: you must request access** | |
| 3.x Aalap | Gated on Hugging Face (401) | |
| 3.x India Code Acts: Companies, RERA, Consumer Protection, Arbitration, Transfer of Property | Official URLs found but downloads failed (504 or blocked); retry later or download manually | |
| Docs | Model cards, dataset card, architecture diagram | `docs/`, README |

## Not done

- **Models:** QLoRA generator, NLI verifier, NER, rhetorical roles, abstractive summariser, document-type router, layout model, ColPali, CLIP fine-tune, IndicTrans2.
- **Infra:** Qdrant, in-store BM25, reranker in the live query path, citation/statute graph, queue workers, MinIO/Postgres, tracing, MLflow/DVC, authentication/roles, admin dashboard, OCR worker, High Court data, Tier 2 scale, Pile of Law and other large corpora.
- **Evaluation:** RAGAS numbers and human rating (need a working LLM key and people), slice and error analysis, sealed-test protocol on trained models, ablation rows A4 to A8.
- **Submission:** demo video, slides, examiner Q&A, updated project report, Docker compose test.

## Running in the background (resumable)

- `scripts/pipeline_part2.sh` (log `data_lake/pipeline2.log`) continues with LegalBench-RAG and then `scripts/reindex_corpus.py`, which rebuilds the corpus vector index (the first one, `data_lake/corpus_store.corrupt`, was corrupted by an abrupt kill). The reindex is resumable. Until it finishes, `/api/search` over the real corpus is not available. Never kill it with `kill -9`.
- Ingestion is finished: 2,596 judgments (2019-2024) in `data_lake/corpus/`, with scanned-PDF OCR worker ready but no scans found (English SC PDFs carry text layers).

## Needs you

- **Fix `backend/.env`.** `LLM_PROVIDER` currently holds what looks like an API key; it must be `openai` or `gemini`, and the key goes on the `LLM_API_KEY` line. That value also appeared in command output during testing, so **treat the key as exposed and rotate it**.
- Hugging Face login and access requests for **IL-TUR** and **Aalap**.
- Manual downloads: Tobacco-800, Roboflow/Kaggle stamp sets, Indian contract samples, the five missing Acts, ACORD, COLIEE (registration), FIRE.
- A lawyer to review the risk rubric; two annotators for 100 scanned pages and answer ratings; a Hindi question set.
- A GPU for full-size embedder, reranker and detector runs.
- **Licences:** IL-TUR, ILDC, HLDC and IL-PCR are CC-BY-NC(-SA): fine for a student project, not for commercial use or redistribution of derived data. Many registry entries still say "verify".
