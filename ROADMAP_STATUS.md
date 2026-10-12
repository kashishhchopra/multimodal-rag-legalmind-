# LegalMind 2.0: roadmap status

Last updated 2026-10-13. Maps to `LegalMind_BigData_Roadmap.pdf`. **Done** = implemented and run or tested here. **Written** = code exists, full run not done. 58 backend tests pass; the frontend typechecks.

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
| 6.1/6.2 Training pairs, embedder v2, reranker | Scripts written; **run queued** (CPU, small sizes) | `backend/scripts/`, `scripts/post_ingest_pipeline.sh` |
| 6.5 Synthetic stamps, detector | 1,500 images generated; **training not run** | `scripts/vision/` |
| 8.x Retrieval ablation, time-split eval, LegalBench-RAG harness, answer eval | Written; ablation run (above); time-split and LegalBench-RAG runs queued; answer eval needs a working LLM key | `backend/eval/` |
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

- `scripts/ingest_judgments.py ... --index` into `data_lake/corpus_store` (log `data_lake/ingest2.log`). Re-run the same command to resume.
- `scripts/post_ingest_pipeline.sh` waits for ingestion, then runs: statute indexing, training pairs, reranker, embedder v2, time-split eval, LegalBench-RAG (log `data_lake/pipeline.log`). Sequential because the machine has 8 GB of RAM.

## Needs you

- **Fix `backend/.env`.** `LLM_PROVIDER` currently holds what looks like an API key; it must be `openai` or `gemini`, and the key goes on the `LLM_API_KEY` line. That value also appeared in command output during testing, so **treat the key as exposed and rotate it**.
- Hugging Face login and access requests for **IL-TUR** and **Aalap**.
- Manual downloads: Tobacco-800, Roboflow/Kaggle stamp sets, Indian contract samples, the five missing Acts, ACORD, COLIEE (registration), FIRE.
- A lawyer to review the risk rubric; two annotators for 100 scanned pages and answer ratings; a Hindi question set.
- A GPU for full-size embedder, reranker and detector runs.
- **Licences:** IL-TUR, ILDC, HLDC and IL-PCR are CC-BY-NC(-SA): fine for a student project, not for commercial use or redistribution of derived data. Many registry entries still say "verify".
