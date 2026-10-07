# LegalMind — Architecture & Build Plan

## 01 · Vision & Problem Statement

Legal work is drowning in documents. Contracts, judgments, notices, and
agreements are long, dense, inconsistent, and frequently arrive as scanned
images rather than clean text. A lawyer or a business owner asking a single
question — *"What is my termination notice period, and does it conflict with
anything else I've signed?"* — may need to read dozens of pages to answer it,
and a wrong reading has real cost.

LegalMind is a retrieval-augmented generation (RAG) system that ingests legal
documents in any form — native PDFs, scanned images, photographs of pages,
embedded tables and signature blocks — and lets a user interrogate them in
natural language. Crucially, every answer is grounded: it points back to the
exact clause, page, and region of the source document, so the user can verify
it rather than trust it blindly.

**The one-line pitch.** Most student "legal AI" projects are a thin wrapper
over text-only PDF chat. LegalMind's differentiators are three: (1) true
multimodality — it reads scanned and image-based documents, not just clean
text; (2) clause-level grounded citation — answers highlight the exact source
region; and (3) higher-order reasoning — cross-document conflict detection,
risk red-flagging, and plain-English translation, not just Q&A.

### Why this wins as a capstone

A capstone is graded on rigor, novelty, and completeness — not just a working
demo. LegalMind is deliberately structured to score on all three: it forces
genuinely hard engineering (multimodal parsing, hybrid retrieval,
structure-aware chunking), it has a defensible novelty story (grounding +
reasoning in a high-stakes domain), and it ships with a real evaluation
harness that most undergraduate projects never build.

### Target users & primary use cases

| User | What they ask LegalMind |
|---|---|
| Small-business owner / founder | "Summarise this vendor contract. What are the risky clauses? Can I get out of it, and how much notice do I need?" |
| Law student / paralegal | "Find every clause about indemnification across these five agreements and flag where they contradict each other." |
| In-house reviewer | "Compare this NDA against our standard playbook and show me every deviation." |
| Non-lawyer citizen | "Translate this legal notice into plain English and tell me what I actually have to do." |

## 02 · Objectives & Success Criteria

**Functional goals**
- Ingest native PDFs, scanned PDFs, and image files end-to-end.
- Answer natural-language questions with an inline, clickable citation to the source clause.
- Detect conflicts across a multi-document set.
- Surface a ranked list of risky / non-standard clauses.
- Produce a plain-English rewrite and an exportable summary memo.

**Quality targets (measurable)**
- Faithfulness ≥ 0.90 (RAGAS) — answers are grounded, not hallucinated.
- Context precision ≥ 0.80 — retrieved chunks are relevant.
- Citation accuracy ≥ 85% on a hand-labelled test set.
- OCR-doc answer parity — accuracy on scanned docs within 10% of native.
- P95 latency < 8s per query on the demo corpus.

**Non-goals.** LegalMind is a decision-support tool, not a source of legal
advice; it does not autonomously file, sign, or send anything; and it is
scoped to a curated document corpus rather than the open web.

## 03 · System Architecture — Layered View

LegalMind is organised as six cooperating layers. Data flows up from raw
documents into a searchable, grounded knowledge layer; queries flow down
through retrieval and generation. A cross-cutting evaluation & observability
plane instruments every stage.

```
EVALUATION & OBSERVABILITY  ·  RAGAS  ·  LangSmith / Langfuse
──────────────────────────────────────────────────────────────
6 · Presentation Layer
    Streamlit / React · chat UI · side-by-side PDF viewer with
    clause highlighting · report export
──────────────────────────────────────────────────────────────
5 · Application & Orchestration (FastAPI)
    Session & document management · async ingestion queue ·
    agent / chain orchestration (LangChain)
──────────────────────────────────────────────────────────────
4 · Reasoning & Generation
    LLM answer synthesis · structured citations ·
    guardrails / "I don't know" · feature modules
    (risk, conflict, compare, translate)
──────────────────────────────────────────────────────────────
3 · Retrieval
    Hybrid search (dense embeddings + BM25) · cross-encoder
    re-ranking · metadata filtering · query rewriting
──────────────────────────────────────────────────────────────
2 · Indexing & Knowledge Store
    Structure-aware chunking · embeddings · vector DB (Qdrant) ·
    metadata store · bounding-box map for highlighting
──────────────────────────────────────────────────────────────
1 · Multimodal Ingestion
    PDF text & layout parsing · OCR for scans · table extraction
    · signature/stamp detection · doc classification
──────────────────────────────────────────────────────────────
Raw sources: native PDFs · scanned PDFs · images/photos of pages
             · DOCX · plain text
```

Instrumentation is not an afterthought bolted on at the end, but a design
concern that touches every stage. This is the single detail that most
distinguishes a research-grade RAG system from a demo.

## 04 · End-to-End Data Flow

Two pipelines run at different times.

**A · Ingestion pipeline (offline, per document)**

```
Upload (PDF/image) → Classify & route (native vs scan) → Parse
(text · OCR · tables · sigs) → Chunk (by clause + metadata) →
Embed (dense + sparse vectors) → Index (Qdrant + metadata)
```

**B · Query pipeline (online, per question)**

```
Question + filters → Rewrite/expand → Hybrid retrieve (top-k) →
Re-rank (cross-encoder) → LLM answer + guardrails → Cited answer
+ clause highlight
```

Every step is traced & scored → evaluation store.

**Design note — grounding is carried in metadata, not re-derived.** At chunk
time, each chunk stores its page number and bounding-box coordinates. That map
travels with the chunk through retrieval, so when the LLM cites a chunk, the
UI already knows exactly where to draw the highlight. Grounding "for free" at
answer time is only possible because you paid for it at ingestion time.

## 05 · Component Breakdown

### 5.1 · Multimodal ingestion

This is where LegalMind earns the word "multimodal." A router first
classifies each incoming file: if a page has an extractable text layer, parse
it directly; if it is a scan or photo, send it through OCR or a vision model.
Never assume clean text.

- PyMuPDF (`fitz`) — fast text + coordinates
- pdfplumber — tables & word boxes
- Docling / `unstructured` — layout parsing
- Tesseract / PaddleOCR — OCR
- A vision LLM (GPT-4o / Qwen2-VL) — hard scans
- Camelot / Table Transformer — tables
- OpenCV — signature/stamp detection

**Key decision:** use a hybrid parser — deterministic extractors for native
text (cheap, exact coordinates) and a vision model only for pages that fail
the text-layer check (accurate on scans, but slower and costlier). Log which
path each page took; it makes the evaluation section far richer.

### 5.2 · Structure-aware chunking

Fixed-size token windows are the default in most tutorials and the wrong
choice for legal text. Contracts are hierarchical — Article → Section →
Clause → sub-clause — and a good chunk respects that boundary. Chunk by
clause/section, keep the section path (e.g. "12 › 12(b) › Termination for
Cause") as metadata, and attach page + bounding box.

- Custom recursive splitter on numbering patterns
- LangChain `RecursiveCharacterTextSplitter` (fallback)
- Parent-document / small-to-big retrieval
- Metadata: `doc_id · type · jurisdiction · section_path · page · bbox`

### 5.3 · Embeddings & vector store

Store both a dense vector (semantics) and keep the raw text for
sparse/keyword search.

- OpenAI `text-embedding-3-large` or BGE-large (open)
- Legal-BERT / InLegalBERT — domain ablation
- Qdrant (recommended) · Chroma · pgvector
- Payload = full metadata for filtering + highlight

### 5.4 · Retrieval

Legal queries hinge on exact terms — "Section 12(b)", a defined party name, a
statute number — that pure semantic search misses. Combine dense retrieval
with BM25 keyword search, fuse the results, then re-rank the top candidates
with a cross-encoder for precision.

- Dense (vector) + BM25 (`rank_bm25` / Qdrant sparse)
- Reciprocal-rank fusion
- `bge-reranker` / Cohere Rerank (cross-encoder)
- Metadata pre-filter (jurisdiction, doc type, date)
- Query rewriting / multi-query for recall

### 5.5 · Reasoning & generation

The generation prompt is a contract of its own: answer only from the
retrieved context, return a structured object containing the answer plus the
chunk IDs it relied on, and abstain when the context is insufficient.
Structured output (JSON / function-calling / an Instructor-style schema) is
what makes reliable citation and highlighting possible.

- GPT-4o / Claude / Llama-3 (swap-able)
- Structured output: `citations = [chunk_id, page, bbox]`
- Guardrail: "insufficient context → say so"
- Per-feature prompt templates

## 06 · Feature Modules

Tagged **CORE** (graded baseline, build first), **WOW** (differentiators),
**STRETCH** (build if time allows).

- **CORE** — Grounded Q&A with clause-level citation. Answer + document
  side by side; clicking a citation scrolls the viewer to the highlighted
  region.
- **WOW** — Cross-document conflict detection. Extract the same clause type
  from each document and run a pairwise LLM comparison over those clauses.
- **WOW** — Risk / red-flag analysis. Auto-renewal traps, unlimited
  liability, one-sided indemnification, missing termination rights, unusual
  governing law — colour-coded red/amber/green with a one-line explanation.
- **WOW** — Clause comparison against a playbook. Redline-style diff plus a
  plain summary of what changed and why it matters.
- **CORE** — Plain-English translation toggle for non-lawyers.
- **STRETCH** — Summary-memo generation & export (parties, key dates,
  obligations, risks, open questions) as PDF/DOCX.

**Optional differentiator — localise to Indian law.** A corpus of contracts
under the Indian Contract Act, judgments, and legal notices unlocks
InLegalBERT, the ILDC judgment corpus, and IndianKanoon as a source, and
reads as more original than another US-contract demo.

## 07 · Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Language / runtime | Python 3.11 | Ecosystem for ML + document processing |
| Orchestration | LangChain or LlamaIndex | LlamaIndex for indexing, LangChain for agentic feature modules |
| Ingestion | PyMuPDF · pdfplumber · Docling/unstructured · Tesseract/PaddleOCR · vision LLM | Hybrid native-text + OCR + vision path |
| Embeddings | `text-embedding-3-large` / BGE-large | Strong retrieval; open option for offline/cost |
| Vector DB | Qdrant | Docker-local, hybrid (dense+sparse), rich metadata filtering |
| Re-ranker | `bge-reranker-v2` / Cohere Rerank | Precision boost on top-k |
| LLM | GPT-4o / Claude / Llama-3 (provider-agnostic wrapper) | Swappable for benchmarking |
| Backend API | FastAPI + Celery/RQ + Redis | Async ingestion so large PDFs don't block the UI |
| Frontend | Streamlit (fast) or React + PDF.js | Streamlit to ship fast; React+PDF.js for polished highlights |
| Evaluation | RAGAS + custom labelled set | Faithfulness, context precision/recall, answer relevancy |
| Observability | LangSmith / Langfuse | Trace every retrieval + generation; latency & cost |
| Packaging | Docker Compose | One command brings up app + Qdrant + Redis |

**Framework advice.** Don't try to master both frameworks. LlamaIndex gets you
a citation-quality query engine fastest; reach for LangChain (or LangGraph)
only for the multi-step feature modules where you orchestrate several LLM
calls.

## 08 · Repository Structure

See top-level tree in [`README.md`](../README.md).
