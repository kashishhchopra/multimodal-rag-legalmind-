"""Generate the synthetic legal corpus and ingest it into the app's vector store.

Run this once (it is idempotent — re-running regenerates and re-ingests) to ship
the app with a large, pre-indexed legal knowledge base so a fresh checkout is
queryable immediately, without requiring the user to upload anything first.

Usage (from backend/, venv active):
    python scripts/build_legal_knowledge_base.py
    MMRAG_FAKE_EMBEDDINGS=1 python scripts/build_legal_knowledge_base.py   # offline/plumbing

What it does:
  1. Generates `data/legal_knowledge_base.pdf` (~130+ pages: contracts, case
     judgments, statute chapters, each with tables + seal/signature images) —
     see app/legal_data/build_pdfs.py.
  2. Generates 3 smaller standalone sample documents for the "upload your own
     document" demo flow (data/legal_samples/).
  3. Registers and ingests all of them through the SAME pipeline a real upload
     goes through (parse -> chunk -> OCR -> embed -> index into ChromaDB), so
     what ships is exactly what the running app would produce.
"""
from __future__ import annotations

import datetime
import shutil
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import get_settings  # noqa: E402
from app.legal_data.build_pdfs import build_legal_knowledge_base, build_standalone_samples  # noqa: E402
from app.schemas import Document  # noqa: E402
from app.services.ingest_service import process_document  # noqa: E402
from app.utils import registry  # noqa: E402

KB_DOCUMENT_ID = "legal_knowledge_base"


def _ingest(document_id: str, pdf_path: Path, filename: str) -> None:
    settings = get_settings()
    out_dir = settings.data_dir / document_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "source.pdf"
    shutil.copy(pdf_path, dest)
    registry.save(Document(
        document_id=document_id,
        filename=filename,
        upload_time=datetime.datetime.utcnow().isoformat() + "Z",
    ))
    print(f"Ingesting {filename} (document_id={document_id}) ...")
    process_document(document_id, str(dest), filename)
    doc = registry.get(document_id)
    if doc and doc.stage.value == "ready":
        print(f"  -> ready: {doc.page_count} pages, {doc.text_chunk_count} text chunks, "
              f"{doc.image_count} visuals, {doc.region_count} OCR regions")
    else:
        print(f"  -> FAILED: {doc.error if doc else 'unknown error'}")


def main() -> None:
    settings = get_settings()
    print(f"Text embedding model: {settings.text_embedding_model}")
    print(f"Image embedding model: {settings.image_embedding_model}")

    kb_pdf = settings.data_dir / "legal_knowledge_base.pdf"
    print("\nGenerating the legal knowledge-base PDF (contracts + case law + statutes)...")
    build_legal_knowledge_base(kb_pdf)

    samples_dir = settings.data_dir / "legal_samples"
    print("Generating standalone sample documents...")
    sample_paths = build_standalone_samples(samples_dir)

    print("\n--- Ingesting the main legal knowledge base ---")
    _ingest(KB_DOCUMENT_ID, kb_pdf, "Legal_Knowledge_Base.pdf")

    print("\n--- Ingesting standalone sample documents ---")
    for p in sample_paths:
        doc_id = f"sample_{p.stem}"
        _ingest(doc_id, p, p.name)

    print("\nDone. Start the backend (uvicorn app.main:app --reload --port 8000) and the "
          "'Legal_Knowledge_Base.pdf' document will already be listed and ready to query.")


if __name__ == "__main__":
    main()
