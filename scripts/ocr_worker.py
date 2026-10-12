"""Consume data_lake/corpus/ocr_queue.jsonl: OCR the queued scanned judgments and chunk them.

    backend/.venv/bin/python scripts/ocr_worker.py --limit 20

Writes data_lake/corpus/ocr_chunks.jsonl (same chunk schema as chunks.jsonl, plus page confidence) and
ocr_done.jsonl; documents already in ocr_done.jsonl are skipped, so it is resumable. Index the result with
scripts/reindex_corpus.py after adding ocr_chunks.jsonl to its inputs (not wired by default: review the
low-confidence report first).
"""
import argparse
import json
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.ingestion.judgments import ParsedJudgment, chunk_judgment  # noqa: E402
from app.ingestion.scan_ocr import ocr_pdf  # noqa: E402

CORPUS = ROOT / "data_lake" / "corpus"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lang", default="eng")
    a = ap.parse_args()
    queue = [json.loads(l) for l in (CORPUS / "ocr_queue.jsonl").open()] if (CORPUS / "ocr_queue.jsonl").exists() else []
    done_f = CORPUS / "ocr_done.jsonl"
    done = {json.loads(l)["doc_id"] for l in done_f.open()} if done_f.exists() else set()
    todo = [q for q in queue if q["doc_id"] not in done][: a.limit or None]
    print(f"{len(todo)} scanned documents to OCR ({len(done)} already done)")
    with (CORPUS / "ocr_chunks.jsonl").open("a") as cf, done_f.open("a") as df:
        for q in todo:
            with tarfile.open(ROOT / q["tar"]) as tf:
                member = next((m for m in tf if Path(m.name).stem == q["doc_id"]), None)
                if not member:
                    continue
                data = tf.extractfile(member).read()
            pages = ocr_pdf(data, lang=a.lang)
            parsed = ParsedJudgment(q["doc_id"], q["doc_id"] + ".pdf", len(pages), "scanned", [p.text for p in pages], "")
            chunks = chunk_judgment(parsed)
            conf = {p.page_number: p.confidence for p in pages}
            for c in chunks:
                cf.write(json.dumps({**asdict(c), "ocr_confidence": conf.get(c.page_number)}) + "\n")
            low = [p.page_number for p in pages if p.low_confidence]
            df.write(json.dumps({"doc_id": q["doc_id"], "pages": len(pages), "chunks": len(chunks), "low_confidence_pages": low}) + "\n")
            print(f"{q['doc_id']}: {len(pages)} pages, {len(chunks)} chunks, low-confidence pages {low}", flush=True)


if __name__ == "__main__":
    main()
