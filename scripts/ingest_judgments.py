"""Bulk-ingest Supreme Court judgments from the AWS tar files into the vector store.

    backend/.venv/bin/python scripts/ingest_judgments.py --years 2024 --limit 200 --index
    backend/.venv/bin/python scripts/ingest_judgments.py --years 2018 2019 2020 --index --workers 6

Stages: read PDF from tar -> classify native/scanned -> clean -> metadata/citations -> dedupe
(sha256) -> structure-aware chunks -> JSONL (resumable) -> [--index] embed + add to Chroma with
filterable metadata (court, year, doc_type, citation, section). Scanned PDFs are NOT OCR'd here:
they are recorded in data_lake/corpus/ocr_queue.jsonl for the OCR worker.

Outputs (data_lake/corpus/): docs.jsonl, chunks.jsonl, quality.jsonl, ocr_queue.jsonl.
Re-running skips documents already in docs.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.ingestion.judgments import chunk_judgment, extract_metadata, parse_pdf, quality_report  # noqa: E402

LAKE = ROOT / "data_lake"
OUT = LAKE / "corpus"


def process(args: tuple[str, bytes, str | None]):
    name, data, nc = args
    try:
        parsed = parse_pdf(data, name)
        meta = extract_metadata(parsed, nc)
        chunks = chunk_judgment(parsed) if parsed.kind != "scanned" else []
        return {"parsed": (parsed.doc_id, parsed.kind, parsed.sha256, parsed.n_pages), "meta": asdict(meta),
                "chunks": [asdict(c) for c in chunks], "quality": quality_report(parsed, chunks, meta)}
    except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the batch
        return {"error": f"{name}: {exc}"}


def tar_items(tar_path: Path, nc_map: dict, seen: set, limit: int):
    n = 0
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".pdf"):
                continue
            doc_id = Path(m.name).stem
            if doc_id in seen:
                continue
            yield (Path(m.name).name, tf.extractfile(m).read(), nc_map.get(doc_id.removesuffix("_EN")))
            n += 1
            if limit and n >= limit:
                return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max new PDFs per year (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--index", action="store_true", help="embed chunks and add to the vector store")
    ap.add_argument("--embed-batch", type=int, default=64)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    docs_f, chunks_f = OUT / "docs.jsonl", OUT / "chunks.jsonl"
    seen_ids, seen_hashes = set(), set()
    if docs_f.exists():
        for line in docs_f.open():
            d = json.loads(line)
            seen_ids.add(d["doc_id"]); seen_hashes.add(d["sha256"])
    if args.index:
        from app.models import embeddings
        from app.retrieval import vector_store

    totals = {"docs": 0, "chunks": 0, "scanned": 0, "dupes": 0, "errors": 0}
    t0 = time.time()
    for year in args.years:
        ydir = LAKE / "sc_judgments" / f"year={year}"
        tar_path = ydir / "english.tar"
        if not tar_path.exists():
            print(f"[{year}] {tar_path} missing: run datasets_cli.py fetch sc_judgments --years {year}")
            continue
        nc_map = {}
        for f in (ydir / "metadata").glob("*.json"):
            try:
                nc_map[f.stem] = json.loads(f.read_text()).get("nc_display")
            except json.JSONDecodeError:
                pass
        with ProcessPoolExecutor(args.workers) as pool, docs_f.open("a") as df, chunks_f.open("a") as cf, \
                (OUT / "quality.jsonl").open("a") as qf, (OUT / "ocr_queue.jsonl").open("a") as of:
            for res in pool.map(process, tar_items(tar_path, nc_map, seen_ids, args.limit), chunksize=4):
                if "error" in res:
                    totals["errors"] += 1; print("ERR", res["error"]); continue
                doc_id, kind, sha, pages = res["parsed"]
                if sha in seen_hashes:
                    totals["dupes"] += 1; continue
                seen_hashes.add(sha); seen_ids.add(doc_id)
                qf.write(json.dumps(res["quality"]) + "\n")
                if kind == "scanned":
                    totals["scanned"] += 1
                    of.write(json.dumps({"doc_id": doc_id, "year": year, "pages": pages, "tar": str(tar_path.relative_to(ROOT))}) + "\n")
                meta, chunks = res["meta"], res["chunks"]
                if args.index and chunks:
                    texts = [c["text"] for c in chunks]
                    vecs = embeddings.embed_text_passages(texts)
                    cm = [{"document_id": doc_id, "page_number": c["page_number"], "section": c["section"], "court": meta["court"],
                           "year": meta["year"], "doc_type": meta["doc_type"], "citation": meta["citation"], "case_title": meta["title"],
                           "language": meta["language"], "source": "sc_aws"} for c in chunks]
                    for i in range(0, len(chunks), 256):
                        vector_store.add_text_chunks([c["chunk_id"] for c in chunks[i:i+256]], vecs[i:i+256], texts[i:i+256], cm[i:i+256])
                df.write(json.dumps({"doc_id": doc_id, "sha256": sha, "kind": kind, "pages": pages, "year": year, **meta}) + "\n")
                for c in chunks:
                    cf.write(json.dumps(c) + "\n")
                totals["docs"] += 1; totals["chunks"] += len(chunks)
                if totals["docs"] % 25 == 0:
                    print(f"[{year}] {totals} {time.time()-t0:.0f}s", flush=True)
    print("DONE", totals, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
