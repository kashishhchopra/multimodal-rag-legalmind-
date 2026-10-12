"""(Re)build the Chroma index for the judgment + statute corpus from data_lake/corpus/*.jsonl.

    TEXT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 VECTOR_DB_PATH=data_lake/corpus_store \
        backend/.venv/bin/python scripts/reindex_corpus.py

Resumable: work is cut into shards of --shard chunks; each finished shard drops a marker in data_lake/corpus/index_done/
and keeps its embeddings in data_lake/embeddings/, so a crash or Ctrl-C costs at most one shard. Never kill this
process during a write with `kill -9`: an abrupt kill is what corrupted the previous store. Judgment chunks need
docs.jsonl (for year/court/citation metadata); statute chunks come from statutes.jsonl.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.models import embeddings  # noqa: E402
from app.retrieval import vector_store  # noqa: E402

CORPUS = ROOT / "data_lake" / "corpus"
EMB = ROOT / "data_lake" / "embeddings"


def rows():
    docs = {d["doc_id"]: d for d in map(json.loads, (CORPUS / "docs.jsonl").open())}
    for c in map(json.loads, (CORPUS / "chunks.jsonl").open()):
        d = docs.get(c["doc_id"])
        if not d:
            continue
        yield c["chunk_id"], c["text"], {"document_id": c["doc_id"], "page_number": c["page_number"], "section": c["section"], "court": d["court"],
                                         "year": d["year"], "doc_type": d["doc_type"], "citation": d["citation"], "case_title": d["title"],
                                         "language": d["language"], "source": "sc_aws"}
    if (CORPUS / "statutes.jsonl").exists():
        for c in map(json.loads, (CORPUS / "statutes.jsonl").open()):
            yield c["chunk_id"], c["text"], {"document_id": f"statute_{c['act']}", "page_number": c["page_number"], "doc_type": "statute", "act": c["act"],
                                             "section": c["section"], "year": c["year"], "court": "", "source": "official-pdf"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=2000)
    a = ap.parse_args()
    (CORPUS / "index_done").mkdir(parents=True, exist_ok=True)
    EMB.mkdir(parents=True, exist_ok=True)
    buf, n = [], 0
    def flush(shard: int, items) -> None:
        marker = CORPUS / "index_done" / f"{shard:05d}"
        if marker.exists():
            return
        ids, texts, metas = zip(*items)
        f = EMB / f"{shard:05d}.npy"
        vecs = np.load(f) if f.exists() else embeddings.embed_text_passages(list(texts))
        if not f.exists():
            np.save(f, vecs)
        for i in range(0, len(ids), 500):
            vector_store.add_text_chunks(list(ids[i:i + 500]), vecs[i:i + 500], list(texts[i:i + 500]), list(metas[i:i + 500]))
        marker.write_text("ok")
        print(f"shard {shard} done ({(shard + 1) * a.shard} chunks)", flush=True)
    for item in rows():
        buf.append(item)
        if len(buf) == a.shard:
            flush(n, buf); buf, n = [], n + 1
    if buf:
        flush(n, buf)
    print("indexed; chroma count:", vector_store.text_count())


if __name__ == "__main__":
    main()
