"""Backfill the FTS5 lexical index for an existing vector store (VECTOR_DB_PATH) from the corpus JSONL files.

    VECTOR_DB_PATH=data_lake/corpus_store backend/.venv/bin/python scripts/build_lexical_index.py

reindex_corpus.py fills it automatically for new indexing; use this for stores built before lexical.py existed. Safe to
re-run: it rebuilds the table from scratch.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
from app.retrieval import lexical  # noqa: E402
from reindex_corpus import rows  # noqa: E402


def main() -> None:
    lexical._conn().execute("DELETE FROM chunks")
    lexical._conn().commit()
    batch = []
    n = 0
    for cid, text, meta in rows():
        batch.append((cid, text, meta))
        if len(batch) == 5000:
            lexical.add(*zip(*batch)); n += len(batch); batch = []
            print(n, flush=True)
    if batch:
        lexical.add(*zip(*batch)); n += len(batch)
    print("lexical rows:", lexical.count())


if __name__ == "__main__":
    main()
