"""Copy the demo/uploaded documents from the old store into the corpus store so one index serves everything.

    TEXT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 VECTOR_DB_PATH=data_lake/corpus_store \
        backend/.venv/bin/python scripts/migrate_demo_to_corpus.py [--old backend/vector_store]

The old store was built with the v1 (synthetic fine-tune) embedder, whose vectors are not comparable with the base
embedder used for the real corpus (never mix embedders in one store). So text chunks and OCR regions are RE-EMBEDDED from their
stored text; CLIP image vectors are model-independent and are copied as they are. Idempotent: ids that already exist are skipped.
"""
import argparse
import sys
from pathlib import Path

import chromadb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.models import embeddings  # noqa: E402
from app.retrieval import vector_store as vs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, default=ROOT / "backend" / "vector_store")
    a = ap.parse_args()
    old = chromadb.PersistentClient(path=str(a.old))
    for name, kind in ((vs.TEXT_COLLECTION, "text"), (vs.REGION_COLLECTION, "text"), (vs.VISUAL_COLLECTION, "image")):
        try:
            col = old.get_collection(name)
        except Exception:  # noqa: BLE001
            print(name, "not in old store, skipped")
            continue
        got = col.get(include=["documents", "metadatas", "embeddings"])
        ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
        have = set(vs._collection(name).get(ids=ids)["ids"]) if ids else set()
        keep = [i for i, _id in enumerate(ids) if _id not in have]
        if not keep:
            print(name, "already migrated"); continue
        ids2, docs2, metas2 = [ids[i] for i in keep], [docs[i] or "" for i in keep], [metas[i] for i in keep]
        vecs = embeddings.embed_text_passages(docs2) if kind == "text" else np.asarray([got["embeddings"][i] for i in keep], dtype=np.float32)
        {vs.TEXT_COLLECTION: vs.add_text_chunks, vs.REGION_COLLECTION: vs.add_regions, vs.VISUAL_COLLECTION: vs.add_visuals}[name](ids2, vecs, docs2, metas2)
        print(f"{name}: migrated {len(ids2)} items ({kind} vectors {'re-embedded' if kind == 'text' else 'copied'})")


if __name__ == "__main__":
    main()
