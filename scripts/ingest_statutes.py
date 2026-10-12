"""Parse statute PDFs section by section into data_lake/corpus/statutes.jsonl (and optionally index them).

    backend/.venv/bin/python scripts/ingest_statutes.py                       # parse only
    TEXT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 VECTOR_DB_PATH=data_lake/corpus_store \
        backend/.venv/bin/python scripts/ingest_statutes.py --index

Sources (government publications; cite the source): data_lake/statutes/{BNS_2023,BNSS_2023_bilingual,BSA_2023}.pdf from mha.gov.in and
Constitution_of_India_2019_english.pdf (Ministry of Law and Justice edition as on 1 April 2019; amendments after that are not included).
Add more Acts from India Code by dropping PDFs in data_lake/statutes/ and listing them in ACTS.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.ingestion.statutes import parse_constitution, parse_sections  # noqa: E402

ACTS = {"BNS": ("BNS_2023.pdf", 2023), "BNSS": ("BNSS_2023_bilingual.pdf", 2023), "BSA": ("BSA_2023.pdf", 2023),
        "COI": ("Constitution_of_India_2019_english.pdf", 2019)}  # COI: articles; Art. 370 is not captured by the parser


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true")
    args = ap.parse_args()
    out = ROOT / "data_lake" / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    allc = []
    for act, (fn, year) in ACTS.items():
        pdf = ROOT / "data_lake" / "statutes" / fn
        if not pdf.exists():
            print("missing", pdf)
            continue
        chunks = parse_constitution(str(pdf), act) if act == "COI" else parse_sections(str(pdf), act)
        print(f"{act}: {len(chunks)} chunks, {len({c.section for c in chunks})} sections")
        allc += [(c, year) for c in chunks]
    with (out / "statutes.jsonl").open("w") as f:
        for c, y in allc:
            f.write(json.dumps({**asdict(c), "year": y}) + "\n")
    if args.index:
        from app.models import embeddings
        from app.retrieval import vector_store

        for i in range(0, len(allc), 128):
            batch = allc[i:i + 128]
            texts = [c.text for c, _ in batch]
            vector_store.add_text_chunks(
                [c.chunk_id for c, _ in batch], embeddings.embed_text_passages(texts), texts,
                [{"document_id": f"statute_{c.act}", "page_number": c.page_number, "doc_type": "statute", "act": c.act,
                  "section": c.section, "year": y, "court": "", "source": "mha.gov.in"} for c, y in batch])
        print("indexed", len(allc))


if __name__ == "__main__":
    main()
