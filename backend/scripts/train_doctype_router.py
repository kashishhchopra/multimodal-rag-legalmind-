"""Train a small document-type router: judgment | statute | contract.

    .venv/bin/python scripts/train_doctype_router.py

Data (all real, first 1,500 characters of each document):
  judgment  Supreme Court judgments in data_lake/corpus (docs.jsonl + chunks.jsonl front chunks)
  statute   section chunks of the Acts parsed from official PDFs (statutes.jsonl)
  contract  CUAD contracts (theatticusproject/cuad-qa)
Scope warning: notices, affidavits and "other" are NOT covered (no labelled data yet; RVL-CDIP is the planned source),
and the three classes come from very different sources, so a high score partly reflects source style. Treat the router as a
routing hint (which analysis tools to offer), not as classification of arbitrary uploads.
Split: judgments by year-disjoint docs is unnecessary here; statutes and contracts are split by Act / contract title.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "app" / "models" / "doctype_router"


def main() -> None:
    rng = random.Random(1)
    rows = []  # (text, label, group)
    docs = {d["doc_id"] for d in map(json.loads, (ROOT / "data_lake/corpus/docs.jsonl").open())}
    seen = set()
    for c in map(json.loads, (ROOT / "data_lake/corpus/chunks.jsonl").open()):
        if c["doc_id"] in docs and c["doc_id"] not in seen and c["section"] in ("front_matter", "issue", "headnotes", "judgment"):
            seen.add(c["doc_id"]); rows.append((c["text"][:1500], "judgment", c["doc_id"]))
    for c in map(json.loads, (ROOT / "data_lake/corpus/statutes.jsonl").open()):
        if len(c["text"]) > 200:
            rows.append((c["text"][:1500], "statute", c["act"]))
    for f in sorted((ROOT / "data_lake/hf/theatticusproject__cuad-qa/default").glob("*/*.parquet")):
        df = pd.read_parquet(f)
        for title, ctx in df.drop_duplicates("title")[["title", "context"]].itertuples(index=False):
            rows.append((ctx[:1500], "contract", title))
    df = pd.DataFrame(rows, columns=["text", "label", "group"])
    # balance classes, then hold out whole groups (Acts / contracts / judgments) for testing
    n = min(df.label.value_counts().min(), 1500)
    df = pd.concat([g.sample(n, random_state=1) for _, g in df.groupby("label")])
    groups = sorted(df.group.unique()); rng.shuffle(groups)
    test_groups = set(groups[: max(3, len(groups) // 5)])
    tr, te = df[~df.group.isin(test_groups)], df[df.group.isin(test_groups)]
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=100_000)),
                     ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pipe.fit(tr.text, tr.label)
    rep = classification_report(te.label, pipe.predict(te.text), output_dict=True, zero_division=0)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUT / "router.joblib")
    result = {"n_train": len(tr), "n_test": len(te), "accuracy": round(rep["accuracy"], 4),
              "per_class_f1": {k: round(rep[k]["f1-score"], 3) for k in ("judgment", "statute", "contract") if k in rep},
              "classes_not_covered": ["notice", "affidavit", "other"]}
    (OUT / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
