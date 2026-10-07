"""Train and evaluate clause-type classifiers on LEDGAR (100 provision types) and CUAD (41 categories).

    .venv/bin/python scripts/train_clause_classifier.py --out app/models/clause_classifier

Baseline = TF-IDF (word 1-2 grams) + linear SVM. It is deliberately simple and fast; a fine-tuned
legal encoder should be compared against these numbers, not replace the reporting. Splits follow the
datasets' own train/validation/test partitions (CUAD split is by contract, so no clause leaks).
Writes the joblib model plus report.json (micro/macro F1, per-label F1) next to it.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

LAKE = Path(__file__).resolve().parents[2] / "data_lake" / "hf"
_CAT = re.compile(r'related to "([^"]+)"')


def load_ledgar():
    base = LAKE / "coastalcph__lex_glue" / "ledgar"
    frames = {s: pd.read_parquet(next((base / s).glob("*.parquet"))) for s in ("train", "validation", "test")}
    names = None
    for p in (LAKE / "coastalcph__lex_glue").glob("**/README.md"):
        m = re.search(r"- name: ledgar.*?names:\n(.*?)\n\s*- name:", p.read_text(), re.S)
        if m:
            names = re.findall(r"'(\d+)': (.+)", m.group(1))
    label_names = {int(i): n.strip() for i, n in names} if names else {}
    return frames, label_names


def load_cuad(neg_per_contract: int = 12, seed: int = 0):
    """Clause spans (text, category) plus a "None" class: random 600-char windows of the same contracts
    that overlap no annotated span, so the model can say "not a tracked clause"."""
    rng = np.random.default_rng(seed)
    out = {}
    for split in ("train", "test"):
        rows, spans, contexts = [], {}, {}
        for f in sorted((LAKE / "theatticusproject__cuad-qa" / "default" / split).glob("*.parquet")):
            df = pd.read_parquet(f)
            for title, ctx, q, ans in zip(df["title"], df["context"], df["question"], df["answers"]):
                contexts[title] = ctx
                cat = _CAT.search(q)
                if not cat:
                    continue
                for t, st in zip(list(ans["text"]), list(ans["answer_start"])):
                    spans.setdefault(title, []).append((int(st), int(st) + len(t)))
                    if len(t.split()) >= 4:
                        rows.append((t, cat.group(1)))
        for title, ctx in contexts.items():
            tries = 0
            got = 0
            while got < neg_per_contract and tries < 60 and len(ctx) > 800:
                tries += 1
                st = int(rng.integers(0, len(ctx) - 600))
                if any(st < b and st + 600 > a for a, b in spans.get(title, [])):
                    continue
                rows.append((ctx[st:st + 600], "None"))
                got += 1
        out[split] = pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates()
    return out


def fit_eval(name: str, tr, te, out_dir: Path) -> dict:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000)),
        ("svm", LinearSVC(C=0.5, class_weight="balanced")),
    ])
    pipe.fit(tr["text"], tr["label"])
    pred = pipe.predict(te["text"])
    rep = classification_report(te["label"], pred, output_dict=True, zero_division=0)
    result = {
        "dataset": name, "n_train": len(tr), "n_test": len(te), "n_labels": int(tr["label"].nunique()),
        "micro_f1": round(f1_score(te["label"], pred, average="micro"), 4),
        "macro_f1": round(f1_score(te["label"], pred, average="macro"), 4),
        "per_label_f1": {k: round(v["f1-score"], 3) for k, v in rep.items() if isinstance(v, dict) and k not in ("macro avg", "weighted avg")},
    }
    joblib.dump(pipe, out_dir / f"{name}.joblib")
    print(f"{name}: micro-F1 {result['micro_f1']}  macro-F1 {result['macro_f1']}  (train {len(tr)}, test {len(te)}, {result['n_labels']} labels)")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("app/models/clause_classifier"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []

    frames, names = load_ledgar()
    lab = lambda df: df.assign(label=df["label"].map(lambda i: names.get(int(i), str(i))))
    results.append(fit_eval("ledgar", lab(frames["train"]), lab(frames["test"]), args.out))

    cuad = load_cuad()
    results.append(fit_eval("cuad", cuad["train"], cuad["test"], args.out))

    (args.out / "report.json").write_text(json.dumps(results, indent=2))
    print("wrote", args.out / "report.json")


if __name__ == "__main__":
    main()
