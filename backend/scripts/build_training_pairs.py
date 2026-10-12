"""Build (query, positive, hard-negative) training pairs for embedder/reranker v2 from REAL data.

    .venv/bin/python scripts/build_training_pairs.py --out ../data_lake/pairs --train-max-year 2022 --val-year 2023

Sources (all offline, no LLM calls):
  cuad      clause-category description  -> annotated clause span            (CUAD train split only)
  ledgar    "<label> clause"             -> provision text                   (LEDGAR train split only)
  issue     "Issue for Consideration"    -> best-matching reasoning chunk    (SC judgments; round-trip filtered)
  citation  citing passage               -> headnote/issue chunk of the cited case (when cited case is in corpus)
  ict       one sentence of a passage    -> the rest of that passage (inverse cloze; label-free, works for every year)
            Headnotes ("Issue for Consideration") exist only in the newest SC reporter format, so ICT is what gives the
            2019-2022 judgments any training signal. Negatives: a same-year chunk of another judgment + the adjacent chunk.
LLM-generated questions are a separate, opt-in step (see --llm-questions): they cost API quota and
send judgment text to a provider, so they are never run implicitly.

Leakage rules: judgment pairs are split by YEAR (train <= --train-max-year, val == --val-year, test later)
and a document never appears in two splits; CUAD/LEDGAR keep their native train splits only so their
test sets stay sealed. Round-trip filter keeps an issue-pair only if BM25 over the whole pool returns the
positive in the top --roundtrip-k, removing vague queries. Hard negatives: BM25 top non-gold from the
same pool, plus a neighbouring chunk of the same judgment.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from app.retrieval.hybrid import BM25Index  # noqa: E402

LAKE = Path(__file__).resolve().parents[2] / "data_lake"
_CAT = re.compile(r'related to "([^"]+)"')
_DETAILS = re.compile(r"Details:\s*(.+)$")
_INSC = re.compile(r"\b(\d{4})\s*INSC\s*(\d+)\b")


def cuad_pairs(rng, max_pairs):
    rows = []
    for f in sorted((LAKE / "hf" / "theatticusproject__cuad-qa" / "default" / "train").glob("*.parquet")):
        df = pd.read_parquet(f)
        for q, ans in zip(df["question"], df["answers"]):
            cat, det = _CAT.search(q), _DETAILS.search(q)
            for t in list(ans["text"]):
                if cat and len(t.split()) >= 6:
                    rows.append({"query": f"{cat.group(1)}: {det.group(1) if det else ''}".strip(), "pos": t[:1500], "source": "cuad", "year": 0, "doc": "cuad"})
    rng.shuffle(rows)
    return rows[:max_pairs]


def ledgar_pairs(rng, max_pairs):
    base = LAKE / "hf" / "coastalcph__lex_glue"
    readme = next(base.glob("**/README.md"), None)
    names = {}
    if readme:
        m = re.search(r"- name: ledgar.*?names:\n(.*?)\n\s*- name:", readme.read_text(), re.S)
        names = {int(i): n.strip() for i, n in re.findall(r"'(\d+)': (.+)", m.group(1))} if m else {}
    df = pd.read_parquet(next((base / "ledgar" / "train").glob("*.parquet")))
    rows = [{"query": f"{names.get(int(l), l)} clause", "pos": t[:1500], "source": "ledgar", "year": 0, "doc": "ledgar"}
            for t, l in zip(df["text"], df["label"]) if len(t.split()) >= 12]
    rng.shuffle(rows)
    return rows[:max_pairs]


def ict_pairs(by_doc, docs, args, rng):
    out = []
    by_year = defaultdict(list)
    for d, cs in by_doc.items():
        by_year[docs[d]["year"]].append(d)
    for year, dids in by_year.items():
        if year > args.val_year or year < 2000:
            continue  # test years are evaluated on issue pairs only; never train on them
        cap = args.max_ict // 6 if year <= args.train_max_year else args.max_ict_val
        rng.shuffle(dids)
        made = 0
        for d in dids:
            cs = [c for c in by_doc[d] if c["section"] in ("judgment", "analysis", "facts") and len(c["text"]) > 700]
            for c in rng.sample(cs, min(4, len(cs))):
                sents = [x for x in re.split(r"(?<=[.!?])\s+", c["text"]) if 15 <= len(x.split()) <= 60]
                if not sents:
                    continue
                q = rng.choice(sents)
                rest = c["text"].replace(q, " ", 1).strip()
                other = rng.choice(by_doc[rng.choice(dids)])
                if other["doc_id"] == d:
                    continue
                idx = by_doc[d].index(c)
                neg = [other["text"]] + ([by_doc[d][idx + 1]["text"]] if idx + 1 < len(by_doc[d]) else [])
                out.append({"query": q, "pos": rest, "neg": neg, "source": "ict", "year": year, "doc": d})
                made += 1
            if made >= cap:
                break
    return out


def judgment_pairs(args, rng):
    docs = {d["doc_id"]: d for d in map(json.loads, (LAKE / "corpus" / "docs.jsonl").open())}
    by_doc = defaultdict(list)
    for c in map(json.loads, (LAKE / "corpus" / "chunks.jsonl").open()):
        if c["doc_id"] in docs:  # chunks of a document killed mid-write have no docs.jsonl record: skip them
            by_doc[c["doc_id"]].append(c)
    pool = [c for cs in by_doc.values() for c in cs if c["section"] in ("judgment", "analysis", "facts")]
    bm25 = BM25Index([c["chunk_id"] for c in pool], [c["text"] for c in pool])
    pos_text = {c["chunk_id"]: c for c in pool}
    out = []
    # (a) issue -> reasoning chunk (round-trip filtered)
    for doc_id, cs in by_doc.items():
        issue = " ".join(c["text"] for c in cs if c["section"] == "issue")[:600]
        if len(issue.split()) < 12:
            continue
        own = {c["chunk_id"] for c in cs}
        hits = bm25.search(issue, max(args.roundtrip_k, 10))
        top = [h for h, _ in hits[: args.roundtrip_k]]
        gold = next((h for h in top if h in own), None)
        if not gold:
            continue  # round trip failed: the query does not retrieve its own case
        negs = [pos_text[h]["text"] for h, _ in hits if h not in own][:3]
        idx = [c["chunk_id"] for c in cs].index(gold)
        if idx + 1 < len(cs):
            negs.append(cs[idx + 1]["text"])
        out.append({"query": issue, "pos": pos_text[gold]["text"], "neg": negs, "source": "issue", "year": docs[doc_id]["year"], "doc": doc_id})
    # (b) citation: citing passage -> cited case's headnote/issue
    cite_to_doc = {docs[d]["citation"]: d for d in docs if docs[d].get("citation")}
    for c in pool:
        for m in _INSC.finditer(c["text"]):
            cited = cite_to_doc.get(f"{m.group(1)} INSC {m.group(2)}")
            if cited and cited != c["doc_id"]:
                heads = [x for x in by_doc[cited] if x["section"] in ("headnotes", "issue")]
                if heads:
                    out.append({"query": c["text"][:900], "pos": heads[0]["text"], "neg": [], "source": "citation", "year": docs[c["doc_id"]]["year"], "doc": c["doc_id"]})
                    break
    out += ict_pairs(by_doc, docs, args, rng)
    return out


def assign_split(rows, train_max, val_year):
    for r in rows:
        r["split"] = "train" if r["year"] == 0 or r["year"] <= train_max else "val" if r["year"] == val_year else "test"
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=LAKE / "pairs")
    ap.add_argument("--train-max-year", type=int, default=2022)
    ap.add_argument("--val-year", type=int, default=2023)
    ap.add_argument("--max-cuad", type=int, default=10000)
    ap.add_argument("--max-ledgar", type=int, default=10000)
    ap.add_argument("--roundtrip-k", type=int, default=20)
    ap.add_argument("--max-ict", type=int, default=9000, help="total ICT pairs across train years")
    ap.add_argument("--max-ict-val", type=int, default=300, help="ICT pairs for the validation year")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = cuad_pairs(rng, args.max_cuad) + ledgar_pairs(rng, args.max_ledgar)
    if (LAKE / "corpus" / "chunks.jsonl").exists():
        rows += judgment_pairs(args, rng)
    else:
        print("no judgment corpus yet (run scripts/ingest_judgments.py); building CUAD/LEDGAR pairs only")
    rows = assign_split(rows, args.train_max_year, args.val_year)
    # a judgment must live in exactly one split
    seen = {}
    for r in rows:
        assert seen.setdefault(r["doc"], r["split"]) == r["split"] or r["doc"] in ("cuad", "ledgar"), r["doc"]
    args.out.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    with (args.out / "pairs.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            counts[(r["source"], r["split"])] += 1
    print(json.dumps({f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())}, indent=1))
    print("wrote", args.out / "pairs.jsonl", len(rows))


if __name__ == "__main__":
    main()
