"""Hindi mode evaluation on docs/hindi_question_set.xlsx (200 Hindi questions, 20 SC judgments, English expected answers).

    .venv/bin/python -m eval.hindi_eval translate            # 1. Hindi -> English (batched LLM calls, cached)
    .venv/bin/python -m eval.hindi_eval retrieval            # 2. which query form finds the right document/passages (no LLM)
    .venv/bin/python -m eval.hindi_eval answers --n 40       # 3. end-to-end Hindi answers from the RUNNING app (cached, resumable)
    .venv/bin/python -m eval.hindi_eval judge                # 4. judge answers vs expected answers (batched LLM calls)

Caveats printed with the results: the Hindi text was drafted by Claude and has NOT been confirmed by a native speaker (column H is
empty); expected answers come from headnotes; the judge is the same model family as the generator (bias risk); small samples.
Everything is cached under data_lake/hindi_eval/ so a quota stop loses nothing.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_lake" / "hindi_eval"
XLSX = ROOT / "docs" / "hindi_question_set.xlsx"
STOP = set("the a an of in to and or for on by with that this is was were be been it its as at from which who whom what are not no into than then they their his her he she under upon any".split())


def load():
    import openpyxl

    ws = openpyxl.load_workbook(XLSX, data_only=True)["Questions"]
    keys = ["id", "hindi", "doc", "case", "citation", "expected", "type", "checked"]
    return [dict(zip(keys, r)) for r in ws.iter_rows(min_row=2, values_only=True) if r[0]]


def jload(name, default):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else default


def jsave(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def provider():
    sys.path.insert(0, str(ROOT / "backend"))
    from app.services import llm

    return llm.get_provider()


def llm_json(p, system, user, tokens=3000):
    raw = p.generate(system, user, [], max_output_tokens=tokens)
    m = re.search(r"[\[{].*[\]}]", raw, re.S)
    return json.loads(m.group(0))


# ---------------------------------------------------------------- 1. translate
def cmd_translate(a):
    rows, cache = load(), jload("translations.json", {})
    todo = [r for r in rows if r["id"] not in cache]
    p = provider()
    for i in range(0, len(todo), 25):
        batch = todo[i:i + 25]
        user = "Translate each Hindi question into natural English, keeping names, section numbers and citations exactly. " \
               'Reply ONLY with JSON: [{"id": "HQ001", "en": "..."}, ...]\n\n' + json.dumps([{"id": r["id"], "hi": r["hindi"]} for r in batch], ensure_ascii=False)
        try:
            for item in llm_json(p, "You are a precise Hindi-to-English legal translator.", user):
                cache[item["id"]] = item["en"]
        except Exception as exc:  # noqa: BLE001
            print("stopped (quota or parse error):", str(exc)[:200]); break
        jsave("translations.json", cache)
        print(f"translated {len(cache)}/{len(rows)}", flush=True)
        time.sleep(4)


# ---------------------------------------------------------------- 2. retrieval
def toks(text):
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 3 and t not in STOP}


def cmd_retrieval(a):
    from sentence_transformers import SentenceTransformer

    rows, tr = load(), jload("translations.json", {})
    chunks = [json.loads(l) for l in (ROOT / "data_lake/corpus/chunks.jsonl").open()]
    docs = {r["doc"] for r in rows}
    pool = [c for c in chunks if c["doc_id"] in docs]
    texts, owner = [c["text"] for c in pool], [c["doc_id"] for c in pool]
    print(f"pool: {len(docs)} judgments, {len(pool)} passages; {len(tr)}/{len(rows)} translated")
    variants = {"bge-small-en, Hindi query as is": ("BAAI/bge-small-en-v1.5", "hindi"),
                "multilingual-e5-small, Hindi query as is": ("intfloat/multilingual-e5-small", "hindi")}
    if tr:
        variants["bge-small-en, LLM-translated query"] = ("BAAI/bge-small-en-v1.5", "en")
    res = {}
    for name, (model_name, form) in variants.items():
        m = SentenceTransformer(model_name, device="cpu")
        e5 = "e5" in model_name
        d = m.encode([("passage: " + t) if e5 else t for t in texts], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        qs = [r for r in rows if form == "hindi" or r["id"] in tr]
        qtext = [("query: " if e5 else ("Represent this sentence for searching relevant passages: " if "bge" in model_name else "")) + (r["hindi"] if form == "hindi" else tr[r["id"]]) for r in qs]
        q = m.encode(qtext, normalize_embeddings=True, show_progress_bar=False)
        sims = q @ d.T
        r1 = r5 = hit = 0
        for i, r in enumerate(qs):
            order = np.argsort(-sims[i])
            ranked_docs = []
            for j in order[:60]:
                if owner[j] not in ranked_docs:
                    ranked_docs.append(owner[j])
            r1 += ranked_docs[:1] == [r["doc"]]
            r5 += r["doc"] in ranked_docs[:5]
            # within the correct judgment: do the top-5 passages cover the expected answer's key words?
            own = [j for j in range(len(pool)) if owner[j] == r["doc"]]
            top = sorted(own, key=lambda j: -sims[i][j])[:5]
            want = toks(r["expected"])
            hit += bool(want) and len(want & set().union(*(toks(texts[j]) for j in top))) / len(want) >= 0.6
        n = len(qs)
        res[name] = {"questions": n, "doc_R@1": round(r1 / n, 3), "doc_R@5": round(r5 / n, 3), "answer_terms_in_top5_of_right_doc": round(hit / n, 3)}
        print(name, res[name], flush=True)
    jsave("retrieval.json", res)


# ---------------------------------------------------------------- 3. answers (from the running app)
def cmd_answers(a):
    rows = load()
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    quota = {"Fact": 14, "Holding": 12, "Reasoning": 8, "Procedure": 3, "Number": 3}
    chosen = [r for t, k in quota.items() for r in by_type[t][:k]][: a.n]
    cache = jload("answers.json", {})
    if a.spread:  # 2 not-yet-answered questions per judgment, covering all 20 documents
        per_doc = defaultdict(list)
        for r in rows:
            if r["id"] not in cache:
                per_doc[r["doc"]].append(r)
        chosen = [r for d in per_doc for r in per_doc[d][3:5]][: a.n]
    for r in chosen:
        if r["id"] in cache and cache[r["id"]].get("ok"):
            continue
        body = json.dumps({"question": r["hindi"], "language": "hi"}).encode()
        req = urllib.request.Request(f"{a.base}/api/documents/{r['doc']}/query", body, {"content-type": "application/json"})
        t0 = time.time()
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300))
        except Exception as exc:  # noqa: BLE001 - record the failure and keep going
            cache[r["id"]] = {"ok": False, "warning": f"request_failed: {str(exc)[:120]}", "answer": "", "latency_s": round(time.time() - t0, 1), "mode": None}
            jsave("answers.json", cache); print(r["id"], "request failed", flush=True); continue
        ok = d.get("warning") not in ("llm_failed", "llm_quota", "missing_api_key")
        cache[r["id"]] = {"ok": ok, "warning": d.get("warning"), "answer": d.get("answer", ""), "latency_s": round(time.time() - t0, 1), "mode": d.get("context_mode")}
        jsave("answers.json", cache)
        print(f"{r['id']} {'ok' if ok else d.get('warning')} {cache[r['id']]['latency_s']}s", flush=True)
        if not ok and d.get("warning") == "llm_quota":
            print("quota reached; rerun later to resume"); break
        time.sleep(a.pause)


# ---------------------------------------------------------------- 4. judge
def cmd_judge(a):
    rows = {r["id"]: r for r in load()}
    ans, tr, cache = jload("answers.json", {}), jload("translations.json", {}), jload("judgements.json", {})
    todo = [i for i, v in ans.items() if v.get("ok") and i not in cache]
    p = provider()
    for k in range(0, len(todo), 8):
        batch = todo[k:k + 8]
        items = [{"id": i, "question_hi": rows[i]["hindi"], "question_en": tr.get(i, ""), "expected_answer": rows[i]["expected"], "system_answer": ans[i]["answer"][:1500]} for i in batch]
        user = ("For each item decide whether system_answer (may be Hindi) correctly answers the question, judged against expected_answer. "
                'verdict: "correct" (same substance), "partial" (some right, something missing or wrong), "incorrect", or "no_answer" (says the document does not say). '
                'Reply ONLY with JSON: [{"id": "...", "verdict": "...", "reason": "<=15 words"}]\n\n' + json.dumps(items, ensure_ascii=False))
        try:
            for it in llm_json(p, "You are a strict grader of legal question answering.", user):
                cache[it["id"]] = it
        except Exception as exc:  # noqa: BLE001
            print("stopped:", str(exc)[:200]); break
        jsave("judgements.json", cache)
        time.sleep(4)
    v = [cache[i]["verdict"] for i in cache if i in ans]
    summary = {k: v.count(k) for k in ("correct", "partial", "incorrect", "no_answer")}
    summary["judged"] = len(v)
    by = defaultdict(lambda: [0, 0])
    for i in cache:
        by[rows[i]["type"]][1] += 1
        by[rows[i]["type"]][0] += cache[i]["verdict"] == "correct"
    summary["correct_by_type"] = {t: f"{c}/{n}" for t, (c, n) in by.items()}
    summary["caveats"] = ["Hindi drafted by Claude, not native-checked", "expected answers from headnotes", "judge = same model family", "small stratified sample"]
    jsave("summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["translate", "retrieval", "answers", "judge"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--pause", type=float, default=5)
    ap.add_argument("--spread", action="store_true", help="sample 2 new questions from every judgment")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    a = ap.parse_args()
    {"translate": cmd_translate, "retrieval": cmd_retrieval, "answers": cmd_answers, "judge": cmd_judge}[a.cmd](a)
