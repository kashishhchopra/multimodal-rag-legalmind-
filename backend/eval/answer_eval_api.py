"""Answer-quality evaluation against the RUNNING app (no second copy of the models in RAM).

    python3 eval/answer_eval_api.py --questions eval/answer_questions.jsonl --out eval/results/answers_api.json

Per question it records: answered (a real LLM answer, not the evidence-only fallback), has page citations, facts present
(`must_contain`: any of these strings, case-insensitive), honest non-answer for `expect_not_found` questions, claim-support ratio
from the app's verifier, context mode and latency. Paced for free-tier rate limits (--pause). Reference-free: it checks
grounding behaviour and known facts, not overall correctness; read the saved answers for that. No RAGAS (needs a judge model).
"""
import argparse
import json
import re
import statistics
import time
import urllib.request

NOT_FOUND = re.compile(r"(does not (contain|mention|address|provide|include|specify|state)|no (mention|information|reference)|not (mentioned|stated|specified|addressed|found|covered|contain)|cannot be (found|determined)|is not (in|part of))", re.I)
CITED = re.compile(r"\(\s*(?:pages?|p\.|clause|section|para)", re.I)


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/answer_questions.jsonl")
    ap.add_argument("--out", default="eval/results/answers_api.json")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--pause", type=float, default=6.0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.questions) if l.strip()]
    out, lat = [], []
    for q in rows:
        t0 = time.time()
        try:
            r = (post(a.base, f"/api/documents/{q['document_id']}/query", {"question": q["question"]}) if q.get("document_id")
                 else post(a.base, "/api/search", {"question": q["question"], "top_k": 8}))
        except Exception as exc:  # noqa: BLE001
            r = {"answer": f"REQUEST FAILED: {exc}", "warning": "request_failed"}
        dt = time.time() - t0
        lat.append(dt)
        ans, warn = r.get("answer", ""), r.get("warning", "")
        llm_ok = warn not in ("llm_failed", "llm_quota", "missing_api_key", "request_failed", "no_evidence")
        v = r.get("verification") or {}
        rec = {"question": q["question"], "document_id": q.get("document_id"), "latency_s": round(dt, 1), "warning": warn, "llm_answered": llm_ok,
               "cited": bool(CITED.search(ans)), "context_mode": r.get("context_mode"), "support_ratio": v.get("support_ratio"), "answer": ans}
        if "must_contain" in q:
            rec["facts_present"] = any(s.lower() in ans.lower() for s in q["must_contain"])
        if q.get("expect_not_found"):
            rec["honest_non_answer"] = bool(NOT_FOUND.search(ans))
        out.append(rec)
        print(f"{'OK ' if llm_ok else 'ERR'} {dt:5.1f}s  {q['question'][:70]}", flush=True)
        time.sleep(a.pause)
    ok = [r for r in out if r["llm_answered"]]
    pct = lambda xs, p: round(sorted(xs)[min(len(xs) - 1, int(p / 100 * (len(xs) - 1) + .5))], 1) if xs else None
    sup = [r["support_ratio"] for r in ok if r["support_ratio"] is not None]
    summary = {
        "questions": len(out), "llm_answered": len(ok),
        "cited_rate": round(sum(r["cited"] for r in ok if r.get("honest_non_answer") is None) / max(1, sum(1 for r in ok if r.get("honest_non_answer") is None)), 3),
        "facts_correct": f"{sum(r['facts_present'] for r in out if 'facts_present' in r)}/{sum(1 for r in out if 'facts_present' in r)}",
        "honest_non_answers": f"{sum(r['honest_non_answer'] for r in out if 'honest_non_answer' in r)}/{sum(1 for r in out if 'honest_non_answer' in r)}",
        "mean_claim_support": round(statistics.mean(sup), 3) if sup else None, "latency_p50_s": pct(lat, 50), "latency_p95_s": pct(lat, 95),
        "context_modes": {m: sum(1 for r in ok if r["context_mode"] == m) for m in {r["context_mode"] for r in ok}},
    }
    json.dump({"summary": summary, "results": out}, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
