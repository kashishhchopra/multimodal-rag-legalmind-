"""End-to-end answer evaluation: latency, refusal rate, claim support, and (optional) RAGAS.

    .venv/bin/python -m eval.answer_eval --questions eval/answer_questions.jsonl --out eval/results/answers.json
    pip install ragas && ... --ragas      # needs LLM_API_KEY; RAGAS judge calls cost API quota

Input JSONL, one object per line: {"question": str, "document_id": str|null, "reference": str (optional)}.
document_id null = cross-document search. Without an LLM key every answer is the evidence-only fallback, so the
claim-support numbers are not meaningful then and the script says so. Use a different model family to judge than to
generate (roadmap 8.3); the RAGAS judge model is set with --judge-model.
Metrics: p50/p95 latency, refusal rate (guard), flagged rate, mean claim-support ratio, and with --ragas faithfulness,
answer_relevancy and context_precision (context_precision/recall need `reference`).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from app.services import query_service


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 3) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("eval/results/answers.json"))
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.questions.open() if l.strip()]
    rows = rows[: a.limit] if a.limit else rows
    per, lat = [], []
    for r in rows:
        t0 = time.perf_counter()
        resp = query_service.answer_question(r.get("document_id"), r["question"])
        dt = time.perf_counter() - t0
        lat.append(dt)
        v = resp.verification
        per.append({
            "question": r["question"], "document_id": r.get("document_id"), "latency_s": round(dt, 3), "warning": resp.warning,
            "status": v.status if v else None, "support_ratio": v.support_ratio if v else None,
            "answer": resp.answer, "contexts": [t.text for t in resp.text_evidence], "reference": r.get("reference"),
        })
    llm_answers = [p for p in per if p["warning"] not in ("missing_api_key", "llm_failed")]
    support = [p["support_ratio"] for p in llm_answers if p["support_ratio"] is not None and p["status"] != "refused"]
    summary = {
        "n": len(per), "llm_answered": len(llm_answers), "latency_p50_s": pct(lat, 50), "latency_p95_s": pct(lat, 95),
        "refusal_rate": round(sum(p["status"] == "refused" for p in per) / max(1, len(per)), 3),
        "flagged_rate": round(sum(p["status"] == "flagged" for p in per) / max(1, len(per)), 3),
        "mean_claim_support": round(statistics.mean(support), 3) if support else None,
    }
    if len(llm_answers) < len(per):
        summary["warning"] = f"{len(per) - len(llm_answers)} answers were evidence-only (no working LLM key); support metrics cover only LLM answers"
    if a.ragas:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, faithfulness

            data = Dataset.from_list([{"question": p["question"], "answer": p["answer"], "contexts": p["contexts"]} for p in llm_answers])
            res = evaluate(data, metrics=[faithfulness, answer_relevancy])
            summary["ragas"] = {k: round(float(v), 4) for k, v in res.items()}
        except Exception as exc:  # noqa: BLE001
            summary["ragas_error"] = f"{type(exc).__name__}: {exc}"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "per_question": per}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
