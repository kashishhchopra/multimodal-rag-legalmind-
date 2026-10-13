import { useState } from "react";
import { GitCompareArrows, Loader2 } from "lucide-react";
import { api } from "../services/api";
import type { CompareResult, DocumentInfo } from "../types";

const STATUS: Record<string, string> = {
  identical: "bg-emerald-50 text-emerald-700",
  similar: "bg-zinc-50 text-zinc-700",
  different: "bg-amber-50 text-amber-700",
  only_in_a: "bg-peach-100 text-peach-500",
  only_in_b: "bg-brand-50 text-brand-700",
};

export default function ComparePage({ documents }: { documents: DocumentInfo[] }) {
  const ready = documents.filter((d) => d.stage === "ready");
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [res, setRes] = useState<CompareResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setErr(null);
    try {
      setRes(await api.compare(a, b));
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Comparison failed");
    } finally {
      setBusy(false);
    }
  }
  const sel = (v: string, set: (s: string) => void, label: string) => (
    <label className="flex-1">
      <div className="label mb-1.5">{label}</div>
      <select value={v} onChange={(e) => set(e.target.value)} className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm">
        <option value="">Choose a document…</option>
        {ready.map((d) => (
          <option key={d.document_id} value={d.document_id}>{d.filename}</option>
        ))}
      </select>
    </label>
  );

  return (
    <div className="mx-auto h-full max-w-[1100px] space-y-5 overflow-y-auto px-8 pb-10 pt-8">
      <div>
        <div className="label">Contract comparison</div>
        <h1 className="mt-1 text-[30px] font-semibold tracking-tight text-slate-900">Compare two documents clause by clause</h1>
        <p className="mt-1 text-sm text-slate-500">Clauses are aligned by detected type (CUAD classifier), then diffed. Missing clause types and risk shifts are flagged.</p>
      </div>
      <div className="card flex flex-wrap items-end gap-4 p-6">
        {sel(a, setA, "Document A")}
        {sel(b, setB, "Document B")}
        <button onClick={run} disabled={!a || !b || a === b || busy} className="flex items-center gap-2 rounded-xl bg-navy-900 px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-40">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitCompareArrows className="h-4 w-4" />} Compare
        </button>
      </div>
      {err && <div className="rounded-xl bg-red-50 px-4 py-2 text-sm text-red-700">{err}</div>}
      {res && (
        <>
          <div className="flex flex-wrap gap-2 text-xs">
            {Object.entries(res.summary).map(([k, v]) => (
              <span key={k} className={`rounded-full px-3 py-1 ${STATUS[k]}`}>{v} {k.replace(/_/g, " ")}</span>
            ))}
          </div>
          {res.rows.length === 0 && <p className="text-sm text-slate-500">No tracked clause types were found in either document.</p>}
          {res.rows.map((r) => (
            <div key={r.category} className="card p-5">
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <span className="text-[15px] font-semibold text-slate-900">{r.category}</span>
                <span className={`rounded-md px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider ${STATUS[r.status]}`}>{r.status.replace(/_/g, " ")}</span>
                {r.status !== "only_in_a" && r.status !== "only_in_b" && <span className="text-xs text-slate-400">similarity {Math.round(r.similarity * 100)}%</span>}
                {r.risk_shift !== "same" && <span className="ml-auto text-xs text-amber-600">{r.risk_shift === "b_riskier" ? "B is riskier" : "A is riskier"}</span>}
              </div>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                {[r.a, r.b].map((side, i) => (
                  <div key={i} className="rounded-xl bg-zinc-50 p-3 text-sm text-slate-600">
                    <div className="label !text-[10px] mb-1">{i === 0 ? "A" : "B"}{side ? ` · page ${side.page} · ${side.risk} risk` : ""}</div>
                    {side ? <p className="line-clamp-5">{side.text}</p> : <p className="text-slate-400">Not present</p>}
                  </div>
                ))}
              </div>
            </div>
          ))}
          <p className="text-xs text-slate-400">Automated analysis for research support only; not legal advice.</p>
        </>
      )}
    </div>
  );
}
