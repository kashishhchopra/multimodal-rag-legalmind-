import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { api } from "../services/api";
import type { AdminStats } from "../types";

function Tile({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div className="card p-5">
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-slate-800">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-slate-400">{hint}</div>}
    </div>
  );
}

function Bars({ data }: { data: Record<string, number> }) {
  const max = Math.max(1, ...Object.values(data));
  return (
    <div className="space-y-1.5">
      {Object.entries(data).map(([k, v]) => (
        <div key={k} className="flex items-center gap-2 text-xs">
          <span className="w-44 truncate text-slate-600">{k}</span>
          <div className="h-3 flex-1 rounded bg-slate-100">
            <div className="h-3 rounded bg-brand-500" style={{ width: `${(v / max) * 100}%` }} />
          </div>
          <span className="w-8 text-right text-slate-500">{v}</span>
        </div>
      ))}
    </div>
  );
}

export default function AdminPanel() {
  const [s, setS] = useState<AdminStats | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.adminStats().then(setS).catch((e) => setErr(e instanceof Error ? e.message : "Failed"));
  }, []);
  if (err) return <div className="m-6 rounded-lg bg-red-50 px-4 py-2 text-sm text-red-700">{err}</div>;
  if (!s)
    return (
      <div className="flex items-center gap-2 p-6 text-sm text-slate-500">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading…
      </div>
    );
  const q = s.queries;
  const fb = s.feedback;
  return (
    <div className="mx-auto h-full max-w-[1000px] space-y-5 overflow-y-auto px-8 pb-10 pt-8">
      <div>
        <div className="label">Admin</div>
        <h1 className="mt-1 text-[30px] font-semibold tracking-tight text-slate-900">System health and usage</h1>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Tile label="Documents" value={s.documents} />
        <Tile label="Indexed chunks" value={s.indexed_text_chunks ?? "n/a"} />
        <Tile label="Queries logged" value={q.queries} />
        <Tile label="Auth" value={s.auth_enabled ? "on" : "off"} />
        <Tile label="Latency p50" value={q.latency_p50_ms != null ? `${q.latency_p50_ms} ms` : "-"} />
        <Tile label="Latency p95" value={q.latency_p95_ms != null ? `${q.latency_p95_ms} ms` : "-"} />
        <Tile label="Helpful / not" value={`${fb.up} / ${fb.down}`} />
        <Tile label="Est. tokens in/out" value={`${((q.est_tokens_in ?? 0) / 1000).toFixed(1)}k / ${((q.est_tokens_out ?? 0) / 1000).toFixed(1)}k`} hint="characters / 4" />
      </div>
      <section>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Documents by stage</h3>
        <Bars data={s.documents_by_stage} />
      </section>
      {q.by_status && (
        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Answer outcomes</h3>
          <Bars data={q.by_status} />
        </section>
      )}
      {Object.keys(fb.down_by_answer_status).length > 0 && (
        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Thumbs-down by answer status</h3>
          <Bars data={fb.down_by_answer_status} />
        </section>
      )}
    </div>
  );
}
