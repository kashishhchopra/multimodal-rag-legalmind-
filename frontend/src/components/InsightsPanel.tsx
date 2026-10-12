import { useEffect, useState } from "react";
import { AlertTriangle, CalendarClock, FileSearch, Gavel, Loader2, ShieldAlert } from "lucide-react";
import { api } from "../services/api";
import type { ClauseReport, DocumentSummary, TimelineEvent } from "../types";

type Tab = "clauses" | "summary" | "timeline";

const RISK_STYLE: Record<string, string> = {
  high: "bg-red-50 text-red-700 border-red-200",
  medium: "bg-amber-50 text-amber-700 border-amber-200",
  low: "bg-emerald-50 text-emerald-700 border-emerald-200",
};

function useLoad<T>(load: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let live = true;
    setLoading(true);
    setErr(null);
    load()
      .then((d) => live && setData(d))
      .catch((e) => live && setErr(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, err, loading };
}

function Status({ loading, err }: { loading: boolean; err: string | null }) {
  if (loading)
    return (
      <div className="flex items-center gap-2 p-6 text-sm text-slate-500">
        <Loader2 className="h-4 w-4 animate-spin" /> Analysing…
      </div>
    );
  if (err) return <div className="m-4 rounded-lg bg-red-50 px-4 py-2 text-sm text-red-700">{err}</div>;
  return null;
}

function Clauses({ documentId }: { documentId: string }) {
  const { data, err, loading } = useLoad<ClauseReport>(() => api.clauses(documentId), [documentId]);
  if (loading || err) return <Status loading={loading} err={err} />;
  if (!data) return null;
  return (
    <div className="space-y-3 p-4">
      {data.warning && <div className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700">{data.warning}</div>}
      <div className="flex gap-2 text-xs">
        {(["high", "medium", "low"] as const).map((l) => (
          <span key={l} className={`rounded-full border px-2.5 py-1 ${RISK_STYLE[l]}`}>
            {data.summary[l]} {l}
          </span>
        ))}
        <span className="ml-auto text-slate-400">{data.clauses.length} clauses detected</span>
      </div>
      {data.clauses.length === 0 && (
        <p className="text-sm text-slate-500">No tracked clause types were detected in this document.</p>
      )}
      {data.clauses.map((c) => (
        <div key={c.chunk_id} className="rounded-xl border border-slate-200 bg-white p-4 text-sm">
          <div className="mb-1 flex items-center gap-2">
            <span className="font-semibold text-slate-800">{c.category}</span>
            <span className={`rounded-full border px-2 py-0.5 text-xs ${RISK_STYLE[c.risk_level]}`}>
              {c.risk_level} risk
            </span>
            <span className="ml-auto text-xs text-slate-400">page {c.page_number}</span>
          </div>
          <p className="line-clamp-3 text-slate-600">{c.text}</p>
          <p className="mt-2 text-xs text-slate-500">{c.risk_reason}</p>
          {c.suggested_edit && (
            <p className="mt-1 text-xs text-slate-500">
              <span className="font-medium">Suggested edit:</span> {c.suggested_edit}
            </p>
          )}
        </div>
      ))}
      <p className="text-xs text-slate-400">{data.disclaimer} The risk rubric is a draft pending legal review.</p>
    </div>
  );
}

function SentenceList({ title, items }: { title: string; items: { text: string; page: number }[] }) {
  if (!items.length) return null;
  return (
    <div>
      <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</h3>
      <ul className="space-y-1.5">
        {items.map((s, i) => (
          <li key={i} className="rounded-lg border border-slate-200 bg-white p-2.5 text-sm text-slate-700">
            {s.text} <span className="text-xs text-slate-400">(p. {s.page})</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Summary({ documentId }: { documentId: string }) {
  const { data, err, loading } = useLoad<DocumentSummary>(() => api.summary(documentId), [documentId]);
  if (loading || err) return <Status loading={loading} err={err} />;
  if (!data) return null;
  return (
    <div className="space-y-4 p-4">
      <SentenceList title="Issue" items={data.issue} />
      <SentenceList title="Headnotes" items={data.headnotes} />
      <SentenceList title="Facts" items={data.facts} />
      <SentenceList title="Holding / order" items={data.holding} />
      {data.statutes_cited.length > 0 && (
        <div>
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Statutes cited</h3>
          <div className="flex flex-wrap gap-1.5">
            {data.statutes_cited.map((s) => (
              <span key={s} className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-xs text-slate-600">
                {s}
              </span>
            ))}
          </div>
        </div>
      )}
      {data.cases_cited.length > 0 && (
        <div>
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Cases cited</h3>
          <div className="flex flex-wrap gap-1.5">
            {data.cases_cited.map((s) => (
              <span key={s} className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-xs text-slate-600">
                {s}
              </span>
            ))}
          </div>
        </div>
      )}
      <p className="text-xs text-slate-400">{data.note}</p>
    </div>
  );
}

function Timeline({ documentId }: { documentId: string }) {
  const { data, err, loading } = useLoad<{ events: TimelineEvent[] }>(() => api.timeline(documentId), [documentId]);
  if (loading || err) return <Status loading={loading} err={err} />;
  if (!data) return null;
  if (!data.events.length) return <p className="p-4 text-sm text-slate-500">No dated events found.</p>;
  return (
    <ol className="space-y-2 p-4">
      {data.events.map((e, i) => (
        <li key={i} className="flex gap-3 rounded-lg border border-slate-200 bg-white p-3 text-sm">
          <span className="w-24 shrink-0 font-mono text-xs text-brand-700">{e.date}</span>
          <span className="text-slate-700">
            {e.text} <span className="text-xs text-slate-400">(p. {e.page_number})</span>
          </span>
        </li>
      ))}
    </ol>
  );
}

export default function InsightsPanel({ documentId }: { documentId: string }) {
  const [tab, setTab] = useState<Tab>("clauses");
  const tabs: { id: Tab; label: string; icon: JSX.Element }[] = [
    { id: "clauses", label: "Clauses & risk", icon: <ShieldAlert className="h-4 w-4" /> },
    { id: "summary", label: "Summary", icon: <Gavel className="h-4 w-4" /> },
    { id: "timeline", label: "Timeline", icon: <CalendarClock className="h-4 w-4" /> },
  ];
  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex gap-1 border-b border-slate-200 bg-white px-4 pt-2">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 rounded-t-lg px-3 py-2 text-sm ${
              tab === t.id ? "border-b-2 border-brand-600 font-medium text-brand-700" : "text-slate-500 hover:text-slate-700"
            }`}
          >
            {t.icon} {t.label}
          </button>
        ))}
        <span className="ml-auto flex items-center gap-1 pb-2 text-xs text-slate-400">
          <FileSearch className="h-3.5 w-3.5" /> automated, verify against source
        </span>
      </div>
      <div className="flex-1 overflow-y-auto">
        {tab === "clauses" && <Clauses documentId={documentId} />}
        {tab === "summary" && <Summary documentId={documentId} />}
        {tab === "timeline" && <Timeline documentId={documentId} />}
      </div>
      <div className="flex items-center gap-1.5 border-t border-slate-200 bg-white px-4 py-2 text-xs text-slate-400">
        <AlertTriangle className="h-3.5 w-3.5" /> Not legal advice.
      </div>
    </div>
  );
}
