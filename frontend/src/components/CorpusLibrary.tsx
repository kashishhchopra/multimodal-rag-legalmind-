import { useEffect, useRef, useState } from "react";
import { BookOpenText, Gavel, Loader2, Search } from "lucide-react";
import { api } from "../services/api";
import type { CorpusEntry, CorpusPage } from "../types";

const PAGE = 25;

export default function CorpusLibrary({ onOpen }: { onOpen: (id: string) => void }) {
  const [q, setQ] = useState("");
  const [kind, setKind] = useState("");
  const [year, setYear] = useState("");
  const [data, setData] = useState<CorpusPage | null>(null);
  const [items, setItems] = useState<CorpusEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  async function load(reset: boolean) {
    setBusy(true);
    setErr(null);
    try {
      const r = await api.corpusDocuments({ q, kind, year, offset: reset ? 0 : items.length, limit: PAGE });
      setData(r);
      setItems(reset ? r.items : [...items, ...r.items]);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Could not load the corpus");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => void load(true), 250); // debounce typing
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, kind, year]);

  const totals = data?.totals ?? {};
  return (
    <div className="mx-auto h-full max-w-[1100px] overflow-y-auto px-8 pb-10 pt-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="label">Corpus library</div>
          <h2 className="mt-1 text-[26px] font-semibold tracking-tight text-slate-900">
            {(totals.judgment ?? 0).toLocaleString()} judgments · {(totals.statute ?? 0).toLocaleString()} statutes
          </h2>
          <p className="text-sm text-slate-500">Open any document to ask questions, see clauses, a summary or a timeline. Corpus documents are read-only.</p>
        </div>
      </div>

      <div className="card mt-4 flex flex-wrap items-center gap-3 p-4">
        <Search className="h-4 w-4 text-slate-400" />
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search by party, citation (e.g. 2024 INSC 735) or act…" className="min-w-[240px] flex-1 bg-transparent text-sm outline-none" />
        <select value={kind} onChange={(e) => setKind(e.target.value)} className="rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm">
          <option value="">All types</option>
          <option value="judgment">Judgments</option>
          <option value="statute">Statutes</option>
        </select>
        <select value={year} onChange={(e) => setYear(e.target.value)} className="rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm">
          <option value="">All years</option>
          {Object.entries(data?.years ?? {}).map(([y, n]) => (
            <option key={y} value={y}>{y} ({n})</option>
          ))}
        </select>
      </div>

      {err && <div className="mt-3 rounded-xl bg-red-50 px-4 py-2 text-sm text-red-700">{err}</div>}
      <div className="mt-4 space-y-2">
        {items.map((e) => (
          <button key={e.document_id} onClick={() => onOpen(e.document_id)} className="card flex w-full items-center gap-4 px-5 py-3.5 text-left transition hover:border-slate-400">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-zinc-100 text-zinc-700">
              {e.kind === "statute" ? <BookOpenText className="h-5 w-5" /> : <Gavel className="h-5 w-5" />}
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate text-[15px] font-medium text-slate-900">{e.title}</span>
              <span className="block truncate text-xs text-slate-500">
                {e.kind === "judgment" ? `${e.citation || "no citation"} · ${e.date || e.year} · ${e.court || "court n/a"}` : `${e.citation} · ${e.year}`}
              </span>
            </span>
            <span className="shrink-0 text-right font-mono text-[11px] text-slate-400">
              {e.pages}p<br />{e.chunks} chunks
            </span>
          </button>
        ))}
        {!busy && items.length === 0 && !err && <p className="py-8 text-center text-sm text-slate-500">No documents match.</p>}
      </div>
      {data && items.length < data.total && (
        <div className="mt-4 flex items-center justify-center gap-3 text-sm text-slate-500">
          <span>Showing {items.length.toLocaleString()} of {data.total.toLocaleString()}</span>
          <button onClick={() => load(false)} disabled={busy} className="rounded-xl bg-navy-900 px-4 py-2 font-semibold text-white disabled:opacity-50">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : "Load more"}
          </button>
        </div>
      )}
    </div>
  );
}
