import { useState } from "react";
import { Loader2, Search } from "lucide-react";
import { api } from "../services/api";
import type { QueryResponse } from "../types";

export default function SearchPanel() {
  const [q, setQ] = useState("");
  const [year, setYear] = useState("");
  const [docType, setDocType] = useState("");
  const [section, setSection] = useState("");
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<QueryResponse | null>(null);
  const [mappings, setMappings] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    if (!q.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      const filters: Record<string, string | number> = {};
      if (year) filters.year = Number(year);
      if (docType) filters.doc_type = docType;
      if (section) filters.section = section;
      const [r, m] = await Promise.all([api.search(q, filters), api.sections(q).catch(() => null)]);
      setRes(r);
      setMappings(m?.mappings ?? []);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Search failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex h-full w-full max-w-4xl flex-col gap-4 overflow-y-auto p-6">
      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex gap-2">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
            placeholder='Search all documents, e.g. "anticipatory bail under Section 438 CrPC"'
            className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-brand-500"
          />
          <button
            onClick={run}
            disabled={busy}
            className="flex items-center gap-1.5 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-60"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />} Search
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          <input value={year} onChange={(e) => setYear(e.target.value)} placeholder="Year (e.g. 2023)" className="w-32 rounded-lg border border-slate-300 px-2 py-1.5" />
          <select value={docType} onChange={(e) => setDocType(e.target.value)} className="rounded-lg border border-slate-300 px-2 py-1.5">
            <option value="">Any type</option>
            <option value="judgment">Judgments</option>
            <option value="statute">Statutes</option>
          </select>
          <input value={section} onChange={(e) => setSection(e.target.value)} placeholder="Statute section (e.g. 318)" className="w-44 rounded-lg border border-slate-300 px-2 py-1.5" />
        </div>
      </div>

      {err && <div className="rounded-lg bg-red-50 px-4 py-2 text-sm text-red-700">{err}</div>}
      {mappings.length > 0 && (
        <div className="rounded-lg bg-brand-50 px-4 py-2 text-xs text-brand-700">
          Section mapping applied: {mappings.join(" · ")}
        </div>
      )}
      {res && (
        <>
          <p className="text-xs text-slate-500">
            {res.text_evidence.length} results · hybrid BM25 + dense ·{" "}
            {res.answer ? "answer generated below" : ""}
          </p>
          {res.answer && (
            <div className="whitespace-pre-wrap rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-800">{res.answer}</div>
          )}
          {res.text_evidence.map((t) => (
            <div key={t.chunk_id} className="rounded-lg border border-slate-200 bg-white p-3 text-sm text-slate-700">
              <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-slate-400">
                <span>{t.document_id || "document"} · page {t.page_number}</span>
                <span className="ml-auto">{(t.score * 100).toFixed(0)}% dense</span>
                {t.lexical_rank != null && <span>BM25 #{t.lexical_rank}</span>}
                {t.dense_rank != null && <span>dense #{t.dense_rank}</span>}
              </div>
              <p className="line-clamp-5">{t.text}</p>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
