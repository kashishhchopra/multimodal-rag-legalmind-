import { ArrowRight, CheckCircle2, Clock, Database, Layers, MessageSquareText, ThumbsUp, FileStack } from "lucide-react";
import type { AdminStats, DocumentInfo } from "../types";
import type { Health, Page } from "./Sidebar";

function greeting(): string {
  const h = new Date().getHours();
  return h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
}

function Stat({ icon, tint, label, value, hint }: { icon: React.ReactNode; tint: string; label: string; value: string; hint: string }) {
  return (
    <div className="card flex items-center gap-4 p-6">
      <div className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl ${tint}`}>{icon}</div>
      <div className="min-w-0">
        <div className="text-[13px] text-slate-500">{label}</div>
        <div className="text-[34px] font-semibold leading-none tracking-tight text-slate-900">{value}</div>
        <div className="mt-1.5 truncate text-xs text-slate-400">{hint}</div>
      </div>
    </div>
  );
}

type Cap = { title: string; desc: string; status: "READY" | "NEEDS KEY" | "DEMO INDEX"; page: Page; dot: string };

export default function OverviewPage({
  documents,
  activeDoc,
  stats,
  health,
  username,
  onNavigate,
  onUpload,
}: {
  documents: DocumentInfo[];
  activeDoc: DocumentInfo | null;
  stats: AdminStats | null;
  health: Health | null;
  username: string;
  onNavigate: (p: Page) => void;
  onUpload: () => void;
}) {
  const corpus = stats?.corpus ?? {};
  const judgments = corpus["judgment"]?.documents ?? 0;
  const statutes = corpus["statute"]?.chunks ?? 0;
  const ready = documents.filter((d) => d.stage === "ready").length;
  const llmOk = !!health && health.llm_key_configured && !health.llm_provider.startsWith("invalid");
  const q = stats?.queries.queries ?? 0;
  const fb = stats?.feedback;
  const rated = fb ? fb.up + fb.down : 0;
  const helpful = rated > 0 && fb ? Math.round((fb.up / rated) * 100) : null;

  const caps: Cap[] = [
    { title: "Hybrid retrieval", desc: "BM25 + dense search with reciprocal rank fusion and metadata filters.", status: "READY", page: "search", dot: "bg-zinc-900 ring-zinc-200" },
    { title: "Clause and risk analysis", desc: "CUAD clause types with a draft risk rubric and suggested edits.", status: "READY", page: "insights", dot: "bg-zinc-700 ring-zinc-200" },
    { title: "Grounded answers", desc: "Citations, claim-support check and refusal when evidence is weak.", status: llmOk ? "READY" : "NEEDS KEY", page: "documents", dot: "bg-zinc-500 ring-zinc-200" },
    { title: "Statute linking", desc: "IPC, CrPC and Evidence Act sections mapped to BNS, BNSS and BSA.", status: "READY", page: "search", dot: "bg-zinc-400 ring-zinc-200" },
    { title: "Real judgment corpus", desc: judgments ? `${judgments.toLocaleString()} Supreme Court judgments (2019-2024) and ${statutes.toLocaleString()} statute chunks are searchable.` : "Only the demo documents are indexed; the judgment corpus is not loaded in this index.", status: judgments ? "READY" : "DEMO INDEX", page: "search", dot: "bg-zinc-300 ring-zinc-200" },
  ];
  const chip = (s: Cap["status"]) =>
    s === "READY" ? "bg-emerald-50 text-emerald-700" : s === "NEEDS KEY" ? "bg-amber-50 text-amber-700" : "bg-brand-50 text-brand-700";

  const tasks = [
    { done: llmOk, task: "Set a valid LLM provider and API key in backend/.env", stream: "Security", pri: "P0" },
    { done: documents.length > 0, task: "Upload a legal document", stream: "Documents", pri: "P0" },
    { done: q > 0, task: "Ask a first grounded question", stream: "Retrieval", pri: "P0" },
    { done: rated > 0, task: "Rate an answer to start collecting feedback", stream: "Evaluation", pri: "P1" },
    { done: false, task: "Have a lawyer review the clause risk rubric", stream: "Safety", pri: "P1" },
  ];
  const doneCount = tasks.filter((t) => t.done).length;

  return (
    <div className="mx-auto h-full max-w-[1400px] space-y-6 overflow-y-auto px-8 pb-10 pt-8">
      <div>
        <div className="label flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-500" /> Project control center
        </div>
        <h1 className="mt-3 text-[44px] font-semibold leading-none tracking-tight text-slate-900">
          {greeting()}
          {username && username !== "Local user" ? `, ${username}.` : "."}
        </h1>
        <p className="mt-3 text-[15px] text-slate-500">
          {ready} of {documents.length} documents ready.{" "}
          {llmOk ? "Grounded answers are on." : <span>Answers are <b className="text-slate-700">evidence-only</b> until an LLM key is configured.</span>}
        </p>
      </div>

      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-4">
        <Stat icon={<FileStack className="h-6 w-6 text-zinc-700" />} tint="bg-zinc-100" label="Documents available" value={(Object.values(corpus).reduce((a, c) => a + c.documents, 0) || documents.length).toLocaleString()} hint={judgments ? `${judgments.toLocaleString()} judgments · ${(corpus["statute"]?.documents ?? 0)} statutes · ${(corpus["document"]?.documents ?? documents.length)} uploaded` : `${ready} ready for questions`} />
        <Stat icon={<Database className="h-6 w-6 text-zinc-700" />} tint="bg-zinc-100" label="Indexed chunks" value={stats?.indexed_text_chunks != null ? stats.indexed_text_chunks.toLocaleString() : "n/a"} hint={judgments ? `${judgments.toLocaleString()} judgments · ${statutes.toLocaleString()} statute chunks` : "Demo vector index"} />
        <Stat icon={<MessageSquareText className="h-6 w-6 text-zinc-700" />} tint="bg-zinc-100" label="Questions answered" value={String(q)} hint={stats?.queries.latency_p50_ms != null ? `median ${Math.round(stats.queries.latency_p50_ms)} ms` : "no queries yet"} />
        <Stat icon={<ThumbsUp className="h-6 w-6 text-zinc-700" />} tint="bg-zinc-100" label="Helpful answers" value={helpful != null ? `${helpful}%` : "n/a"} hint={rated ? `${rated} rated` : "no ratings yet"} />
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1fr_380px]">
        <section className="card p-7">
          <div className="flex items-start justify-between">
            <div>
              <div className="label">Platform</div>
              <h2 className="mt-1 text-[22px] font-semibold tracking-tight text-slate-900">What is available</h2>
            </div>
            <button onClick={() => onNavigate("admin")} className="flex items-center gap-1.5 text-sm font-medium text-brand-600 hover:text-brand-700">
              View system stats <ArrowRight className="h-4 w-4" />
            </button>
          </div>
          <ol className="relative mt-6 space-y-5 before:absolute before:bottom-3 before:left-[11px] before:top-3 before:w-px before:bg-slate-200">
            {caps.map((c) => (
              <li key={c.title} className="relative flex items-center gap-5">
                <span className={`relative z-10 h-6 w-6 shrink-0 rounded-full ring-4 ${c.dot}`} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[16px] font-medium text-slate-900">{c.title}</span>
                    <span className={`rounded-md px-2 py-0.5 font-mono text-[10px] font-medium tracking-wider ${chip(c.status)}`}>{c.status}</span>
                  </div>
                  <div className="mt-0.5 text-sm text-slate-500">{c.desc}</div>
                </div>
                <button onClick={() => onNavigate(c.page)} className="rounded-xl border border-slate-200 p-2.5 text-slate-500 transition hover:border-brand-300 hover:text-brand-600" aria-label={`Open ${c.title}`}>
                  <ArrowRight className="h-4 w-4" />
                </button>
              </li>
            ))}
          </ol>
        </section>

        <section className="card p-7">
          <div className="flex items-start justify-between">
            <div>
              <div className="label">Active document</div>
              <h2 className="mt-1 max-w-[230px] truncate text-[22px] font-semibold tracking-tight text-slate-900">{activeDoc ? activeDoc.filename : "Nothing selected"}</h2>
            </div>
            <div className="flex h-14 w-14 items-center justify-center rounded-full border border-zinc-200 bg-zinc-100 font-mono text-xs font-medium text-zinc-700">
              {activeDoc ? `${activeDoc.page_count}p` : "—"}
            </div>
          </div>
          {activeDoc ? (
            <>
              <div className="mt-5 rounded-2xl bg-zinc-50 p-4">
                <div className="flex items-center justify-between text-sm text-slate-500">
                  <span>Processing</span>
                  <span className="font-semibold text-slate-800">{activeDoc.stage === "ready" ? "Ready" : `${Math.round(activeDoc.progress * 100)}%`}</span>
                </div>
                <div className="mt-2 h-2 rounded-full bg-slate-200">
                  <div className="h-2 rounded-full bg-brand-500 transition-all" style={{ width: `${activeDoc.stage === "ready" ? 100 : Math.round(activeDoc.progress * 100)}%` }} />
                </div>
              </div>
              <dl className="mt-5 grid grid-cols-3 gap-3 text-center">
                {[
                  ["Chunks", activeDoc.text_chunk_count],
                  ["Figures", activeDoc.figure_count],
                  ["Visuals", activeDoc.image_count],
                ].map(([k, v]) => (
                  <div key={k as string} className="rounded-xl border border-slate-100 py-3">
                    <dd className="text-lg font-semibold text-slate-800">{v}</dd>
                    <dt className="text-xs text-slate-400">{k}</dt>
                  </div>
                ))}
              </dl>
              <div className="mt-5 flex gap-2">
                <button onClick={() => onNavigate("documents")} className="flex-1 rounded-xl bg-navy-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-navy-800">Ask questions</button>
                <button onClick={() => onNavigate("insights")} className="flex-1 rounded-xl border border-slate-200 px-4 py-2.5 text-sm font-semibold text-slate-700 hover:border-brand-300">Insights</button>
              </div>
            </>
          ) : (
            <div className="mt-6 text-sm text-slate-500">
              <p>Upload a contract, judgment or statute PDF to begin.</p>
              <button onClick={onUpload} className="mt-4 rounded-xl bg-navy-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-navy-800">Upload PDF</button>
            </div>
          )}
          <div className="mt-6 flex items-center gap-2 border-t border-slate-100 pt-4 text-sm text-slate-500">
            <Clock className="h-4 w-4" /> {documents.length ? `${ready} ready · ${documents.length - ready} processing or failed` : "No documents yet"}
          </div>
        </section>
      </div>

      <section className="card p-7">
        <div className="flex items-end justify-between">
          <div>
            <div className="label">Immediate action</div>
            <h2 className="mt-1 text-[22px] font-semibold tracking-tight text-slate-900">Getting started queue</h2>
          </div>
          <div className="rounded-full bg-slate-100 px-3 py-1.5 text-sm text-slate-600">
            <b className="text-slate-900">{doneCount}</b> / {tasks.length} done
          </div>
        </div>
        <div className="mt-5 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="label !text-[10px] border-b border-slate-100">
                <th className="py-3 font-normal">Task</th>
                <th className="py-3 font-normal">Workstream</th>
                <th className="py-3 font-normal">Priority</th>
                <th className="py-3 font-normal">Status</th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((t) => (
                <tr key={t.task} className="border-b border-slate-50 last:border-0">
                  <td className="py-4">
                    <span className="flex items-center gap-3">
                      {t.done ? <CheckCircle2 className="h-6 w-6 text-emerald-500" /> : <span className="h-6 w-6 rounded-lg border-2 border-slate-200" />}
                      <span className={t.done ? "text-slate-400 line-through" : "font-medium text-slate-800"}>{t.task}</span>
                    </span>
                  </td>
                  <td className="py-4 text-slate-500">{t.stream}</td>
                  <td className="py-4">
                    <span className={`rounded-md px-2 py-0.5 font-mono text-[11px] font-medium ${t.pri === "P0" ? "bg-peach-100 text-zinc-700" : "bg-zinc-100 text-zinc-700"}`}>{t.pri}</span>
                  </td>
                  <td className="py-4 text-slate-500">{t.done ? "Done" : "Open"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <div className="flex items-center justify-between pb-2 font-mono text-[10px] uppercase tracking-[0.14em] text-slate-400">
        <span><Layers className="mr-1 inline h-3 w-3" />LegalMind · research support, not legal advice</span>
        <span>Live data from this backend</span>
      </div>
    </div>
  );
}
