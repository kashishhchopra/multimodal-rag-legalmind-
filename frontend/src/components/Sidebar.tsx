import { useState } from "react";
import {
  Activity,
  BookOpen,
  ChevronDown,
  CheckCircle2,
  Command,
  FileStack,
  GitCompareArrows,
  LayoutGrid,
  Loader2,
  PanelLeftClose,
  ScanSearch,
  Settings,
  ShieldCheck,
  Sparkles,
  AlertTriangle,
  Plus,
} from "lucide-react";
import type { DocumentInfo } from "../types";

export type Page = "overview" | "documents" | "insights" | "compare" | "search" | "safety" | "admin";

const NAV: { id: Page; label: string; icon: React.ReactNode; group: "workspace" | "system" }[] = [
  { id: "overview", label: "Overview", icon: <LayoutGrid className="h-[18px] w-[18px]" />, group: "workspace" },
  { id: "documents", label: "Documents", icon: <FileStack className="h-[18px] w-[18px]" />, group: "workspace" },
  { id: "insights", label: "Insights", icon: <Sparkles className="h-[18px] w-[18px]" />, group: "workspace" },
  { id: "compare", label: "Compare", icon: <GitCompareArrows className="h-[18px] w-[18px]" />, group: "workspace" },
  { id: "search", label: "Corpus search", icon: <ScanSearch className="h-[18px] w-[18px]" />, group: "workspace" },
  { id: "safety", label: "Safety", icon: <ShieldCheck className="h-[18px] w-[18px]" />, group: "system" },
  { id: "admin", label: "Admin", icon: <Activity className="h-[18px] w-[18px]" />, group: "system" },
];

export type Health = { llm_provider: string; llm_key_configured: boolean; ocr_available: boolean };

export default function Sidebar({
  page,
  onNavigate,
  documents,
  activeDoc,
  onSelectDoc,
  onUpload,
  health,
  chunks,
  username,
  onShowArchitecture,
  onShowSettings,
  onCollapse,
}: {
  page: Page;
  onNavigate: (p: Page) => void;
  documents: DocumentInfo[];
  activeDoc: DocumentInfo | null;
  onSelectDoc: (id: string) => void;
  onUpload: () => void;
  health: Health | null;
  chunks: number | null;
  username: string;
  onShowArchitecture: () => void;
  onShowSettings: () => void;
  onCollapse: () => void;
}) {
  const [open, setOpen] = useState(false);
  const llmOk = !!health && health.llm_key_configured && !health.llm_provider.startsWith("invalid");
  const initials = username.slice(0, 2).toUpperCase();

  return (
    <aside className="flex h-full w-[272px] shrink-0 flex-col bg-gradient-to-b from-navy-900 via-navy-900 to-navy-950 text-slate-200">
      <div className="flex items-center gap-3 px-6 pb-4 pt-6">
        <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-zinc-600 to-zinc-800 text-white shadow-lg shadow-black/30">
          <Command className="h-6 w-6" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-[17px] font-semibold leading-tight text-white">LegalMind</div>
          <div className="label !text-[10px] !text-slate-400">Research OS</div>
        </div>
        <button onClick={onCollapse} className="rounded-lg p-1.5 text-slate-400 hover:bg-white/10 hover:text-white" aria-label="Collapse sidebar">
          <PanelLeftClose className="h-5 w-5" />
        </button>
      </div>

      {/* Workspace = the active document */}
      <div className="relative px-4">
        <button
          onClick={() => setOpen((o) => !o)}
          className="flex w-full items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-3.5 py-3 text-left transition hover:bg-white/10"
        >
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white/10 text-xs font-semibold text-zinc-300">
            {activeDoc ? "DOC" : "NEW"}
          </div>
          <div className="min-w-0 flex-1">
            <div className="label !text-[10px] !text-slate-400">Active document</div>
            <div className="truncate text-sm font-medium text-white">{activeDoc ? activeDoc.filename : "None selected"}</div>
          </div>
          <ChevronDown className={`h-4 w-4 text-slate-400 transition ${open ? "rotate-180" : ""}`} />
        </button>
        {open && (
          <div className="absolute left-4 right-4 top-[calc(100%+6px)] z-20 max-h-72 overflow-y-auto rounded-2xl border border-white/10 bg-navy-800 p-1.5 shadow-2xl">
            {documents.length === 0 && <div className="px-3 py-2 text-xs text-slate-400">No documents yet.</div>}
            {documents.map((d) => (
              <button
                key={d.document_id}
                onClick={() => {
                  onSelectDoc(d.document_id);
                  setOpen(false);
                }}
                className={`flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-sm hover:bg-white/10 ${activeDoc?.document_id === d.document_id ? "bg-white/10 text-white" : "text-slate-300"}`}
              >
                <span className="min-w-0 flex-1 truncate">{d.filename}</span>
                {d.stage === "ready" ? <CheckCircle2 className="h-4 w-4 text-emerald-400" /> : d.stage === "failed" ? <AlertTriangle className="h-4 w-4 text-red-400" /> : <Loader2 className="h-4 w-4 animate-spin text-zinc-300" />}
              </button>
            ))}
            <button
              onClick={() => {
                setOpen(false);
                onUpload();
              }}
              className="mt-1 flex w-full items-center gap-2 rounded-xl border border-dashed border-white/15 px-3 py-2 text-sm text-slate-300 hover:bg-white/10"
            >
              <Plus className="h-4 w-4" /> Upload a PDF
            </button>
          </div>
        )}
      </div>

      <nav className="mt-6 flex-1 overflow-y-auto px-4">
        {(["workspace", "system"] as const).map((g) => (
          <div key={g} className="mb-5">
            <div className="label !text-[10px] !text-slate-500 px-2 pb-2">{g === "workspace" ? "Workspace" : "System"}</div>
            <div className="space-y-1">
              {NAV.filter((n) => n.group === g).map((n) => {
                const active = page === n.id;
                return (
                  <button
                    key={n.id}
                    onClick={() => onNavigate(n.id)}
                    className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-[15px] transition ${
                      active ? "bg-white/10 font-medium text-white ring-1 ring-white/20" : "text-slate-300 hover:bg-white/5 hover:text-white"
                    }`}
                  >
                    <span className={active ? "text-zinc-300" : "text-slate-400"}>{n.icon}</span>
                    {n.label}
                    {n.id === "documents" && (
                      <span className="ml-auto rounded-md bg-white/10 px-1.5 py-0.5 text-[11px] text-slate-300" title="documents open in this session; the corpus library holds many more">{documents.length} open</span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}

        <div className="relative mt-2 overflow-hidden rounded-2xl border border-white/10 bg-white/5 p-4">
          <div className="absolute -right-10 -top-10 h-24 w-24 rounded-full border border-white/10" />
          <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-xl bg-white/10 text-zinc-300">
            <Sparkles className="h-4 w-4" />
          </div>
          <div className="text-sm font-semibold text-white">Index status</div>
          <div className="mt-1 text-xs text-slate-400">{chunks != null ? `${chunks.toLocaleString()} text chunks indexed` : "Index size unavailable"}</div>
          <div className={`mt-2 inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] ${llmOk ? "bg-emerald-500/15 text-emerald-300" : "bg-amber-500/15 text-amber-300"}`}>
            <span className={`h-1.5 w-1.5 rounded-full ${llmOk ? "bg-emerald-400" : "bg-amber-400"}`} />
            {llmOk ? "LLM connected" : "LLM not configured"}
          </div>
        </div>
      </nav>

      <div className="space-y-1 px-4 pb-3">
        <button onClick={onShowArchitecture} className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-[15px] text-slate-300 hover:bg-white/5">
          <BookOpen className="h-[18px] w-[18px] text-slate-400" /> Documentation
        </button>
        <button onClick={onShowSettings} className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-[15px] text-slate-300 hover:bg-white/5">
          <Settings className="h-[18px] w-[18px] text-slate-400" /> Settings
        </button>
      </div>

      <div className="flex items-center gap-3 border-t border-white/10 px-6 py-4">
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white/15 text-sm font-semibold text-white">{initials}</div>
        <div className="min-w-0">
          <div className="truncate text-sm font-medium text-white">{username}</div>
          <div className="text-xs text-slate-400">Researcher</div>
        </div>
      </div>
    </aside>
  );
}
