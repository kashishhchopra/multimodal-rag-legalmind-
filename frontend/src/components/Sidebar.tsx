import { FilePlus2, FileText, Trash2, Network, CheckCircle2, Loader2, AlertTriangle } from "lucide-react";
import type { DocumentInfo } from "../types";

function StageBadge({ doc }: { doc: DocumentInfo }) {
  if (doc.stage === "ready")
    return <CheckCircle2 className="h-4 w-4 text-emerald-500" aria-label="Ready" />;
  if (doc.stage === "failed")
    return <AlertTriangle className="h-4 w-4 text-red-500" aria-label="Failed" />;
  return <Loader2 className="h-4 w-4 animate-spin text-brand-400" aria-label="Processing" />;
}

export default function Sidebar({
  documents,
  activeId,
  onSelect,
  onNew,
  onDelete,
  onShowArchitecture,
  health,
}: {
  documents: DocumentInfo[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onShowArchitecture: () => void;
  health: { llm_provider: string; llm_key_configured: boolean; ocr_available: boolean } | null;
}) {
  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-slate-800 bg-slate-900 text-slate-100">
      <div className="flex items-center gap-2.5 px-5 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600">
          <Network className="h-5 w-5" />
        </div>
        <div>
          <div className="text-sm font-semibold leading-tight">LegalMind</div>
          <div className="text-[11px] text-slate-400">Multimodal Legal Document AI</div>
        </div>
      </div>

      <div className="px-3">
        <button
          onClick={onNew}
          className="flex w-full items-center gap-2 rounded-lg bg-brand-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-brand-700"
        >
          <FilePlus2 className="h-4 w-4" /> Upload Legal Document
        </button>
      </div>

      <div className="mt-5 px-5 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
        Uploaded Documents
      </div>
      <nav className="mt-2 flex-1 space-y-1 overflow-y-auto px-3 pb-3">
        {documents.length === 0 && (
          <div className="px-2 py-3 text-xs text-slate-500">No documents yet.</div>
        )}
        {documents.map((doc) => (
          <div
            key={doc.document_id}
            className={`group flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm transition ${
              activeId === doc.document_id ? "bg-slate-800" : "hover:bg-slate-800/60"
            }`}
          >
            <button
              onClick={() => onSelect(doc.document_id)}
              className="flex min-w-0 flex-1 items-center gap-2 text-left"
            >
              <FileText className="h-4 w-4 shrink-0 text-slate-400" />
              <span className="truncate text-slate-200">{doc.filename}</span>
            </button>
            <StageBadge doc={doc} />
            <button
              onClick={() => onDelete(doc.document_id)}
              className="opacity-0 transition group-hover:opacity-100"
              title="Delete document"
            >
              <Trash2 className="h-4 w-4 text-slate-500 hover:text-red-400" />
            </button>
          </div>
        ))}
      </nav>

      <div className="border-t border-slate-800 px-4 py-3">
        <button
          onClick={onShowArchitecture}
          className="mb-3 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-xs text-slate-300 hover:bg-slate-800"
        >
          <Network className="h-3.5 w-3.5" /> How it works (architecture)
        </button>
        {health && (
          <div className="space-y-1 text-[11px] text-slate-400">
            <div className="flex items-center justify-between">
              <span>LLM provider</span>
              <span className="font-medium text-slate-300">{health.llm_provider}</span>
            </div>
            <div className="flex items-center justify-between">
              <span>API key</span>
              <span className={health.llm_key_configured ? "text-emerald-400" : "text-amber-400"}>
                {health.llm_key_configured ? "configured" : "not set"}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span>OCR</span>
              <span className={health.ocr_available ? "text-emerald-400" : "text-amber-400"}>
                {health.ocr_available ? "available" : "unavailable"}
              </span>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
