import { useState } from "react";
import { FileText, Image as ImageIcon, MapPin, Sparkles, AlertTriangle, ThumbsUp, ThumbsDown, Download, ShieldCheck } from "lucide-react";
import { api } from "../services/api";
import type { Citation, QueryResponse } from "../types";
import VisualEvidenceCard from "./VisualEvidenceCard";

const WARNINGS: Record<string, string> = {
  missing_api_key:
    "No LLM API key is configured, so this shows the retrieved evidence only. Set LLM_API_KEY in backend/.env for a full grounded answer.",
  llm_failed: "The LLM provider failed, so this shows the retrieved evidence only.",
  llm_quota:
    "The LLM's quota or rate limit is used up (free tiers allow only a few requests per day or minute). This shows the retrieved evidence only. Wait and retry, add credits, or switch LLM_MODEL / LLM_PROVIDER in backend/.env.",
  no_evidence: "No relevant evidence was found in this document for that question.",
  low_retrieval_confidence:
    "The best match in the document was too weak to answer reliably, so no answer was generated. Try rephrasing or naming a clause, section or party.",
  unsupported_claims:
    "Some statements in this answer are not clearly supported by the retrieved text. Check the flagged items below against the cited pages.",
};

function Verification({ r }: { r: QueryResponse }) {
  const v = r.verification;
  if (!v || v.status === "no_claims") return null;
  const ok = v.status === "ok";
  return (
    <div className={`mt-3 rounded-lg px-3 py-2 text-xs ${ok ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
      <div className="flex items-center gap-1.5 font-medium">
        <ShieldCheck className="h-3.5 w-3.5" />
        {ok ? "All statements matched the retrieved evidence" : `${v.unsupported.length} statement(s) not clearly supported`}
        <span className="ml-auto font-normal opacity-70">lexical check, not a legal opinion</span>
      </div>
      {!ok && (
        <ul className="mt-1.5 list-disc space-y-1 pl-5">
          {v.claims
            .filter((c) => !c.supported)
            .map((c, i) => (
              <li key={i}>
                {c.claim}
                {c.missing_numbers.length > 0 && <span> — not found in evidence: {c.missing_numbers.join(", ")}</span>}
                {c.bad_page_refs.length > 0 && <span> — page {c.bad_page_refs.join(", ")} was not retrieved</span>}
              </li>
            ))}
        </ul>
      )}
    </div>
  );
}

function Actions({ r, documentId }: { r: QueryResponse; documentId: string }) {
  const [sent, setSent] = useState<1 | -1 | null>(null);
  const [err, setErr] = useState<string | null>(null);
  async function rate(rating: 1 | -1) {
    setSent(rating);
    try {
      await api.feedback({
        document_id: documentId,
        question: r.question,
        answer: r.answer,
        rating,
        status: r.warning || r.verification?.status || "",
      });
    } catch {
      setSent(null);
      setErr("Could not save feedback");
    }
  }
  return (
    <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-slate-100 pt-3 text-xs text-slate-500">
      <button onClick={() => rate(1)} disabled={sent !== null} aria-label="Helpful" className={`rounded-md p-1.5 hover:bg-slate-100 ${sent === 1 ? "text-emerald-600" : ""}`}>
        <ThumbsUp className="h-4 w-4" />
      </button>
      <button onClick={() => rate(-1)} disabled={sent !== null} aria-label="Not helpful" className={`rounded-md p-1.5 hover:bg-slate-100 ${sent === -1 ? "text-red-600" : ""}`}>
        <ThumbsDown className="h-4 w-4" />
      </button>
      {sent && <span>Thanks, saved.</span>}
      {err && <span className="text-red-600">{err}</span>}
      <span className="ml-auto flex items-center gap-1">
        <Download className="h-3.5 w-3.5" /> Memo:
        {(["pdf", "docx", "md"] as const).map((f) => (
          <button key={f} onClick={() => api.downloadMemo(f, r).catch((e) => setErr(e.message))} className="rounded-md px-1.5 py-0.5 uppercase hover:bg-slate-100">
            {f}
          </button>
        ))}
      </span>
    </div>
  );
}

function CitationChip({ c }: { c: Citation }) {
  const icon =
    c.kind === "text" ? (
      <FileText className="h-3.5 w-3.5" />
    ) : c.kind === "visual" ? (
      <ImageIcon className="h-3.5 w-3.5" />
    ) : (
      <MapPin className="h-3.5 w-3.5" />
    );
  const emoji = c.kind === "text" ? "📄" : c.kind === "visual" ? "🖼️" : "🔍";
  return (
    <span
      title={c.snippet}
      className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-600"
    >
      <span aria-hidden>{emoji}</span>
      <span className="sr-only">{icon}</span>
      {c.label}
    </span>
  );
}

export default function AnswerCard({
  documentId,
  response,
  onZoom,
}: {
  documentId: string;
  response: QueryResponse;
  onZoom: (url: string, caption: string) => void;
}) {
  const confidencePct = Math.round(response.combined_confidence * 100);

  return (
    <div className="animate-fade-in space-y-4">
      {/* Answer */}
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-brand-700">
          <Sparkles className="h-4 w-4" /> Answer
          {response.combined_confidence > 0 && (
            <span className="ml-auto text-xs font-normal text-slate-400">
              retrieval confidence {confidencePct}%
            </span>
          )}
        </div>
        <div className="whitespace-pre-wrap text-[15px] leading-relaxed text-slate-800">
          {response.answer}
        </div>
        {response.context_mode && response.context_chunks ? (
          <div className="mt-2 font-mono text-[11px] uppercase tracking-wider text-slate-400">
            {response.context_mode === "full_document"
              ? `Answered from the whole document · ${response.context_chunks} passages read`
              : `Answered from the ${response.context_chunks} most relevant passages`}
          </div>
        ) : null}
        {response.warning && WARNINGS[response.warning] && (
          <div className="mt-3 flex items-start gap-2 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            {WARNINGS[response.warning]}
          </div>
        )}
        <Verification r={response} />
        {response.answer && <Actions r={response} documentId={documentId} />}
      </div>

      {/* Visual evidence */}
      {response.visual_evidence.length > 0 && (
        <div>
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            <ImageIcon className="h-3.5 w-3.5" /> Visual evidence
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {response.visual_evidence.map((v) => (
              <VisualEvidenceCard
                key={v.visual_id}
                documentId={documentId}
                visual={v}
                onZoom={onZoom}
              />
            ))}
          </div>
        </div>
      )}

      {/* Text evidence */}
      {response.text_evidence.length > 0 && (
        <div>
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            <FileText className="h-3.5 w-3.5" /> Text evidence
          </div>
          <div className="space-y-2">
            {response.text_evidence.map((t) => (
              <div
                key={t.chunk_id}
                className="rounded-lg border border-slate-200 bg-white p-3 text-sm text-slate-700"
              >
                <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
                  <span>📄 Page {t.page_number}</span>
                  <span>{(t.score * 100).toFixed(0)}% match</span>
                </div>
                <p className="line-clamp-4">{t.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Citations */}
      {response.citations.length > 0 && (
        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Sources
          </div>
          <div className="flex flex-wrap gap-2">
            {response.citations.map((c) => (
              <CitationChip key={`${c.kind}-${c.source_id}`} c={c} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
