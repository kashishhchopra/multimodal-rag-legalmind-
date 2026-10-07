import { FileText, Image as ImageIcon, MapPin, Sparkles, AlertTriangle } from "lucide-react";
import type { Citation, QueryResponse } from "../types";
import VisualEvidenceCard from "./VisualEvidenceCard";

const WARNINGS: Record<string, string> = {
  missing_api_key:
    "No LLM API key is configured, so this shows the retrieved evidence only. Set LLM_API_KEY in backend/.env for a full grounded answer.",
  llm_failed: "The LLM provider failed, so this shows the retrieved evidence only.",
  no_evidence: "No relevant evidence was found in this document for that question.",
};

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
        {response.warning && WARNINGS[response.warning] && (
          <div className="mt-3 flex items-start gap-2 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            {WARNINGS[response.warning]}
          </div>
        )}
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
