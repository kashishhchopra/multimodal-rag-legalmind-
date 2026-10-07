import { Check, Loader2, Circle } from "lucide-react";
import type { DocumentInfo, ProcessingStage } from "../types";

// Ordered pipeline stages shown to the user. These map to REAL backend stages
// reported via /status — the bar reflects actual work, not a timer.
const STEPS: { stage: ProcessingStage; label: string }[] = [
  { stage: "reading", label: "Reading document" },
  { stage: "extracting_text", label: "Extracting text" },
  { stage: "detecting_figures", label: "Detecting figures" },
  { stage: "embedding", label: "Creating embeddings" },
  { stage: "indexing", label: "Indexing document" },
  { stage: "ready", label: "Ready" },
];

const ORDER: ProcessingStage[] = [
  "queued",
  "reading",
  "extracting_text",
  "detecting_figures",
  "extracting_images",
  "ocr",
  "embedding",
  "indexing",
  "ready",
];

function rank(stage: ProcessingStage): number {
  const i = ORDER.indexOf(stage);
  return i < 0 ? 0 : i;
}

export default function ProcessingSteps({ doc }: { doc: DocumentInfo }) {
  const current = rank(doc.stage);

  if (doc.stage === "failed") {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        <div className="font-semibold">Processing failed</div>
        <div className="mt-1">{doc.error || "An unexpected error occurred."}</div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-700">Processing “{doc.filename}”</h3>
        <span className="text-xs text-slate-400">{Math.round(doc.progress * 100)}%</span>
      </div>

      <div className="mb-4 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full bg-brand-500 transition-all duration-500"
          style={{ width: `${Math.max(5, doc.progress * 100)}%` }}
        />
      </div>

      <ul className="space-y-2.5">
        {STEPS.map((step) => {
          const stepRank = rank(step.stage);
          const done = current > stepRank || doc.stage === "ready";
          const active = doc.stage === step.stage || (current === stepRank && doc.stage !== "ready");
          return (
            <li key={step.stage} className="flex items-center gap-3 text-sm">
              {done ? (
                <Check className="h-4 w-4 shrink-0 text-emerald-500" />
              ) : active ? (
                <Loader2 className="h-4 w-4 shrink-0 animate-spin text-brand-500" />
              ) : (
                <Circle className="h-4 w-4 shrink-0 text-slate-300" />
              )}
              <span className={done ? "text-slate-500" : active ? "font-medium text-slate-800" : "text-slate-400"}>
                {step.label}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
