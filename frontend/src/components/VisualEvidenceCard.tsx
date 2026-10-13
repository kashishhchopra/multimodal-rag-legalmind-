import { useState } from "react";
import { Crop, Eye, Maximize2, MapPin } from "lucide-react";
import type { RetrievedVisual } from "../types";
import { api } from "../services/api";

const TYPE_EMOJI: Record<string, string> = {
  schematic: "🔌",
  diagram: "📐",
  chart: "📊",
  table: "📋",
  figure: "🖼️",
  image: "🖼️",
};

export default function VisualEvidenceCard({
  documentId,
  visual,
  onZoom,
}: {
  documentId: string;
  visual: RetrievedVisual;
  onZoom: (url: string, caption: string) => void;
}) {
  const [grounded, setGrounded] = useState(false);
  const label =
    visual.figure_label || `${visual.image_type[0].toUpperCase()}${visual.image_type.slice(1)}`;
  const src = grounded
    ? api.groundingUrl(documentId, visual.visual_id, "highlight")
    : visual.image_url;

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
      <div className="flex items-center justify-between border-b border-slate-100 px-3 py-2">
        <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
          <span>{TYPE_EMOJI[visual.image_type] || "🖼️"}</span>
          <span>{label}</span>
          <span className="text-slate-400">· Page {visual.page_number}</span>
        </div>
        <span className="rounded bg-brand-50 px-1.5 py-0.5 text-[10px] font-semibold text-brand-600">
          {(visual.score * 100).toFixed(0)}% match
        </span>
      </div>

      <div className="relative bg-zinc-50">
        <img
          src={src}
          alt={label}
          className="max-h-64 w-full object-contain"
          loading="lazy"
        />
        <button
          onClick={() => onZoom(src, `${label} — Page ${visual.page_number}`)}
          className="absolute right-2 top-2 rounded-md bg-white/90 p-1.5 text-slate-600 shadow-sm hover:bg-white"
          title="Enlarge"
        >
          <Maximize2 className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="flex items-center justify-between px-3 py-2">
        <button
          onClick={() => setGrounded((g) => !g)}
          className={`flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium transition ${
            grounded ? "bg-orange-100 text-orange-700" : "bg-slate-100 text-slate-600 hover:bg-slate-200"
          }`}
        >
          {grounded ? <Eye className="h-3.5 w-3.5" /> : <MapPin className="h-3.5 w-3.5" />}
          {grounded ? "Showing region" : "Highlight region"}
        </button>
        <a
          href={api.groundingUrl(documentId, visual.visual_id, "crop")}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-brand-600"
          title="Open cropped region"
        >
          <Crop className="h-3.5 w-3.5" /> Crop
        </a>
      </div>

      {visual.ocr_preview && (
        <div className="border-t border-slate-100 px-3 py-1.5 text-[11px] italic text-slate-400">
          OCR: {visual.ocr_preview}
        </div>
      )}
    </div>
  );
}
