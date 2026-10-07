import { FileText, Layers, Image as ImageIcon, ScanSearch, Crop } from "lucide-react";
import type { DocumentInfo } from "../types";

function Stat({
  icon,
  value,
  label,
}: {
  icon: React.ReactNode;
  value: number;
  label: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-50 text-brand-600">
        {icon}
      </div>
      <div>
        <div className="text-lg font-semibold leading-none text-slate-800">{value}</div>
        <div className="mt-1 text-xs text-slate-500">{label}</div>
      </div>
    </div>
  );
}

export default function DocumentStats({ doc }: { doc: DocumentInfo }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      <Stat icon={<FileText className="h-5 w-5" />} value={doc.page_count} label="Pages" />
      <Stat icon={<Layers className="h-5 w-5" />} value={doc.text_chunk_count} label="Text chunks" />
      <Stat icon={<ImageIcon className="h-5 w-5" />} value={doc.figure_count} label="Figures" />
      <Stat icon={<ScanSearch className="h-5 w-5" />} value={doc.image_count} label="Visuals" />
      <Stat icon={<Crop className="h-5 w-5" />} value={doc.region_count} label="Indexed regions" />
    </div>
  );
}
