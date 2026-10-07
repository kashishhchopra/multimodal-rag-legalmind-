import { X } from "lucide-react";

export default function Lightbox({
  url,
  caption,
  onClose,
}: {
  url: string;
  caption: string;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-slate-900/80 p-6"
      onClick={onClose}
    >
      <button
        onClick={onClose}
        className="absolute right-5 top-5 rounded-lg bg-white/10 p-2 text-white hover:bg-white/20"
      >
        <X className="h-5 w-5" />
      </button>
      <img
        src={url}
        alt={caption}
        className="max-h-[80vh] max-w-full rounded-lg bg-white object-contain shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      />
      <div className="mt-3 text-sm text-slate-200">{caption}</div>
    </div>
  );
}
