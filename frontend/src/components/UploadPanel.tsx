import { useCallback, useRef, useState } from "react";
import { UploadCloud, FileText, AlertCircle } from "lucide-react";

export default function UploadPanel({
  onUpload,
  disabled,
}: {
  onUpload: (file: File) => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateAndUpload = useCallback(
    (file: File | undefined) => {
      setError(null);
      if (!file) return;
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        setError("Only PDF files are supported in this version.");
        return;
      }
      onUpload(file);
    },
    [onUpload]
  );

  return (
    <div className="mx-auto max-w-2xl">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          validateAndUpload(e.dataTransfer.files?.[0]);
        }}
        onClick={() => !disabled && inputRef.current?.click()}
        className={`cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition-colors ${
          dragging ? "border-brand-500 bg-brand-50" : "border-slate-300 bg-white hover:border-brand-300"
        } ${disabled ? "pointer-events-none opacity-60" : ""}`}
      >
        <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-brand-50 text-brand-600">
          <UploadCloud className="h-8 w-8" />
        </div>
        <h2 className="text-lg font-semibold text-slate-800">Upload a legal document to begin</h2>
        <p className="mt-1 text-sm text-slate-500">
          Drag &amp; drop a contract, judgment, or statute PDF here, or click to browse. The
          system will extract clauses, tables, seals and signatures, then let you ask
          questions grounded in the document's actual text and visual evidence.
        </p>
        <div className="mt-4 inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white">
          <FileText className="h-4 w-4" /> Choose PDF
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          className="hidden"
          onChange={(e) => validateAndUpload(e.target.files?.[0])}
        />
      </div>

      {error && (
        <div className="mt-3 flex items-center gap-2 rounded-lg bg-red-50 px-4 py-2 text-sm text-red-700">
          <AlertCircle className="h-4 w-4" /> {error}
        </div>
      )}
    </div>
  );
}
