import { X } from "lucide-react";

export default function ArchitectureModal({ onClose }: { onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4"
      onClick={onClose}
    >
      <div
        className="max-h-[85vh] w-full max-w-3xl overflow-y-auto rounded-2xl bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-800">
            Traditional RAG vs. Multimodal RAG
          </h2>
          <button onClick={onClose} className="rounded-lg p-1 text-slate-400 hover:bg-slate-100">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <h3 className="mb-2 text-sm font-semibold text-slate-600">Traditional RAG</h3>
            <pre className="whitespace-pre text-xs leading-relaxed text-slate-500">{`Question
   ↓
Text retrieval
   ↓
   LLM
   ↓
 Answer`}</pre>
            <p className="mt-3 text-xs text-slate-500">
              Only text chunks are retrieved. Seals, stamps, signatures and clause tables are
              invisible to the model.
            </p>
          </div>

          <div className="rounded-xl border border-brand-200 bg-brand-50 p-4">
            <h3 className="mb-2 text-sm font-semibold text-brand-700">This system</h3>
            <pre className="whitespace-pre text-xs leading-relaxed text-brand-700">{`      Question
         ↓
   ┌─────┴─────┐
   ↓           ↓
 Text        Visual
 retrieval   retrieval
 (BGE)       (CLIP)
   ↓           ↓
   └─────┬─────┘
         ↓
  Multimodal LLM
         ↓
  Grounded answer
         ↓
Text + Visual citations`}</pre>
          </div>
        </div>

        <div className="mt-4 space-y-2 text-sm text-slate-600">
          <p>
            <strong>Text path:</strong> the question is embedded with a BGE text model and
            matched against text/table chunks.
          </p>
          <p>
            <strong>Visual path:</strong> the same question is embedded with CLIP into a shared
            image–text space and matched directly against seal/stamp/signature embeddings.
          </p>
          <p>
            <strong>Grounding:</strong> OCR line-regions inside each image let the UI highlight
            the specific region relevant to your question (e.g. a signatory's name on a seal).
          </p>
          <p>
            <strong>Citations</strong> only ever reference evidence that was actually retrieved —
            the system never invents sources.
          </p>
        </div>
      </div>
    </div>
  );
}
