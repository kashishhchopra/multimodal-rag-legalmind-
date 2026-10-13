import { useState } from "react";
import { Loader2, ShieldCheck } from "lucide-react";
import { api } from "../services/api";

const SAMPLE = "Contact the complainant on 9876543210 or ravi.k@example.com. PAN ABCDE1234F, Aadhaar 2345 6789 0123, account number 123456789012.";

export default function SafetyPage() {
  const [text, setText] = useState(SAMPLE);
  const [out, setOut] = useState<{ text: string; redactions: Record<string, number>; note: string } | null>(null);
  const [busy, setBusy] = useState(false);
  async function run() {
    setBusy(true);
    try {
      setOut(await api.redact(text));
    } finally {
      setBusy(false);
    }
  }
  const rules = [
    ["Retrieval gate", "If the best retrieved chunk scores below 0.30 similarity, no answer is generated."],
    ["Claim support check", "Each answer sentence must share its content words with retrieved text; numbers, section numbers and cited pages must appear in the evidence. This is lexical, not legal reasoning."],
    ["Citations", "Every answer carries the pages and chunks it was built from; unsupported statements are listed under the answer."],
    ["Disclaimer", "Answers end with a not-legal-advice line and memos carry it too."],
    ["PII redaction", "Memo exports mask Aadhaar, PAN, phone, e-mail, IFSC and account numbers. Names, addresses and victim identity are not detected."],
  ];
  return (
    <div className="mx-auto h-full max-w-[1000px] space-y-5 overflow-y-auto px-8 pb-10 pt-8">
      <div>
        <div className="label">Safety</div>
        <h1 className="mt-1 text-[30px] font-semibold tracking-tight text-slate-900">Guardrails in this system</h1>
      </div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {rules.map(([t, d]) => (
          <div key={t} className="card flex gap-4 p-5">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600"><ShieldCheck className="h-5 w-5" /></div>
            <div>
              <div className="font-semibold text-slate-900">{t}</div>
              <p className="mt-1 text-sm text-slate-500">{d}</p>
            </div>
          </div>
        ))}
      </div>
      <section className="card p-6">
        <div className="label">Try it</div>
        <h2 className="mb-3 mt-1 text-[20px] font-semibold text-slate-900">PII redaction</h2>
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} className="w-full rounded-xl border border-slate-200 p-3 text-sm outline-none focus:border-brand-500" />
        <button onClick={run} disabled={busy} className="mt-3 flex items-center gap-2 rounded-xl bg-navy-900 px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-50">
          {busy && <Loader2 className="h-4 w-4 animate-spin" />} Redact
        </button>
        {out && (
          <div className="mt-4 rounded-xl bg-zinc-50 p-4 text-sm">
            <p className="whitespace-pre-wrap text-slate-800">{out.text}</p>
            <p className="mt-2 text-xs text-slate-500">{Object.entries(out.redactions).map(([k, v]) => `${v} ${k}`).join(" · ") || "Nothing found"} — {out.note}</p>
          </div>
        )}
      </section>
    </div>
  );
}
