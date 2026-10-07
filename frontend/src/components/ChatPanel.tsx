import { useEffect, useRef, useState } from "react";
import { Send, Loader2, User, Bot } from "lucide-react";
import type { ChatMessage } from "../types";
import AnswerCard from "./AnswerCard";

const EXAMPLES = [
  "Who are the parties to this agreement and what is the effective date?",
  "What does the Termination clause say?",
  "What was the court's holding and what relief was granted?",
  "What is the monthly rent and security deposit in the key-terms table?",
  "Explain the Indemnification clause in plain language.",
  "What does Section 73 of the General Contract Code say about damages?",
];

export default function ChatPanel({
  documentId,
  messages,
  onSend,
  busy,
  onZoom,
}: {
  documentId: string;
  messages: ChatMessage[];
  onSend: (question: string) => void;
  busy: boolean;
  onZoom: (url: string, caption: string) => void;
}) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const submit = () => {
    const q = input.trim();
    if (!q || busy) return;
    onSend(q);
    setInput("");
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-6 overflow-y-auto px-1 py-4">
        {messages.length === 0 && (
          <div className="mx-auto max-w-xl pt-6 text-center">
            <h3 className="text-base font-semibold text-slate-700">Ask about this document</h3>
            <p className="mt-1 text-sm text-slate-500">
              Questions can reference clauses, parties, tables, seals or signatures.
            </p>
            <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => onSend(ex)}
                  disabled={busy}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-left text-sm text-slate-600 transition hover:border-brand-300 hover:bg-brand-50 disabled:opacity-50"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="flex justify-end">
              <div className="flex max-w-2xl items-start gap-2">
                <div className="rounded-2xl rounded-tr-sm bg-brand-600 px-4 py-2.5 text-sm text-white">
                  {m.content}
                </div>
                <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-200 text-slate-500">
                  <User className="h-4 w-4" />
                </div>
              </div>
            </div>
          ) : (
            <div key={m.id} className="flex items-start gap-2">
              <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-brand-100 text-brand-600">
                <Bot className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                {m.pending ? (
                  <div className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-500">
                    <Loader2 className="h-4 w-4 animate-spin text-brand-500" />
                    Retrieving text &amp; visual evidence…
                  </div>
                ) : m.response ? (
                  <AnswerCard documentId={documentId} response={m.response} onZoom={onZoom} />
                ) : (
                  <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                    {m.content}
                  </div>
                )}
              </div>
            </div>
          )
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-slate-200 bg-white px-1 py-3">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder="Ask a question about the document…"
            className="max-h-32 flex-1 resize-none rounded-xl border border-slate-300 px-4 py-2.5 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
          />
          <button
            onClick={submit}
            disabled={busy || !input.trim()}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-brand-600 text-white transition hover:bg-brand-700 disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
          </button>
        </div>
      </div>
    </div>
  );
}
