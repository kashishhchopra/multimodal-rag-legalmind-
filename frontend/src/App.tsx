import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import UploadPanel from "./components/UploadPanel";
import ProcessingSteps from "./components/ProcessingSteps";
import DocumentStats from "./components/DocumentStats";
import ChatPanel from "./components/ChatPanel";
import ArchitectureModal from "./components/ArchitectureModal";
import Lightbox from "./components/Lightbox";
import InsightsPanel from "./components/InsightsPanel";
import SearchPanel from "./components/SearchPanel";
import { api } from "./services/api";
import type { ChatMessage, DocumentInfo } from "./types";

type Health = { llm_provider: string; llm_key_configured: boolean; ocr_available: boolean };

export default function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [messagesByDoc, setMessagesByDoc] = useState<Record<string, ChatMessage[]>>({});
  const [busy, setBusy] = useState(false);
  const [showArch, setShowArch] = useState(false);
  const [lightbox, setLightbox] = useState<{ url: string; caption: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const [view, setView] = useState<"document" | "search">("document");
  const [docTab, setDocTab] = useState<"chat" | "insights">("chat");

  const activeDoc = useMemo(
    () => documents.find((d) => d.document_id === activeId) || null,
    [documents, activeId]
  );

  // Initial load
  useEffect(() => {
    (async () => {
      try {
        setHealth(await api.health());
      } catch {
        setError("Cannot reach the backend. Is it running on port 8000?");
      }
      try {
        const docs = await api.listDocuments();
        setDocuments(docs);
        const firstReady = docs.find((d) => d.stage === "ready");
        if (firstReady) setActiveId(firstReady.document_id);
      } catch {
        /* handled above */
      }
    })();
  }, []);

  // Poll status for any documents still processing
  useEffect(() => {
    const anyProcessing = documents.some((d) => d.stage !== "ready" && d.stage !== "failed");
    if (!anyProcessing) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }
    if (pollRef.current) return;
    pollRef.current = window.setInterval(async () => {
      setDocuments((prev) => prev); // trigger nothing; real work below
      const processing = documents.filter((d) => d.stage !== "ready" && d.stage !== "failed");
      const updates = await Promise.all(
        processing.map((d) => api.getStatus(d.document_id).catch(() => null))
      );
      setDocuments((prev) =>
        prev.map((d) => {
          const u = updates.find((x) => x && x.document_id === d.document_id);
          return u || d;
        })
      );
    }, 1200);
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [documents]);

  const handleUpload = useCallback(async (file: File) => {
    setError(null);
    try {
      const doc = await api.uploadDocument(file);
      setDocuments((prev) => [doc, ...prev]);
      setActiveId(doc.document_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    }
  }, []);

  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await api.deleteDocument(id);
      } catch {
        /* ignore */
      }
      setDocuments((prev) => prev.filter((d) => d.document_id !== id));
      setMessagesByDoc((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      if (activeId === id) setActiveId(null);
    },
    [activeId]
  );

  const handleSend = useCallback(
    async (question: string) => {
      if (!activeId) return;
      const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: question };
      const pendingId = crypto.randomUUID();
      const pendingMsg: ChatMessage = { id: pendingId, role: "assistant", content: "", pending: true };
      setMessagesByDoc((prev) => ({
        ...prev,
        [activeId]: [...(prev[activeId] || []), userMsg, pendingMsg],
      }));
      setBusy(true);
      try {
        const response = await api.query(activeId, question);
        setMessagesByDoc((prev) => ({
          ...prev,
          [activeId]: (prev[activeId] || []).map((m) =>
            m.id === pendingId ? { ...m, pending: false, response, content: response.answer } : m
          ),
        }));
      } catch (e) {
        setMessagesByDoc((prev) => ({
          ...prev,
          [activeId]: (prev[activeId] || []).map((m) =>
            m.id === pendingId
              ? { ...m, pending: false, content: e instanceof Error ? e.message : "Query failed" }
              : m
          ),
        }));
      } finally {
        setBusy(false);
      }
    },
    [activeId]
  );

  const messages = activeId ? messagesByDoc[activeId] || [] : [];

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        documents={documents}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={() => setActiveId(null)}
        onDelete={handleDelete}
        onShowArchitecture={() => setShowArch(true)}
        health={health}
      />

      <main className="flex flex-1 flex-col overflow-hidden bg-slate-50">
        <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-3.5">
          <div>
            <h1 className="text-base font-semibold text-slate-800">
              {activeDoc ? activeDoc.filename : "LegalMind — Multimodal Legal Document AI"}
            </h1>
            <p className="text-xs text-slate-500">
              Retrieve and reason over clauses, tables, seals, signatures and case citations.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex rounded-lg bg-slate-100 p-0.5 text-xs">
              {(["document", "search"] as const).map((v) => (
                <button
                  key={v}
                  onClick={() => setView(v)}
                  className={`rounded-md px-3 py-1.5 ${view === v ? "bg-white font-medium text-brand-700 shadow-sm" : "text-slate-500"}`}
                >
                  {v === "document" ? "Document" : "Search corpus"}
                </button>
              ))}
            </div>
            {health && (
              <div className="hidden items-center gap-2 rounded-lg bg-slate-100 px-3 py-1.5 text-xs text-slate-500 sm:flex">
                <span className="h-2 w-2 rounded-full bg-emerald-500" /> Backend connected
              </div>
            )}
          </div>
        </header>

        {error && (
          <div className="mx-6 mt-4 rounded-lg bg-red-50 px-4 py-2 text-sm text-red-700">{error}</div>
        )}

        <div className="flex-1 overflow-hidden">
          {view === "search" ? (
            <SearchPanel />
          ) : !activeDoc ? (
            <div className="flex h-full items-center justify-center p-6">
              <UploadPanel onUpload={handleUpload} />
            </div>
          ) : activeDoc.stage !== "ready" ? (
            <div className="mx-auto max-w-2xl space-y-4 p-6">
              <ProcessingSteps doc={activeDoc} />
            </div>
          ) : (
            <div className="flex h-full flex-col">
              <div className="border-b border-slate-200 bg-white px-6 py-4">
                <DocumentStats doc={activeDoc} />
              </div>
              <div className="flex gap-1 border-b border-slate-200 bg-white px-6 text-sm">
                {(["chat", "insights"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setDocTab(t)}
                    className={`px-3 py-2 ${docTab === t ? "border-b-2 border-brand-600 font-medium text-brand-700" : "text-slate-500"}`}
                  >
                    {t === "chat" ? "Ask questions" : "Insights"}
                  </button>
                ))}
              </div>
              {docTab === "insights" ? (
                <div className="mx-auto w-full max-w-4xl flex-1 overflow-hidden">
                  <InsightsPanel documentId={activeDoc.document_id} />
                </div>
              ) : (
                <div className="mx-auto flex w-full max-w-4xl flex-1 flex-col overflow-hidden px-6">
                  <ChatPanel
                    documentId={activeDoc.document_id}
                    messages={messages}
                    onSend={handleSend}
                    busy={busy}
                    onZoom={(url, caption) => setLightbox({ url, caption })}
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {showArch && <ArchitectureModal onClose={() => setShowArch(false)} />}
      {lightbox && (
        <Lightbox url={lightbox.url} caption={lightbox.caption} onClose={() => setLightbox(null)} />
      )}
    </div>
  );
}
