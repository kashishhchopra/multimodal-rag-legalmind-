import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { X } from "lucide-react";
import Sidebar, { type Health, type Page } from "./components/Sidebar";
import Topbar, { type Notice } from "./components/Topbar";
import OverviewPage from "./components/OverviewPage";
import UploadPanel from "./components/UploadPanel";
import ProcessingSteps from "./components/ProcessingSteps";
import DocumentStats from "./components/DocumentStats";
import ChatPanel from "./components/ChatPanel";
import ArchitectureModal from "./components/ArchitectureModal";
import Lightbox from "./components/Lightbox";
import InsightsPanel from "./components/InsightsPanel";
import SearchPanel from "./components/SearchPanel";
import AdminPanel from "./components/AdminPanel";
import ComparePage from "./components/ComparePage";
import SafetyPage from "./components/SafetyPage";
import LoginPanel from "./components/LoginPanel";
import CorpusLibrary from "./components/CorpusLibrary";
import { api, auth } from "./services/api";
import type { AdminStats, ChatMessage, DocumentInfo } from "./types";

function usernameFromToken(): string {
  try {
    const t = auth.token;
    if (!t) return "Local user";
    const body = JSON.parse(atob(t.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
    return body.sub || "Local user";
  } catch {
    return "Local user";
  }
}

export default function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [messagesByDoc, setMessagesByDoc] = useState<Record<string, ChatMessage[]>>({});
  const [busy, setBusy] = useState(false);
  const [showArch, setShowArch] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [lightbox, setLightbox] = useState<{ url: string; caption: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const PAGES: Page[] = ["overview", "documents", "insights", "compare", "search", "safety", "admin"];
  const [page, setPageState] = useState<Page>(() => {
    const h = window.location.hash.replace("#", "") as Page;
    return PAGES.includes(h) ? h : "overview";
  });
  const setPage = useCallback((p: Page) => {
    setPageState(p);
    window.location.hash = p; // linkable pages, back button friendly
  }, []);
  useEffect(() => {
    const on = () => {
      const h = window.location.hash.replace("#", "") as Page;
      if (PAGES.includes(h)) setPageState(h);
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const [needLogin, setNeedLogin] = useState(false);
  const [sidebarHidden, setSidebarHidden] = useState(false);
  const [searchSeed, setSearchSeed] = useState("");
  const [uploading, setUploading] = useState(false);
  const [docView, setDocView] = useState<"mine" | "corpus">("mine");
  const pollRef = useRef<number | null>(null);
  const username = useMemo(usernameFromToken, [needLogin]);

  const activeDoc = useMemo(() => documents.find((d) => d.document_id === activeId) || null, [documents, activeId]);

  const refreshStats = useCallback(() => {
    api.adminStats().then(setStats).catch(() => setStats(null));
  }, []);

  // A 401 from any API call (auth enabled, no/expired token) shows the login screen
  useEffect(() => {
    const on = () => {
      auth.token = null;
      setNeedLogin(true);
    };
    window.addEventListener("legalmind:unauthorized", on);
    return () => window.removeEventListener("legalmind:unauthorized", on);
  }, []);

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
      refreshStats();
    })();
  }, [refreshStats]);

  useEffect(() => {
    if (page === "overview") refreshStats();
  }, [page, refreshStats]);

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
      const processing = documents.filter((d) => d.stage !== "ready" && d.stage !== "failed");
      const updates = await Promise.all(processing.map((d) => api.getStatus(d.document_id).catch(() => null)));
      setDocuments((prev) => prev.map((d) => updates.find((x) => x && x.document_id === d.document_id) || d));
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
      setUploading(false);
      setPage("documents");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    }
  }, []);

  const openCorpusDoc = useCallback(async (id: string) => {
    setError(null);
    try {
      const doc = await api.getStatus(id);
      setDocuments((prev) => (prev.some((d) => d.document_id === id) ? prev : [doc, ...prev]));
      setActiveId(id);
      setUploading(false);
      setDocView("mine");
      setPage("documents");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not open the document");
    }
  }, [setPage]);

  const goUpload = useCallback(() => {
    setDocView("mine");
    setUploading(true);
    setPage("documents");
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
    async (question: string, opts: { plainEnglish: boolean; language: "en" | "hi" }) => {
      if (!activeId) return;
      const history = (messagesByDoc[activeId] || [])
        .filter((m) => !m.pending && m.content)
        .slice(-6)
        .map((m) => ({ role: m.role, content: m.content.slice(0, 1500) }));
      const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: question };
      const pendingId = crypto.randomUUID();
      const pendingMsg: ChatMessage = { id: pendingId, role: "assistant", content: "", pending: true };
      setMessagesByDoc((prev) => ({ ...prev, [activeId]: [...(prev[activeId] || []), userMsg, pendingMsg] }));
      setBusy(true);
      try {
        const response = await api.query(activeId, question, { history, ...opts });
        setMessagesByDoc((prev) => ({
          ...prev,
          [activeId]: (prev[activeId] || []).map((m) => (m.id === pendingId ? { ...m, pending: false, response, content: response.answer } : m)),
        }));
      } catch (e) {
        setMessagesByDoc((prev) => ({
          ...prev,
          [activeId]: (prev[activeId] || []).map((m) =>
            m.id === pendingId ? { ...m, pending: false, content: e instanceof Error ? e.message : "Query failed" } : m
          ),
        }));
      } finally {
        setBusy(false);
      }
    },
    [activeId, messagesByDoc]
  );

  const notices: Notice[] = [];
  if (health && (health.llm_provider.startsWith("invalid") || !health.llm_key_configured))
    notices.push({ id: "llm", tone: "warn", text: "LLM not configured: set LLM_PROVIDER (openai or gemini) and LLM_API_KEY in backend/.env. Answers are evidence-only until then." });
  if (documents.some((d) => d.stage === "failed")) notices.push({ id: "failed", tone: "warn", text: "One or more documents failed to process." });
  if (error) notices.push({ id: "err", tone: "warn", text: error });

  const messages = activeId ? messagesByDoc[activeId] || [] : [];
  const needsDoc = (title: string) => (
    <div className="flex h-full items-center justify-center p-8">
      <div className="card max-w-md p-8 text-center">
        <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
        <p className="mt-2 text-sm text-slate-500">Select or upload a processed document first.</p>
        <button onClick={goUpload} className="mt-5 rounded-xl bg-navy-900 px-5 py-2.5 text-sm font-semibold text-white">Upload PDF</button>
      </div>
    </div>
  );

  const content = needLogin ? (
    <LoginPanel onDone={() => window.location.reload()} />
  ) : page === "overview" ? (
    <OverviewPage documents={documents} activeDoc={activeDoc} stats={stats} health={health} username={username} onNavigate={setPage} onUpload={goUpload} />
  ) : page === "documents" ? (
    <div className="flex h-full flex-col">
      <div className="flex gap-1 border-b border-slate-200/70 bg-white/60 px-8 pt-2 text-sm">
        {(["mine", "corpus"] as const).map((v) => (
          <button key={v} onClick={() => setDocView(v)} className={`px-3 py-2 ${docView === v ? "border-b-2 border-zinc-800 font-medium text-slate-900" : "text-slate-500"}`}>
            {v === "mine" ? `Open documents (${documents.length})` : `Corpus library (${(stats?.corpus?.judgment?.documents ?? 0).toLocaleString()} judgments)`}
          </button>
        ))}
      </div>
      {docView === "corpus" ? (
        <div className="min-h-0 flex-1"><CorpusLibrary onOpen={openCorpusDoc} /></div>
      ) : (<>
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-200/70 bg-white/60 px-8 py-3">
        {documents.map((d) => (
          <span key={d.document_id} className={`group flex items-center gap-1 rounded-xl border px-3 py-1.5 text-sm ${d.document_id === activeId && !uploading ? "border-brand-500 bg-brand-50 text-brand-700" : "border-slate-200 bg-white text-slate-600 hover:border-brand-300"}`}>
            <button onClick={() => { setActiveId(d.document_id); setUploading(false); }}>{d.filename}</button>
            <button onClick={() => handleDelete(d.document_id)} className="opacity-0 transition group-hover:opacity-100" title="Delete document"><X className="h-3.5 w-3.5 text-slate-400 hover:text-red-500" /></button>
          </span>
        ))}
        <button onClick={() => setUploading(true)} className="rounded-xl border border-dashed border-slate-300 px-3 py-1.5 text-sm text-slate-500 hover:border-brand-400 hover:text-brand-600">+ Upload</button>
      </div>
      <div className="min-h-0 flex-1">
        {uploading || !activeDoc ? (
          <div className="flex h-full items-center justify-center p-8"><UploadPanel onUpload={handleUpload} /></div>
        ) : activeDoc.stage !== "ready" ? (
          <div className="mx-auto max-w-2xl space-y-4 p-8"><ProcessingSteps doc={activeDoc} /></div>
        ) : (
          <div className="flex h-full flex-col">
            <div className="px-8 pt-4"><DocumentStats doc={activeDoc} /></div>
            <div className="mx-auto flex w-full max-w-[1000px] min-h-0 flex-1 flex-col overflow-hidden px-8">
              <ChatPanel documentId={activeDoc.document_id} messages={messages} onSend={handleSend} busy={busy} onZoom={(url, caption) => setLightbox({ url, caption })} />
            </div>
          </div>
        )}
      </div>
      </>)}
    </div>
  ) : page === "insights" ? (
    activeDoc && activeDoc.stage === "ready" ? (
      <div className="mx-auto h-full max-w-[1000px] px-8 pb-6 pt-8">
        <div className="label">Insights</div>
        <h1 className="mb-4 mt-1 truncate text-[30px] font-semibold tracking-tight text-slate-900">{activeDoc.filename}</h1>
        <div className="card h-[calc(100%-90px)] overflow-hidden"><InsightsPanel documentId={activeDoc.document_id} /></div>
      </div>
    ) : needsDoc("No document selected")
  ) : page === "compare" ? (
    <ComparePage documents={documents} />
  ) : page === "search" ? (
    <SearchPanel initialQuery={searchSeed} key={searchSeed} />
  ) : page === "safety" ? (
    <SafetyPage />
  ) : (
    <AdminPanel />
  );

  return (
    <div className="flex h-screen overflow-hidden">
      {!sidebarHidden && (
        <Sidebar
          page={page}
          onNavigate={(p) => { setPage(p); if (p !== "search") setSearchSeed(""); }}
          documents={documents}
          activeDoc={activeDoc}
          onSelectDoc={(id) => { setActiveId(id); setUploading(false); }}
          onUpload={goUpload}
          health={health}
          chunks={stats?.indexed_text_chunks ?? null}
          username={username}
          onShowArchitecture={() => setShowArch(true)}
          onShowSettings={() => setShowSettings(true)}
          onCollapse={() => setSidebarHidden(true)}
        />
      )}
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <Topbar
          onSearch={(q) => { setSearchSeed(q); setPage("search"); }}
          onUpload={goUpload}
          notices={notices}
          sidebarHidden={sidebarHidden}
          onShowSidebar={() => setSidebarHidden(false)}
        />
        {error && page !== "overview" && <div className="mx-8 mt-4 rounded-xl bg-red-50 px-4 py-2 text-sm text-red-700">{error}</div>}
        <div className="min-h-0 flex-1">{content}</div>
      </main>

      {showArch && <ArchitectureModal onClose={() => setShowArch(false)} />}
      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-navy-950/50 p-4" onClick={() => setShowSettings(false)}>
          <div className="card w-full max-w-md p-6" onClick={(e) => e.stopPropagation()}>
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-slate-900">Settings</h2>
              <button onClick={() => setShowSettings(false)} aria-label="Close"><X className="h-5 w-5 text-slate-400" /></button>
            </div>
            <dl className="space-y-2 text-sm">
              {[
                ["LLM provider", health?.llm_provider ?? "unknown"],
                ["API key", health?.llm_key_configured ? "set" : "not set"],
                ["OCR (Tesseract)", health?.ocr_available ? "available" : "unavailable"],
                ["Signed in as", username],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between rounded-xl bg-zinc-50 px-3 py-2"><dt className="text-slate-500">{k}</dt><dd className="font-medium text-slate-800">{v}</dd></div>
              ))}
            </dl>
            <p className="mt-4 text-xs text-slate-400">Configuration lives in backend/.env (LLM_PROVIDER, LLM_API_KEY, AUTH_ENABLED). Restart the backend after changing it.</p>
            {auth.token && <button onClick={() => { auth.token = null; window.location.reload(); }} className="mt-4 rounded-xl border border-slate-200 px-4 py-2 text-sm">Sign out</button>}
          </div>
        </div>
      )}
      {lightbox && <Lightbox url={lightbox.url} caption={lightbox.caption} onClose={() => setLightbox(null)} />}
    </div>
  );
}
