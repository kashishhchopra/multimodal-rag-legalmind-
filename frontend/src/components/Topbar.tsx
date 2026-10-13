import { useEffect, useRef, useState } from "react";
import { Bell, Command, PanelLeft, Search, Upload } from "lucide-react";

export type Notice = { id: string; text: string; tone: "warn" | "info" };

export default function Topbar({
  onSearch,
  onUpload,
  notices,
  sidebarHidden,
  onShowSidebar,
}: {
  onSearch: (q: string) => void;
  onUpload: () => void;
  notices: Notice[];
  sidebarHidden: boolean;
  onShowSidebar: () => void;
}) {
  const [q, setQ] = useState("");
  const [bell, setBell] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        input.current?.focus();
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  return (
    <header className="relative flex items-center gap-3 border-b border-slate-200/70 bg-white/70 px-8 py-4 backdrop-blur">
      {sidebarHidden && (
        <button onClick={onShowSidebar} className="rounded-xl border border-slate-200 bg-white p-2 text-slate-500 hover:text-slate-800" aria-label="Show sidebar">
          <PanelLeft className="h-5 w-5" />
        </button>
      )}
      <form
        className="flex max-w-xl flex-1 items-center gap-3"
        onSubmit={(e) => {
          e.preventDefault();
          if (q.trim()) {
            onSearch(q.trim());
            setQ("");
          }
        }}
      >
        <Search className="h-[18px] w-[18px] text-slate-400" />
        <input
          ref={input}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search judgments, statutes, clauses…"
          className="flex-1 bg-transparent text-[15px] text-slate-700 outline-none placeholder:text-slate-400"
        />
        <kbd className="flex items-center gap-0.5 rounded-lg border border-slate-200 bg-white px-2 py-1 font-mono text-[11px] text-slate-400">
          <Command className="h-3 w-3" />K
        </kbd>
      </form>

      <div className="ml-auto flex items-center gap-3">
        <div className="relative">
          <button onClick={() => setBell((b) => !b)} className="relative rounded-xl border border-slate-200 bg-white p-2.5 text-slate-600 hover:text-slate-900" aria-label="Notifications">
            <Bell className="h-5 w-5" />
            {notices.length > 0 && <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-peach-500 ring-2 ring-white" />}
          </button>
          {bell && (
            <div className="absolute right-0 top-[calc(100%+8px)] z-30 w-80 rounded-2xl border border-slate-200 bg-white p-2 shadow-xl">
              {notices.length === 0 && <div className="px-3 py-4 text-sm text-slate-500">Nothing needs attention.</div>}
              {notices.map((n) => (
                <div key={n.id} className={`rounded-xl px-3 py-2.5 text-sm ${n.tone === "warn" ? "text-amber-800" : "text-slate-600"}`}>
                  {n.text}
                </div>
              ))}
            </div>
          )}
        </div>
        <button onClick={onUpload} className="flex items-center gap-2 rounded-2xl bg-navy-900 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-navy-900/20 transition hover:bg-navy-800">
          <Upload className="h-4 w-4" /> Upload PDF
        </button>
      </div>
    </header>
  );
}
