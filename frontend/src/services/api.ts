// Thin API client for the FastAPI backend. Paths are same-origin and proxied
// to the backend by Vite in dev (see vite.config.ts).

import type { AdminStats, CorpusPage, CompareResult, ClauseReport, DocumentInfo, DocumentSummary, QueryResponse, TimelineEvent } from "../types";

const BASE = "/api";

// Bearer token for deployments with AUTH_ENABLED=1; harmless when auth is off.
const TOKEN_KEY = "legalmind.token";
export const auth = {
  get token(): string | null {
    try {
      return localStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },
  set token(v: string | null) {
    try {
      v ? localStorage.setItem(TOKEN_KEY, v) : localStorage.removeItem(TOKEN_KEY);
    } catch {
      /* storage blocked */
    }
  },
};

// Every request goes through this so the token and 401 handling live in one place.
const rawFetch = window.fetch.bind(window);
window.fetch = (input: RequestInfo | URL, init: RequestInit = {}) => {
  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  if (url.startsWith(BASE) && auth.token) {
    init = { ...init, headers: { ...(init.headers as Record<string, string>), Authorization: `Bearer ${auth.token}` } };
  }
  return rawFetch(input, init).then((r) => {
    if (r.status === 401 && url.startsWith(BASE) && !url.startsWith(`${BASE}/auth/`)) window.dispatchEvent(new Event("legalmind:unauthorized"));
    return r;
  });
};

async function handle<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    let message = `Request failed (${resp.status})`;
    try {
      const body = await resp.json();
      message = body?.error?.message || body?.detail || message;
    } catch {
      /* ignore parse errors */
    }
    throw new Error(message);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  async health(): Promise<{ status: string; llm_provider: string; llm_key_configured: boolean; ocr_available: boolean }> {
    return handle(await fetch(`${BASE}/health`));
  },

  async listDocuments(): Promise<DocumentInfo[]> {
    return handle(await fetch(`${BASE}/documents`));
  },

  async getStatus(documentId: string): Promise<DocumentInfo> {
    return handle(await fetch(`${BASE}/documents/${documentId}/status`));
  },

  async uploadDocument(file: File): Promise<DocumentInfo> {
    const form = new FormData();
    form.append("file", file);
    return handle(await fetch(`${BASE}/documents/upload`, { method: "POST", body: form }));
  },

  async deleteDocument(documentId: string): Promise<void> {
    await handle(await fetch(`${BASE}/documents/${documentId}`, { method: "DELETE" }));
  },

  async query(
    documentId: string,
    question: string,
    opts: { history?: { role: "user" | "assistant"; content: string }[]; plainEnglish?: boolean; language?: "en" | "hi" } = {}
  ): Promise<QueryResponse> {
    return handle(
      await fetch(`${BASE}/documents/${documentId}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          history: opts.history ?? [],
          plain_english: opts.plainEnglish ?? false,
          language: opts.language ?? "en",
        }),
      })
    );
  },

  async corpusDocuments(params: { q?: string; year?: string; kind?: string; offset?: number; limit?: number }): Promise<CorpusPage> {
    const u = new URLSearchParams();
    if (params.q) u.set("q", params.q);
    if (params.year) u.set("year", params.year);
    if (params.kind) u.set("kind", params.kind);
    u.set("offset", String(params.offset ?? 0));
    u.set("limit", String(params.limit ?? 25));
    return handle(await fetch(`${BASE}/corpus/documents?${u}`));
  },

  async login(username: string, password: string): Promise<{ access_token: string; role: string }> {
    return handle(
      await fetch(`${BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      })
    );
  },

  async register(username: string, password: string): Promise<{ role: string }> {
    return handle(
      await fetch(`${BASE}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      })
    );
  },

  async compare(a: string, b: string): Promise<CompareResult> {
    return handle(
      await fetch(`${BASE}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_a: a, document_b: b }),
      })
    );
  },

  async redact(text: string): Promise<{ text: string; redactions: Record<string, number>; note: string }> {
    return handle(
      await fetch(`${BASE}/redact`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      })
    );
  },

  async adminStats(): Promise<AdminStats> {
    return handle(await fetch(`${BASE}/admin/stats`));
  },

  async search(question: string, filters: Record<string, string | number>, topK = 8): Promise<QueryResponse> {
    return handle(
      await fetch(`${BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: topK, filters }),
      })
    );
  },

  async sections(q: string): Promise<{ expanded_query: string; mappings: string[] }> {
    return handle(await fetch(`${BASE}/legal/sections?q=${encodeURIComponent(q)}`));
  },

  async clauses(documentId: string): Promise<ClauseReport> {
    return handle(await fetch(`${BASE}/documents/${documentId}/clauses`));
  },

  async summary(documentId: string): Promise<DocumentSummary> {
    return handle(await fetch(`${BASE}/documents/${documentId}/summary`));
  },

  async timeline(documentId: string): Promise<{ events: TimelineEvent[] }> {
    return handle(await fetch(`${BASE}/documents/${documentId}/timeline`));
  },

  async feedback(body: {
    document_id: string;
    question: string;
    answer: string;
    rating: 1 | -1;
    reason?: string;
    status?: string;
  }): Promise<{ id: number }> {
    return handle(
      await fetch(`${BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
    );
  },

  async downloadMemo(
    format: "pdf" | "docx" | "md",
    response: QueryResponse
  ): Promise<void> {
    const resp = await fetch(`${BASE}/memo`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: response.question,
        answer: response.answer,
        citations: response.citations,
        verification: response.verification ?? null,
        format,
      }),
    });
    if (!resp.ok) throw new Error(`Memo export failed (${resp.status})`);
    const url = URL.createObjectURL(await resp.blob());
    const a = document.createElement("a");
    a.href = url;
    a.download = `memo.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  },

  groundingUrl(documentId: string, visualId: string, mode: "highlight" | "crop" = "highlight"): string {
    return `${BASE}/documents/${documentId}/visuals/${visualId}/grounding?mode=${mode}`;
  },
};
