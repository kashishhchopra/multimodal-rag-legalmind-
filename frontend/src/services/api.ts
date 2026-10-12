// Thin API client for the FastAPI backend. Paths are same-origin and proxied
// to the backend by Vite in dev (see vite.config.ts).

import type { ClauseReport, DocumentInfo, DocumentSummary, QueryResponse, TimelineEvent } from "../types";

const BASE = "/api";

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

  async query(documentId: string, question: string): Promise<QueryResponse> {
    return handle(
      await fetch(`${BASE}/documents/${documentId}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      })
    );
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
