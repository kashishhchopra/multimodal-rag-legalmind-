// Thin API client for the FastAPI backend. Paths are same-origin and proxied
// to the backend by Vite in dev (see vite.config.ts).

import type { DocumentInfo, QueryResponse } from "../types";

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

  groundingUrl(documentId: string, visualId: string, mode: "highlight" | "crop" = "highlight"): string {
    return `${BASE}/documents/${documentId}/visuals/${visualId}/grounding?mode=${mode}`;
  },
};
