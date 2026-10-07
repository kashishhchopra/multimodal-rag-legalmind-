import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies API + static image requests to the FastAPI backend,
// so the frontend can call same-origin paths like `/api/...` and `/static/...`.
const BACKEND = process.env.VITE_BACKEND_URL || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: BACKEND, changeOrigin: true },
      "/static": { target: BACKEND, changeOrigin: true },
    },
  },
});
