import { useState } from "react";
import { api, auth } from "../services/api";

export default function LoginPanel({ onDone }: { onDone: () => void }) {
  const [u, setU] = useState("");
  const [p, setP] = useState("");
  const [mode, setMode] = useState<"login" | "register">("login");
  const [err, setErr] = useState<string | null>(null);
  async function go() {
    setErr(null);
    try {
      if (mode === "register") await api.register(u, p);
      auth.token = (await api.login(u, p)).access_token;
      onDone();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed");
    }
  }
  return (
    <div className="flex h-full items-center justify-center p-6">
      <div className="w-full max-w-sm space-y-3 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-base font-semibold text-slate-800">{mode === "login" ? "Sign in" : "Create account"}</h2>
        <input value={u} onChange={(e) => setU(e.target.value)} placeholder="Username" className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
        <input value={p} onChange={(e) => setP(e.target.value)} type="password" placeholder="Password (10+ characters)" onKeyDown={(e) => e.key === "Enter" && go()} className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
        {err && <div className="rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700">{err}</div>}
        <button onClick={go} className="w-full rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white">
          {mode === "login" ? "Sign in" : "Register and sign in"}
        </button>
        <button onClick={() => setMode(mode === "login" ? "register" : "login")} className="w-full text-xs text-slate-500">
          {mode === "login" ? "No account? Register" : "Have an account? Sign in"}
        </button>
        <p className="text-xs text-slate-400">The first account created becomes the admin.</p>
      </div>
    </div>
  );
}
