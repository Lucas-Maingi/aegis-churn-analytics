"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { apiAuth, saveSession, ApiError } from "@/lib/api";
import { ErrorNote } from "@/components/ui";

interface AuthResponse {
  access_token: string;
  organization_name: string;
}

// Baked in at build time for hosted demos ("email / password"); undefined
// in normal builds, which hides the demo box entirely.
const DEMO_LOGIN = process.env.NEXT_PUBLIC_DEMO_LOGIN;

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  function fillDemo() {
    if (!DEMO_LOGIN) return;
    const [demoEmail, demoPassword] = DEMO_LOGIN.split(" / ");
    setEmail(demoEmail ?? "");
    setPassword(demoPassword ?? "");
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const res = await apiAuth<AuthResponse>("/api/v1/auth/login", {
        email,
        password,
      });
      saveSession(res.access_token, res.organization_name);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not reach the server.");
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold tracking-tight">
            <span className="bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">
              Aegis
            </span>
          </h1>
          <p className="mt-2 text-sm text-slate-400">
            Churn intelligence for subscription businesses
          </p>
        </div>

        <form
          onSubmit={submit}
          className="space-y-4 rounded-2xl border border-slate-700/60 bg-slate-800/40 p-8"
        >
          <h2 className="text-lg font-semibold">Sign in</h2>
          {DEMO_LOGIN && (
            <div className="rounded-lg border border-sky-500/40 bg-sky-500/10 px-4 py-3 text-sm text-sky-200">
              <p>
                Just exploring? A demo ISP with 400 scored customers is ready.
              </p>
              <button
                type="button"
                onClick={fillDemo}
                className="mt-2 rounded-md bg-sky-500 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-sky-400"
              >
                Use demo account
              </button>
            </div>
          )}
          <ErrorNote message={error} />
          <div>
            <label className="mb-1 block text-sm text-slate-400">Email</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-lg border border-slate-600 bg-slate-900/60 px-3 py-2 text-sm outline-none focus:border-sky-500"
              placeholder="you@company.com"
            />
          </div>
          <div>
            <label className="mb-1 block text-sm text-slate-400">Password</label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-lg border border-slate-600 bg-slate-900/60 px-3 py-2 text-sm outline-none focus:border-sky-500"
              placeholder="••••••••"
            />
          </div>
          <button
            type="submit"
            disabled={busy}
            className="w-full rounded-lg bg-sky-500 py-2.5 text-sm font-semibold text-white transition hover:bg-sky-400 disabled:opacity-50"
          >
            {busy ? "Signing in…" : "Sign in"}
          </button>
          <p className="text-center text-sm text-slate-400">
            New here?{" "}
            <Link href="/signup" className="text-sky-400 hover:underline">
              Create your organization
            </Link>
          </p>
        </form>
      </div>
    </main>
  );
}
