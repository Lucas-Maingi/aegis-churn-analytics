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

export default function SignupPage() {
  const router = useRouter();
  const [form, setForm] = useState({
    organization_name: "",
    industry: "ISP",
    full_name: "",
    email: "",
    password: "",
  });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  function set(field: string, value: string) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const res = await apiAuth<AuthResponse>("/api/v1/auth/signup", form);
      saveSession(res.access_token, res.organization_name);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not reach the server.");
      setBusy(false);
    }
  }

  const inputCls =
    "w-full rounded-lg border border-slate-600 bg-slate-900/60 px-3 py-2 text-sm outline-none focus:border-sky-500";

  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-10">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold tracking-tight">
            <span className="bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">
              Aegis
            </span>
          </h1>
          <p className="mt-2 text-sm text-slate-400">
            See who is about to churn, why, and win them back in one click.
          </p>
        </div>

        <form
          onSubmit={submit}
          className="space-y-4 rounded-2xl border border-slate-700/60 bg-slate-800/40 p-8"
        >
          <h2 className="text-lg font-semibold">Create your organization</h2>
          <ErrorNote message={error} />
          <div>
            <label className="mb-1 block text-sm text-slate-400">
              Company name
            </label>
            <input
              required
              value={form.organization_name}
              onChange={(e) => set("organization_name", e.target.value)}
              className={inputCls}
              placeholder="Nairobi Fiber Co"
            />
          </div>
          <div>
            <label className="mb-1 block text-sm text-slate-400">Industry</label>
            <select
              value={form.industry}
              onChange={(e) => set("industry", e.target.value)}
              className={inputCls}
            >
              <option value="ISP">Internet Service Provider</option>
              <option value="telecom">Telecom / MVNO</option>
              <option value="cable">Cable / TV operator</option>
              <option value="other">Other subscription business</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-sm text-slate-400">Your name</label>
            <input
              value={form.full_name}
              onChange={(e) => set("full_name", e.target.value)}
              className={inputCls}
              placeholder="Jane Doe"
            />
          </div>
          <div>
            <label className="mb-1 block text-sm text-slate-400">Email</label>
            <input
              type="email"
              required
              value={form.email}
              onChange={(e) => set("email", e.target.value)}
              className={inputCls}
              placeholder="you@company.com"
            />
          </div>
          <div>
            <label className="mb-1 block text-sm text-slate-400">
              Password <span className="text-slate-500">(8+ characters)</span>
            </label>
            <input
              type="password"
              required
              minLength={8}
              value={form.password}
              onChange={(e) => set("password", e.target.value)}
              className={inputCls}
              placeholder="••••••••"
            />
          </div>
          <button
            type="submit"
            disabled={busy}
            className="w-full rounded-lg bg-sky-500 py-2.5 text-sm font-semibold text-white transition hover:bg-sky-400 disabled:opacity-50"
          >
            {busy ? "Creating…" : "Create account"}
          </button>
          <p className="text-center text-sm text-slate-400">
            Already have an account?{" "}
            <Link href="/login" className="text-sky-400 hover:underline">
              Sign in
            </Link>
          </p>
        </form>
      </div>
    </main>
  );
}
