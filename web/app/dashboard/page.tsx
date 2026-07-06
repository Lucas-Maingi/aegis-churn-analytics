"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { apiGet, ApiError, CustomerList } from "@/lib/api";
import {
  ErrorNote,
  ProbabilityBar,
  RiskPill,
  Spinner,
  StatCard,
} from "@/components/ui";
import CustomerDrawer from "@/components/CustomerDrawer";

const TIER_FILTERS = ["ALL", "HIGH", "MEDIUM", "LOW"] as const;

export default function CustomersPage() {
  const [data, setData] = useState<CustomerList | null>(null);
  const [error, setError] = useState("");
  const [tier, setTier] = useState<(typeof TIER_FILTERS)[number]>("ALL");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<number | null>(null);

  const load = useCallback(async () => {
    setError("");
    try {
      const params = new URLSearchParams({ page: String(page), page_size: "25" });
      if (tier !== "ALL") params.set("risk_tier", tier);
      if (search.trim()) params.set("search", search.trim());
      setData(await apiGet<CustomerList>(`/api/v1/customers?${params}`));
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Could not reach the server.",
      );
    }
  }, [page, tier, search]);

  useEffect(() => {
    const t = setTimeout(load, search ? 300 : 0);
    return () => clearTimeout(t);
  }, [load, search]);

  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;
  const scoredTotal = data
    ? data.tier_counts.HIGH + data.tier_counts.MEDIUM + data.tier_counts.LOW
    : 0;

  return (
    <div>
      <div className="mb-6 flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-bold">Customers</h1>
          <p className="mt-1 text-sm text-slate-400">
            Ranked by churn risk — most at-risk first.
          </p>
        </div>
        <Link
          href="/dashboard/upload"
          className="rounded-lg bg-sky-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-sky-400"
        >
          + Import customers
        </Link>
      </div>

      <ErrorNote message={error} />

      {data && (
        <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
          <StatCard
            label="At high risk"
            value={String(data.tier_counts.HIGH)}
            accent="text-red-400"
            sub={`of ${scoredTotal} scored customers`}
          />
          <StatCard
            label="Medium risk"
            value={String(data.tier_counts.MEDIUM)}
            accent="text-amber-400"
          />
          <StatCard
            label="Low risk"
            value={String(data.tier_counts.LOW)}
            accent="text-emerald-400"
          />
          <StatCard
            label="Revenue at risk"
            value={`$${data.revenue_at_risk.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
            accent="text-red-400"
            sub="monthly, across high-risk customers"
          />
        </div>
      )}

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="flex rounded-lg border border-slate-700 p-0.5">
          {TIER_FILTERS.map((t) => (
            <button
              key={t}
              onClick={() => {
                setTier(t);
                setPage(1);
              }}
              className={`rounded-md px-3 py-1.5 text-xs font-semibold transition ${
                tier === t
                  ? "bg-slate-700 text-slate-100"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {t === "ALL" ? "All" : t}
            </button>
          ))}
        </div>
        <input
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(1);
          }}
          placeholder="Search by ID, name, or email…"
          className="w-72 rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm outline-none focus:border-sky-500"
        />
      </div>

      {!data && !error && <Spinner />}

      {data && data.total === 0 && (
        <div className="rounded-xl border border-dashed border-slate-700 py-16 text-center">
          <p className="text-slate-400">No customers yet.</p>
          <p className="mt-1 text-sm text-slate-500">
            <Link href="/dashboard/upload" className="text-sky-400 hover:underline">
              Import a CSV
            </Link>{" "}
            from your billing system to see churn risk for every customer.
          </p>
        </div>
      )}

      {data && data.total > 0 && (
        <>
          <div className="overflow-hidden rounded-xl border border-slate-700/60">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700/60 bg-slate-800/60 text-left text-xs uppercase tracking-wide text-slate-400">
                  <th className="px-4 py-3">Customer</th>
                  <th className="px-4 py-3">Churn probability</th>
                  <th className="px-4 py-3">Risk</th>
                  <th className="px-4 py-3">Tenure</th>
                  <th className="px-4 py-3">Monthly</th>
                  <th className="px-4 py-3">Contract</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((c) => (
                  <tr
                    key={c.id}
                    onClick={() => setSelected(c.id)}
                    className="cursor-pointer border-b border-slate-800/60 transition last:border-0 hover:bg-slate-800/40"
                  >
                    <td className="px-4 py-3">
                      <p className="font-medium text-slate-200">
                        {c.name || c.external_id}
                      </p>
                      <p className="text-xs text-slate-500">
                        {c.name ? c.external_id : c.email}
                      </p>
                    </td>
                    <td className="px-4 py-3">
                      <ProbabilityBar value={c.churn_probability} />
                    </td>
                    <td className="px-4 py-3">
                      <RiskPill tier={c.risk_tier} />
                    </td>
                    <td className="px-4 py-3 text-slate-300">{c.tenure} mo</td>
                    <td className="px-4 py-3 tabular-nums text-slate-300">
                      ${c.monthly_charges.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-slate-300">{c.contract || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 flex items-center justify-between text-sm text-slate-400">
            <p>
              {data.total} customer{data.total === 1 ? "" : "s"}
              {tier !== "ALL" ? ` at ${tier} risk` : ""}
            </p>
            <div className="flex items-center gap-2">
              <button
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
                className="rounded-lg border border-slate-700 px-3 py-1.5 transition hover:bg-slate-800 disabled:opacity-40"
              >
                ← Prev
              </button>
              <span>
                {page} / {totalPages}
              </span>
              <button
                disabled={page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
                className="rounded-lg border border-slate-700 px-3 py-1.5 transition hover:bg-slate-800 disabled:opacity-40"
              >
                Next →
              </button>
            </div>
          </div>
        </>
      )}

      {selected !== null && (
        <CustomerDrawer customerId={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}
