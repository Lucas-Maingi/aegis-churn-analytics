"use client";

/** Small shared UI atoms for the dashboard. */

export function RiskPill({ tier }: { tier: string | null }) {
  if (!tier)
    return (
      <span className="inline-block rounded-full border border-slate-600 px-3 py-0.5 text-xs font-semibold text-slate-400">
        UNSCORED
      </span>
    );
  const styles: Record<string, string> = {
    HIGH: "border-red-500/50 bg-red-500/15 text-red-300",
    MEDIUM: "border-amber-500/50 bg-amber-500/15 text-amber-300",
    LOW: "border-emerald-500/50 bg-emerald-500/15 text-emerald-300",
  };
  return (
    <span
      className={`inline-block rounded-full border px-3 py-0.5 text-xs font-semibold tracking-wide ${styles[tier] ?? ""}`}
    >
      {tier}
    </span>
  );
}

export function ProbabilityBar({ value }: { value: number | null }) {
  if (value === null || value === undefined)
    return <span className="text-slate-500 text-sm">—</span>;
  const color =
    value >= 70 ? "bg-red-500" : value >= 40 ? "bg-amber-500" : "bg-emerald-500";
  return (
    <div className="flex items-center gap-2 min-w-[120px]">
      <div className="h-1.5 w-20 rounded-full bg-slate-700 overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${Math.min(100, value)}%` }}
        />
      </div>
      <span className="text-sm tabular-nums text-slate-200">
        {value.toFixed(1)}%
      </span>
    </div>
  );
}

export function StatCard({
  label,
  value,
  accent,
  sub,
}: {
  label: string;
  value: string;
  accent?: string;
  sub?: string;
}) {
  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-800/40 p-5">
      <p className="text-sm text-slate-400">{label}</p>
      <p className={`mt-1 text-3xl font-bold ${accent ?? "text-slate-100"}`}>
        {value}
      </p>
      {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
    </div>
  );
}

export function Spinner() {
  return (
    <div className="flex justify-center py-12">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-600 border-t-sky-400" />
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  if (!message) return null;
  return (
    <p className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-2.5 text-sm text-red-300">
      {message}
    </p>
  );
}
