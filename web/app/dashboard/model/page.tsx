"use client";

import { useCallback, useEffect, useState } from "react";
import { apiGet, apiPost, ApiError, RetrainResult, Scorecard } from "@/lib/api";
import { ErrorNote, Spinner, StatCard } from "@/components/ui";

function pct(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : `${(v * 100).toFixed(0)}%`;
}

export default function ModelPage() {
  const [card, setCard] = useState<Scorecard | null>(null);
  const [error, setError] = useState("");
  const [retraining, setRetraining] = useState(false);
  const [result, setResult] = useState<RetrainResult | null>(null);

  const load = useCallback(async () => {
    setError("");
    try {
      setCard(await apiGet<Scorecard>("/api/v1/model/scorecard"));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load the scorecard.");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function retrain() {
    setRetraining(true);
    setResult(null);
    setError("");
    try {
      const res = await apiPost<RetrainResult>("/api/v1/model/retrain", {});
      setResult(res);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Retrain failed.");
    } finally {
      setRetraining(false);
    }
  }

  const tierOrder = ["HIGH", "MEDIUM", "LOW"] as const;

  return (
    <div className="max-w-4xl">
      <div className="mb-1 flex items-center gap-3">
        <h1 className="text-2xl font-bold">Model performance</h1>
        {card && (
          <span
            className={`rounded-full border px-3 py-0.5 text-xs font-semibold ${
              card.active_model === "tenant"
                ? "border-indigo-500/50 bg-indigo-500/15 text-indigo-300"
                : "border-slate-600 bg-slate-700/40 text-slate-300"
            }`}
          >
            {card.active_model === "tenant"
              ? "Your custom model"
              : "Base model"}
          </span>
        )}
      </div>
      <p className="mb-6 text-sm text-slate-400">
        How the scoring model has held up against the outcomes you&apos;ve
        recorded — and, once you have enough, retrain it on your own results.
      </p>

      <ErrorNote message={error} />
      {!card && !error && <Spinner />}

      {card && card.n_outcomes === 0 && (
        <div className="rounded-xl border border-dashed border-slate-700 py-16 text-center">
          <p className="text-slate-400">No recorded outcomes yet.</p>
          <p className="mt-1 text-sm text-slate-500">
            Open a customer and mark whether they churned or were retained.
            Once outcomes accumulate, this page scores the model against reality.
          </p>
        </div>
      )}

      {card && card.n_outcomes > 0 && (
        <>
          <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard
              label="Recorded outcomes"
              value={String(card.n_outcomes)}
              sub={`${card.n_churned} churned · ${card.n_retained} retained`}
            />
            <StatCard
              label="Caught actual churners"
              value={pct(card.recall)}
              accent="text-emerald-400"
              sub="recall — of those who churned"
            />
            <StatCard
              label="High-risk was right"
              value={pct(card.high_risk_precision)}
              accent="text-sky-400"
              sub="of those flagged, % churned"
            />
            <StatCard
              label="Overall accuracy"
              value={pct(card.accuracy)}
              sub="predictions vs. reality"
            />
          </div>

          {/* Per-tier reality check */}
          {card.tier_actual_churn && (
            <div className="mb-6 rounded-xl border border-slate-700/60 bg-slate-800/40 p-5">
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
                Did the risk tiers hold up?
              </h2>
              <div className="space-y-3">
                {tierOrder.map((tier) => {
                  const t = card.tier_actual_churn![tier];
                  if (!t || t.total === 0) return null;
                  const rate = t.rate ?? 0;
                  const color =
                    tier === "HIGH"
                      ? "bg-red-500"
                      : tier === "MEDIUM"
                        ? "bg-amber-500"
                        : "bg-emerald-500";
                  return (
                    <div key={tier} className="flex items-center gap-3">
                      <span className="w-16 text-sm font-semibold text-slate-300">
                        {tier}
                      </span>
                      <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-700">
                        <div
                          className={`h-full rounded-full ${color}`}
                          style={{ width: `${rate * 100}%` }}
                        />
                      </div>
                      <span className="w-32 text-right text-sm tabular-nums text-slate-300">
                        {(rate * 100).toFixed(0)}% churned ({t.churned}/{t.total})
                      </span>
                    </div>
                  );
                })}
              </div>
              <p className="mt-3 text-xs text-slate-500">
                A well-calibrated model shows a much higher actual churn rate in
                HIGH than in LOW.
              </p>
            </div>
          )}

          {/* Retrain */}
          <div className="rounded-xl border border-slate-700/60 bg-slate-800/40 p-5">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
              Train a model on your own data
            </h2>
            <p className="mt-2 text-sm text-slate-400">
              Your customers may churn for reasons the base model (trained on a
              public telecom dataset) hasn&apos;t seen. Retraining fits a model
              to <em>your</em> recorded outcomes and promotes it only if it beats
              the base model on held-out data.
            </p>

            {card.validated_improvement && (
              <div className="mt-4 rounded-lg border border-indigo-500/40 bg-indigo-500/10 p-4 text-sm">
                <p className="font-semibold text-indigo-200">
                  Your custom model is live.
                </p>
                <p className="mt-1 text-slate-300">
                  Validated on {card.validated_improvement.n_eval} held-out
                  customers: AUC{" "}
                  <span className="font-semibold text-slate-400">
                    {card.validated_improvement.base_auc?.toFixed(3)}
                  </span>{" "}
                  (base) →{" "}
                  <span className="font-semibold text-emerald-300">
                    {card.validated_improvement.tenant_auc?.toFixed(3)}
                  </span>{" "}
                  (yours).
                </p>
              </div>
            )}

            {result && (
              <div
                className={`mt-4 rounded-lg border p-4 text-sm ${
                  result.promoted
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
                    : "border-slate-600 bg-slate-700/30 text-slate-300"
                }`}
              >
                {result.detail}
              </div>
            )}

            <div className="mt-4 flex items-center gap-3">
              <button
                onClick={retrain}
                disabled={!card.can_retrain || retraining}
                className="rounded-lg bg-indigo-500 px-5 py-2 text-sm font-semibold text-white transition hover:bg-indigo-400 disabled:opacity-50"
              >
                {retraining ? "Training…" : "Retrain on my data"}
              </button>
              {!card.can_retrain && (
                <span className="text-xs text-slate-500">{card.retrain_hint}</span>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
