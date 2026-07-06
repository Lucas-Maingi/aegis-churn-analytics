"use client";

import { useEffect, useState } from "react";
import {
  apiGet,
  apiPost,
  ApiError,
  CustomerDetail,
  OutreachTemplate,
} from "@/lib/api";
import { ErrorNote, ProbabilityBar, RiskPill, Spinner } from "@/components/ui";

interface SendResponse {
  detail: string;
  message: { status: string };
}

/**
 * Slide-over panel: full customer picture — churn probability, the
 * plain-English "why", and one-click retention actions.
 */
export default function CustomerDrawer({
  customerId,
  onClose,
}: {
  customerId: number;
  onClose: () => void;
}) {
  const [customer, setCustomer] = useState<CustomerDetail | null>(null);
  const [templates, setTemplates] = useState<OutreachTemplate[]>([]);
  const [error, setError] = useState("");
  const [sending, setSending] = useState("");
  const [sentNote, setSentNote] = useState("");

  useEffect(() => {
    setCustomer(null);
    setSentNote("");
    setError("");
    Promise.all([
      apiGet<CustomerDetail>(`/api/v1/customers/${customerId}`),
      apiGet<OutreachTemplate[]>("/api/v1/outreach/templates"),
    ])
      .then(([cust, tpls]) => {
        setCustomer(cust);
        setTemplates(tpls);
      })
      .catch((err) =>
        setError(err instanceof ApiError ? err.message : "Failed to load customer."),
      );
  }, [customerId]);

  async function send(templateKey: string) {
    setSending(templateKey);
    setSentNote("");
    setError("");
    try {
      const res = await apiPost<SendResponse>("/api/v1/outreach/send", {
        customer_id: customerId,
        template_key: templateKey,
      });
      setSentNote(res.detail);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to send.");
    } finally {
      setSending("");
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div
        className="absolute inset-0 bg-black/60"
        onClick={onClose}
        data-testid="drawer-backdrop"
      />
      <div className="relative flex h-full w-full max-w-lg flex-col overflow-y-auto border-l border-slate-700 bg-slate-900 p-6 shadow-2xl">
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h2 className="text-xl font-bold">
              {customer?.name || customer?.external_id || "Customer"}
            </h2>
            {customer && (
              <p className="text-sm text-slate-400">
                {customer.external_id}
                {customer.email ? ` · ${customer.email}` : ""}
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="rounded-lg px-2.5 py-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <ErrorNote message={error} />

        {!customer && !error && <Spinner />}

        {customer && (
          <>
            <div className="mb-5 rounded-xl border border-slate-700/60 bg-slate-800/40 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">Churn probability</p>
                  <div className="mt-1">
                    <ProbabilityBar value={customer.churn_probability} />
                  </div>
                </div>
                <RiskPill tier={customer.risk_tier} />
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2 text-sm">
                <div>
                  <p className="text-slate-500">Tenure</p>
                  <p className="text-slate-200">{customer.tenure} mo</p>
                </div>
                <div>
                  <p className="text-slate-500">Monthly</p>
                  <p className="text-slate-200">
                    ${customer.monthly_charges.toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-slate-500">Contract</p>
                  <p className="text-slate-200">{customer.contract || "—"}</p>
                </div>
              </div>
            </div>

            <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-400">
              Why this score
            </h3>
            <div className="mb-6 space-y-2">
              {(customer.explanations ?? []).map((exp, i) => (
                <div
                  key={i}
                  className="rounded-lg border border-slate-700/60 bg-slate-800/30 p-3"
                >
                  <p className="text-sm leading-relaxed text-slate-200">
                    {exp.plain_english}
                  </p>
                  <p
                    className={`mt-1 text-xs font-medium ${
                      exp.direction.startsWith("increases")
                        ? "text-red-400"
                        : "text-emerald-400"
                    }`}
                  >
                    {exp.direction.startsWith("increases") ? "▲" : "▼"}{" "}
                    {exp.direction}
                  </p>
                </div>
              ))}
              {(customer.explanations ?? []).length === 0 && (
                <p className="text-sm text-slate-500">
                  No explanation available — re-import this customer to score them.
                </p>
              )}
            </div>

            <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-400">
              Win them back
            </h3>
            {sentNote && (
              <p className="mb-3 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-4 py-2.5 text-sm text-emerald-300">
                {sentNote}
              </p>
            )}
            <div className="space-y-2">
              {templates.map((tpl) => (
                <div
                  key={tpl.key}
                  className="flex items-center justify-between gap-3 rounded-lg border border-slate-700/60 bg-slate-800/30 p-3"
                >
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-slate-200">
                      {tpl.label}
                    </p>
                    <p className="mt-0.5 text-xs leading-relaxed text-slate-500">
                      {tpl.description}
                    </p>
                  </div>
                  <button
                    onClick={() => send(tpl.key)}
                    disabled={sending !== ""}
                    className="shrink-0 rounded-lg bg-sky-500 px-4 py-1.5 text-sm font-semibold text-white transition hover:bg-sky-400 disabled:opacity-50"
                  >
                    {sending === tpl.key ? "Sending…" : "Send"}
                  </button>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
