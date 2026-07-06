"use client";

import { useEffect, useState } from "react";
import { apiGet, ApiError, OutreachMessage } from "@/lib/api";
import { ErrorNote, Spinner } from "@/components/ui";

const STATUS_STYLES: Record<string, string> = {
  sent: "border-emerald-500/50 bg-emerald-500/15 text-emerald-300",
  simulated: "border-sky-500/50 bg-sky-500/15 text-sky-300",
  failed: "border-red-500/50 bg-red-500/15 text-red-300",
};

export default function OutreachPage() {
  const [messages, setMessages] = useState<OutreachMessage[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    apiGet<OutreachMessage[]>("/api/v1/outreach/history?limit=100")
      .then(setMessages)
      .catch((err) =>
        setError(err instanceof ApiError ? err.message : "Could not load history."),
      );
  }, []);

  return (
    <div className="max-w-4xl">
      <h1 className="text-2xl font-bold">Outreach log</h1>
      <p className="mt-1 mb-6 text-sm text-slate-400">
        Every retention message sent from the customer view.
      </p>

      <ErrorNote message={error} />
      {!messages && !error && <Spinner />}

      {messages && messages.length === 0 && (
        <div className="rounded-xl border border-dashed border-slate-700 py-16 text-center">
          <p className="text-slate-400">No outreach yet.</p>
          <p className="mt-1 text-sm text-slate-500">
            Open a high-risk customer and send them an offer — it will show up
            here.
          </p>
        </div>
      )}

      {messages && messages.length > 0 && (
        <div className="overflow-hidden rounded-xl border border-slate-700/60">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700/60 bg-slate-800/60 text-left text-xs uppercase tracking-wide text-slate-400">
                <th className="px-4 py-3">Customer</th>
                <th className="px-4 py-3">Subject</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">When</th>
              </tr>
            </thead>
            <tbody>
              {messages.map((m) => (
                <tr
                  key={m.id}
                  className="border-b border-slate-800/60 last:border-0"
                >
                  <td className="px-4 py-3">
                    <p className="font-medium text-slate-200">
                      {m.customer_name || m.customer_external_id}
                    </p>
                    <p className="text-xs text-slate-500">
                      {m.customer_external_id}
                    </p>
                  </td>
                  <td className="max-w-md truncate px-4 py-3 text-slate-300">
                    {m.subject}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-block rounded-full border px-3 py-0.5 text-xs font-semibold ${STATUS_STYLES[m.status] ?? "border-slate-600 text-slate-400"}`}
                    >
                      {m.status}
                    </span>
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-slate-400">
                    {new Date(m.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
