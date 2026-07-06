"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { apiUpload, ApiError } from "@/lib/api";
import { ErrorNote } from "@/components/ui";

interface PreviewResponse {
  columns: string[];
  row_count: number;
  sample_rows: Record<string, string>[];
  suggested_mapping: Record<string, string>;
}

interface UploadResponse {
  imported: number;
  updated: number;
  scored: number;
  skipped: number;
  errors: string[];
}

/** Model fields the uploader can map, in display order. */
const MAPPABLE_FIELDS: {
  field: string;
  label: string;
  required?: boolean;
  hint: string;
}[] = [
  { field: "external_id", label: "Customer ID", required: true, hint: "Unique subscriber/account ID" },
  { field: "name", label: "Name", hint: "For personalized outreach" },
  { field: "email", label: "Email", hint: "Where retention offers are sent" },
  { field: "tenure", label: "Months as customer", required: true, hint: "e.g. 14" },
  { field: "MonthlyCharges", label: "Monthly charge", required: true, hint: "e.g. 45.00" },
  { field: "Contract", label: "Contract type", hint: "monthly / 1 year / 2 year" },
  { field: "InternetService", label: "Service type", hint: "fiber / dsl / none" },
  { field: "PaymentMethod", label: "Payment method", hint: "mpesa, card, direct debit…" },
  { field: "TotalCharges", label: "Total billed", hint: "Lifetime revenue (optional)" },
  { field: "PaperlessBilling", label: "Paperless billing", hint: "yes / no" },
  { field: "TechSupport", label: "Support add-on", hint: "yes / no" },
  { field: "OnlineSecurity", label: "Security add-on", hint: "yes / no" },
  { field: "StreamingTV", label: "TV add-on", hint: "yes / no" },
  { field: "PhoneService", label: "Phone service", hint: "yes / no" },
];

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function selectFile(f: File | null) {
    setFile(f);
    setPreview(null);
    setResult(null);
    setError("");
    if (!f) return;
    setBusy(true);
    try {
      const res = await apiUpload<PreviewResponse>(
        "/api/v1/customers/upload/preview",
        f,
      );
      setPreview(res);
      setMapping(res.suggested_mapping);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not read the file.");
    } finally {
      setBusy(false);
    }
  }

  async function runImport() {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const res = await apiUpload<UploadResponse>("/api/v1/customers/upload", file, {
        mapping: JSON.stringify(mapping),
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Import failed.");
    } finally {
      setBusy(false);
    }
  }

  const missingRequired = MAPPABLE_FIELDS.filter(
    (f) => f.required && !mapping[f.field],
  );

  return (
    <div className="max-w-3xl">
      <h1 className="text-2xl font-bold">Import customers</h1>
      <p className="mt-1 mb-6 text-sm text-slate-400">
        Export your customer list from your billing system as a CSV, upload it
        here, and every customer is scored the moment the import finishes.
      </p>

      <ErrorNote message={error} />

      {/* Step 1 — choose file */}
      <div className="mt-4 rounded-xl border border-slate-700/60 bg-slate-800/40 p-6">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
          1 · Choose your CSV
        </h2>
        <input
          type="file"
          accept=".csv,text/csv"
          onChange={(e) => selectFile(e.target.files?.[0] ?? null)}
          className="block w-full text-sm text-slate-400 file:mr-4 file:rounded-lg file:border-0 file:bg-sky-500 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white hover:file:bg-sky-400"
        />
        {preview && (
          <p className="mt-3 text-sm text-slate-400">
            Found <span className="font-semibold text-slate-200">{preview.row_count}</span>{" "}
            rows and {preview.columns.length} columns
            {Object.keys(preview.suggested_mapping).length > 0 &&
              " — columns matched automatically where possible."}
          </p>
        )}
      </div>

      {/* Step 2 — map columns */}
      {preview && !result && (
        <div className="mt-4 rounded-xl border border-slate-700/60 bg-slate-800/40 p-6">
          <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide text-slate-400">
            2 · Match your columns
          </h2>
          <p className="mb-4 text-sm text-slate-500">
            Tell Aegis which of your columns holds each detail. Anything left
            unmatched uses a sensible default.
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            {MAPPABLE_FIELDS.map(({ field, label, required, hint }) => (
              <div key={field}>
                <label className="mb-1 block text-sm text-slate-300">
                  {label}
                  {required && <span className="text-red-400"> *</span>}
                  <span className="ml-2 text-xs text-slate-500">{hint}</span>
                </label>
                <select
                  value={mapping[field] ?? ""}
                  onChange={(e) =>
                    setMapping((m) => {
                      const next = { ...m };
                      if (e.target.value) next[field] = e.target.value;
                      else delete next[field];
                      return next;
                    })
                  }
                  className="w-full rounded-lg border border-slate-600 bg-slate-900/60 px-3 py-2 text-sm outline-none focus:border-sky-500"
                >
                  <option value="">— not in my file —</option>
                  {preview.columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>

          <button
            onClick={runImport}
            disabled={busy || missingRequired.length > 0}
            className="mt-6 w-full rounded-lg bg-sky-500 py-2.5 text-sm font-semibold text-white transition hover:bg-sky-400 disabled:opacity-50"
          >
            {busy
              ? "Importing & scoring…"
              : missingRequired.length > 0
                ? `Match required columns first: ${missingRequired.map((f) => f.label).join(", ")}`
                : `Import & score ${preview.row_count} customers`}
          </button>
        </div>
      )}

      {/* Step 3 — result */}
      {result && (
        <div className="mt-4 rounded-xl border border-emerald-500/40 bg-emerald-500/10 p-6">
          <h2 className="text-lg font-semibold text-emerald-300">
            Import complete
          </h2>
          <p className="mt-2 text-sm text-slate-300">
            {result.imported} new customer{result.imported === 1 ? "" : "s"} added
            {result.updated > 0 && `, ${result.updated} updated`}
            {result.skipped > 0 && `, ${result.skipped} skipped`} —{" "}
            <span className="font-semibold">{result.scored} scored</span> for churn
            risk.
          </p>
          {result.errors.length > 0 && (
            <ul className="mt-2 list-inside list-disc text-xs text-amber-300">
              {result.errors.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          )}
          <button
            onClick={() => router.push("/dashboard")}
            className="mt-4 rounded-lg bg-sky-500 px-5 py-2 text-sm font-semibold text-white transition hover:bg-sky-400"
          >
            View ranked customers →
          </button>
        </div>
      )}
    </div>
  );
}
