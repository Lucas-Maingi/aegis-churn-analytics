/**
 * API client for the Aegis FastAPI backend.
 * The JWT session token lives in localStorage under "aegis_token".
 */

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const TOKEN_KEY = "aegis_token";
const ORG_KEY = "aegis_org";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function getOrgName(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem(ORG_KEY) || "";
}

export function saveSession(token: string, orgName: string) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(ORG_KEY, orgName);
}

export function clearSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(ORG_KEY);
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (res.status === 401 && typeof window !== "undefined") {
    clearSession();
    window.location.href = "/login";
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : "Something went wrong. Please try again.";
    throw new ApiError(res.status, detail);
  }
  return body as T;
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  return handle<T>(res);
}

export async function apiPost<T>(path: string, payload: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getToken()}`,
    },
    body: JSON.stringify(payload),
  });
  return handle<T>(res);
}

export async function apiUpload<T>(
  path: string,
  file: File,
  fields: Record<string, string> = {},
): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  for (const [key, value] of Object.entries(fields)) form.append(key, value);
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { Authorization: `Bearer ${getToken()}` },
    body: form,
  });
  return handle<T>(res);
}

/** Unauthenticated call for login/signup. */
export async function apiAuth<T>(path: string, payload: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : "Something went wrong. Please try again.";
    throw new ApiError(res.status, detail);
  }
  return body as T;
}

// ── Shared response types ───────────────────────────────────────────────────

export interface Explanation {
  feature_name: string;
  shap_value: number;
  feature_value: number | string;
  direction: string;
  plain_english: string;
}

export interface CustomerSummary {
  id: number;
  external_id: string;
  name: string;
  email: string;
  tenure: number;
  monthly_charges: number;
  contract: string;
  churn_probability: number | null;
  risk_tier: "HIGH" | "MEDIUM" | "LOW" | null;
}

export interface CustomerDetail extends CustomerSummary {
  features: Record<string, string | number>;
  explanations: Explanation[] | null;
  scored_at: string | null;
}

export interface CustomerList {
  items: CustomerSummary[];
  total: number;
  page: number;
  page_size: number;
  tier_counts: { HIGH: number; MEDIUM: number; LOW: number };
  revenue_at_risk: number;
}

export interface OutreachTemplate {
  key: string;
  label: string;
  description: string;
  subject: string;
  body: string;
}

export interface OutreachMessage {
  id: number;
  customer_id: number;
  customer_external_id: string;
  customer_name: string;
  template_key: string;
  channel: string;
  subject: string;
  status: string;
  created_at: string;
}
