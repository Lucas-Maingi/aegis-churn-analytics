---
title: Aegis Churn Analytics
emoji: 🔮
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Aegis — Churn Intelligence for Subscription Businesses (Live Demo)

This Space runs the **full multi-tenant SaaS**: the Next.js dashboard and the
FastAPI scoring engine in a single container.

**Try it in two clicks:** open the app, press **"Use demo account"** on the
login page (`demo@aegis.app` / `aegis-demo-2026`), and you'll land on a
demo ISP with 400 customers already scored for churn risk and ~300 outcomes
recorded. From there you can:

- browse the ranked at-risk customer list, each with plain-English reasons;
- send one-click retention offers (recorded as simulated sends in this demo);
- record what actually happened to a customer (churned / retained);
- open the **Model** page to see how the model performed against those
  outcomes, and retrain a tenant-specific model that gets promoted only if it
  beats the base model on held-out data.

You can also create your own organization and import a customer CSV — column
mapping is automatic for common billing-export headers.

- **Source, architecture & benchmarks:** https://github.com/Lucas-Maingi/aegis-churn-analytics
- **Base model:** tuned XGBoost (ROC-AUC 0.847) on the IBM Telco churn dataset,
  with SHAP-based per-customer explanations.
- **API docs:** `/docs` on this Space.

> Demo storage is ephemeral: organizations you create here reset when the
> Space restarts. The production configuration points `DATABASE_URL` at
> Postgres instead.
