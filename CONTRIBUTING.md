# Contributing to Aegis

Most contributions land in one of three places: the ingestion layer (new billing-CSV shapes), the model layer, or the retention-offer engine.

## Ground rules

- **Tenant isolation is non-negotiable.** Every query touching customer data is scoped by organization; a new endpoint or job that can read across tenants is a security bug even if no current UI exposes it. Tests for new data paths must include a second-tenant negative case.
- **Explanations come from SHAP, not vibes.** The "why" shown for a customer's risk is computed from the model's actual attribution — never template text keyed off the score. If a feature isn't in the model, it can't appear as a driver.
- **Metrics are measured or absent.** The scorecard shows held-out-split numbers from the actual trained artifact. No aspirational metrics in UI or docs.
- **Retraining is gated, per-tenant.** A tenant retrain uses only that tenant's recorded outcomes, and promotion requires the new model to beat the incumbent on the held-out split — silently swapping in a worse model because it's newer defeats the feedback loop's purpose.

## Adding support for a new billing-CSV shape

Ingestion does column auto-matching and value normalization. To support a new export format:

1. Extend the column-alias tables rather than adding format-specific branches — "MonthlyCharges" / "monthly_fee" / "mrc" should all converge on the same canonical field.
2. Add a small anonymized sample CSV under `docs/` and an ingestion test asserting the mapped output, including the rows that *fail* mapping (unmappable rows must surface in the import report, not vanish).

## Touching the model

- Preprocessing lives in the pipeline artifact, not in ad-hoc code before it — the serving path and the training path must transform identically (that's what the tracked `preprocessor.joblib` is).
- Any model change re-runs the training script and updates `model_metadata.json` (metrics, split, date) in the same commit. The scorecard page reads that file; stale metadata is a lie in the UI.
- Keep the model honest about scope: it's trained on Telco-shaped data. If you add features that only exist in one tenant's CSV, they need a missing-value story for every other tenant.

## Running checks

```bash
pip install -r requirements.txt -r requirements-api.txt
ruff check .
pytest                     # 43 tests: preprocessing, API, SaaS layer
```

CI runs the same. The web dashboard builds with `npm ci && npm run build` in `web/`.

## Commit style

Small commits, present tense, explain *why* when the diff can't.
