# How to record the demo GIF

The README embeds `docs/demo.gif`. Replace it with a real recording of the live
dashboard so the project shows something the moment a recruiter opens it.

## Tool (Windows)

Use **[ScreenToGif](https://www.screentogif.com/)** — free, records a screen region
straight to an optimized `.gif`.

## What to capture (~15–25 seconds, keep it tight)

Open the live Space (or run locally) and record this flow:

1. Fill in a customer's details (e.g. a month-to-month, fiber, no-support customer).
2. Click **Predict** and show the **churn probability** and **risk tier**.
3. Highlight the **top-3 SHAP drivers** rendered in plain English.
4. (Optional) Run a **batch prediction** to show several customers ranked by risk.

## Keep the file small

- Target **under ~8 MB** so it loads fast on GitHub (crop to the app, ~12–15 fps).
- In ScreenToGif use *File → Save as → GIF* and enable the built-in optimizer, or
  export MP4 and convert. Save it as **`docs/demo.gif`** (exact path/name).

## Then

```bash
git add docs/demo.gif && git commit -m "docs: add live demo GIF" && git push
```

Also update the **▶️ Try it live** link at the top of the README with your real
Hugging Face Space URL.
