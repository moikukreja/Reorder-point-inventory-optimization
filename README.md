# AI Reorder Point Control Tower

*AI in Operations: AI Applications in Reorder Point & Inventory Management (Exercise)*

An interactive dashboard that enhances the classic reorder-point calculator with:
1. **Demand forecasting:** Holt-Winters, Ridge regression, an LSTM and an adaptive ensemble, with new data added and forecasts updated live.
2. **Dynamic safety stock:** the slide formula `SS = Z·√(σ_D²·L + D̄²·σ_L²)`, plus Monte Carlo risk simulation.
3. **Bayesian optimisation** of the reorder point and order quantity under constraints.

Everything runs in the browser (Web Workers). The site ships **no data**: you load your own prepared file, or use the synthetic demo.

## Repository layout

```
notebooks/01_prepare_and_validate.ipynb   data cleaning, subset export, Python reference implementation
app/                                      React + TypeScript dashboard (Vite)
  src/core/        forecasting, safety stock, Monte Carlo, simulation, GP / Bayesian optimisation
  src/engine/      Web Workers (analysis, optimiser)
  src/pages/ ui/   dashboard pages and components
  public/models/   pre-trained LSTM weights (the only precomputed artefact)
  tests/unit/      Vitest (68 tests)   tests/e2e/  Playwright (10 tests)
docs/            implementation plan, test report
netlify.toml     Netlify build and security headers
data/            NOT committed (git-ignored): raw Kaggle files and derived subset
```

## Run locally

Requirements: Node 20+ (and Python 3.11 with pandas/scipy/jupyter for the notebook).

```bash
# 1) Prepare data once: put the Kaggle "Store Sales" files in data/raw/favorita, then
jupyter nbconvert --to notebook --execute --inplace notebooks/01_prepare_and_validate.ipynb
# 2) Dashboard
cd app
npm install
npm run dev            # http://localhost:5173 → "Load local dev dataset"
```

Other scripts (in `app/`):

| Script | What it does |
|---|---|
| `npm test` | Unit and integration tests (real-data tests skip without `data/derived`) |
| `npm run build && npm run test:e2e` | End-to-end tests on the production build |
| `npm run typecheck` | Strict TypeScript check |
| `npm run pretrain` | Re-train the 9 LSTMs (writes `public/models/`) |
| `npm run verify:models` / `verify:risk` / `verify:optim` | Verification reports on real data |

## Deploy to Netlify (free tier)

1. **Push to GitHub.** `data/` is git-ignored, so no course data is uploaded. Check with `git status` before the first commit: nothing under `data/` should appear.
2. In Netlify: **Add new site → Import an existing project → GitHub**, and pick the repository.
3. The build settings are read from `netlify.toml`: base `app`, command `npm ci && npm run build`, publish `dist`, Node 20. No environment variables are needed. Click **Deploy**.
4. Open the site. It shows the load screen. Drop your `data/derived/favorita_subset.json` (processed only in your browser), or click **Explore with synthetic demo data**.
5. Optional: smoke-test the live site from your machine.
   ```bash
   cd app && BASE_URL=https://<your-site>.netlify.app npx playwright test
   ```

Netlify serves static files only: no functions, no database. Security headers (a strict CSP that allows WebAssembly for TF.js) and caching are set in `netlify.toml`.

## Data handling

The exercise rule is "No sharing of data with peers allowed", and the Kaggle competition rules also restrict redistribution. So:
- Raw and derived data are never committed or deployed.
- The dashboard processes the file in the visitor's browser and keeps it only in that browser's IndexedDB.
- **Clear local data** (Data page) removes it.

Dataset: Kaggle, *Store Sales – Time Series Forecasting* (Corporación Favorita).
