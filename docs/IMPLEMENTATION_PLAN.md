# Implementation Plan: AI-Enhanced Reorder Point & Inventory Dashboard

*AI in Operations: "AI Applications in Reorder Point & Inventory Management, Exercise" (Session II, slide 35)*
*Phase 2 deliverable. No development code has been written.*

---

## 0. Summary

A static React + TypeScript dashboard deployed on Netlify (free tier). All modelling runs **in the user's browser inside Web Workers**: forecasting (including a TensorFlow.js LSTM), Monte Carlo, Bayesian optimisation, and retraining when new data arrives. Netlify only serves files, so there is no server that could hold or leak your data, and the UI stays responsive during heavy computation. A Python notebook does the one-time data preparation and serves as an independent reference implementation for cross-checking every number.

---

## 1. Dataset inspection (Favorita "Store Sales – Time Series Forecasting")

### 1.1 Files and structure

| File | Rows | Content | Use |
|---|---|---|---|
| `train.csv` | 3,000,888 | date, store_nbr, family, **sales** (units), **onpromotion** (number of items on promotion) | Demand history |
| `stores.csv` | 54 | city, state, type (A–E), cluster | Picks the store and its local holidays |
| `oil.csv` | 1,218 | Daily WTI oil price (trading days only) | External factor |
| `holidays_events.csv` | 350 | National, regional and local holidays, transfers, bridges, work days, events | Holiday and event features |
| `transactions.csv` | 83,488 | Store traffic per day | Diagnostic only (not known ahead of time) |
| `test.csv` | 28,512 | Promotions for 16–31 Aug 2017, no sales | Optional "known future promotions" |
| `sample_submission.csv` | n/a | Kaggle submission format | Not used |

- **Coverage:** 2013‑01‑01 to 2017‑08‑15, which is 1,684 days for each of 54 stores × 33 families.
- **Integrity:** no nulls and no duplicate (date, store, family) keys.

### 1.2 Quality issues and the cleaning each needs

| Finding | Evidence | Cleaning |
|---|---|---|
| 25 Dec missing every year | 4 dates absent | Insert as **closed** days (sales = NaN, excluded from fitting). Demand on closed days is zero, not missing, in the inventory simulation. |
| 1 Jan near-zero across all stores | Most stores report 0 | Flag as a closure, handled the same way |
| `onpromotion` is always 0 before 2014‑04‑01 | First non-zero value is 2014‑04‑01 | Promotion-aware models train from **2014‑04‑01** (about 3.4 years, ≈1,230 days). Earlier data is used only for long-run seasonality checks. |
| `oil.csv` has gaps | 43 nulls, and no weekend rows (1,218 of 1,704 calendar days) | Reindex to the calendar, interpolate over time, back-fill the first value |
| Holiday table needs logic | 12 transferred, 38 dates with two or more records, "Work Day" entries | Use non-transferred Holiday, Transfer, Bridge, Additional and Event entries. Match national entries to all stores and regional/local entries to the store's state or city. "Work Day" entries are regular working days. |
| Stores opened late | e.g. 52 (2017‑04), 20/21/22/29/42 (2015) | Avoid these stores. Default stores are open throughout. |
| Fractional sales | Sold by weight: MEATS, POULTRY, PRODUCE, DELI, BREAD | Default families have integer units (BEVERAGES, CLEANING). GROCERY I is 99.8% integer. |
| 2016‑04‑16 earthquake | Known demand shock (relief buying) | Add an event flag and exclude those weeks from error statistics |
| Oil is confounded with trend | corr(sales, oil) = −0.28 for store 44 BEVERAGES, partly because both trend over time | Keep it as a regressor but report its effect honestly as weak and confounded |

### 1.3 Signal check: what makes this dataset fit the exercise

I ran these checks on Store 44 (Quito, type A, the largest store) × BEVERAGES, with a holdout from 2017‑05‑01 to 2017‑08‑15:

- Average demand is 9,013 units/day with a standard deviation of 3,697.
- **Weekly seasonality:** Saturday and Sunday average 12,300–13,300 units, versus 6,500–8,000 on weekdays. The autocorrelation at a 7‑day lag is 0.83.
- **Yearly seasonality:** demand is higher from September to December.
- **Promotions:** average sales by promotion quartile are 6,878 → 8,788 → 9,969 → 10,806 units.
- **Holidays:** national holidays average 10,272 units, versus 8,898 on other days.

Quick model comparison (a sanity check, not the final models):

| Model | MAE (units/day) | WAPE |
|---|---|---|
| Recent 28‑day mean (flat) | 2,790 | 26.2% |
| Seasonal naive (same day last week) | 1,803 | 16.9% |
| Ridge regression (calendar, promotions, holidays, oil, lags) | 1,458 | 13.7% |
| Gradient boosting | 1,294 | 12.1% |

The data has real, learnable structure, so AI forecasting can show a measurable improvement. The first dataset (`retail_store_inventory.csv`) failed exactly this check. It stays out of the project; the extracted copy can be deleted.

### 1.4 Series offered in the dashboard

- **Default:** Store 44 × BEVERAGES.
- **Selectable:** stores {44, 45, 3} × {BEVERAGES, GROCERY I, CLEANING}, which is 9 series. These cover strong and weak promotion effects and two store types.

### 1.5 What the data does not contain: documented assumptions

| Missing | How it is modelled |
|---|---|
| **Supplier lead time** | L ~ **Gamma(mean L̄, sd σ_L)** in days. The defaults are L̄ = 4 days and σ_L = 1.2 days, both stated as assumptions and set with sliders. The gamma distribution is positive and skewed toward late deliveries, like real lead times. It can be switched to lognormal or normal for comparison. The **Add data** panel also accepts observed lead times (order date → receipt date), and L̄ and σ_L are then re-estimated from those observations. This lets lead-time uncertainty learn from data too. |
| **Costs** | Holding cost h (per unit per day), ordering cost K (per order) and stockout penalty p (per unit short) are user inputs with stated defaults. They are needed only for the optimiser's objective. |
| **Inventory position** | Simulated from the policy. The starting inventory is an input. |

### 1.6 Units (kept consistent everywhere)

- D̄ and σ_D are in **units/day**.
- L and σ_L are in **days**.
- The formula Z·√(σ_D²·L + D̄²·σ_L²) then has units √(u²/d²·d + u²/d²·d²). This is only dimensionally exact when everything is on the daily basis, and the codebase enforces that: one period = one day, and every function's signature says so.

---

## 2. System architecture (resolving the Netlify constraint)

```
┌─────────────────────────── Browser (the user's device) ───────────────────────────┐
│  React UI (main thread: rendering only)                                         │
│     │  Comlink RPC (typed)                                                      │
│     ├──► forecast.worker   Holt-Winters · Ridge · LSTM (TF.js) · ensemble       │
│     ├──► risk.worker       Monte Carlo demand during lead time · policy sim     │
│     └──► optimise.worker   Gaussian-process Bayesian optimisation               │
│  IndexedDB: dataset · rows you add · LSTM weights · run history                 │
└─────────────────────────────────────────────────────────────────────────────────┘
          ▲ static files only (HTML/JS/CSS/WASM, pretrained LSTM weights)
   Netlify CDN (free tier): no functions, no database
```

### 2.1 Where each computation runs, and why

| Computation | Runs in | Cost | Justification |
|---|---|---|---|
| Data preparation (3M rows → 9 series) | **Python notebook, offline, once** | ~10 s locally | A 120 MB CSV is too big to ship. The notebook outputs a small JSON file of about 0.5 MB. |
| Seasonal naive, Holt-Winters, Ridge | Browser, forecast worker | < 50 ms | Closed-form or recursive. Instant refits. |
| **LSTM** initial training | **Precomputed at build time** with a Node script using TF.js (the same code and library as the browser) | ~1–3 min once | Avoids a slow first load. Using TF.js on both sides means no Python→JS conversion mismatch. **This is the one precomputed artefact, and the dashboard labels it as such.** |
| LSTM update on new data | Browser, forecast worker, **warm-start fine-tuning** (a few epochs over a recent window) | ~2–10 s | Uses the WebGL backend in the worker if available, otherwise WASM or CPU. A progress bar shows and it can be cancelled. |
| "Retrain LSTM from scratch" button | Browser, forecast worker | ~30–90 s | Proves the browser can train the full model, not only fine-tune it |
| Monte Carlo (100k–200k draws) | Browser, risk worker (typed arrays, seeded random numbers) | ~50–200 ms | |
| Inventory policy simulation (e.g. 365 days × 500 replications) | Browser, risk worker | ~0.3–1 s | Also called by the optimiser as its objective function |
| Bayesian optimisation (≈40 evaluations, each a 200‑replication simulation) | Browser, optimise worker | ~5–15 s, streamed iteration by iteration | |

### 2.2 Why not Netlify Functions?

- Functions are limited to about 10 s of runtime (26 s at most), start slowly when idle, have no GPU, and have a 50 MB bundle limit. `tfjs-node` doesn't fit, and LSTM training would time out.
- A function would also mean sending your data to a server, which works against the slide's "no sharing" note.
- The browser is faster for this problem size and keeps the data on your device.

### 2.3 Honest trade-offs

- Rows you add are stored in **that browser's IndexedDB**. They don't sync across devices or viewers. There is an **Export / Import dataset** button (JSON/CSV) to move them.
- The LSTM starts from precomputed weights. Everything else is computed live, and every later LSTM update happens live.
- Results depend on random draws, so a **fixed seed is shown and editable**, which makes them reproducible.
- On a slow laptop, fine-tuning takes longer. It can be turned off ("Classical models only") without breaking anything.

### 2.4 Data handling ("No sharing of data with peers allowed") ⚠ decision for you

**Never committed or deployed:** raw Kaggle files. `data/raw/` is git-ignored. Kaggle competition rules also restrict redistribution.

The open question is what the *public* site ships:

| Option | What deploys | Pros | Cons |
|---|---|---|---|
| **A (recommended)** | **No data.** On first visit the site asks you to load `favorita_subset.json` (exported by the notebook) or the raw `train.csv` (parsed in a worker). Everything stays on your device in IndexedDB. | No data is ever public, which meets the slide note and the Kaggle rules | Anyone else opening the URL sees an empty "load your data" screen until they supply a file, so you load it before presenting |
| B | Bundles the small derived subset (9 series, about 0.5 MB) | Works for anyone instantly | Redistributes data derived from the competition publicly |

A **"Load sample-shaped synthetic data"** button will also exist for a public walkthrough that uses no real data. It is clearly labelled as synthetic.

---

## 3. Forecasting design (slide step 1)

### 3.1 Features (all known in advance, no leakage)

- Day of week and month (one-hot), day of year (Fourier terms, K = 3)
- **onpromotion**
- National, regional and local **holiday or event** flags, plus a bridge/transfer flag
- **Payday** (15th and month end, which is when public-sector salaries are paid in Ecuador)
- **Oil price** (lagged 1 day)
- Earthquake window flag
- Lags of 7, 14, 28 and 364 days, and a rolling 28‑day mean shifted by 7 days

`transactions` is excluded because it isn't known ahead of time.

### 3.2 Models compared

| # | Model | Slide category | Notes |
|---|---|---|---|
| M0 | Seasonal naive (same day last week) | Benchmark | The scale for MASE |
| M1 | Holt-Winters additive, weekly season, damped trend | Time series | Parameters fitted by minimising one-step SSE (Nelder–Mead). State updated recursively as new data arrives. |
| M2 | Ridge regression on features §3.1 | Regression | Closed form (XᵀX + λI)⁻¹Xᵀy. λ chosen by time-series cross-validation. Captures promotion, holiday and oil effects with readable coefficients. |
| M3 | **LSTM** (TF.js) | Deep learning | Input: a 28‑day window of [scaled sales + exogenous features]. Architecture: 1 × LSTM(32) → Dense(16) → Dense(H), forecasting the next H = 14 days in one pass. Trained on log1p-scaled data with early stopping. |
| M4 | **Adaptive ensemble** | "Adaptive decision-making" | Weights are proportional to 1/MAE over a rolling 28‑day window. They update every time new data arrives, so the system leans toward whichever model is currently best. |

The gradient boosting model from §1.3 is left out of the browser build because it is heavy to train in JS and adds little over Ridge on this data. The notebook keeps it as a reference point.

### 3.3 Evaluation

- **Rolling-origin backtest:** 6 origins, each forecasting 14 days ahead, over 2017‑03‑01 to 2017‑08‑15. Earthquake weeks are excluded.
- **Metrics:** **MAE, RMSE, WAPE, MASE** (relative to seasonal naive) and **Bias** (mean error). MAPE is not used as a headline because it inflates on low days.
- **Uncertainty bands:** split-conformal prediction intervals from out-of-sample residuals, at 80% and 95%. The dashboard reports **empirical coverage** next to the targets.
- A leaderboard and the per-origin errors are shown, and the notebook reproduces the same numbers.

### 3.4 Adding new data and updating forecasts

There are three ways to add data, and all go through the same validated pipeline:

1. **Form:** add one day (date, units sold, items on promotion, holiday flag, oil price optional). Validation covers the date following the last one, values being non-negative, and duplicates.
2. **CSV upload:** a batch of rows. A preview with validation errors is shown before committing.
3. **Live replay:** the last ~45 days are held back as "future" data and can be streamed one day at a time with play, pause and speed controls. This demonstrates continuous updating with real data.

Each new row triggers this pipeline in the worker:
- Append the row to IndexedDB, then recompute features.
- Update the Holt-Winters state (O(1)) and refit Ridge (closed form).
- Fine-tune the LSTM on a recent window. This is debounced, so a batch triggers one fine-tune.
- Score each model on the new actual and update the ensemble weights.
- Recompute the conformal residuals, then D̄, σ_D, safety stock and ROP.

An **update log** records what changed and by how much, for example "ROP 41,230 → 42,010 (+1.9%): σ_D rose after a promotion-day miss".

---

## 4. Dynamic safety stock and Monte Carlo (slide step 2)

### 4.1 The slide formula, implemented exactly

**SS = Z · √( σ_D² · L + D̄² · σ_L² )**, and **ROP = D̄ · L + SS**, where:

- Z = Φ⁻¹(service level), using an inverse normal CDF accurate to about 1e‑9.
- D̄ = mean daily demand over the coming lead-time window.
- σ_D = standard deviation of daily demand.
- L = mean lead time (days).
- σ_L = standard deviation of lead time (days).

The dashboard shows the formula with today's numbers substituted step by step, so it can be checked by hand.

### 4.2 Baseline versus AI-enhanced inputs (the "enhancement")

| | Baseline (the class notebook's approach) | AI-enhanced |
|---|---|---|
| D̄ | Historical average | **Forecast** mean demand over the next L̄ days (accounts for upcoming weekends and promotions) |
| σ_D | Standard deviation of raw historical demand (inflated by the predictable weekly pattern) | Standard deviation of **forecast errors** (only the uncertainty that remains). Recomputed as data arrives. |
| σ_L | Ignored (SS = Z·σ·√L, as in the notebook) | Included, as the slide's formula requires |
| SS / ROP | Fixed | Recomputed after every update, and checked by Monte Carlo |
| Policy choice | Manual sweep | Bayesian optimisation under constraints |

### 4.3 Monte Carlo design (risk worker)

- **Demand during lead time (DDLT):**
  - Draw L ~ Gamma(L̄, σ_L). Allow fractional days by weighting the final day.
  - For each day in the lead time, draw demand = forecast for that day + a resampled forecast error. This is a bootstrap of empirical residuals, so it keeps their fat tails and skew instead of assuming normality.
  - Sum the days. N = 100,000 runs by default.
- **Outputs:**
  - Histogram and CDF of DDLT.
  - **P(stockout) = P(DDLT > ROP)**, with the stockout tail shaded.
  - The service level actually achieved at the formula's ROP.
  - The **Monte Carlo ROP**, which is the SL-quantile of DDLT.
  - Expected units short per cycle.
- **Formula versus Monte Carlo:** a side-by-side comparison. The worked check below shows the formula can under-cover when lead times are skewed. That gap is what "dynamic" safety stock captures.
- **Risk scenarios:** preset buttons ("supplier delay +2 days", "σ_L doubles", "promotion week", "holiday surge") plus custom settings. Each re-runs the Monte Carlo and the policy simulation.
- **Inventory policy simulation:** an (s, Q) continuous-review policy over the forecast horizon with random lead times. It plots on-hand inventory against the ROP line, with order and receipt markers and red stockout markers, and reports fill rate, cycle service level, average inventory, number of orders and total cost.

### 4.4 Verification already done (hand calculation)

With D̄ = 100 u/d, σ_D = 20, L = 5 d, σ_L = 1.5 d and SL = 95% (Z = 1.6449):

- √(20²·5 + 100²·1.5²) = √(2,000 + 22,500) = √24,500 = 156.52
- **SS = 1.6449 × 156.52 = 257.46**, so **ROP = 757.46**

Monte Carlo with 200,000 gamma lead-time draws gives:
- Mean 499.3 (theory: 500).
- Standard deviation 155.7 (theory: 156.5). ✔
- At ROP = 757.46, **P(stockout) = 6.2%**, not 5%. The Monte Carlo 95% ROP is 778.6, which means SS = 278.6.

These exact numbers will be unit tests (§9).

---

## 5. Bayesian optimisation design (slide step 3)

- **Decision variables:**
  - Reorder point **s**, in the range [0.5·D̄L̄, D̄L̄ + 6·σ_DDLT].
  - Order quantity **Q**, in [1, 30] days of demand. Q is optional: "s only" mode gives a 1‑D view that is easy to explain.
- **Objective:** minimise the expected cost per day = h·(average on-hand) + K·(orders per day) + p·(units short per day). It is estimated by the policy simulation with common random numbers. This is a noisy, expensive black box, which is exactly where Bayesian optimisation helps.
- **Constraints** (sliders):
  1. Fill rate ≥ target (default 98%), or cycle service level ≥ target (toggle).
  2. Average inventory value ≤ budget (working capital).
  3. Peak on-hand ≤ warehouse capacity.
  4. Orders per 30 days ≤ maximum.
- **Method:**
  - A Gaussian process with a Matérn‑5/2 kernel, inputs normalised to [0,1], and a noise term. Hyperparameters are fitted by maximising the marginal likelihood (multi-start L-BFGS or Nelder–Mead).
  - Constraint feasibility is modelled with a second GP on the constraint slack.
  - **Constrained Expected Improvement:** EI(x) × P(feasible | x).
  - 8 Latin-hypercube starting points, then about 32 iterations. The acquisition function is maximised by random search plus local refinement.
- **Benchmarks:** random search and a full grid search on the same budget and seed. The dashboard shows Bayesian optimisation reaching within about 1% of the grid optimum with far fewer simulations.
- **Visuals:**
  - Convergence chart (best feasible cost against iteration, for BO, random search and grid).
  - GP surrogate mean and uncertainty over (s, Q), with sampled points, the infeasible region hatched and the optimum starred.
  - The acquisition function at the current iteration.
  - An animated replay of the search.

---

## 6. Sensitivity analysis views

1. **Tornado chart:** change in ROP and SS for ±20% changes in D̄, σ_D, L̄, σ_L and SL, showing which input drives risk the most.
2. **Heatmap:** ROP over (L̄ × σ_L) at the current service level, with the current position marked.
3. **Service-level curve:** SS, expected cost and P(stockout) against SL from 80% to 99.9%. This mirrors the class notebook's trade-off plot, now including σ_L and cost.
4. **Optimal-policy sensitivity:** each step re-runs a warm-started optimisation. For example, "if the penalty cost doubles, the optimal s rises by X". This is the AI replacement for manual sweeps that the slide asks for.
5. **Variance decomposition:** the share of DDLT variance from σ_D²·L versus D̄²·σ_L². It makes clear that lead-time variability usually dominates.

---

## 7. Dashboard pages and visuals

| Page | Purpose | Key visuals and controls |
|---|---|---|
| **1. Control Tower** (overview) | One-screen answer for an operations manager | KPI tiles (baseline vs AI: ROP, SS, P(stockout), fill rate, cost per day, days of cover), a "what changed" feed, series selector, a plain-English recommendation ("Reorder when stock falls to 42,010 units") |
| **2. Data & Live Feed** | Add data and watch it arrive | Data table, add-day form, CSV upload with validation preview, replay player, lead-time observations, export/import, update log |
| **3. Demand Forecast** | Step 1 | Forecast vs actual with 80/95% bands, model leaderboard, ensemble weights over time, rolling accuracy, promotion/holiday/day-of-week effects, residual diagnostics, LSTM fine-tune progress |
| **4. Safety Stock & Risk** | Step 2 | Live formula with substituted values, lead-time distribution, Monte Carlo DDLT histogram with stockout tail, formula vs Monte Carlo panel, risk scenario buttons, inventory sawtooth vs ROP |
| **5. Bayesian Optimiser** | Step 3 | Constraint panel, convergence chart, surrogate surface (3D WebGL, lazy-loaded, with a 2D heatmap fallback), acquisition plot, optimum card |
| **6. Sensitivity** | AI-driven sensitivity analysis | Tornado chart, L×σ_L heatmap, service-level curve, optimal-policy sensitivity, variance decomposition |
| **7. Baseline vs AI** | The objective's "enhancement" | Backtest of both policies on Mar–Aug 2017 with the same simulated lead times: stockouts, fill rate, average inventory, total cost, side-by-side sawtooth charts |
| **8. Method & Assumptions** | Honesty and traceability | Architecture diagram, the "where it runs" table, assumptions, units, the precomputed-LSTM note, the traceability table |

**Global controls** (in a sticky side panel): service level, L̄, σ_L, lead-time distribution, h/K/p costs, constraint targets and the random seed.

Every chart has an **"What this means"** caption in plain language, generated from the live numbers. For example: "At 95% service you'll run out about once every 20 replenishment cycles; holding 3,100 extra units costs ≈ $X/day."

---

## 8. Visual design language

- **Theme:** a night-shift distribution centre or control tower. The background is deep graphite-navy (#0B1018 → #111A27) with a faint isometric rack-grid pattern.
- **Glass panels:** semi-transparent surfaces (rgba(255,255,255,0.04–0.07)), `backdrop-filter: blur(14px)`, 1 px hairline borders and a soft inner glow.
- **Semantic colours** (validated for contrast):

  | Colour | Hex | Used for |
  |---|---|---|
  | Cyan | #38BDF8 | Actual demand |
  | Teal | #2DD4BF | AI forecast and bands |
  | Hazard amber | #F5B83D | ROP / safety lines (like warehouse floor tape) |
  | Signal red | #F43F5E | Stockouts |
  | Slate | #94A3B8 | Baseline |
  | Violet | #A78BFA | Optimiser |

- **Type:** Inter for UI text and JetBrains Mono (tabular figures) for numbers.
- **Motion (Framer Motion), always purposeful:**
  - Panels fade in on page change, and KPI numbers count up only when their values change.
  - The replay animates each new point onto the chart.
  - The optimiser convergence chart draws progressively.
  - Animation is off when `prefers-reduced-motion` is set.
- **Responsive:** 12-column grid on desktop, 2 columns on tablet, 1 column on phone. The controls panel becomes a drawer on small screens.
- **Accessibility:** WCAG AA contrast, keyboard-reachable controls, and a data-table alternative for each chart.

---

## 9. Technology and folder structure

**Stack:**
- Vite 5, React 18, TypeScript (strict)
- Tailwind CSS, Framer Motion, Apache ECharts (canvas, handles bands, heatmaps and large series)
- Zustand (state), Comlink (worker RPC), TensorFlow.js (LSTM; WebGL/WASM backends)
- idb-keyval (IndexedDB), PapaParse (CSV in a worker)
- @react-three/fiber + drei, lazy-loaded for the 3D surrogate surface only
- Vitest for unit tests, Playwright for end-to-end tests

```
Lab 2/
├─ notebooks/
│  └─ 01_prepare_and_validate.ipynb   # cleaning, EDA, subset export, reference calcs
├─ data/                              # git-ignored (raw Kaggle files never committed)
│  └─ raw/favorita/…
├─ app/
│  ├─ index.html · vite.config.ts · tailwind.config.ts · tsconfig.json
│  ├─ scripts/pretrain-lstm.ts        # build-time TF.js pretraining → public/models/
│  ├─ public/models/lstm-<series>/    # model.json + weights.bin (precomputed)
│  ├─ src/
│  │  ├─ core/                        # pure, tested maths (no React)
│  │  │  ├─ stats.ts        (invNormCdf, quantiles, rng/seeding, gamma sampler)
│  │  │  ├─ safetyStock.ts  (slide formula, ROP, baseline formula)
│  │  │  ├─ features.ts     (calendar/holiday/promo/oil features)
│  │  │  ├─ models/         (seasonalNaive, holtWinters, ridge, lstm, ensemble)
│  │  │  ├─ metrics.ts      (MAE, RMSE, WAPE, MASE, bias, coverage)
│  │  │  ├─ conformal.ts
│  │  │  ├─ monteCarlo.ts   (DDLT, scenario engine)
│  │  │  ├─ policySim.ts    ((s,Q) inventory simulation, costs, service)
│  │  │  ├─ gp.ts · bayesOpt.ts (GP, constrained EI, LHS)
│  │  │  └─ sensitivity.ts
│  │  ├─ workers/  forecast.worker.ts · risk.worker.ts · optimise.worker.ts
│  │  ├─ data/     schema.ts · loader.ts · store(IndexedDB).ts · validate.ts
│  │  ├─ state/    useAppStore.ts
│  │  ├─ components/  (GlassPanel, KpiTile, ChartFrame, Explain, ControlsDrawer…)
│  │  ├─ charts/      (ForecastBands, McHistogram, Sawtooth, Convergence, Surface3D, Tornado, Heatmap…)
│  │  ├─ pages/       (ControlTower, DataFeed, Forecast, Risk, Optimiser, Sensitivity, BaselineVsAI, Method)
│  │  └─ App.tsx · main.tsx · theme.css
│  └─ tests/ unit/ (vitest) · e2e/ (playwright)
├─ docs/IMPLEMENTATION_PLAN.md
└─ netlify.toml
```

### Why a Python notebook

1. Turning 120 MB and 3M rows into a 0.5 MB subset is easiest and most transparent in pandas.
2. It is an **independent reference implementation** in NumPy, SciPy and scikit-learn of the formula, the Monte Carlo, the Ridge and Holt-Winters metrics, and a GP-BO run. Its outputs become golden test fixtures that the TypeScript code must match within tolerance.
3. It provides EDA evidence (seasonality, promotion and holiday effects) for your write-up.

---

## 10. Testing approach

- **Unit tests (Vitest)** in `core/`:
  - The slide formula matches the hand calculation (257.46 / 757.46) to 0.01.
  - invNormCdf at 0.90/0.95/0.99 equals 1.2816/1.6449/2.3263.
  - The gamma sampler's mean and standard deviation are within 1%.
  - The Monte Carlo DDLT mean matches D̄L̄ and its standard deviation matches √(σ_D²L + D̄²σ_L²) when inputs are normal (within 1%).
  - Metrics match fixtures, and Ridge and Holt-Winters match the notebook's MAE within 0.5%.
  - The GP recovers a known 1‑D function, and Bayesian optimisation finds the minimum of a test function (Branin) within tolerance.
  - The policy simulation conserves inventory (start + receipts − sales = end).
- **Golden fixtures:** the notebook writes `tests/fixtures/*.json`.
- **Integration:** add a row, then check that the forecast, SS and ROP change and the update log is written. Upload a CSV with bad rows and check they are rejected with messages.
- **End-to-end (Playwright, against `vite preview` of the production build):**
  - Visit every page and **fail on any console error or warning**.
  - Run the replay for 10 days.
  - Run the optimiser.
  - Check the main thread is never blocked for more than 200 ms during computation (Long Tasks API).
- **Performance budget:** initial JS ≤ 350 KB gzipped (TF.js and three.js lazy-loaded) and Lighthouse score ≥ 90.
- **Manual walk-through** against the traceability table (§12) in Phase 4.

---

## 11. Netlify deployment (free tier)

1. Push to GitHub, without `data/`. I'll ask for the repository link in Phase 4.
2. `netlify.toml`:
   - base `app/`, command `npm run build`, publish `app/dist`
   - Node 20
   - SPA redirect `/* → /index.html 200`
   - Long cache headers for hashed assets and `/models/*`
   - Security headers (CSP allowing `wasm-unsafe-eval` for the TF.js WASM backend)
3. In Netlify: **Add new site → Import from Git → select repo**. The settings are picked up from `netlify.toml`, then **Deploy**.
4. Smoke-test the live URL with the same Playwright suite (`BASE_URL=<netlify url>`).
5. **No environment variables, functions or paid add-ons** are needed.

---

## 12. Traceability: slide requirements → features

| Slide requirement | Dashboard feature | Page | Verified by |
|---|---|---|---|
| Objective: enhance the reorder-point calculator | Baseline (notebook method) vs AI ROP, side-by-side backtest | 1, 7 | Backtest KPIs, e2e |
| Objective: predictive intelligence | Forecast-driven D̄ and σ_D | 3, 4 | Leaderboard metrics |
| Objective: adaptive decision-making | Ensemble reweighting, SS/ROP recomputed after each update, lead-time re-estimation | 2, 3, 4 | Integration test (add row → change) |
| Objective: real-time optimisation | Bayesian optimisation re-run on demand or on data changes, streaming iterations | 5 | e2e, convergence chart |
| 1. ML models (time series, regression, LSTM) | Holt-Winters, Ridge, LSTM (TF.js), plus seasonal naive benchmark and ensemble | 3 | Unit tests and notebook fixtures |
| 1. Historical data | Favorita daily sales 2013–2017 | 2, 3 | Data panel |
| 1. Seasonality | Day-of-week, Fourier yearly and month features, Holt-Winters weekly season | 3 | Effects chart |
| 1. Promotions | `onpromotion` feature and promotion effect chart | 3 | Ridge coefficient, effect chart |
| 1. External factors | Oil price, holidays/events, payday, earthquake flag | 3 | Effect chart |
| 1. Keep the possibility of adding new data | Form, CSV upload, replay stream, lead-time observations, import/export | 2 | Integration and e2e |
| 1. Continuously update forecasts | Incremental Holt-Winters, Ridge refit, LSTM fine-tune, rolling accuracy, update log | 2, 3 | e2e replay test |
| 2. Probabilistic models (demand variability) | Forecast-error distribution, conformal bands | 3, 4 | Coverage metric |
| 2. Lead-time uncertainty | Gamma/lognormal/normal L, σ_L slider, learned from observations | 4 | Unit tests (sampler) |
| 2. Monte Carlo risk scenarios and buffers | DDLT simulation, P(stockout), Monte Carlo ROP, scenario presets, policy simulation | 4 | Unit test vs theory |
| 2. Formula SS = Z·√(σ_D²L + D̄²σ_L²) | Live formula card with substituted values | 4 | Hand-calc unit test |
| 3. Bayesian optimisation instead of manual sweeps | GP and constrained EI over (s, Q), compared with random and grid search | 5 | Branin test, convergence vs grid |
| 3. Optimal reorder points under constraints | Fill-rate, budget, capacity and order-frequency constraints | 5 | Feasibility check in e2e |
| Sensitivity analysis (objective) | Tornado chart, heatmap, SL curve, optimal-policy sensitivity, variance split | 6 | Unit tests |
| Note: no sharing of data | No data deployed (Option A), local IndexedDB, raw files git-ignored | 8 | Build artefact check (no CSV/JSON data in `dist/`) |

---

## 13. Development parts (Phase 3, each gated on your "continue")

1. **Data and models:**
   - The notebook (cleaning, subset export, reference metrics).
   - `core/` stats, features, Holt-Winters, Ridge and metrics, plus the LSTM pretraining script.
   - Verification printouts compared with the notebook.
2. **Safety stock and Monte Carlo:** the formula, DDLT simulation, policy simulation and scenarios, plus hand-calculation and theory checks.
3. **Bayesian optimisation and sensitivity:** GP, constrained EI, benchmarks and sensitivity, plus the Branin test and comparison with grid search.
4. **Dashboard UI:** the design system, the 8 pages and the charts.
5. **Integration:** workers, the new-data pipeline, persistence and the replay. Then Phase 4 (testing and deployment).

---

## 14. Decisions for you at approval

1. **Data on the public site:** Option A (nothing deployed; recommended) or Option B (bundle the small derived subset). See §2.4.
2. **Default series:** Store 44 × BEVERAGES (recommended), or another from §1.4.
3. **Lead-time defaults:** L̄ = 4 days, σ_L = 1.2 days, gamma distribution. Change them if your course used other values.
