# Test report: Phase 4

**Date:** 2026-11-11 · **Build:** production (`vite build`, served by `vite preview`) · **Browser:** Chromium (Playwright) plus manual checks in the desktop app's browser

## Summary

| Suite | Command | Result |
|---|---|---|
| Unit and integration (Vitest) | `npm test` | **68 / 68 passed** (8 files) |
| End-to-end, production build, Netlify CSP injected (Playwright) | `npm run build && npm run test:e2e` | **10 / 10 passed**, with **0 console errors, warnings or CSP violations** |
| Type-check (strict TS) | `npm run typecheck` | **0 errors** |
| Python reference notebook | `jupyter nbconvert --execute` | Runs clean (13 code cells, no errors or warnings) |
| Verification scripts (real data) | `npm run verify:models`, `verify:risk`, `verify:optim`, `npx tsx scripts/verify-replay.ts` | Results below |

Every e2e test fails if the page logs any console error or warning, throws, or violates the production Content-Security-Policy, which is read straight from `netlify.toml`.

## Traceability: slide requirement → feature → how it was verified

| # | Slide requirement | Feature | Verified by | Result |
|---|---|---|---|---|
| O1 | Enhance the reorder-point calculator | Baseline (class notebook) vs AI ROP, simulated forward and **replayed on actual 2017 demand** | `verify-replay.ts`; e2e T2 (Compare page) | Real data, 24 May – 15 Aug 2017: baseline **$502.56/day** (fill 94.48%), AI formula **$262.27** (−48%, fill 97.61%), AI Monte Carlo **$170.70** (−66%, fill 98.83%) |
| O2 | Predictive intelligence | Forecast-driven D̄ and forecast-error σ_D | `engine.test` (formula on live inputs) | ✅ |
| O3 | Adaptive decision-making | Ensemble re-weighting, SS/ROP recomputed on every change | `livefeed.test` ("new day refits"); e2e T4, T5 | ✅ ROP changes on a new day and on settings changes |
| O4 | Real-time optimisation | Bayesian optimiser streaming from a worker | e2e T8 (live "Running… n/40", longest main-thread task < 250 ms) | ✅ (manual check: 43 ms max main-thread stall) |
| 1a | ML models: time series, regression, LSTM | Holt-Winters, Ridge, LSTM (TF.js), seasonal-naive benchmark, adaptive ensemble | `models.test` (TS = Python reference to ≤1.7e-7 WAPE points on all 9 series); `lstm.test`; `verify-models` | Mean WAPE over 9 series: benchmark 16.9%, Holt-Winters 15.0%, Ridge 11.2%, LSTM 11.9%, **ensemble 10.9%** |
| 1b | Historical data, seasonality, promotions, external factors | Weekday and yearly Fourier terms, onpromotion, holidays/events, payday, oil, earthquake flag | `models.test` (feature rows = notebook); Forecast page effect charts | ✅ |
| 1c | Keep the possibility of adding new data | Add-day form, CSV upload with validation preview, replay, import/export, persisted per browser | `livefeed.test` (validation, CSV round-trip, append); e2e **T5** (form, reload persistence, removal), **T6** (bad CSV rejected with line message, valid CSV added) | ✅ |
| 1d | Continuously update forecasts | Refit on every new day; LSTM fine-tuned in the browser every 7 days; live replay | `livefeed.test` (replay restores exactly; fine-tune leaves the frozen backtest model unchanged); e2e **T7** | ✅ LSTM catch-up on 252 days: **5.2 s in the browser** (WASM). From-scratch training: **18 s** |
| 2a | Probabilistic models: demand variability and lead-time uncertainty | Conformal 80/95% bands; gamma/lognormal/normal lead times | `stats.test` (samplers within 1%); `models.test` (conformal = notebook) | 80% band coverage reported per model (ensemble 75%; LSTM 81%) |
| 2b | Monte Carlo risk scenarios and buffers | DDLT simulation (normal / bootstrap / block bootstrap), P(stockout), MC ROP, 6 scenarios, policy simulation | `risk.test` (fixed-L 5.0% ±0.2%; gamma-L SD within 1% of theory; matches notebook 6.28%; simulation bracketed by MC) | ✅ |
| 2c | SS = Z·√(σ_D²·L + D̄²·σ_L²) | Live formula card with substituted values | `risk.test` hand calculation **257.46 / 757.46**; e2e **T3** | ✅ |
| 3a | Bayesian optimisation instead of manual sweeps | GP (Matérn-5/2) + constrained EI; grid and random benchmarks | `optim.test` (Branin < 0.6 in 40 evals, beats random; constrained optimum on boundary); `verify-optim` | Within 1% of a 625-point grid after **12** simulations; out-of-sample feasible |
| 3b | Optimal reorder points under constraints | Fill rate / cycle SL, budget, capacity, order frequency | `verify-optim` out-of-sample check; e2e T8 | BO optimum $98.31/day, all constraints met (baseline + EOQ $282.84 fails fill rate) |
| S | Sensitivity analysis | Tornado, elasticities, L̄×σL heatmap, target vs achieved SL, optimal-policy sensitivity | `optim.test` (elasticities = finite differences to 4 dp); e2e **T9** | ✅ |
| N | "No sharing of data with peers" | No data in the deployed site; local file and IndexedDB only; raw data git-ignored | e2e **T1**; `grep` of `dist/` (no data, no dev route); `.gitignore` | ✅ `dist/` holds only code, LSTM weights and WASM |
| UI | Responsive, no console errors, UI responsive during heavy computation | Workers; mobile drawers | e2e **T10** (375 px: no horizontal scroll on all 8 pages); T2–T9 console-clean; T8 long-task check | ✅ |

## Bugs found by testing and fixed

| Found by | Problem | Fix |
|---|---|---|
| e2e (CSP) | The production CSP blocked fonts that Vite inlines as `data:` URIs. The site would have rendered with fallback fonts and console errors on Netlify | `font-src 'self' data:` |
| Browser | LSTM training stalled in background tabs (TF.js waits on `requestAnimationFrame`) | Worker shim that yields through `setTimeout` |
| Browser | LSTM fine-tuning slow on the pure-JS backend | WebAssembly backend in the worker, with binaries shipped in `/tfjs-wasm/` |
| Unit test | Replay restore overwrote known future oil prices | Replayed days restore only their own value |
| Out-of-sample check | "Peak on-hand" grew with the replication count | 95th percentile of per-run peaks |
| Browser | ECharts heatmap error on value axes; overlapping ROP labels; reload jumped to the home page; stale optimiser results shown | Custom cell renderer; staggered labels; initial page from the URL hash; stale results hidden |

## Not covered or limitations

- Lighthouse was not run. The initial JS is about 325 KB gzipped; three.js (the 3D view) and TF.js load lazily or only inside the worker.
- The e2e suite uses the synthetic demo dataset, so it runs without course data. Real-data behaviour is covered by the unit, integration and verification scripts (which skip automatically without the local data).
- Only Chromium was automated. Firefox and Safari were not tested.
- After deployment, run the same e2e suite against the live URL: `BASE_URL=https://<site>.netlify.app npm run test:e2e`.
