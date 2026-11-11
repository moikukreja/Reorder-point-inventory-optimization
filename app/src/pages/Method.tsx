/**
 * Method & assumptions: architecture (where each computation runs and why), units, assumptions,
 * honest limitations, and the traceability table from the slide to dashboard features.
 */
import { useApp, type PageId } from '../state/store';
import { fmtMoney, fmtPct } from '../charts/EChart';
import { DataTable, Note, Panel } from '../ui/primitives';
import { PageHeader } from './common';

const TRACE: { req: string; feature: string; page: PageId; pageLabel: string }[] = [
  { req: 'Objective: enhance the reorder-point calculator', feature: 'Baseline (class notebook) vs AI ROP, simulated side by side', page: 'compare', pageLabel: 'Baseline vs AI' },
  { req: 'Objective: predictive intelligence', feature: 'Forecast-driven D̄ and forecast-error σ_D', page: 'forecast', pageLabel: 'Demand Forecast' },
  { req: 'Objective: adaptive decision-making', feature: 'Ensemble re-weighting by recent accuracy; SS/ROP recomputed on every change', page: 'forecast', pageLabel: 'Demand Forecast' },
  { req: 'Objective: real-time optimisation', feature: 'Bayesian optimiser streaming iterations from a worker', page: 'optimiser', pageLabel: 'Bayesian Optimiser' },
  { req: '1. ML models: time series, regression, LSTM', feature: 'Holt-Winters, Ridge, LSTM (TF.js), seasonal-naive benchmark, adaptive ensemble', page: 'forecast', pageLabel: 'Demand Forecast' },
  { req: '1. Historical data, seasonality', feature: 'Favorita daily sales 2013–2017; weekday dummies, yearly Fourier terms, HW weekly season', page: 'forecast', pageLabel: 'Demand Forecast' },
  { req: '1. Promotions', feature: 'onpromotion feature; promotion-quartile effect chart', page: 'forecast', pageLabel: 'Demand Forecast' },
  { req: '1. External factors', feature: 'Oil price, holidays/events, paydays, earthquake period', page: 'forecast', pageLabel: 'Demand Forecast' },
  { req: '1. Keep the possibility of adding new data', feature: 'Add-day form, CSV upload, replay stream (integration step)', page: 'data', pageLabel: 'Data & Live Feed' },
  { req: '1. Continuously update forecasts', feature: 'Refit on each new row; LSTM fine-tuning (integration step)', page: 'data', pageLabel: 'Data & Live Feed' },
  { req: '2. Probabilistic models (demand variability)', feature: 'Conformal 80/95% bands; bootstrapped forecast errors', page: 'risk', pageLabel: 'Safety Stock & Risk' },
  { req: '2. Lead-time uncertainty', feature: 'Gamma / lognormal / normal lead times, σL control', page: 'risk', pageLabel: 'Safety Stock & Risk' },
  { req: '2. Monte Carlo risk scenarios and buffers', feature: 'DDLT simulation, P(stockout), MC ROP, 6 scenarios, policy simulation', page: 'risk', pageLabel: 'Safety Stock & Risk' },
  { req: '2. SS = Z·√(σ_D²·L + D̄²·σ_L²)', feature: 'Live formula card with substituted values; unit-tested hand calculation (257.46)', page: 'risk', pageLabel: 'Safety Stock & Risk' },
  { req: '3. Bayesian optimisation instead of manual sweeps', feature: 'GP + constrained EI over (s, Q); compared with grid and random search', page: 'optimiser', pageLabel: 'Bayesian Optimiser' },
  { req: '3. Optimal reorder points under constraints', feature: 'Fill rate / cycle SL, budget, capacity, order-frequency constraints', page: 'optimiser', pageLabel: 'Bayesian Optimiser' },
  { req: 'Sensitivity analysis', feature: 'Tornado, elasticities, L̄×σL heatmap, service curve, optimal-policy sensitivity', page: 'sensitivity', pageLabel: 'Sensitivity' },
  { req: 'Note: no sharing of data', feature: 'No data in the deployed site; local file, IndexedDB only; raw data git-ignored', page: 'data', pageLabel: 'Data & Live Feed' },
];

export function Method() {
  const st = useApp((s) => s.settings);
  const setPage = useApp((s) => s.setPage);
  const carrying = (st.costs.holdingPerUnitDay * 365) / st.costs.unitCost;
  const box = 'rounded-xl border border-white/10 bg-white/[0.03] p-3 text-[12.5px] leading-snug';
  return (
    <>
      <PageHeader eyebrow="Method & assumptions" title="How this dashboard works">
        Netlify's free tier only hosts static files, so all modelling runs in your browser inside Web Workers. That keeps the interface responsive and the data on your device.
      </PageHeader>

      <Panel title="Architecture">
        <div className="grid items-stretch gap-3 lg:grid-cols-[1fr_auto_1.3fr_auto_1fr]">
          <div className={box}><div className="font-semibold text-ink">Python notebook (offline, once)</div><div className="mt-1 text-ink-3">Cleans 3M Kaggle rows → 9-series subset (175 KB). Reference implementation → golden test fixtures.</div></div>
          <div className="hidden items-center text-ink-3 lg:flex" aria-hidden>→</div>
          <div className={`${box} border-ai/30`}>
            <div className="font-semibold text-ink">Your browser</div>
            <div className="mt-2 grid gap-2 sm:grid-cols-3">
              <div className="rounded-lg bg-white/[0.04] p-2"><div className="text-ink-2">UI thread</div><div className="text-ink-3">React, charts only</div></div>
              <div className="rounded-lg bg-white/[0.04] p-2"><div className="text-ink-2">Analysis worker</div><div className="text-ink-3">Backtest, forecasts, LSTM, Monte Carlo, policy sims, sensitivity</div></div>
              <div className="rounded-lg bg-white/[0.04] p-2"><div className="text-ink-2">Optimiser worker</div><div className="text-ink-3">GP-based Bayesian optimisation, benchmarks</div></div>
            </div>
            <div className="mt-2 text-ink-3">IndexedDB: your dataset (this device only)</div>
          </div>
          <div className="hidden items-center text-ink-3 lg:flex" aria-hidden>←</div>
          <div className={box}><div className="font-semibold text-ink">Netlify CDN (static)</div><div className="mt-1 text-ink-3">HTML/JS/CSS and pre-trained LSTM weights (553 KB). No functions, no database, no data.</div></div>
        </div>
      </Panel>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Panel title="Where each computation runs">
          <DataTable caption="Computation placement" columns={[{ key: 'c', label: 'Computation' }, { key: 'w', label: 'Runs in' }, { key: 'y', label: 'Why' }]} rows={[
            { c: 'Data preparation', w: 'Notebook, offline', y: '120 MB CSV is too large to ship; done once' },
            { c: 'Holt-Winters, Ridge, seasonal naive', w: 'Analysis worker', y: 'Closed-form or recursive: milliseconds' },
            { c: 'LSTM initial training', w: 'Build time (Node, TF.js)', y: 'Precomputed so first load is fast. Same code and library as the browser' },
            { c: 'LSTM inference / fine-tuning', w: 'Analysis worker', y: 'Small network; updates on new data in seconds' },
            { c: 'Monte Carlo (≤400k runs)', w: 'Analysis worker', y: 'Typed arrays, seeded RNG: ~0.1–0.5 s' },
            { c: 'Policy simulation', w: 'Both workers', y: 'Also the optimiser’s objective function' },
            { c: 'Bayesian optimisation', w: 'Optimiser worker', y: 'Streams iterations; never blocks the UI' },
            { c: 'Netlify Functions', w: 'Not used', y: '~10 s limit, no GPU, and it would send your data to a server' },
          ]} />
          <div className="mt-3"><Note>Precomputed: only the LSTM's initial weights, trained on data up to 2016-12-06 (the first backtest calibration origin), so its backtest is leak-free. Everything else is computed live from the file you load.</Note></div>
        </Panel>
        <Panel title="Assumptions & units (current values)">
          <DataTable caption="Assumptions" columns={[{ key: 'k', label: 'Item' }, { key: 'v', label: 'Value' }, { key: 'n', label: 'Note' }]} rows={[
            { k: 'Period', v: '1 day', n: 'D̄, σ_D in units/day; L, σL in days' },
            { k: 'Lead time', v: `${st.leadTime.dist}, L̄ ${st.leadTime.mean} d, σL ${st.leadTime.sd} d`, n: 'Not in the data: an assumption, set in Controls' },
            { k: 'Service level', v: fmtPct(st.serviceLevel), n: 'Cycle service level, Z = Φ⁻¹(SL)' },
            { k: 'Unit cost / carrying rate', v: `${fmtMoney(st.costs.unitCost)} · ${fmtPct(carrying, 0)}/yr`, n: `Holding ${fmtMoney(st.costs.holdingPerUnitDay, 5)}/unit/day` },
            { k: 'Order cost', v: fmtMoney(st.costs.orderCost, 0), n: 'Per replenishment' },
            { k: 'Lost-sale penalty', v: fmtMoney(st.costs.stockoutPenalty), n: 'Per unit not served (lost sales, no backorders)' },
            { k: 'Review', v: 'Daily, continuous (s, Q)', n: 'Order Q when inventory position ≤ s' },
            { k: 'Future calendar', v: 'No holidays assumed beyond the file', n: 'Promotions: weekday mean of last 4 weeks; oil: last price' },
          ]} />
        </Panel>
      </div>

      <Panel className="mt-4" title="Honest limitations">
        <ul className="list-disc space-y-1.5 pl-5 text-[13px] leading-relaxed text-ink-2">
          <li>Lead-time behaviour and all costs are assumptions; results scale with them. Sensitivity analysis shows how much.</li>
          <li>The slide formula assumes normal, independent daily errors and continuous review. Monte Carlo and the simulation show where that under-protects, so the dashboard reports both rather than hiding the gap.</li>
          <li>The block bootstrap draws from about 12 recent 14-day error paths: realistic, but a small sample.</li>
          <li>The LSTM's backtest uses frozen weights (conservative), while Holt-Winters and Ridge refit at each origin.</li>
          <li>Results use seeded random numbers. Optimal policies sit in a flat cost valley, so ±1–2% differences are noise.</li>
          <li>Data you add is stored per browser; use export/import to move it.</li>
        </ul>
      </Panel>

      <Panel className="mt-4" title="Traceability: slide requirement → dashboard feature">
        <DataTable caption="Traceability table" maxHeight={600} columns={[{ key: 'r', label: 'Slide requirement' }, { key: 'f', label: 'Feature' }, { key: 'p', label: 'Where' }]}
          rows={TRACE.map((t) => ({ r: t.req, f: <span className="whitespace-normal">{t.feature}</span>, p: <button className="text-ai hover:underline" onClick={() => setPage(t.page)}>{t.pageLabel}</button> }))} />
      </Panel>
    </>
  );
}
