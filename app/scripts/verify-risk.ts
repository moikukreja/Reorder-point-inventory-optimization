/**
 * Verification report for Part 2 (safety stock and Monte Carlo) on real data.
 *
 *  1. Hand calculation of the slide formula.
 *  2. Baseline (class notebook) vs AI inputs for Store 44 × BEVERAGES, with the substituted formula.
 *  3. Monte Carlo (forecast + bootstrapped errors): the real stockout risk of each ROP.
 *  4. Check of the iid-error assumption: realised L-day forecast-error SD vs √L·σ_D.
 *  5. Risk scenarios.
 *  6. Forward (s, Q) policy simulation: baseline ROP vs AI ROP.
 *
 * Usage: npx tsx scripts/verify-risk.ts [seriesKey]
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { loadSubset, MODELS_DIR } from './lib/local-data';
import { initNodeBackend, loadModel } from './lib/node-tf';
import { buildCalendar, extendCalendar, indexOf, toSeriesData, TRAIN_START_DATE } from '../src/core/features';
import { DEFAULT_BACKTEST, runBacktest } from '../src/core/backtest';
import { forecastNow } from '../src/core/forecastNow';
import { lstmForecast, modelSlug, type LstmMeta } from '../src/core/models/lstm';
import { compareSafetyStock } from '../src/core/riskInputs';
import { slideSafetyStock, type SafetyStockResult } from '../src/core/safetyStock';
import { simulateDdlt, stockoutProbability, expectedShortage, monteCarloRop } from '../src/core/monteCarlo';
import { SCENARIOS, estimateUplifts } from '../src/core/scenarios';
import { eoq, simulatePolicy } from '../src/core/policySim';
import { DEFAULT_COSTS, DEFAULT_LEAD_TIME, DEFAULT_SERVICE_LEVEL } from '../src/core/assumptions';
import { std } from '../src/core/stats';

const f0 = (v: number) => Math.round(v).toLocaleString('en-US');
const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
const pad = (s: string | number, n: number) => String(s).padStart(n);

function showFormula(label: string, r: SafetyStockResult) {
  console.log(`${label}`);
  console.log(`  Z = Φ⁻¹(${r.serviceLevel}) = ${r.z.toFixed(4)} | D̄ = ${r.dBar.toFixed(1)} u/d | σ_D = ${r.sdD.toFixed(1)} u/d | L = ${r.L} d | σL = ${r.sdL} d`);
  console.log(`  SS = ${r.z.toFixed(4)} × √(${r.sdD.toFixed(1)}² × ${r.L} + ${r.dBar.toFixed(1)}² × ${r.sdL}²) = ${r.z.toFixed(4)} × √(${f0(r.varDemand)} + ${f0(r.varLeadTime)}) = ${r.z.toFixed(4)} × ${r.sdDdlt.toFixed(1)} = ${f0(r.safetyStock)} units`);
  console.log(`  ROP = D̄·L + SS = ${f0(r.cycleStock)} + ${f0(r.safetyStock)} = ${f0(r.reorderPoint)} units   (lead-time share of variance: ${pct(r.varLeadTime / (r.varDemand + r.varLeadTime))})`);
}

async function main() {
  await initNodeBackend();
  const key = process.argv[2] ?? '44|BEVERAGES';
  const sub = loadSubset();
  const raw = toSeriesData(sub.series.find((x) => x.key === key)!);
  // Forecasts need 28+ days of calendar beyond the history; extend with the documented assumptions
  const { cal, series: [s] } = extendCalendar(buildCalendar(sub), [raw], raw.y.length + 60);
  const TS = indexOf(cal, TRAIN_START_DATE);
  const lt = DEFAULT_LEAD_TIME, SL = DEFAULT_SERVICE_LEVEL;

  console.log('=== 1. Hand calculation (D̄ = 100, σ_D = 20, L = 5, σL = 1.5, SL = 95%)');
  showFormula('', slideSafetyStock({ serviceLevel: 0.95, dBar: 100, sdD: 20, L: 5, sdL: 1.5 }));

  const dir = join(MODELS_DIR, modelSlug(key));
  let lstm: ((T: number) => Float64Array) | undefined;
  if (existsSync(join(dir, 'model.json'))) {
    const model = await loadModel(dir);
    const meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf-8')) as LstmMeta;
    lstm = (T) => lstmForecast(model, meta, s, cal, T);
  }
  const bt = runBacktest(s, cal, { ...DEFAULT_BACKTEST, trainStart: TS }, lstm);
  const fc = forecastNow(s, cal, bt, 28, lstm);
  const cmp = compareSafetyStock(s, fc, lt, SL);

  console.log(`\n=== 2. ${key}: reorder decision on ${fc.originDate} (lead time ${lt.dist}, L̄ = ${lt.mean} d, σL = ${lt.sd} d, target SL ${pct(SL, 0)})`);
  console.log(`Forecast for the next 7 days (ensemble): ${Array.from(fc.ensemble.slice(0, 7), f0).join(', ')}`);
  showFormula(`BASELINE: ${cmp.baseline.source}`, cmp.baseline);
  showFormula(`AI-ENHANCED: ${cmp.ai.source}`, cmp.ai);

  console.log('\n=== 3. Monte Carlo: what each ROP really delivers (200,000 lead-time scenarios, forecast + bootstrapped errors)');
  const mc = simulateDdlt({ mode: 'bootstrap', n: 200_000, seed: 42, leadTime: lt, forecast: fc.ensemble, residuals: fc.residuals });
  const mcB = simulateDdlt({ mode: 'bootstrap', n: 200_000, seed: 42, leadTime: lt, forecast: fc.ensemble, residuals: fc.residuals, errorPaths: fc.errorPaths });
  const mcN = simulateDdlt({ mode: 'normal', n: 200_000, seed: 42, leadTime: lt, dBar: cmp.ai.dBar, sdD: cmp.ai.sdD });
  console.log(`DDLT SD: formula ${f0(cmp.ai.sdDdlt)} | MC normal ${f0(mcN.sd)} | MC iid bootstrap ${f0(mc.sd)} | MC block bootstrap ${f0(mcB.sd)}   (mean ${f0(mcB.mean)} vs D̄L ${f0(cmp.ai.cycleStock)})`);
  for (const [name, m] of [['iid bootstrap', mc], ['block bootstrap', mcB]] as const) {
    for (const [lbl, r] of [['Baseline ROP', cmp.baseline], ['AI formula ROP', cmp.ai]] as const) {
      const p = stockoutProbability(m.samples, r.reorderPoint);
      console.log(`  [${name}] ${lbl.padEnd(15)} ${pad(f0(r.reorderPoint), 7)} → P(stockout per cycle) ${pad(pct(p), 6)}, achieved SL ${pct(1 - p)}, expected short ${f0(expectedShortage(m.samples, r.reorderPoint))} u/cycle`);
    }
    console.log(`  [${name}] Monte Carlo ROP for exactly ${pct(SL, 0)}: ${f0(monteCarloRop(m.samples, SL))}`);
  }
  const mcRop = monteCarloRop(mcB.samples, SL);
  console.log(`  Lead times beyond the 28-day horizon: ${pct(mc.horizonOverflow, 3)}`);

  console.log('\n=== 4. Assumption check: are daily forecast errors independent within a lead time?');
  const L = Math.round(lt.mean);
  const sums: number[] = [];
  bt.actuals.forEach((a, i) => {
    const f = bt.forecasts.ensemble![i];
    for (let h = 0; h + L <= a.length; h += L) {
      let e = 0, ok = true;
      for (let k = h; k < h + L; k++) { if (Number.isNaN(a[k])) ok = false; e += a[k] - f[k]; }
      if (ok) sums.push(e);
    }
  });
  const real = std(sums, 1), iid = Math.sqrt(L) * fc.residualSd;
  console.log(`  Realised SD of ${L}-day error sums: ${f0(real)} vs √L·σ_D = ${f0(iid)} (ratio ${(real / iid).toFixed(2)}, n = ${sums.length} non-overlapping windows)`);
  console.log(`  ${real / iid > 1.1 ? 'Errors are positively correlated, so the iid assumption understates risk; the dashboard flags this.' : 'Close to independent, so the formula’s √L scaling is reasonable for this series.'}`);

  console.log('\n=== 5. Risk scenarios (AI ROP recomputed per scenario; Monte Carlo risk if the current AI ROP is kept)');
  const u = estimateUplifts(s, cal, TS, s.y.length - 1);
  console.log(`  Data-derived uplifts: promotion ×${u.promo.toFixed(2)}, holiday ×${u.holiday.toFixed(2)}`);
  console.log('  (Monte Carlo uses the block bootstrap)');
  console.log(`  ${'Scenario'.padEnd(30)}${pad('Formula ROP', 13)}${pad('MC ROP', 10)}${pad('P(stockout) @ current ROP', 28)}`);
  for (const sc of SCENARIOS) {
    const e = sc.apply(lt, u);
    const r = slideSafetyStock({ serviceLevel: SL, dBar: cmp.ai.dBar * e.demandMultiplier, sdD: cmp.ai.sdD * e.residualMultiplier, L: e.leadTime.mean, sdL: e.leadTime.sd });
    const m = simulateDdlt({ mode: 'bootstrap', n: 100_000, seed: 7, leadTime: e.leadTime, forecast: fc.ensemble, residuals: fc.residuals, errorPaths: fc.errorPaths, demandMultiplier: e.demandMultiplier, residualMultiplier: e.residualMultiplier });
    console.log(`  ${sc.label.padEnd(30)}${pad(f0(r.reorderPoint), 13)}${pad(f0(monteCarloRop(m.samples, SL)), 10)}${pad(pct(stockoutProbability(m.samples, cmp.ai.reorderPoint)), 28)}`);
  }

  console.log('\n=== 6. Forward policy simulation: 182 days × 300 replications, lost sales, Q = EOQ');
  const Q = eoq(cmp.ai.dBar, DEFAULT_COSTS.orderCost, DEFAULT_COSTS.holdingPerUnitDay);
  console.log(`  EOQ = √(2 × ${f0(cmp.ai.dBar)} × ${DEFAULT_COSTS.orderCost} / ${DEFAULT_COSTS.holdingPerUnitDay.toFixed(5)}) = ${f0(Q)} units (${(Q / cmp.ai.dBar).toFixed(1)} days of demand)`);
  console.log(`  ${'Policy'.padEnd(22)}${pad('s', 8)}${pad('Fill rate', 11)}${pad('Cycle SL', 10)}${pad('Avg stock', 11)}${pad('Orders/30d', 12)}${pad('Cost/day', 10)}`);
  console.log('  (demand = 28-day ensemble forecast profile + block-bootstrapped real forecast errors)');
  for (const [lbl, sVal] of [['Baseline (notebook)', cmp.baseline.reorderPoint], ['AI formula', cmp.ai.reorderPoint], ['AI Monte Carlo (block)', mcRop]] as const) {
    const r = simulatePolicy({ s: sVal, Q }, { horizon: 196, warmup: 14, replications: 300, seed: 1, leadTime: lt, costs: DEFAULT_COSTS, demand: { kind: 'generated', mean: fc.ensemble, residuals: fc.residuals, errorPaths: fc.errorPaths } });
    const m = r.mean;
    console.log(`  ${lbl.padEnd(22)}${pad(f0(sVal), 8)}${pad(pct(m.fillRate, 2), 11)}${pad(pct(m.cycleServiceLevel), 10)}${pad(f0(m.avgOnHand), 11)}${pad(m.ordersPer30.toFixed(1), 12)}${pad('$' + m.totalCostPerDay.toFixed(2), 10)}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
