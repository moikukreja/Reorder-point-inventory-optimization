/**
 * Verification report for Part 3 (Bayesian optimisation and sensitivity) on real data.
 *
 *  1. Optimal (s, Q) under constraints: Bayesian optimisation vs random search vs an exhaustive grid.
 *  2. Out-of-sample check: the winners re-simulated with fresh random numbers, next to the formula policies.
 *  3. s-only mode (Q fixed at EOQ), the direct "optimal reorder point" question.
 *  4. Sensitivity: tornado, elasticities, L̄×σL heatmap, service-level curve, variance split.
 *  5. Optimal-policy sensitivity: the optimiser re-run under perturbed economics.
 *
 * Usage: npx tsx scripts/verify-optim.ts [seriesKey]
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
import { DEFAULT_COSTS, DEFAULT_LEAD_TIME, DEFAULT_SERVICE_LEVEL } from '../src/core/assumptions';
import { eoq, simulatePolicy, type PolicySimConfig } from '../src/core/policySim';
import { bayesianOptimise, bestRecord, gridSearch, randomSearch, type BoRecord } from '../src/core/bayesOpt';
import { constraintLabels, defaultBounds, defaultConstraints, policyProblem, standardPerturbations, type PolicyEvalInfo } from '../src/core/policyOptimizer';
import { elasticities, INPUT_LABELS, ropHeatmap, serviceLevelCurve, tornado, varianceSplit } from '../src/core/sensitivity';

const f0 = (v: number) => Math.round(v).toLocaleString('en-US');
const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
const pad = (s: string | number, n: number) => String(s).padStart(n);

function evalsToWithin(h: BoRecord[], target: number, tol = 0.01): number | null {
  const i = h.findIndex((r) => r.bestFeasible <= target * (1 + tol));
  return i < 0 ? null : i + 1;
}

async function main() {
  await initNodeBackend();
  const key = process.argv[2] ?? '44|BEVERAGES';
  const sub = loadSubset();
  const raw = toSeriesData(sub.series.find((x) => x.key === key)!);
  const { cal, series: [s] } = extendCalendar(buildCalendar(sub), [raw], raw.y.length + 60);
  const TS = indexOf(cal, TRAIN_START_DATE);
  const lt = DEFAULT_LEAD_TIME, costs = DEFAULT_COSTS;
  const dir = join(MODELS_DIR, modelSlug(key));
  let lstm: ((T: number) => Float64Array) | undefined;
  if (existsSync(join(dir, 'model.json'))) {
    const model = await loadModel(dir);
    const meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf-8')) as LstmMeta;
    lstm = (T) => lstmForecast(model, meta, s, cal, T);
  }
  const bt = runBacktest(s, cal, { ...DEFAULT_BACKTEST, trainStart: TS }, lstm);
  const fc = forecastNow(s, cal, bt, 28, lstm);
  const cmp = compareSafetyStock(s, fc, lt, DEFAULT_SERVICE_LEVEL);
  const dBar = cmp.ai.dBar;

  const sim: Omit<PolicySimConfig, 'trace'> = { horizon: 196, warmup: 14, replications: 200, seed: 1, leadTime: lt, costs, demand: { kind: 'generated', mean: fc.ensemble, residuals: fc.residuals, errorPaths: fc.errorPaths } };
  const bounds = defaultBounds(dBar, lt.mean, cmp.ai.sdDdlt);
  const cons = defaultConstraints(dBar, costs.unitCost);
  const Qeoq = eoq(dBar, costs.orderCost, costs.holdingPerUnitDay);
  console.log(`=== ${key} | decision date ${fc.originDate} | D̄ = ${f0(dBar)} u/d | lead time ${lt.dist}(${lt.mean} d, σ ${lt.sd} d)`);
  console.log(`Search box: s ∈ [${f0(bounds.s[0])}, ${f0(bounds.s[1])}], Q ∈ [${f0(bounds.Q[0])}, ${f0(bounds.Q[1])}]`);
  console.log(`Constraints: ${constraintLabels(cons).join(' · ')}`);
  console.log(`Objective: expected cost/day (holding $${costs.holdingPerUnitDay.toFixed(5)}/u/d + $${costs.orderCost}/order + $${costs.stockoutPenalty}/lost unit), ${sim.replications} reps × ${sim.horizon - (sim.warmup ?? 0)} days per evaluation\n`);

  // 1) Grid (ground truth), random and BO
  let t0 = Date.now();
  const gridP = policyProblem(sim, bounds, cons, 'sQ', Qeoq);
  const grid = gridSearch(gridP, 25);
  const tGrid = (Date.now() - t0) / 1000;
  const gBest = bestRecord(grid)! as BoRecord<PolicyEvalInfo>;
  t0 = Date.now();
  const boP = policyProblem(sim, bounds, cons, 'sQ', Qeoq);
  const bo = bayesianOptimise(boP, { budget: 40, nInit: 8, seed: 11 });
  const tBo = (Date.now() - t0) / 1000;
  const rnd = randomSearch(policyProblem(sim, bounds, cons, 'sQ', Qeoq), 40, 11);
  const rBest = bestRecord(rnd) as BoRecord<PolicyEvalInfo> | null;
  const bBest = bo.best! as BoRecord<PolicyEvalInfo>;
  console.log('=== 1. Optimal (s, Q) under constraints');
  console.log(`${'Method'.padEnd(26)}${pad('Evals', 7)}${pad('s', 9)}${pad('Q', 9)}${pad('Cost/day', 10)}${pad('Fill', 8)}${pad('Inv value', 11)}${pad('Peak', 9)}${pad('Ord/30', 8)}${pad('Evals to ≤1% of grid', 22)}`);
  const row = (lbl: string, n: number, r: BoRecord<PolicyEvalInfo> | null, h: BoRecord[]) => {
    if (!r) return console.log(`${lbl.padEnd(26)}${pad(n, 7)}   no feasible point found`);
    const m = r.info!.metrics;
    const w = r.phase === 'grid' ? h.length : evalsToWithin(h, gBest.objective);
    console.log(`${lbl.padEnd(26)}${pad(n, 7)}${pad(f0(r.info!.s), 9)}${pad(f0(r.info!.Q), 9)}${pad('$' + r.objective.toFixed(2), 10)}${pad(pct(m.fillRate, 2), 8)}${pad('$' + f0(m.avgInventoryValue), 11)}${pad(f0(m.peakOnHand), 9)}${pad(m.ordersPer30.toFixed(1), 8)}${pad(w ? `${w}` : '—', 22)}`);
  };
  row('Grid 25×25 (manual sweep)', grid.length, gBest, grid);
  row('Random search', rnd.length, rBest, rnd);
  row('Bayesian optimisation', bo.history.length, bBest, bo.history);
  console.log(`Runtime: grid ${tGrid.toFixed(1)} s, BO ${tBo.toFixed(1)} s (including GP fitting). Best-so-far by BO eval: ${bo.history.filter((_, i) => i % 5 === 4).map((r) => '$' + r.bestFeasible.toFixed(0)).join(' → ')}`);
  const hyp = bo.surrogates.objective.hyper;
  console.log(`Objective GP: length-scales s=${hyp.ls[0].toFixed(2)}, Q=${hyp.ls[1].toFixed(2)} (unit box), noise SD ${hyp.sn.toFixed(3)} (standardised)\n`);

  // 2) Out-of-sample: fresh random numbers, more replications
  console.log('=== 2. Out-of-sample check (seed 999, 500 replications), against the formula policies with Q = EOQ');
  const oos: Omit<PolicySimConfig, 'trace'> = { ...sim, seed: 999, replications: 500 };
  console.log(`${'Policy'.padEnd(30)}${pad('s', 9)}${pad('Q', 9)}${pad('Cost/day', 10)}${pad('Fill', 8)}${pad('Cycle SL', 10)}${pad('Inv value', 11)}${pad('Feasible', 10)}`);
  for (const [lbl, sv, qv] of [
    ['Baseline (notebook) + EOQ', cmp.baseline.reorderPoint, Qeoq],
    ['AI formula + EOQ', cmp.ai.reorderPoint, Qeoq],
    ['Grid optimum', gBest.info!.s, gBest.info!.Q],
    ['Bayesian optimum', bBest.info!.s, bBest.info!.Q],
  ] as const) {
    const m = simulatePolicy({ s: sv, Q: qv }, oos).mean;
    const feas = m.fillRate >= cons.minService && m.avgInventoryValue <= cons.maxInventoryValue! && m.peakOnHand <= cons.maxPeakOnHand! && m.ordersPer30 <= cons.maxOrdersPer30!;
    console.log(`${lbl.padEnd(30)}${pad(f0(sv), 9)}${pad(f0(qv), 9)}${pad('$' + m.totalCostPerDay.toFixed(2), 10)}${pad(pct(m.fillRate, 2), 8)}${pad(pct(m.cycleServiceLevel), 10)}${pad('$' + f0(m.avgInventoryValue), 11)}${pad(feas ? 'yes' : 'NO', 10)}`);
  }

  // 3) s-only
  const sGrid = gridSearch(policyProblem(sim, bounds, cons, 's', Qeoq), 60);
  const sBo = bayesianOptimise(policyProblem(sim, bounds, cons, 's', Qeoq), { budget: 15, nInit: 4, seed: 3 });
  const sg = bestRecord(sGrid) as BoRecord<PolicyEvalInfo>, sb = sBo.best as BoRecord<PolicyEvalInfo>;
  console.log(`\n=== 3. Optimal reorder point only (Q = EOQ = ${f0(Qeoq)})`);
  console.log(`  Grid (60 evals): s* = ${f0(sg.info!.s)}, cost $${sg.objective.toFixed(2)}/day | BO (15 evals): s* = ${f0(sb.info!.s)}, cost $${sb.objective.toFixed(2)}/day, within 1% after ${evalsToWithin(sBo.history, sg.objective) ?? '—'} evals`);
  console.log(`  Compare: formula ROP ${f0(cmp.ai.reorderPoint)} → BO says hold ${f0(sb.info!.s - cmp.ai.reorderPoint)} more units (the gap the simulation found: undershoot, lead-time skew, correlated errors)`);

  // 4) Sensitivity
  const base = { serviceLevel: DEFAULT_SERVICE_LEVEL, dBar, sdD: cmp.ai.sdD, L: lt.mean, sdL: lt.sd };
  console.log('\n=== 4. Sensitivity of the AI reorder point (slide formula)');
  const el = elasticities(base);
  for (const b of tornado(base)) console.log(`  ${b.label.padEnd(26)} ROP ${pad(f0(b.ropLow), 8)} … ${pad(f0(b.ropHigh), 8)}  swing ${pad(f0(b.swing), 7)}  elasticity ${el[b.key].toFixed(2)}`);
  const vs = varianceSplit(base);
  console.log(`  Variance of lead-time demand: ${pct(vs.leadTime)} from lead-time uncertainty, ${pct(vs.demand)} from demand uncertainty`);
  const hm = ropHeatmap(base, [2, 4, 6, 8], [0, 0.6, 1.2, 2.4]);
  console.log(`  Heatmap ROP (rows σL = ${hm.ys.join(', ')}; cols L̄ = ${hm.xs.join(', ')}):`);
  hm.rop.forEach((r, i) => console.log(`    σL ${String(hm.ys[i]).padEnd(4)} ${r.map((v) => pad(f0(v), 9)).join('')}`));
  console.log('  Service-level curve (Monte Carlo achieved SL at the formula ROP):');
  for (const p of serviceLevelCurve(base, costs.holdingPerUnitDay, { mode: 'bootstrap', seed: 5, leadTime: lt, forecast: fc.ensemble, residuals: fc.residuals, errorPaths: fc.errorPaths }))
    console.log(`    target ${pad(pct(p.serviceLevel), 6)} Z ${p.z.toFixed(3)}  SS ${pad(f0(p.safetyStock), 7)}  ROP ${pad(f0(p.reorderPoint), 7)}  SS holding $${p.safetyHoldingPerDay.toFixed(2)}/day  achieved ${pct(p.mcAchieved)}`);

  // 5) Optimal-policy sensitivity
  console.log('\n=== 5. How the optimal policy moves (BO, 25 evaluations each)');
  console.log(`${'  Change'.padEnd(32)}${pad('s*', 9)}${pad('Q*', 9)}${pad('Cost/day', 10)}${pad('Δs*', 9)}`);
  let sBase = 0;
  for (const p of standardPerturbations(dBar)) {
    const a = p.apply(sim, cons, bounds);
    const r = bayesianOptimise(policyProblem(a.sim, a.bounds, a.cons, 'sQ', Qeoq), { budget: 25, nInit: 6, seed: 21 });
    const b = r.best as BoRecord<PolicyEvalInfo> | null;
    if (!b) { console.log(`  ${p.label.padEnd(30)} no feasible policy within the constraints`); continue; }
    if (p.id === 'base') sBase = b.info!.s;
    console.log(`  ${p.label.padEnd(30)}${pad(f0(b.info!.s), 9)}${pad(f0(b.info!.Q), 9)}${pad('$' + b.objective.toFixed(2), 10)}${pad((b.info!.s - sBase >= 0 ? '+' : '') + f0(b.info!.s - sBase), 9)}`);
  }
  void INPUT_LABELS;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
