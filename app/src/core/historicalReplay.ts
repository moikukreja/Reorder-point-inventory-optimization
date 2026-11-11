/**
 * Historical replay: baseline vs AI on REAL demand over the backtest test window.
 *
 * At each test origin (every 14 days), each method recomputes its reorder point using only data
 * known at that moment:
 *  - Baseline: mean and SD of the previous 365 days, SS = Z·σ·√L (σL ignored).
 *  - AI formula: the ensemble's out-of-sample forecast made at that origin, σ_D from the previous
 *    12 origins' forecast errors, and the slide formula with σL.
 *  - AI Monte Carlo: the SL-quantile of simulated lead-time demand from that forecast and those errors.
 * The resulting reorder-point schedules are run against the actual daily sales (lost sales). Only
 * supplier lead times are random, and every policy sees the same delays.
 */
import type { BacktestResult } from './backtest';
import type { CostAssumption, LeadTimeAssumption } from './assumptions';
import type { SeriesData } from './features';
import { baselineSafetyStock, forecastMeanOverLeadTime, historicalDemandStats, slideSafetyStock } from './safetyStock';
import { monteCarloRop, simulateDdlt } from './monteCarlo';
import { simulatePolicy, type PolicyMetrics, type TraceDay } from './policySim';
import { std } from './stats';

export interface ReplayPolicy {
  id: 'baseline' | 'ai-formula' | 'ai-mc';
  label: string;
  sByDay: number[];
  metrics: PolicyMetrics;
  trace: TraceDay[];
}

export interface HistoricalReplay {
  from: number;
  to: number;
  origins: number[];
  actual: number[];
  Q: number;
  policies: ReplayPolicy[];
}

export function historicalReplay(
  s: SeriesData, bt: BacktestResult, lt: LeadTimeAssumption, serviceLevel: number, costs: CostAssumption, Q: number,
  opts: { replications: number; seed: number; block: boolean; window?: number },
): HistoricalReplay {
  const nC = bt.calOrigins.length;
  const origins = bt.testOrigins;
  const H = bt.config.H;
  const L = Math.min(lt.mean, H);
  const win = opts.window ?? 12;
  const from = origins[0] + 1;
  const to = origins[origins.length - 1] + H;
  const days = to - from + 1;
  const sched = { baseline: new Array<number>(days), 'ai-formula': new Array<number>(days), 'ai-mc': new Array<number>(days) };

  const allOrigins = [...bt.calOrigins, ...bt.testOrigins];
  origins.forEach((o, k) => {
    const i = nC + k; // index of this origin in the backtest arrays
    const f = bt.forecasts.ensemble![i];
    // Forecast errors known at origin o: the previous `win` origins' windows (only fully observed days)
    const resid: number[] = [];
    const paths: number[][] = [];
    for (let j = Math.max(0, i - win); j < i; j++) {
      const p: number[] = [];
      for (let h = 0; h < H; h++) {
        const a = bt.actuals[j][h];
        if (allOrigins[j] + h + 1 > o) break; // not yet observed at origin o
        p.push(a - bt.forecasts.ensemble![j][h]);
        if (!Number.isNaN(a)) resid.push(a - bt.forecasts.ensemble![j][h]);
      }
      if (p.length) paths.push(p);
    }
    const hist = historicalDemandStats(s.y, o, 365);
    const base = baselineSafetyStock({ serviceLevel, dBar: hist.dBar, sdD: hist.sdD, L: lt.mean, sdL: lt.sd });
    const ai = slideSafetyStock({ serviceLevel, dBar: forecastMeanOverLeadTime(f, L), sdD: std(resid, 1), L: lt.mean, sdL: lt.sd });
    const mc = simulateDdlt({ mode: 'bootstrap', n: 30_000, seed: opts.seed + k, leadTime: lt, forecast: f, residuals: resid, errorPaths: opts.block ? paths : undefined });
    const mcRop = monteCarloRop(mc.samples, serviceLevel);
    for (let h = 0; h < H; h++) {
      const d = o + h + 1 - from;
      if (d < 0 || d >= days) continue;
      sched.baseline[d] = base.reorderPoint;
      sched['ai-formula'][d] = ai.reorderPoint;
      sched['ai-mc'][d] = mcRop;
    }
  });

  const actual = Array.from({ length: days }, (_, d) => s.y[from + d]);
  const labels = { baseline: 'Baseline (class notebook)', 'ai-formula': 'AI: slide formula', 'ai-mc': 'AI: Monte Carlo ROP' } as const;
  const policies = (Object.keys(sched) as ReplayPolicy['id'][]).map((id) => {
    const r = simulatePolicy({ s: sched[id][0], Q }, {
      horizon: days, warmup: 0, replications: opts.replications, seed: opts.seed, leadTime: lt, costs,
      demand: { kind: 'actual', values: actual }, sByDay: sched[id], trace: true,
    });
    return { id, label: labels[id], sByDay: sched[id], metrics: r.mean, trace: r.trace! };
  });
  return { from, to, origins, actual: actual.map((v) => (Number.isNaN(v) ? 0 : v)), Q, policies };
}
