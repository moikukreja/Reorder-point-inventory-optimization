/**
 * Monte Carlo simulation of demand during lead time (DDLT).
 *
 * Two demand modes:
 *  - 'normal': demand during lead time | L ~ Normal(D̄·L, σ_D²·L). These are exactly the
 *    assumptions behind the slide formula, except that L follows the chosen distribution. It
 *    isolates the effect of lead-time skew and is the theory cross-check (mean D̄L̄,
 *    variance σ_D²L̄ + D̄²σ_L²).
 *  - 'bootstrap': day-by-day demand = forecast[h] + a resampled out-of-sample forecast error,
 *    clipped at 0. This keeps the weekly pattern, known promotions, and the real (skewed,
 *    fat-tailed) error distribution, with no normality assumption.
 *
 * Lead time is continuous: full days count fully and the last partial day counts fractionally.
 */
import { quantile, Rng, sampleLeadTime, type LeadTimeDist } from './stats';

export interface DdltConfig {
  mode: 'normal' | 'bootstrap';
  n: number;
  seed: number;
  leadTime: { mean: number; sd: number; dist: LeadTimeDist };
  /** normal mode */
  dBar?: number;
  sdD?: number;
  /** bootstrap mode: daily forecast for days 1..H, and a pool of residuals (actual − forecast) */
  forecast?: ArrayLike<number>;
  residuals?: ArrayLike<number>;
  /**
   * Optional block bootstrap: consecutive out-of-sample error paths (one per backtest origin).
   * When given, each run draws a random path and start day and uses consecutive errors, which keeps
   * the day-to-day correlation of forecast misses. Independent draws understate lead-time risk
   * when misses persist.
   */
  errorPaths?: ArrayLike<number>[];
  /** Scenario multipliers (1 = none). */
  demandMultiplier?: number;
  residualMultiplier?: number;
}

/**
 * Returns a function that yields consecutive errors from randomly chosen paths. When a path runs
 * out it jumps to another random path and start day. NaN entries (closed days) are skipped.
 */
export function blockErrorStream(rng: Rng, paths: ArrayLike<number>[]): { reset: () => void; next: () => number } {
  let p = paths[0], i = 0;
  const reset = () => {
    p = paths[rng.int(paths.length)];
    i = rng.int(p.length);
  };
  const next = (): number => {
    for (let guard = 0; guard < 1000; guard++) {
      if (i >= p.length) reset();
      const v = p[i++];
      if (!Number.isNaN(v)) return v;
    }
    throw new Error('blockErrorStream: no finite errors in paths');
  };
  return { reset, next };
}

export interface Histogram {
  edges: Float64Array; // bins + 1
  counts: Uint32Array;
}

export interface DdltResult {
  /** Sorted DDLT samples (units). */
  samples: Float64Array;
  /** Sorted lead-time samples (days). */
  leadTimes: Float64Array;
  mean: number;
  sd: number;
  /** Share of runs where the lead time exceeded the forecast horizon (the last forecast day is reused). */
  horizonOverflow: number;
}

export function simulateDdlt(c: DdltConfig): DdltResult {
  const rng = new Rng(c.seed);
  const out = new Float64Array(c.n);
  const Ls = new Float64Array(c.n);
  const dm = c.demandMultiplier ?? 1;
  const rm = c.residualMultiplier ?? 1;
  const block = c.mode === 'bootstrap' && c.errorPaths?.length ? blockErrorStream(rng, c.errorPaths) : null;
  let overflow = 0;
  for (let i = 0; i < c.n; i++) {
    const L = sampleLeadTime(rng, c.leadTime.mean, c.leadTime.sd, c.leadTime.dist);
    Ls[i] = L;
    if (c.mode === 'normal') {
      const dBar = c.dBar! * dm;
      const sdD = c.sdD! * rm;
      out[i] = dBar * L + sdD * Math.sqrt(L) * rng.normal();
    } else {
      const f = c.forecast!, r = c.residuals ?? [];
      const H = f.length, nr = r.length;
      const full = Math.floor(L);
      const frac = L - full;
      if (full + (frac > 0 ? 1 : 0) > H) overflow++;
      const err = block ? (block.reset(), block.next) : () => r[rng.int(nr)];
      let s = 0;
      for (let h = 0; h < full; h++) s += Math.max(0, f[Math.min(h, H - 1)] * dm + rm * err());
      if (frac > 0) s += frac * Math.max(0, f[Math.min(full, H - 1)] * dm + rm * err());
      out[i] = s;
    }
  }
  out.sort();
  Ls.sort();
  let m = 0;
  for (let i = 0; i < c.n; i++) m += out[i] / c.n;
  let v = 0;
  for (let i = 0; i < c.n; i++) v += (out[i] - m) ** 2;
  return { samples: out, leadTimes: Ls, mean: m, sd: Math.sqrt(v / (c.n - 1)), horizonOverflow: overflow / c.n };
}

/** P(DDLT > ROP): the probability of a stockout before the replenishment arrives. */
export function stockoutProbability(sorted: Float64Array, rop: number): number {
  // Binary search for the first sample > rop
  let lo = 0, hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid] <= rop) lo = mid + 1;
    else hi = mid;
  }
  return (sorted.length - lo) / sorted.length;
}

/** E[(DDLT − ROP)⁺]: expected units short per replenishment cycle. */
export function expectedShortage(sorted: Float64Array, rop: number): number {
  let s = 0;
  for (let i = sorted.length - 1; i >= 0 && sorted[i] > rop; i--) s += sorted[i] - rop;
  return s / sorted.length;
}

/** The ROP that achieves the target cycle service level under the simulated distribution. */
export const monteCarloRop = (sorted: Float64Array, serviceLevel: number) => quantile(sorted, serviceLevel, true);

/** Equal-width histogram over the [pLo, pHi] quantile range, for charting. */
export function histogram(sorted: Float64Array, bins = 60, pLo = 0.0005, pHi = 0.9995): Histogram {
  const a = quantile(sorted, pLo, true), b = quantile(sorted, pHi, true);
  const w = (b - a) / bins || 1;
  const edges = Float64Array.from({ length: bins + 1 }, (_, i) => a + i * w);
  const counts = new Uint32Array(bins);
  for (const v of sorted) {
    if (v < a || v > b) continue;
    counts[Math.min(bins - 1, Math.floor((v - a) / w))]++;
  }
  return { edges, counts };
}

/** Summary used by the dashboard's "formula vs Monte Carlo" panel. */
export interface RiskSummary {
  formulaRop: number;
  pStockoutAtFormulaRop: number;
  achievedServiceAtFormulaRop: number;
  expectedShortageAtFormulaRop: number;
  mcRop: number;
  mcSafetyStock: number;
  ddltMean: number;
  ddltSd: number;
}

export function summariseRisk(r: DdltResult, formulaRop: number, serviceLevel: number, cycleStock: number): RiskSummary {
  const p = stockoutProbability(r.samples, formulaRop);
  const mcRop = monteCarloRop(r.samples, serviceLevel);
  return {
    formulaRop,
    pStockoutAtFormulaRop: p,
    achievedServiceAtFormulaRop: 1 - p,
    expectedShortageAtFormulaRop: expectedShortage(r.samples, formulaRop),
    mcRop,
    mcSafetyStock: mcRop - cycleStock,
    ddltMean: r.mean,
    ddltSd: r.sd,
  };
}
