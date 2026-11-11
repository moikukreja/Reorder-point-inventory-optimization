/**
 * Split-conformal prediction intervals. Out-of-sample residuals (actual − forecast) from
 * calibration origins are pooled. The interval is forecast + [q_lo, q_hi], using NumPy's
 * linear quantiles, and coverage is measured on separate test origins. This makes no
 * normality assumption, so the asymmetric risk of promotion spikes stays in the band.
 */
import { quantile, std } from './stats';

export interface IntervalBounds {
  lo: number;
  hi: number;
  coverage?: number;
}

export type IntervalLevel = '0.8' | '0.95';
export const INTERVAL_LEVELS: IntervalLevel[] = ['0.8', '0.95'];

export interface ConformalResult {
  levels: Record<IntervalLevel, IntervalBounds>;
  residualSd: number;
  residuals: Float64Array;
}

/** Pools residuals over (actual, forecast) pairs, skipping NaN actuals. */
export function pooledResiduals(actuals: ArrayLike<number>[], forecasts: ArrayLike<number>[]): Float64Array {
  const r: number[] = [];
  actuals.forEach((a, i) => {
    const f = forecasts[i];
    for (let h = 0; h < a.length; h++) if (!Number.isNaN(a[h])) r.push(a[h] - f[h]);
  });
  return Float64Array.from(r);
}

export function conformalFromResiduals(residuals: Float64Array): ConformalResult {
  const sorted = Float64Array.from(residuals).sort();
  const levels = {} as Record<IntervalLevel, IntervalBounds>;
  for (const lv of INTERVAL_LEVELS) {
    const a = 1 - Number(lv);
    levels[lv] = { lo: quantile(sorted, a / 2, true), hi: quantile(sorted, 1 - a / 2, true) };
  }
  return { levels, residualSd: std(residuals, 1), residuals };
}

/** Share of actuals that fall inside forecast + [lo, hi]. */
export function empiricalCoverage(actuals: ArrayLike<number>[], forecasts: ArrayLike<number>[], b: IntervalBounds): number {
  let inside = 0, n = 0;
  actuals.forEach((a, i) => {
    for (let h = 0; h < a.length; h++) {
      if (Number.isNaN(a[h])) continue;
      const f = forecasts[i][h];
      if (a[h] >= f + b.lo && a[h] <= f + b.hi) inside++;
      n++;
    }
  });
  return inside / n;
}
