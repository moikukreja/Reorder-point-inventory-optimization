/**
 * Safety stock and reorder point.
 *
 * Slide formula (implemented exactly):
 *     SS = Z · √( σ_D² · L + D̄² · σ_L² )        ROP = D̄ · L + SS
 *   Z   = service-level factor Φ⁻¹(SL)
 *   σ_D = standard deviation of daily demand (units/day)
 *   L   = mean lead time (days)
 *   σ_L = lead-time standard deviation (days)
 *   D̄   = average daily demand (units/day)
 * Units: σ_D²·L has units (u/d)²·d, and D̄²·σ_L² has units (u/d)²·d². Both are consistent
 * only on a daily basis, which is used everywhere in this codebase.
 *
 * Two input policies are compared:
 *  - BASELINE (the class notebook): SS = Z·σ·√L, ROP = d·L + SS. It uses the historical mean d
 *    and the SD σ of raw daily demand, and ignores lead-time variability.
 *  - AI-ENHANCED: D̄ = forecast mean demand over the coming lead time, σ_D = SD of the
 *    forecast errors (only the uncertainty the model cannot explain), and the slide's full
 *    formula including σ_L. It is recomputed whenever the forecast or the data change.
 */
import { invNormCdf, mean, std } from './stats';

export interface SafetyStockInputs {
  serviceLevel: number;
  dBar: number;
  sdD: number;
  L: number;
  sdL: number;
}

export interface SafetyStockResult extends SafetyStockInputs {
  z: number;
  /** σ_D²·L: variance contributed by demand uncertainty. */
  varDemand: number;
  /** D̄²·σ_L²: variance contributed by lead-time uncertainty. */
  varLeadTime: number;
  /** √(σ_D²L + D̄²σ_L²): standard deviation of demand during lead time. */
  sdDdlt: number;
  safetyStock: number;
  /** D̄·L: expected demand during lead time. */
  cycleStock: number;
  reorderPoint: number;
}

/** Slide formula: SS = Z·√(σ_D²·L + D̄²·σ_L²), ROP = D̄·L + SS. */
export function slideSafetyStock(i: SafetyStockInputs): SafetyStockResult {
  if (i.serviceLevel <= 0 || i.serviceLevel >= 1) throw new RangeError('Service level must be in (0, 1).');
  if (i.L <= 0 || i.sdL < 0 || i.sdD < 0 || i.dBar < 0) throw new RangeError('Demand and lead-time inputs must be non-negative (L > 0).');
  const z = invNormCdf(i.serviceLevel);
  const varDemand = i.sdD ** 2 * i.L;
  const varLeadTime = i.dBar ** 2 * i.sdL ** 2;
  const sdDdlt = Math.sqrt(varDemand + varLeadTime);
  const safetyStock = z * sdDdlt;
  const cycleStock = i.dBar * i.L;
  return { ...i, z, varDemand, varLeadTime, sdDdlt, safetyStock, cycleStock, reorderPoint: cycleStock + safetyStock };
}

/** Class-notebook baseline: SS = Z·σ·√L (no lead-time variability term). */
export function baselineSafetyStock(i: SafetyStockInputs): SafetyStockResult {
  return slideSafetyStock({ ...i, sdL: 0 });
}

/** Static baseline statistics: mean and SD of raw daily demand over the last `window` open days up to T. */
export function historicalDemandStats(y: ArrayLike<number>, T: number, window: number): { dBar: number; sdD: number; n: number } {
  const xs: number[] = [];
  for (let t = T; t >= 0 && xs.length < window; t--) if (!Number.isNaN(y[t])) xs.push(y[t]);
  return { dBar: mean(xs), sdD: std(xs, 1), n: xs.length };
}

/**
 * Forecast mean daily demand over the next L days (a continuous L counts the last day fractionally).
 * The forecast must cover at least ⌈L⌉ days.
 */
export function forecastMeanOverLeadTime(forecast: ArrayLike<number>, L: number): number {
  const full = Math.floor(L);
  const frac = L - full;
  if (forecast.length < Math.ceil(L)) throw new RangeError(`Forecast covers ${forecast.length} days but L = ${L}.`);
  let s = 0;
  for (let h = 0; h < full; h++) s += forecast[h];
  if (frac > 0) s += frac * forecast[full];
  return s / L;
}
