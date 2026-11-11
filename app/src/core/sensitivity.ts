/**
 * Sensitivity analysis of the reorder point. This replaces the class notebook's
 * one-at-a-time manual sweeps with:
 *  - a tornado chart: ROP / SS at ±x% for every input, ranked by impact
 *  - analytic elasticities: % change in ROP per 1% change in each input
 *  - a heatmap: ROP over (L̄ × σL)
 *  - a service-level curve: SS, its holding cost, and the Monte Carlo service actually achieved
 *  - optimal-policy sensitivity: the Bayesian optimiser re-run under perturbed economics
 *    (in policySensitivity, run by the optimiser worker)
 */
import { slideSafetyStock, type SafetyStockInputs, type SafetyStockResult } from './safetyStock';
import { invNormCdf, normPdf } from './stats';
import { simulateDdlt, stockoutProbability, type DdltConfig } from './monteCarlo';

export type InputKey = 'dBar' | 'sdD' | 'L' | 'sdL' | 'serviceLevel';
export const INPUT_LABELS: Record<InputKey, string> = {
  dBar: 'Average demand D̄',
  sdD: 'Demand variability σ_D',
  L: 'Lead time L̄',
  sdL: 'Lead-time variability σL',
  serviceLevel: 'Service level',
};

export interface TornadoBar {
  key: InputKey;
  label: string;
  lowInput: number;
  highInput: number;
  ropLow: number;
  ropHigh: number;
  ssLow: number;
  ssHigh: number;
  /** |ROP(high) − ROP(low)|, used for ranking. */
  swing: number;
}

/**
 * Perturbs each input by ±`pct` (service level by ±`slPoints` percentage points, capped to (0.5, 0.999)).
 * Bars are sorted by swing, largest first.
 */
export function tornado(base: SafetyStockInputs, pct = 0.2, slPoints = 0.025): TornadoBar[] {
  const keys: InputKey[] = ['dBar', 'sdD', 'L', 'sdL', 'serviceLevel'];
  const bars = keys.map((k) => {
    const v = base[k];
    const lo = k === 'serviceLevel' ? Math.max(0.5, v - slPoints) : v * (1 - pct);
    const hi = k === 'serviceLevel' ? Math.min(0.999, v + slPoints) : v * (1 + pct);
    const rLo = slideSafetyStock({ ...base, [k]: lo });
    const rHi = slideSafetyStock({ ...base, [k]: hi });
    return { key: k, label: INPUT_LABELS[k], lowInput: lo, highInput: hi, ropLow: rLo.reorderPoint, ropHigh: rHi.reorderPoint, ssLow: rLo.safetyStock, ssHigh: rHi.safetyStock, swing: Math.abs(rHi.reorderPoint - rLo.reorderPoint) };
  });
  return bars.sort((a, b) => b.swing - a.swing);
}

/**
 * Analytic elasticities of ROP = D̄L + Z·√(σ_D²L + D̄²σL²): (∂ROP/∂x)·(x/ROP).
 * For the service level the derivative is taken through Z = Φ⁻¹(SL): dZ/dSL = 1/φ(Z).
 */
export function elasticities(i: SafetyStockInputs): Record<InputKey, number> {
  const r = slideSafetyStock(i);
  const S = r.sdDdlt, z = r.z, rop = r.reorderPoint;
  const d = {
    dBar: i.L + (z * i.dBar * i.sdL ** 2) / S,
    sdD: (z * i.sdD * i.L) / S,
    L: i.dBar + (z * i.sdD ** 2) / (2 * S),
    sdL: (z * i.dBar ** 2 * i.sdL) / S,
    serviceLevel: S / normPdf(invNormCdf(i.serviceLevel)),
  };
  return Object.fromEntries((Object.keys(d) as InputKey[]).map((k) => [k, (d[k] * i[k]) / rop])) as Record<InputKey, number>;
}

export interface Heatmap {
  xs: number[]; // L̄ values
  ys: number[]; // σL values
  rop: number[][]; // rop[iy][ix]
}

export function ropHeatmap(base: SafetyStockInputs, Ls: number[], sdLs: number[]): Heatmap {
  return { xs: Ls, ys: sdLs, rop: sdLs.map((sdL) => Ls.map((L) => slideSafetyStock({ ...base, L, sdL }).reorderPoint)) };
}

export interface ServiceCurvePoint {
  serviceLevel: number;
  z: number;
  safetyStock: number;
  reorderPoint: number;
  /** Holding cost of the safety stock alone ($/day). */
  safetyHoldingPerDay: number;
  /** Service level actually achieved at the formula ROP under the Monte Carlo distribution. */
  mcAchieved: number;
}

/**
 * SL from 80% to 99.9%. The Monte Carlo distribution is simulated once and reused for every
 * point, because only the ROP changes.
 */
export function serviceLevelCurve(base: SafetyStockInputs, holdingPerUnitDay: number, mc: Omit<DdltConfig, 'n'> & { n?: number }, levels = [0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]): ServiceCurvePoint[] {
  const sim = simulateDdlt({ ...mc, n: mc.n ?? 100_000 });
  return levels.map((sl) => {
    const r: SafetyStockResult = slideSafetyStock({ ...base, serviceLevel: sl });
    return { serviceLevel: sl, z: r.z, safetyStock: r.safetyStock, reorderPoint: r.reorderPoint, safetyHoldingPerDay: r.safetyStock * holdingPerUnitDay, mcAchieved: 1 - stockoutProbability(sim.samples, r.reorderPoint) };
  });
}

/** Shares of lead-time-demand variance: demand uncertainty σ_D²L vs lead-time uncertainty D̄²σL². */
export function varianceSplit(i: SafetyStockInputs): { demand: number; leadTime: number } {
  const r = slideSafetyStock(i);
  const tot = r.varDemand + r.varLeadTime;
  return { demand: r.varDemand / tot, leadTime: r.varLeadTime / tot };
}
