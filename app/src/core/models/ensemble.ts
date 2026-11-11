/**
 * Adaptive ensemble ("adaptive decision-making"). Each member's weight is proportional to
 * 1 / MAE over its most recent out-of-sample errors. When a model starts to drift, for
 * example after a promotion regime change, its weight falls automatically as new actuals
 * arrive. For horizons a member does not cover (the LSTM covers 14 days), the remaining
 * weights are renormalised.
 */

export type MemberName = 'holtwinters' | 'ridge' | 'lstm';

export type Weights = Partial<Record<MemberName, number>>;

/** Inverse-MAE weights from recent absolute errors per member. Falls back to equal weights. */
export function inverseMaeWeights(recentAbsErrors: Partial<Record<MemberName, number[]>>, members: MemberName[]): Weights {
  const inv: Weights = {};
  let total = 0;
  for (const m of members) {
    const e = recentAbsErrors[m];
    if (!e || e.length === 0) continue;
    const mae = e.reduce((s, v) => s + v, 0) / e.length;
    const w = 1 / Math.max(mae, 1e-9);
    inv[m] = w;
    total += w;
  }
  if (total === 0) {
    const w: Weights = {};
    for (const m of members) w[m] = 1 / members.length;
    return w;
  }
  for (const m of Object.keys(inv) as MemberName[]) inv[m]! /= total;
  return inv;
}

/** Combines member forecasts horizon by horizon. A member's array may be shorter than H. */
export function combine(forecasts: Partial<Record<MemberName, ArrayLike<number>>>, weights: Weights, H: number): Float64Array {
  const out = new Float64Array(H);
  for (let h = 0; h < H; h++) {
    let s = 0, w = 0;
    for (const m of Object.keys(weights) as MemberName[]) {
      const f = forecasts[m];
      if (!f || h >= f.length || !Number.isFinite(f[h])) continue;
      s += weights[m]! * f[h];
      w += weights[m]!;
    }
    out[h] = w > 0 ? s / w : NaN;
  }
  return out;
}
