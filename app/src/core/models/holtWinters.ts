/**
 * Holt-Winters exponential smoothing: additive weekly season (m = 7) with a damped additive trend.
 *
 *   ℓt = α(yt − st−m) + (1 − α)(ℓt−1 + φbt−1)
 *   bt = β(ℓt − ℓt−1) + (1 − β)φbt−1
 *   st = γ(yt − ℓt−1 − φbt−1) + (1 − γ)st−m
 *   ŷt+h|t = ℓt + (φ + … + φʰ)bt + st+h−m(k+1)
 *
 * On closed days (NaN) the state moves forward without an update. Parameters are fitted
 * by Nelder–Mead on the one-step-ahead SSE, with the same parametrisation and start point
 * as the reference notebook. The state can be updated one observation at a time, so new
 * data is absorbed in O(1).
 */
import { impute } from '../features';
import { nelderMead } from '../optim';

export interface HWParams {
  alpha: number;
  beta: number;
  gamma: number;
  phi: number;
}

export interface HWState {
  level: number;
  trend: number;
  /** Seasonal terms for the last m days, oldest first. */
  season: number[];
  /** One-step SSE over fitted observations (from t = 2m). */
  sse: number;
  count: number;
}

const M = 7;

/** Runs the recursions over y (a training window) and returns the final state. */
export function hwRun(y: ArrayLike<number>, p: HWParams): HWState {
  const n = y.length;
  const y0 = impute(Array.prototype.slice.call(y, 0, 2 * M) as number[]);
  let l = 0;
  for (let i = 0; i < M; i++) l += y0[i] / M;
  let w2 = 0;
  for (let i = M; i < 2 * M; i++) w2 += y0[i] / M;
  let b = (w2 - l) / M;
  const s: number[] = [];
  for (let i = 0; i < M; i++) s.push(y0[i] - l);
  const { alpha: a, beta: bb, gamma: g, phi } = p;
  let sse = 0, cnt = 0;
  for (let t = M; t < n; t++) {
    const st = s[t - M];
    const yt = y[t];
    let lNew: number, bNew: number, sNew: number;
    if (Number.isNaN(yt)) {
      lNew = l + phi * b;
      bNew = phi * b;
      sNew = st;
    } else {
      const e = yt - (l + phi * b + st);
      if (t >= 2 * M) {
        sse += e * e;
        cnt++;
      }
      lNew = a * (yt - st) + (1 - a) * (l + phi * b);
      bNew = bb * (lNew - l) + (1 - bb) * phi * b;
      sNew = g * (yt - l - phi * b) + (1 - g) * st;
    }
    l = lNew;
    b = bNew;
    s.push(sNew);
  }
  return { level: l, trend: b, season: s.slice(-M), sse, count: cnt };
}

/** Absorbs one new observation into an existing state (used when new data arrives). */
export function hwUpdate(state: HWState, yt: number, p: HWParams): HWState {
  const { level: l, trend: b } = state;
  const st = state.season[0];
  const { alpha: a, beta: bb, gamma: g, phi } = p;
  if (Number.isNaN(yt)) return { ...state, level: l + phi * b, trend: phi * b, season: [...state.season.slice(1), st] };
  const e = yt - (l + phi * b + st);
  const lNew = a * (yt - st) + (1 - a) * (l + phi * b);
  const bNew = bb * (lNew - l) + (1 - bb) * phi * b;
  const sNew = g * (yt - l - phi * b) + (1 - g) * st;
  return { level: lNew, trend: bNew, season: [...state.season.slice(1), sNew], sse: state.sse + e * e, count: state.count + 1 };
}

export function hwForecast(state: HWState, H: number, phi: number): Float64Array {
  const out = new Float64Array(H);
  let damp = 0;
  for (let h = 1; h <= H; h++) {
    damp += phi ** h;
    out[h - 1] = state.level + damp * state.trend + state.season[(h - 1) % M];
  }
  return out;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Unconstrained → constrained: α, γ ∈ (0,1); β ∈ (0, 0.2); φ ∈ (0.8, 0.98). */
export function hwUnpack(x: number[]): HWParams {
  return { alpha: sigmoid(x[0]), beta: sigmoid(x[1]) * 0.2, gamma: sigmoid(x[2]), phi: 0.8 + 0.18 * sigmoid(x[3]) };
}

/** Start point: α = 0.2, β = 0.04, γ = 0.1, φ = 0.89 (identical to the notebook). */
export const HW_X0 = [-1.386294, -1.386294, -2.197225, 0.0];

/** Fits parameters by minimising one-step SSE. Returns the parameters and the fitted final state. */
export function hwFit(y: ArrayLike<number>): { params: HWParams; state: HWState; sse: number } {
  const res = nelderMead((x) => hwRun(y, hwUnpack(x)).sse, HW_X0, { maxiter: 600, xatol: 1e-4, fatol: 1e-6 });
  const params = hwUnpack(res.x);
  const state = hwRun(y, params);
  return { params, state, sse: res.fun };
}

/** Convenience: fit on y[from..T] and forecast H days ahead (negative values clipped to 0). */
export function holtWintersForecast(y: ArrayLike<number>, from: number, T: number, H: number): { forecast: Float64Array; params: HWParams } {
  const win = Array.prototype.slice.call(y, from, T + 1) as number[];
  const { params, state } = hwFit(win);
  const f = hwForecast(state, H, params.phi);
  for (let i = 0; i < H; i++) f[i] = Math.max(0, f[i]);
  return { forecast: f, params };
}
