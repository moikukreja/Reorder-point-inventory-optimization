/**
 * Ridge regression demand model on the feature set in features.ts.
 *
 * - Features are standardised with the training mean and standard deviation (population SD, as np.std).
 * - The intercept is unpenalised: β = (XsᵀXs + λI)⁻¹ Xsᵀ(y − ȳ), b0 = ȳ.
 * - Multi-step forecasts are recursive: each day's prediction feeds the later lags.
 *
 * Refitting is closed-form (milliseconds), so the model is refitted every time new data arrives.
 */
import { solveSPD } from '../linalg';
import { featureRow, impute, N_FEATURES, type Calendar } from '../features';

export interface RidgeModel {
  lambda: number;
  b0: number;
  beta: Float64Array;
  mu: Float64Array;
  sd: Float64Array;
}

/** Training design for open days in [from, T]: rows of X (flattened) and targets. */
export interface Design {
  X: Float64Array;
  y: Float64Array;
  rows: number;
}

export function buildDesign(y: ArrayLike<number>, promo: ArrayLike<number>, cal: Calendar, from: number, T: number): Design {
  const z = impute(y, T);
  const idx: number[] = [];
  for (let t = from; t <= T; t++) if (!Number.isNaN(y[t])) idx.push(t);
  const X = new Float64Array(idx.length * N_FEATURES);
  const yy = new Float64Array(idx.length);
  const row = new Float64Array(N_FEATURES);
  idx.forEach((t, r) => {
    featureRow(t, z, promo, cal, row);
    X.set(row, r * N_FEATURES);
    yy[r] = y[t];
  });
  return { X, y: yy, rows: idx.length };
}

export function ridgeFit(d: Design, lambda: number): RidgeModel {
  const p = N_FEATURES, n = d.rows;
  const mu = new Float64Array(p), sd = new Float64Array(p);
  for (let r = 0; r < n; r++) for (let j = 0; j < p; j++) mu[j] += d.X[r * p + j] / n;
  for (let r = 0; r < n; r++) for (let j = 0; j < p; j++) sd[j] += (d.X[r * p + j] - mu[j]) ** 2 / n;
  for (let j = 0; j < p; j++) {
    sd[j] = Math.sqrt(sd[j]);
    if (sd[j] < 1e-12) sd[j] = 1;
  }
  let ym = 0;
  for (let r = 0; r < n; r++) ym += d.y[r] / n;
  const A = new Float64Array(p * p);
  const b = new Float64Array(p);
  const xs = new Float64Array(p);
  for (let r = 0; r < n; r++) {
    for (let j = 0; j < p; j++) xs[j] = (d.X[r * p + j] - mu[j]) / sd[j];
    const yc = d.y[r] - ym;
    for (let i = 0; i < p; i++) {
      b[i] += xs[i] * yc;
      for (let j = 0; j <= i; j++) A[i * p + j] += xs[i] * xs[j];
    }
  }
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < i; j++) A[j * p + i] = A[i * p + j];
    A[i * p + i] += lambda;
  }
  return { lambda, b0: ym, beta: solveSPD(A, p, b), mu, sd };
}

export function ridgePredictRow(m: RidgeModel, row: ArrayLike<number>): number {
  let v = m.b0;
  for (let j = 0; j < row.length; j++) v += ((row[j] - m.mu[j]) / m.sd[j]) * m.beta[j];
  return v;
}

/**
 * Recursive H-step forecast from origin T. Predictions are clipped at 0 and fed back into the lags.
 * `closed(t)` marks future closed days, which get a forecast of 0 (no demand).
 */
export function ridgeForecast(
  m: RidgeModel, y: ArrayLike<number>, promo: ArrayLike<number>, cal: Calendar, T: number, H: number,
  closed: (t: number) => boolean = () => false,
): Float64Array {
  const z = new Float64Array(T + H + 1);
  impute(y, T, z);
  const out = new Float64Array(H);
  const row = new Float64Array(N_FEATURES);
  for (let h = 1; h <= H; h++) {
    const t = T + h;
    const p = closed(t) ? 0 : Math.max(0, ridgePredictRow(m, featureRow(t, z, promo, cal, row)));
    out[h - 1] = p;
    // A closed day's lag value is imputed from the previous week, as in impute()
    z[t] = closed(t) ? z[t - 7] : p;
  }
  return out;
}

/** λ grid searched by time-series validation. */
export const RIDGE_LAMBDAS = [0.1, 1, 10, 100, 1000];
