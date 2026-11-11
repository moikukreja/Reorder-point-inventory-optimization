/**
 * Rolling-origin backtest and model comparison. This is the same protocol as notebook §6.3.
 *
 * - Origins every 14 days back from the end of the history.
 *   The 12 calibration origins (k = 18..7) are used to choose Ridge λ and for conformal residuals.
 *   The 6 test origins (k = 6..1) produce the leaderboard and the interval coverage.
 * - Holt-Winters and Ridge are refitted at every origin, using only data up to that origin.
 * - The LSTM uses frozen pre-trained weights whose training cut-off is the first calibration
 *   origin. That makes its backtest leak-free and conservative, because it is not refitted.
 * - The ensemble at origin i weights members by 1/MAE over origins i−4..i−1 only (no look-ahead).
 */
import { accuracy, seasonalNaiveScale, type AccuracyMetrics } from './metrics';
import { conformalFromResiduals, empiricalCoverage, INTERVAL_LEVELS, pooledResiduals, type IntervalBounds, type IntervalLevel } from './conformal';
import { isClosed, type Calendar, type SeriesData } from './features';
import { seasonalNaiveForecast } from './models/seasonalNaive';
import { holtWintersForecast, type HWParams } from './models/holtWinters';
import { buildDesign, ridgeFit, ridgeForecast, RIDGE_LAMBDAS, type Design } from './models/ridge';
import { combine, inverseMaeWeights, type MemberName, type Weights } from './models/ensemble';

export type ModelName = 'snaive' | MemberName | 'ensemble';
export const MODEL_LABELS: Record<ModelName, string> = {
  snaive: 'Seasonal naive (benchmark)',
  holtwinters: 'Holt-Winters (time series)',
  ridge: 'Ridge regression',
  lstm: 'LSTM (deep learning)',
  ensemble: 'Adaptive ensemble',
};

export interface BacktestConfig {
  H: number;
  step: number;
  nTest: number;
  nCal: number;
  trainStart: number;
  ensembleWindow: number;
}

export const DEFAULT_BACKTEST: Omit<BacktestConfig, 'trainStart'> = { H: 14, step: 14, nTest: 6, nCal: 12, ensembleWindow: 4 };

export interface BacktestResult {
  config: BacktestConfig;
  calOrigins: number[];
  testOrigins: number[];
  lambda: number;
  lambdaScores: Record<string, number>;
  models: ModelName[];
  /** forecasts[model][i] for origins calOrigins ++ testOrigins */
  forecasts: Partial<Record<ModelName, Float64Array[]>>;
  actuals: Float64Array[];
  board: Partial<Record<ModelName, AccuracyMetrics>>;
  conformal: Partial<Record<ModelName, { levels: Record<IntervalLevel, IntervalBounds>; residualSd: number }>>;
  ensembleWeights: Weights[];
  hwParams: HWParams[];
  maseScale: number;
}

/** Optional LSTM forecaster: returns up to H forecasts from origin T. */
export type ExternalForecaster = (T: number) => Float64Array;

export function originsFor(lastIndex: number, cfg: Omit<BacktestConfig, 'trainStart'>) {
  const test: number[] = [], cal: number[] = [];
  for (let k = cfg.nTest; k >= 1; k--) test.push(lastIndex - cfg.step * k);
  for (let k = cfg.nCal + cfg.nTest; k > cfg.nTest; k--) cal.push(lastIndex - cfg.step * k);
  return { test, cal };
}

const actualWindow = (y: ArrayLike<number>, T: number, H: number) => Float64Array.from({ length: H }, (_, h) => (T + h + 1 < y.length ? y[T + h + 1] : NaN));

export function runBacktest(
  s: SeriesData, cal: Calendar, cfg: BacktestConfig, lstm?: ExternalForecaster, onProgress?: (done: number, total: number) => void,
): BacktestResult {
  const { H } = cfg;
  const last = s.y.length - 1;
  const { test, cal: calO } = originsFor(last, cfg);
  const all = [...calO, ...test];
  const closed = (t: number) => isClosed(s, cal, t);

  // 1) Ridge λ by mean MAE over the calibration origins (designs cached per origin)
  const designs = new Map<number, Design>();
  const design = (T: number) => {
    let d = designs.get(T);
    if (!d) designs.set(T, (d = buildDesign(s.y, s.promo, cal, cfg.trainStart, T)));
    return d;
  };
  const lambdaScores: Record<string, number> = {};
  for (const lam of RIDGE_LAMBDAS) {
    let tot = 0;
    for (const T of calO) {
      const f = ridgeForecast(ridgeFit(design(T), lam), s.y, s.promo, cal, T, H, closed);
      const a = actualWindow(s.y, T, H);
      let e = 0, n = 0;
      for (let h = 0; h < H; h++) if (!Number.isNaN(a[h])) { e += Math.abs(f[h] - a[h]); n++; }
      tot += e / n;
    }
    lambdaScores[String(lam)] = tot / calO.length;
  }
  const lambda = RIDGE_LAMBDAS.reduce((b, l) => (lambdaScores[String(l)] < lambdaScores[String(b)] ? l : b), RIDGE_LAMBDAS[0]);

  // 2) Forecasts at every origin
  const models: ModelName[] = ['snaive', 'holtwinters', 'ridge', ...(lstm ? (['lstm'] as const) : []), 'ensemble'];
  const fc: Partial<Record<ModelName, Float64Array[]>> = {};
  for (const m of models) fc[m] = [];
  const actuals: Float64Array[] = [];
  const hwParams: HWParams[] = [];
  const members: MemberName[] = lstm ? ['holtwinters', 'ridge', 'lstm'] : ['holtwinters', 'ridge'];
  const ensembleWeights: Weights[] = [];
  all.forEach((T, i) => {
    const a = actualWindow(s.y, T, H);
    actuals.push(a);
    fc.snaive!.push(seasonalNaiveForecast(s.y, T, H));
    const hw = holtWintersForecast(s.y, cfg.trainStart, T, H);
    hwParams.push(hw.params);
    fc.holtwinters!.push(hw.forecast);
    fc.ridge!.push(ridgeForecast(ridgeFit(design(T), lambda), s.y, s.promo, cal, T, H, closed));
    if (lstm) fc.lstm!.push(lstm(T).slice(0, H));
    // Ensemble weights from the previous `ensembleWindow` origins' errors only
    const errs: Partial<Record<MemberName, number[]>> = {};
    for (let j = Math.max(0, i - cfg.ensembleWindow); j < i; j++) {
      for (const m of members) {
        const arr = (errs[m] ??= []);
        for (let h = 0; h < H; h++) if (!Number.isNaN(actuals[j][h])) arr.push(Math.abs(fc[m]![j][h] - actuals[j][h]));
      }
    }
    const w = inverseMaeWeights(errs, members);
    ensembleWeights.push(w);
    fc.ensemble!.push(combine(Object.fromEntries(members.map((m) => [m, fc[m]![i]])), w, H));
    onProgress?.(i + 1, all.length);
  });

  // 3) Leaderboard on the test origins, and conformal bands calibrated on the calibration origins
  const nC = calO.length;
  const maseScale = seasonalNaiveScale(s.y, cfg.trainStart, test[0]);
  const A = concat(actuals.slice(nC));
  const board: BacktestResult['board'] = {};
  const conf: BacktestResult['conformal'] = {};
  for (const m of models) {
    board[m] = accuracy(A, concat(fc[m]!.slice(nC)), maseScale);
    const c = conformalFromResiduals(pooledResiduals(actuals.slice(0, nC), fc[m]!.slice(0, nC)));
    const levels = {} as Record<IntervalLevel, IntervalBounds>;
    for (const lv of INTERVAL_LEVELS) levels[lv] = { ...c.levels[lv], coverage: empiricalCoverage(actuals.slice(nC), fc[m]!.slice(nC), c.levels[lv]) };
    conf[m] = { levels, residualSd: c.residualSd };
  }
  return { config: cfg, calOrigins: calO, testOrigins: test, lambda, lambdaScores, models, forecasts: fc, actuals, board, conformal: conf, ensembleWeights, hwParams, maseScale };
}

function concat(xs: Float64Array[]): Float64Array {
  const out = new Float64Array(xs.reduce((n, x) => n + x.length, 0));
  let o = 0;
  for (const x of xs) {
    out.set(x, o);
    o += x.length;
  }
  return out;
}
