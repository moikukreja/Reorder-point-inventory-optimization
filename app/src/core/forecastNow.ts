/**
 * Production forecast from the latest observed day T. It feeds the safety stock, the Monte Carlo
 * and the optimiser.
 *
 * - Holt-Winters and Ridge are refitted on all data up to T (Ridge λ comes from the backtest).
 * - The LSTM (if loaded) contributes days 1–14, and the ensemble renormalises beyond that.
 * - Ensemble weights are 1/MAE over the most recent `ensembleWindow` backtest origins.
 * - Intervals and the residual pool use the ensemble's most recent 12 backtest origins
 *   (rolling split-conformal), so they adapt as new data arrives.
 */
import { conformalFromResiduals, pooledResiduals, type IntervalLevel } from './conformal';
import { dateAt, isClosed, type Calendar, type SeriesData } from './features';
import { holtWintersForecast } from './models/holtWinters';
import { buildDesign, ridgeFit, ridgeForecast } from './models/ridge';
import { seasonalNaiveForecast } from './models/seasonalNaive';
import { combine, inverseMaeWeights, type MemberName, type Weights } from './models/ensemble';
import type { BacktestResult, ExternalForecaster, ModelName } from './backtest';

export interface CurrentForecast {
  origin: number;
  originDate: string;
  H: number;
  dates: string[];
  models: Partial<Record<ModelName, Float64Array>>;
  ensemble: Float64Array;
  weights: Weights;
  bands: Record<IntervalLevel, { lo: Float64Array; hi: Float64Array }>;
  /** Pool of recent out-of-sample ensemble errors (actual − forecast), units/day. */
  residuals: Float64Array;
  /** The same recent errors kept as consecutive 14-day paths (one per origin), for the block bootstrap. */
  errorPaths: Float64Array[];
  /** σ_D for the AI safety stock: SD of the ensemble's daily forecast errors. */
  residualSd: number;
}

export function forecastNow(s: SeriesData, cal: Calendar, bt: BacktestResult, H = 28, lstm?: ExternalForecaster, recentOrigins = 12): CurrentForecast {
  const T = s.y.length - 1;
  if (T + H >= cal.length) throw new RangeError(`Calendar ends ${cal.length - 1 - T} days after the data; ${H} days needed.`);
  const closed = (t: number) => isClosed(s, cal, t);
  const models: CurrentForecast['models'] = {};
  models.snaive = seasonalNaiveForecast(s.y, T, H);
  models.holtwinters = holtWintersForecast(s.y, bt.config.trainStart, T, H).forecast;
  models.ridge = ridgeForecast(ridgeFit(buildDesign(s.y, s.promo, cal, bt.config.trainStart, T), bt.lambda), s.y, s.promo, cal, T, H, closed);
  if (lstm) models.lstm = lstm(T);
  for (const m of Object.keys(models) as ModelName[]) {
    const f = models[m]!;
    for (let h = 0; h < f.length; h++) if (closed(T + h + 1)) f[h] = 0;
  }

  // Ensemble weights from the most recent origins' errors
  const members: MemberName[] = lstm ? ['holtwinters', 'ridge', 'lstm'] : ['holtwinters', 'ridge'];
  const nO = bt.actuals.length;
  const errs: Partial<Record<MemberName, number[]>> = {};
  for (let j = Math.max(0, nO - bt.config.ensembleWindow); j < nO; j++) {
    for (const m of members) {
      const arr = (errs[m] ??= []);
      for (let h = 0; h < bt.config.H; h++) if (!Number.isNaN(bt.actuals[j][h])) arr.push(Math.abs(bt.forecasts[m]![j][h] - bt.actuals[j][h]));
    }
  }
  const weights = inverseMaeWeights(errs, members);
  const ensemble = combine(Object.fromEntries(members.map((m) => [m, models[m]!])), weights, H);
  for (let h = 0; h < H; h++) if (closed(T + h + 1)) ensemble[h] = 0;

  // Rolling conformal on the most recent origins
  const from = Math.max(0, nO - recentOrigins);
  const residuals = pooledResiduals(bt.actuals.slice(from), bt.forecasts.ensemble!.slice(from));
  const errorPaths = bt.actuals.slice(from).map((a, i) => a.map((v, h) => v - bt.forecasts.ensemble![from + i][h]));
  const conf = conformalFromResiduals(residuals);
  const band = (lv: IntervalLevel) => ({
    lo: ensemble.map((v, h) => (closed(T + h + 1) ? 0 : Math.max(0, v + conf.levels[lv].lo))),
    hi: ensemble.map((v, h) => (closed(T + h + 1) ? 0 : v + conf.levels[lv].hi)),
  });
  return {
    origin: T,
    originDate: dateAt(cal, T),
    H,
    dates: Array.from({ length: H }, (_, h) => dateAt(cal, T + h + 1)),
    models,
    ensemble,
    weights,
    bands: { '0.8': band('0.8'), '0.95': band('0.95') },
    residuals,
    errorPaths,
    residualSd: conf.residualSd,
  };
}
