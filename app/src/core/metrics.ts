/**
 * Forecast accuracy metrics (definitions identical to the reference notebook §6.3).
 * NaN actuals (closed days) are ignored. Bias = mean(forecast − actual), so a positive
 * value means over-forecasting, which leads to excess stock.
 */

export interface AccuracyMetrics {
  mae: number;
  rmse: number;
  wape: number; // percent
  mase: number; // MAE ÷ in-sample MAE of the weekly seasonal-naive forecast
  bias: number;
  n: number;
}

export function accuracy(actual: ArrayLike<number>, forecast: ArrayLike<number>, maseScale: number): AccuracyMetrics {
  let sa = 0, se = 0, se2 = 0, sy = 0, n = 0;
  for (let i = 0; i < actual.length; i++) {
    const y = actual[i];
    if (Number.isNaN(y)) continue;
    const e = forecast[i] - y;
    sa += Math.abs(e);
    se += e;
    se2 += e * e;
    sy += y;
    n++;
  }
  const mae = sa / n;
  return { mae, rmse: Math.sqrt(se2 / n), wape: (sa / sy) * 100, mase: mae / maseScale, bias: se / n, n };
}

/** In-sample MAE of the lag-7 seasonal naive over y[from..to], which is the MASE denominator. */
export function seasonalNaiveScale(y: ArrayLike<number>, from: number, to: number): number {
  let s = 0, n = 0;
  for (let t = from + 7; t <= to; t++) {
    const d = Math.abs(y[t] - y[t - 7]);
    if (!Number.isNaN(d)) {
      s += d;
      n++;
    }
  }
  return s / n;
}
