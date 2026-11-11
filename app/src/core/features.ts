/**
 * Calendar and feature engineering. This mirrors section 6.1 of the reference notebook exactly
 * (same order, same definitions), so Ridge coefficients and forecasts match the Python
 * results to floating-point precision.
 *
 * Every feature is known before the day it describes, so there is no leakage: calendar
 * flags, planned promotions, yesterday's oil price, and sales lags of at least 7 days.
 */
import { isoAt, type SubsetFile, type SubsetSeries } from '../data/schema';

export const FEATURE_NAMES = [
  'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6',
  'f_sin1', 'f_cos1', 'f_sin2', 'f_cos2', 'f_sin3', 'f_cos3',
  'promo_log', 'holiday', 'event', 'quake', 'payday', 'oil', 'trend',
  'lag7', 'lag14', 'lag28', 'lag364', 'roll28_7',
] as const;

/** Human-readable labels for the dashboard's feature-effect chart. */
export const FEATURE_LABELS: Record<(typeof FEATURE_NAMES)[number], string> = {
  dow_1: 'Tuesday', dow_2: 'Wednesday', dow_3: 'Thursday', dow_4: 'Friday', dow_5: 'Saturday', dow_6: 'Sunday',
  f_sin1: 'Yearly cycle (sin 1)', f_cos1: 'Yearly cycle (cos 1)', f_sin2: 'Yearly cycle (sin 2)',
  f_cos2: 'Yearly cycle (cos 2)', f_sin3: 'Yearly cycle (sin 3)', f_cos3: 'Yearly cycle (cos 3)',
  promo_log: 'Promotions (log items)', holiday: 'Holiday', event: 'Event (Black Friday etc.)', quake: 'Earthquake period',
  payday: 'Payday', oil: 'Oil price (t−1)', trend: 'Trend',
  lag7: 'Sales 7 days ago', lag14: 'Sales 14 days ago', lag28: 'Sales 28 days ago', lag364: 'Sales 52 weeks ago',
  roll28_7: '28-day mean (to t−7)',
};

export const N_FEATURES = FEATURE_NAMES.length;

/** Calendar inputs aligned to day indices 0..length-1. */
export interface Calendar {
  start: string;
  length: number;
  dow: Int8Array; // 0 = Monday … 6 = Sunday
  payday: Float64Array; // 15th or last day of month
  holiday: Float64Array;
  event: Float64Array;
  quake: Float64Array;
  oil: Float64Array;
  name: (string | null)[];
}

/** One demand series in model-ready form. */
export interface SeriesData {
  key: string;
  store: number;
  family: string;
  city: string;
  storeType: string;
  /** Observed sales, length = history days; NaN = closed (no demand observed). */
  y: Float64Array;
  /** Items on promotion, length = calendar length (includes known future promotions). */
  promo: Float64Array;
}

/** Day index where model training starts (2014-04-01: onpromotion is all-zero before). */
export const TRAIN_START_DATE = '2014-04-01';

export function buildCalendar(f: SubsetFile): Calendar {
  const n = f.calendar.holiday.length;
  const dow = new Int8Array(n);
  const payday = new Float64Array(n);
  const t0 = Date.parse(`${f.startDate}T00:00:00Z`);
  for (let t = 0; t < n; t++) {
    const d = new Date(t0 + t * 86_400_000);
    dow[t] = (d.getUTCDay() + 6) % 7;
    const next = new Date(t0 + (t + 1) * 86_400_000);
    payday[t] = d.getUTCDate() === 15 || next.getUTCDate() === 1 ? 1 : 0;
  }
  return {
    start: f.startDate,
    length: n,
    dow,
    payday,
    holiday: Float64Array.from(f.calendar.holiday),
    event: Float64Array.from(f.calendar.event),
    quake: Float64Array.from(f.calendar.quake),
    oil: Float64Array.from(f.calendar.oil),
    name: f.calendar.name.slice(),
  };
}

export function toSeriesData(s: SubsetSeries): SeriesData {
  return {
    key: s.key,
    store: s.store,
    family: s.family,
    city: s.city,
    storeType: s.storeType,
    y: Float64Array.from(s.sales, (v) => (v === null ? NaN : v)),
    promo: Float64Array.from(s.onpromotion),
  };
}

/** Day index for an ISO date relative to the calendar start. */
export function indexOf(cal: Calendar, iso: string): number {
  return Math.round((Date.parse(`${iso}T00:00:00Z`) - Date.parse(`${cal.start}T00:00:00Z`)) / 86_400_000);
}

export const dateAt = (cal: Calendar, t: number): string => isoAt(cal.start, t);

/**
 * Imputed copy of y[0..upto] (inclusive), used for lags only. A closed day takes the
 * same-weekday value one week earlier (0 if none). It is never used as a training target.
 */
export function impute(y: ArrayLike<number>, upto = y.length - 1, out?: Float64Array): Float64Array {
  const z = out ?? new Float64Array(upto + 1);
  for (let t = 0; t <= upto; t++) {
    const v = y[t];
    z[t] = Number.isNaN(v) ? (t >= 7 ? z[t - 7] : 0) : v;
  }
  return z;
}

/**
 * Feature row for day index t. `z` must contain values (observed, imputed or predicted)
 * for every day before t. Writes into `out` if given (avoids allocations in loops).
 */
export function featureRow(t: number, z: ArrayLike<number>, promo: ArrayLike<number>, cal: Calendar, out?: Float64Array): Float64Array {
  const r = out ?? new Float64Array(N_FEATURES);
  const dow = cal.dow[t];
  for (let k = 1; k <= 6; k++) r[k - 1] = dow === k ? 1 : 0;
  let i = 6;
  for (let k = 1; k <= 3; k++) {
    const a = (2 * Math.PI * k * t) / 365.25;
    r[i++] = Math.sin(a);
    r[i++] = Math.cos(a);
  }
  r[i++] = Math.log1p(promo[t]);
  r[i++] = cal.holiday[t];
  r[i++] = cal.event[t];
  r[i++] = cal.quake[t];
  r[i++] = cal.payday[t];
  r[i++] = cal.oil[Math.max(t - 1, 0)];
  r[i++] = t / 365.25;
  r[i++] = z[t - 7];
  r[i++] = z[t - 14];
  r[i++] = z[t - 28];
  r[i++] = z[t - 364];
  let s = 0;
  for (let j = t - 34; j <= t - 7; j++) s += z[j];
  r[i++] = s / 28;
  return r;
}

/**
 * Extends the calendar (and each series' promotion plan) to `length` days, for forecasting past
 * the dates the file covers. These assumptions are documented in the dashboard:
 *  - holiday / event / quake = 0 (none known; the user can flag holidays when adding data)
 *  - oil = last known price (carried forward)
 *  - promotions = same-weekday mean of the last 28 days with known promotions, rounded
 * Days that already exist are never changed. Returns new objects and leaves the inputs untouched.
 */
export function extendCalendar(cal: Calendar, series: SeriesData[], length: number): { cal: Calendar; series: SeriesData[]; extendedFrom: number } {
  const n0 = cal.length;
  if (length <= n0) return { cal, series, extendedFrom: n0 };
  const grow = (a: Float64Array, fill: (t: number) => number) => {
    const out = new Float64Array(length);
    out.set(a);
    for (let t = n0; t < length; t++) out[t] = fill(t);
    return out;
  };
  const t0 = Date.parse(`${cal.start}T00:00:00Z`);
  const dow = new Int8Array(length);
  dow.set(cal.dow);
  const payday = grow(cal.payday, () => 0);
  for (let t = n0; t < length; t++) {
    const d = new Date(t0 + t * 86_400_000);
    dow[t] = (d.getUTCDay() + 6) % 7;
    payday[t] = d.getUTCDate() === 15 || new Date(t0 + (t + 1) * 86_400_000).getUTCDate() === 1 ? 1 : 0;
  }
  const lastOil = cal.oil[n0 - 1];
  const out: Calendar = {
    ...cal,
    length,
    dow,
    payday,
    holiday: grow(cal.holiday, () => 0),
    event: grow(cal.event, () => 0),
    quake: grow(cal.quake, () => 0),
    oil: grow(cal.oil, () => lastOil),
    name: [...cal.name, ...new Array<string | null>(length - n0).fill(null)],
  };
  const ext = series.map((s) => {
    const known = s.promo.length;
    const byDow = new Array<number>(7).fill(0), cnt = new Array<number>(7).fill(0);
    for (let t = Math.max(0, known - 28); t < known; t++) {
      byDow[out.dow[t]] += s.promo[t];
      cnt[out.dow[t]]++;
    }
    const promo = new Float64Array(length);
    promo.set(s.promo.subarray(0, Math.min(known, length)));
    for (let t = known; t < length; t++) promo[t] = Math.round(byDow[out.dow[t]] / Math.max(1, cnt[out.dow[t]]));
    return { ...s, promo };
  });
  return { cal: out, series: ext, extendedFrom: n0 };
}

/** True when day t is a closed day (no demand). Future closures follow the historical pattern (25 Dec, 1 Jan). */
export function isClosed(series: SeriesData, cal: Calendar, t: number): boolean {
  if (t < series.y.length) return Number.isNaN(series.y[t]);
  const d = dateAt(cal, t).slice(5);
  return d === '12-25' || d === '01-01';
}
