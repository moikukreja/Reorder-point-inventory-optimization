/**
 * Dataset schema. The dashboard reads the subset exported by
 * notebooks/01_prepare_and_validate.ipynb (schema "favorita-subset/v1"), validates it,
 * and converts it into typed arrays for the models.
 *
 * Units: sales are units/day, and lead times are days. One period is one day everywhere.
 */

export interface SubsetFile {
  schema: 'favorita-subset/v1';
  source: string;
  generatedAt: string;
  startDate: string; // ISO date of day index 0
  historyEnd: string; // last date with observed sales
  calendarEnd: string; // last date with known calendar/promotion inputs
  units: { sales: string; leadTime: string };
  calendar: {
    holiday: number[];
    event: number[];
    quake: number[];
    oil: number[];
    name: (string | null)[];
  };
  series: SubsetSeries[];
}

export interface SubsetSeries {
  key: string; // "store|family"
  store: number;
  family: string;
  city: string;
  state: string;
  storeType: string;
  cluster: number;
  sales: (number | null)[]; // length = history days; null = closed
  onpromotion: number[]; // length = calendar days (includes known future promotions)
}

/** Validation error with a readable message for the UI. */
export class DataValidationError extends Error {}

/** Parses and validates a subset file. Throws DataValidationError with a specific reason. */
export function validateSubset(raw: unknown): SubsetFile {
  const f = raw as Partial<SubsetFile>;
  const fail = (msg: string): never => {
    throw new DataValidationError(msg);
  };
  if (!f || typeof f !== 'object') fail('File is not a JSON object.');
  if (f.schema !== 'favorita-subset/v1') fail(`Unsupported schema "${String(f.schema)}" (expected favorita-subset/v1).`);
  for (const k of ['startDate', 'historyEnd', 'calendarEnd'] as const) {
    if (typeof f[k] !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(f[k] as string)) fail(`Field "${k}" must be an ISO date.`);
  }
  const calLen = daysBetween(f.startDate!, f.calendarEnd!) + 1;
  const histLen = daysBetween(f.startDate!, f.historyEnd!) + 1;
  if (histLen < 400) fail('At least 400 days of history are required (for the 364-day lag).');
  if (calLen < histLen) fail('calendarEnd must not be before historyEnd.');
  const c = f.calendar;
  if (!c) fail('Missing "calendar".');
  for (const k of ['holiday', 'event', 'quake', 'oil', 'name'] as const) {
    if (!Array.isArray(c![k]) || c![k].length !== calLen) fail(`calendar.${k} must have ${calLen} entries.`);
  }
  if (!Array.isArray(f.series) || f.series.length === 0) fail('No series found.');
  for (const s of f.series!) {
    if (typeof s.key !== 'string') fail('Series without a key.');
    if (!Array.isArray(s.sales) || s.sales.length !== histLen) fail(`Series ${s.key}: sales must have ${histLen} entries.`);
    if (!Array.isArray(s.onpromotion) || s.onpromotion.length !== calLen) fail(`Series ${s.key}: onpromotion must have ${calLen} entries.`);
    if (s.sales.some((v) => v !== null && (!Number.isFinite(v) || v < 0))) fail(`Series ${s.key}: sales must be non-negative numbers or null.`);
  }
  return f as SubsetFile;
}

/** Whole days from ISO date a to ISO date b (UTC, so no DST issues). */
export function daysBetween(a: string, b: string): number {
  return Math.round((Date.parse(`${b}T00:00:00Z`) - Date.parse(`${a}T00:00:00Z`)) / 86_400_000);
}

/** ISO date for a day index relative to start. */
export function isoAt(start: string, t: number): string {
  return new Date(Date.parse(`${start}T00:00:00Z`) + t * 86_400_000).toISOString().slice(0, 10);
}
