/**
 * New observations ("keep the possibility of adding new data"). It covers the row format,
 * CSV parsing, and validation against the series' current last date. It is pure and has no DOM
 * dependency, so it is unit-tested in Node and also used by the worker.
 *
 * CSV columns (header required, case-insensitive):
 *   date, units_sold, onpromotion, holiday[, oil]
 *   date        YYYY-MM-DD; rows must continue day by day from the last date in the data
 *   units_sold  ≥ 0, or empty / "closed" for a closed day (no demand)
 *   onpromotion integer ≥ 0 (items on promotion that day)
 *   holiday     0 or 1
 *   oil         optional WTI price (> 0); the last known price is carried forward when blank
 */
import { daysBetween } from './schema';

export interface NewRow {
  date: string;
  sales: number | null;
  onpromotion: number;
  holiday: 0 | 1;
  oil?: number | null;
}

export interface RowIssue {
  line: number; // 1-based data line (excluding the header)
  message: string;
}

export interface ValidationResult {
  rows: NewRow[];
  issues: RowIssue[];
}

const nextDay = (iso: string) => new Date(Date.parse(`${iso}T00:00:00Z`) + 86_400_000).toISOString().slice(0, 10);

/** Minimal CSV parser for simple comma-separated files (quoted fields supported). */
export function parseCsv(text: string): string[][] {
  const out: string[][] = [];
  for (const raw of text.replace(/^﻿/, '').split(/\r?\n/)) {
    if (!raw.trim()) continue;
    const cells: string[] = [];
    let cur = '', q = false;
    for (let i = 0; i < raw.length; i++) {
      const ch = raw[i];
      if (q) {
        if (ch === '"' && raw[i + 1] === '"') { cur += '"'; i++; }
        else if (ch === '"') q = false;
        else cur += ch;
      } else if (ch === '"') q = true;
      else if (ch === ',') { cells.push(cur.trim()); cur = ''; }
      else cur += ch;
    }
    cells.push(cur.trim());
    out.push(cells);
  }
  return out;
}

const COLS = { date: ['date'], sales: ['units_sold', 'sales', 'units'], promo: ['onpromotion', 'promo', 'items_on_promotion'], holiday: ['holiday', 'is_holiday'], oil: ['oil', 'dcoilwtico', 'oil_price'] };

/** Converts CSV text to candidate rows, then validates them. */
export function rowsFromCsv(text: string, lastDate: string): ValidationResult {
  const table = parseCsv(text);
  if (table.length < 2) return { rows: [], issues: [{ line: 0, message: 'The file needs a header row and at least one data row.' }] };
  const header = table[0].map((h) => h.toLowerCase().replace(/\s+/g, '_'));
  const col = (names: string[]) => header.findIndex((h) => names.includes(h));
  const idx = { date: col(COLS.date), sales: col(COLS.sales), promo: col(COLS.promo), holiday: col(COLS.holiday), oil: col(COLS.oil) };
  const missing = (['date', 'sales', 'promo', 'holiday'] as const).filter((k) => idx[k] < 0);
  if (missing.length) return { rows: [], issues: [{ line: 0, message: `Missing column(s): ${missing.map((m) => ({ date: 'date', sales: 'units_sold', promo: 'onpromotion', holiday: 'holiday' })[m]).join(', ')}.` }] };
  const candidates = table.slice(1).map((r) => ({
    date: r[idx.date] ?? '',
    sales: r[idx.sales] ?? '',
    onpromotion: r[idx.promo] ?? '',
    holiday: r[idx.holiday] ?? '',
    oil: idx.oil >= 0 ? r[idx.oil] ?? '' : '',
  }));
  return validateRows(candidates, lastDate);
}

export interface RawRow {
  date: string;
  sales: string | number | null;
  onpromotion: string | number;
  holiday: string | number | boolean;
  oil?: string | number | null;
}

/** Validates rows that must continue the series day by day from `lastDate`. */
export function validateRows(raw: RawRow[], lastDate: string): ValidationResult {
  const rows: NewRow[] = [];
  const issues: RowIssue[] = [];
  let expected = nextDay(lastDate);
  raw.forEach((r, i) => {
    const line = i + 1;
    const problems: string[] = [];
    const date = String(r.date).trim();
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || Number.isNaN(Date.parse(`${date}T00:00:00Z`))) problems.push(`date "${date}" is not YYYY-MM-DD`);
    else if (date !== expected) {
      const gap = daysBetween(expected, date);
      problems.push(gap < 0 ? `date ${date} is already in the data or repeated (expected ${expected})` : `date ${date} skips ${gap} day(s): expected ${expected}`);
    }
    const sRaw = r.sales === null ? '' : String(r.sales).trim().toLowerCase();
    let sales: number | null = null;
    if (sRaw !== '' && sRaw !== 'closed') {
      sales = Number(sRaw);
      if (!Number.isFinite(sales) || sales < 0) problems.push(`units sold "${r.sales}" must be a number ≥ 0 (or blank/"closed")`);
    }
    const promo = Number(String(r.onpromotion).trim() || '0');
    if (!Number.isInteger(promo) || promo < 0) problems.push(`items on promotion "${r.onpromotion}" must be a whole number ≥ 0`);
    const hRaw = String(r.holiday).trim().toLowerCase();
    const holiday = hRaw === '1' || hRaw === 'true' || hRaw === 'yes' ? 1 : hRaw === '0' || hRaw === 'false' || hRaw === 'no' || hRaw === '' ? 0 : -1;
    if (holiday < 0) problems.push(`holiday "${r.holiday}" must be 0 or 1`);
    let oil: number | null = null;
    const oRaw = r.oil === undefined || r.oil === null ? '' : String(r.oil).trim();
    if (oRaw !== '') {
      oil = Number(oRaw);
      if (!Number.isFinite(oil) || oil <= 0) problems.push(`oil "${r.oil}" must be a positive number (or blank)`);
    }
    if (problems.length) issues.push({ line, message: problems.join('; ') });
    else {
      rows.push({ date, sales, onpromotion: promo, holiday: holiday as 0 | 1, oil });
      expected = nextDay(date);
    }
  });
  return { rows, issues };
}

/** CSV export of rows in the same format the importer reads. */
export function rowsToCsv(rows: NewRow[]): string {
  return ['date,units_sold,onpromotion,holiday,oil', ...rows.map((r) => [r.date, r.sales ?? 'closed', r.onpromotion, r.holiday, r.oil ?? ''].join(','))].join('\n');
}
