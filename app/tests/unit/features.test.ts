/** Data-free tests for the calendar extension used when forecasting beyond the data file. */
import { describe, expect, it } from 'vitest';
import { extendCalendar, isClosed, type Calendar, type SeriesData } from '../../src/core/features';

function makeCal(n: number): Calendar {
  const t0 = Date.parse('2017-08-01T00:00:00Z');
  const dow = Int8Array.from({ length: n }, (_, t) => (new Date(t0 + t * 86_400_000).getUTCDay() + 6) % 7);
  return {
    start: '2017-08-01', length: n, dow, payday: new Float64Array(n), holiday: new Float64Array(n).fill(1),
    event: new Float64Array(n), quake: new Float64Array(n), oil: Float64Array.from({ length: n }, (_, t) => 40 + t), name: new Array(n).fill(null),
  };
}

describe('extendCalendar', () => {
  const cal = makeCal(28);
  const s: SeriesData = { key: 'x', store: 1, family: 'F', city: 'Q', storeType: 'A', y: new Float64Array(20).fill(100), promo: Float64Array.from({ length: 28 }, (_, t) => (cal.dow[t] === 5 ? 30 : 10)) };
  const { cal: c2, series: [s2], extendedFrom } = extendCalendar(cal, [s], 60);

  it('keeps existing days unchanged and extends to the requested length', () => {
    expect(extendCalendar(cal, [s], 10).cal).toBe(cal);
    expect(extendedFrom).toBe(28);
    expect(c2.length).toBe(60);
    expect(Array.from(c2.holiday.subarray(0, 28))).toEqual(Array.from(cal.holiday));
    expect(Array.from(s2.promo.subarray(0, 28))).toEqual(Array.from(s.promo));
  });
  it('assumes no holidays, carries oil forward and repeats the weekday promotion pattern', () => {
    expect(c2.holiday[40]).toBe(0);
    expect(c2.oil[59]).toBe(cal.oil[27]);
    for (let t = 28; t < 60; t++) expect(s2.promo[t]).toBe(c2.dow[t] === 5 ? 30 : 10);
  });
  it('computes paydays for new days (15th and month-end only)', () => {
    expect(c2.payday[45]).toBe(1); // 2017-09-15
    expect(c2.payday[59]).toBe(0); // 2017-09-29
    expect(extendCalendar(cal, [s], 61).cal.payday[60]).toBe(1); // 2017-09-30 (month-end)
  });
  it('treats future 25 Dec / 1 Jan as closed', () => {
    const c3 = extendCalendar(cal, [s], 200).cal;
    expect(isClosed(s, c3, 146)).toBe(true); // 2017-12-25
    expect(isClosed(s, c3, 147)).toBe(false);
  });
});
