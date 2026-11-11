/**
 * Synthetic demo dataset in the same schema as the Favorita subset. Anyone opening the public
 * site can explore the dashboard without the course data (which is never deployed).
 * It is clearly labelled as synthetic everywhere it appears.
 *
 * Demand = level × trend × weekday profile × yearly cycle × promotion lift × holiday effect × noise,
 * with persistent (AR(1)) log-noise. That gives the models real, learnable structure, and gives
 * the safety stock correlated forecast misses, like real retail data.
 */
import { Rng } from '../core/stats';
import type { SubsetFile, SubsetSeries } from './schema';

const START = '2013-01-01';
const HIST_DAYS = 1688; // 2013-01-01 → 2017-08-15, like the real subset
const CAL_DAYS = 1704; // + 16 days of known future promotions

const iso = (t: number) => new Date(Date.parse(`${START}T00:00:00Z`) + t * 86_400_000).toISOString().slice(0, 10);

/** Fixed-date public holidays (month-day) plus a moving "event" (late-November sales day). */
const HOLIDAYS = ['01-01', '02-12', '04-14', '05-01', '05-24', '08-10', '10-09', '11-02', '11-03', '12-06', '12-24', '12-25', '12-31'];

export function generateDemoData(seed = 2026): SubsetFile {
  const rng = new Rng(seed);
  const holiday: number[] = [], event: number[] = [], quake: number[] = [], oil: number[] = [], name: (string | null)[] = [];
  let o = 95;
  for (let t = 0; t < CAL_DAYS; t++) {
    const d = iso(t), md = d.slice(5);
    const isHol = HOLIDAYS.includes(md) ? 1 : 0;
    const dt = new Date(`${d}T00:00:00Z`);
    const isEvt = dt.getUTCMonth() === 10 && dt.getUTCDay() === 5 && dt.getUTCDate() >= 23 && dt.getUTCDate() <= 29 ? 1 : 0; // last Friday of November
    holiday.push(isHol);
    event.push(isEvt);
    quake.push(0);
    o = Math.max(25, o + 0.9 * rng.normal() - (t > 600 && t < 760 ? 0.25 : 0)); // random walk with a 2014–15 slump
    oil.push(Math.round(o * 100) / 100);
    name.push(isHol ? 'Demo public holiday' : isEvt ? 'Demo sales event' : null);
  }
  const specs = [
    { key: 'DEMO-1|BEVERAGES', store: 101, family: 'BEVERAGES', level: 6000, promoLift: 0.07, dow: [0.85, 0.8, 0.88, 0.78, 0.95, 1.45, 1.55] },
    { key: 'DEMO-1|GROCERY I', store: 101, family: 'GROCERY I', level: 8000, promoLift: 0.04, dow: [0.95, 0.9, 0.95, 0.85, 0.95, 1.3, 1.4] },
    { key: 'DEMO-2|CLEANING', store: 102, family: 'CLEANING', level: 2000, promoLift: 0.03, dow: [0.95, 0.9, 1.0, 0.85, 0.95, 1.2, 1.3] },
  ];
  const series: SubsetSeries[] = specs.map((sp) => {
    const r = new Rng(seed + sp.store * 7 + sp.level);
    const sales: (number | null)[] = [];
    const promo: number[] = [];
    let e = 0;
    let burst = 0;
    for (let t = 0; t < CAL_DAYS; t++) {
      const d = iso(t), md = d.slice(5);
      // Promotions start in 2014-04 (as in the real data) and come in bursts of a few days
      if (t >= 455) {
        if (burst <= 0 && r.next() < 0.08) burst = 2 + r.int(6);
        promo.push(burst > 0 ? 10 + r.int(60) : r.int(6));
        burst--;
      } else promo.push(0);
      if (t >= HIST_DAYS) continue;
      if (md === '12-25' || md === '01-01') { sales.push(null); continue; }
      const dow = (new Date(`${d}T00:00:00Z`).getUTCDay() + 6) % 7;
      const year = 1 + 0.08 * Math.sin((2 * Math.PI * (t - 60)) / 365.25) + 0.12 * Math.max(0, Math.cos((2 * Math.PI * (t - 350)) / 365.25)) ** 6;
      const trend = 1 + 0.12 * (t / 1688);
      e = 0.55 * e + 0.11 * r.normal();
      const lift = (1 + sp.promoLift * Math.log1p(promo[t])) * (holiday[t] ? 1.18 : 1) * (event[t] ? 1.35 : 1);
      sales.push(Math.round(sp.level * trend * sp.dow[dow] * year * lift * Math.exp(e)));
    }
    return { key: sp.key, store: sp.store, family: sp.family, city: 'Demo City', state: 'Demo', storeType: 'A', cluster: 0, sales, onpromotion: promo };
  });
  return {
    schema: 'favorita-subset/v1',
    source: 'SYNTHETIC demo data generated in the browser: not real sales, for exploring the dashboard only.',
    generatedAt: new Date().toISOString().slice(0, 19),
    startDate: START,
    historyEnd: iso(HIST_DAYS - 1),
    calendarEnd: iso(CAL_DAYS - 1),
    units: { sales: 'units/day', leadTime: 'days' },
    calendar: { holiday, event, quake, oil, name },
    series,
  };
}

export const isDemo = (source: string) => source.startsWith('SYNTHETIC');
