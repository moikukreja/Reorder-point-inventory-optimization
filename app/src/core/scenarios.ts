/**
 * Risk scenarios for the Monte Carlo and policy simulations. Each scenario changes the
 * lead-time distribution and/or demand. The promotion and holiday uplifts are estimated from
 * the series' own history rather than invented.
 */
import type { LeadTimeAssumption } from './assumptions';
import type { Calendar, SeriesData } from './features';

export interface ScenarioEffect {
  leadTime: LeadTimeAssumption;
  /** Multiplies forecast demand. */
  demandMultiplier: number;
  /** Multiplies forecast errors (more volatile demand). */
  residualMultiplier: number;
}

export interface Scenario {
  id: ScenarioId;
  label: string;
  /** Plain-language description for operations managers. */
  description: string;
  apply: (base: LeadTimeAssumption, u: Uplifts) => ScenarioEffect;
}

export type ScenarioId = 'base' | 'supplier-delay' | 'unreliable-supplier' | 'promo-push' | 'holiday-surge' | 'perfect-storm';

export interface Uplifts {
  /** Mean sales on high-promotion days ÷ overall mean (data-derived). */
  promo: number;
  /** Mean sales on holidays ÷ mean on normal days (data-derived). */
  holiday: number;
}

export const SCENARIOS: Scenario[] = [
  {
    id: 'base', label: 'Business as usual',
    description: 'Current forecast and the supplier lead-time assumptions as set.',
    apply: (lt) => ({ leadTime: { ...lt }, demandMultiplier: 1, residualMultiplier: 1 }),
  },
  {
    id: 'supplier-delay', label: 'Supplier delay +2 days',
    description: 'The supplier takes two extra days on average (e.g. port congestion or a strike). More demand must be covered before stock arrives.',
    apply: (lt) => ({ leadTime: { ...lt, mean: lt.mean + 2 }, demandMultiplier: 1, residualMultiplier: 1 }),
  },
  {
    id: 'unreliable-supplier', label: 'Unreliable supplier (σL × 2)',
    description: 'Average lead time is unchanged but deliveries are twice as erratic. This is the D̄²·σL² term in the formula.',
    apply: (lt) => ({ leadTime: { ...lt, sd: lt.sd * 2 }, demandMultiplier: 1, residualMultiplier: 1 }),
  },
  {
    id: 'promo-push', label: 'Promotion push',
    description: 'A heavy promotion period. Demand rises by the uplift seen historically on high-promotion days, and forecast errors grow by 25%.',
    apply: (lt, u) => ({ leadTime: { ...lt }, demandMultiplier: u.promo, residualMultiplier: 1.25 }),
  },
  {
    id: 'holiday-surge', label: 'Holiday surge',
    description: 'A national-holiday period, with demand at the historical holiday uplift and forecast errors 25% larger.',
    apply: (lt, u) => ({ leadTime: { ...lt }, demandMultiplier: u.holiday, residualMultiplier: 1.25 }),
  },
  {
    id: 'perfect-storm', label: 'Perfect storm',
    description: 'Promotion push + supplier delay + erratic deliveries at the same time. This is the stress test for the buffer.',
    apply: (lt, u) => ({ leadTime: { ...lt, mean: lt.mean + 2, sd: lt.sd * 2 }, demandMultiplier: u.promo, residualMultiplier: 1.25 }),
  },
];

/** Estimates promotion and holiday uplifts from open days in [from, to]. */
export function estimateUplifts(s: SeriesData, cal: Calendar, from: number, to: number): Uplifts {
  const promoVals: number[] = [];
  for (let t = from; t <= to; t++) if (!Number.isNaN(s.y[t])) promoVals.push(s.promo[t]);
  const sorted = promoVals.slice().sort((a, b) => a - b);
  const q75 = sorted[Math.floor(sorted.length * 0.75)];
  let all = 0, nAll = 0, hi = 0, nHi = 0, hol = 0, nHol = 0, norm = 0, nNorm = 0;
  for (let t = from; t <= to; t++) {
    const v = s.y[t];
    if (Number.isNaN(v)) continue;
    all += v; nAll++;
    if (s.promo[t] > q75) { hi += v; nHi++; }
    if (cal.holiday[t] === 1) { hol += v; nHol++; } else { norm += v; nNorm++; }
  }
  const promo = nHi ? hi / nHi / (all / nAll) : 1;
  const holiday = nHol ? hol / nHol / (norm / nNorm) : 1;
  return { promo: Math.max(1, promo), holiday: Math.max(1, holiday) };
}
