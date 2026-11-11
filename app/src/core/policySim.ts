/**
 * Daily inventory simulation of a continuous-review (s, Q) policy with stochastic lead times
 * and lost sales (unmet retail demand walks away).
 *
 * The sequence each day t is:
 *   1. receive orders due today
 *   2. serve demand: served = min(on-hand, demand), and the rest is lost
 *   3. review: while inventory position (on-hand + on-order) ≤ s, order Q
 *      (an order placed at the end of day t is exposed to the demand of days t+1 … t+L, so its
 *      lead-time demand is exactly the DDLT the safety-stock formula covers)
 *   4. pay holding cost on end-of-day on-hand
 *
 * Continuous lead times are converted to whole days by stochastic rounding (⌊L⌋ + Bernoulli(frac)),
 * which keeps E[exposure] = L. Common random numbers (one seeded demand stream per replication) mean
 * that different (s, Q) policies face identical demand, so the optimiser compares policies, not luck.
 */
import type { CostAssumption, LeadTimeAssumption } from './assumptions';
import { blockErrorStream } from './monteCarlo';
import { quantile, Rng, sampleLeadTime } from './stats';

export type DemandSource =
  | {
      kind: 'generated';
      /** Mean daily demand profile, used cyclically (t mod length): e.g. the forecast for the next 28 days. */
      mean: ArrayLike<number>;
      /** Out-of-sample forecast errors to bootstrap (actual − forecast). */
      residuals: ArrayLike<number>;
      /** Optional consecutive error paths for a block bootstrap (keeps day-to-day correlation of misses). */
      errorPaths?: ArrayLike<number>[];
      demandMultiplier?: number;
      residualMultiplier?: number;
    }
  | {
      kind: 'actual';
      /** Realised daily demand (NaN = closed, no demand). Every replication sees the same demand; only lead times vary. */
      values: ArrayLike<number>;
    };

export interface PolicyParams {
  s: number;
  Q: number;
}

export interface PolicySimConfig {
  horizon: number;
  replications: number;
  seed: number;
  leadTime: LeadTimeAssumption;
  costs: CostAssumption;
  demand: DemandSource;
  /** On-hand stock at day 0. Defaults to s + Q/2 (mid-cycle). */
  initialOnHand?: number;
  /** Days excluded from the statistics while the system settles (default 14). */
  warmup?: number;
  /** Keep a day-by-day trace of replication 0 for charts. */
  trace?: boolean;
  /**
   * Optional time-varying reorder point (one value per day). It overrides policy.s, so a
   * dynamic policy that is recomputed as new data arrives can be replayed over history.
   */
  sByDay?: ArrayLike<number>;
}

export interface PolicyMetrics {
  /** Units served ÷ units demanded. */
  fillRate: number;
  /** Share of replenishment cycles with no stockout while the order was outstanding. */
  cycleServiceLevel: number;
  avgOnHand: number;
  /** Highest on-hand stock (after receipts). Aggregated: 95th percentile across replications. */
  peakOnHand: number;
  avgInventoryValue: number;
  ordersPer30: number;
  lostUnitsPerDay: number;
  stockoutDaysShare: number;
  holdingCostPerDay: number;
  orderingCostPerDay: number;
  penaltyCostPerDay: number;
  totalCostPerDay: number;
}

export interface TraceDay {
  t: number;
  demand: number;
  served: number;
  lost: number;
  onHand: number; // end of day
  position: number; // inventory position after review
  received: number;
  ordered: number;
}

export interface PolicySimResult {
  policy: PolicyParams;
  mean: PolicyMetrics;
  /** Standard error of the replication means (Monte Carlo noise), used by the Bayesian optimiser. */
  stdErr: Pick<PolicyMetrics, 'totalCostPerDay' | 'fillRate' | 'cycleServiceLevel'>;
  perReplication: PolicyMetrics[];
  trace?: TraceDay[];
}

interface OpenOrder {
  arrive: number;
  qty: number;
  stockout: boolean;
}

export function simulatePolicy(p: PolicyParams, c: PolicySimConfig): PolicySimResult {
  if (!(p.Q > 0)) throw new RangeError('Order quantity Q must be positive.');
  const warm = c.warmup ?? 14;
  const reps: PolicyMetrics[] = [];
  let trace: TraceDay[] | undefined;
  const { holdingPerUnitDay: h, orderCost: K, stockoutPenalty: pen, unitCost } = c.costs;

  for (let r = 0; r < c.replications; r++) {
    const rngD = new Rng((c.seed * 7919 + r * 104729) >>> 0);
    const rngL = new Rng((c.seed * 6271 + r * 15485863 + 1) >>> 0);
    let onHand = c.initialOnHand ?? (c.sByDay ? c.sByDay[0] : p.s) + p.Q / 2;
    let onOrder = 0;
    let open: OpenOrder[] = [];
    let demanded = 0, served = 0, lost = 0, stockDays = 0, holding = 0, orders = 0, cycles = 0, badCycles = 0, peak = 0, onHandSum = 0;
    const days = c.horizon - warm;
    const tr: TraceDay[] | undefined = c.trace && r === 0 ? [] : undefined;
    const g0 = c.demand.kind === 'generated' ? c.demand : null;
    const block = g0?.errorPaths?.length ? blockErrorStream(rngD, g0.errorPaths) : null;
    block?.reset();

    for (let t = 0; t < c.horizon; t++) {
      const counted = t >= warm;
      // 1) receipts
      let received = 0;
      if (open.length) {
        const still: OpenOrder[] = [];
        for (const o of open) {
          if (o.arrive === t) {
            received += o.qty;
            if (counted) {
              cycles++;
              if (o.stockout) badCycles++;
            }
          } else still.push(o);
        }
        open = still;
        onHand += received;
        onOrder -= received;
      }
      if (counted) peak = Math.max(peak, onHand);
      // 2) demand
      let d: number;
      if (c.demand.kind === 'actual') {
        const v = c.demand.values[t];
        d = Number.isNaN(v) || v === undefined ? 0 : v;
      } else {
        const g = c.demand;
        const m = g.mean[t % g.mean.length] * (g.demandMultiplier ?? 1);
        const e = block ? block.next() : g.residuals[rngD.int(g.residuals.length)];
        d = Math.max(0, m + (g.residualMultiplier ?? 1) * e);
      }
      const sv = Math.min(onHand, d);
      const ls = d - sv;
      onHand -= sv;
      if (ls > 1e-9) for (const o of open) o.stockout = true;
      // 3) review
      let ordered = 0;
      const sNow = c.sByDay ? c.sByDay[Math.min(t, c.sByDay.length - 1)] : p.s;
      while (onHand + onOrder <= sNow) {
        const L = sampleLeadTime(rngL, c.leadTime.mean, c.leadTime.sd, c.leadTime.dist);
        const full = Math.floor(L);
        const exposure = full + (rngL.next() < L - full ? 1 : 0);
        open.push({ arrive: t + exposure + 1, qty: p.Q, stockout: false });
        onOrder += p.Q;
        ordered += p.Q;
        if (counted) orders++;
      }
      // 4) statistics
      if (counted) {
        demanded += d;
        served += sv;
        lost += ls;
        if (ls > 1e-9) stockDays++;
        holding += h * onHand;
        onHandSum += onHand;
      }
      tr?.push({ t, demand: d, served: sv, lost: ls, onHand, position: onHand + onOrder, received, ordered });
    }
    const avgOnHand = onHandSum / days;
    const holdingCostPerDay = holding / days;
    const orderingCostPerDay = (orders * K) / days;
    const penaltyCostPerDay = (lost * pen) / days;
    reps.push({
      fillRate: demanded > 0 ? served / demanded : 1,
      cycleServiceLevel: cycles > 0 ? 1 - badCycles / cycles : 1,
      avgOnHand,
      peakOnHand: peak,
      avgInventoryValue: avgOnHand * unitCost,
      ordersPer30: (orders / days) * 30,
      lostUnitsPerDay: lost / days,
      stockoutDaysShare: stockDays / days,
      holdingCostPerDay,
      orderingCostPerDay,
      penaltyCostPerDay,
      totalCostPerDay: holdingCostPerDay + orderingCostPerDay + penaltyCostPerDay,
    });
    if (tr) trace = tr;
  }
  const keys = Object.keys(reps[0]) as (keyof PolicyMetrics)[];
  const mean = Object.fromEntries(keys.map((k) => [k, reps.reduce((s, x) => s + x[k], 0) / reps.length])) as unknown as PolicyMetrics;
  const se = (k: keyof PolicyMetrics) => {
    if (reps.length < 2) return 0;
    const m = mean[k];
    return Math.sqrt(reps.reduce((s, x) => s + (x[k] - m) ** 2, 0) / (reps.length - 1) / reps.length);
  };
  // Peak on-hand (for the capacity constraint) is the 95th percentile of the per-replication peaks.
  // This is a high-but-stable worst case. The maximum would keep growing with the replication count.
  mean.peakOnHand = quantile(reps.map((x) => x.peakOnHand), 0.95);
  return { policy: p, mean, stdErr: { totalCostPerDay: se('totalCostPerDay'), fillRate: se('fillRate'), cycleServiceLevel: se('cycleServiceLevel') }, perReplication: reps, trace };
}

/** Economic order quantity Q* = √(2·D·K / h), with D in units/day and h in $/unit/day. */
export function eoq(dailyDemand: number, orderCost: number, holdingPerUnitDay: number): number {
  return Math.sqrt((2 * dailyDemand * orderCost) / holdingPerUnitDay);
}
