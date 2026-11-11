/**
 * Turns the (s, Q) inventory simulation into a constrained black-box problem for the optimiser.
 *
 *   minimise   expected cost per day = holding + ordering + lost-sales penalty
 *   subject to fill rate ≥ target (or cycle service level ≥ target)
 *              average inventory value ≤ working-capital budget
 *              peak on-hand ≤ warehouse capacity
 *              orders per 30 days ≤ maximum (receiving-dock / supplier limit)
 *
 * Each evaluation runs the full policy simulation. Common random numbers make the objective
 * a smooth, repeatable function of (s, Q).
 */
import type { BoProblem, Evaluation } from './bayesOpt';
import { simulatePolicy, type PolicyMetrics, type PolicySimConfig } from './policySim';

export interface PolicyConstraints {
  serviceMetric: 'fillRate' | 'cycleServiceLevel';
  minService: number;
  maxInventoryValue: number | null;
  maxPeakOnHand: number | null;
  maxOrdersPer30: number | null;
}

export interface PolicyBounds {
  s: [number, number];
  Q: [number, number];
}

export type SearchMode = 'sQ' | 's';

export interface PolicyEvalInfo {
  s: number;
  Q: number;
  metrics: PolicyMetrics;
}

/** Default search box: s from half the lead-time demand to 6 DDLT SDs above it; Q from 1 to 20 days of demand. */
export function defaultBounds(dBar: number, meanL: number, sdDdlt: number): PolicyBounds {
  return { s: [0.5 * dBar * meanL, dBar * meanL + 6 * sdDdlt], Q: [dBar, 20 * dBar] };
}

/**
 * Default constraints, scaled to the series' demand: 98% fill rate, an inventory value of at most
 * 6 days of demand at cost, a warehouse capacity of 12 days of demand, and at most 10 orders a month.
 */
export function defaultConstraints(dBar: number, unitCost: number): PolicyConstraints {
  return { serviceMetric: 'fillRate', minService: 0.98, maxInventoryValue: 6 * dBar * unitCost, maxPeakOnHand: 12 * dBar, maxOrdersPer30: 10 };
}

export const constraintLabels = (c: PolicyConstraints): string[] => [
  `${c.serviceMetric === 'fillRate' ? 'Fill rate' : 'Cycle service level'} ≥ ${(c.minService * 100).toFixed(1)}%`,
  ...(c.maxInventoryValue !== null ? [`Avg inventory value ≤ $${Math.round(c.maxInventoryValue).toLocaleString('en-US')}`] : []),
  ...(c.maxPeakOnHand !== null ? [`Peak on-hand ≤ ${Math.round(c.maxPeakOnHand).toLocaleString('en-US')} units`] : []),
  ...(c.maxOrdersPer30 !== null ? [`Orders ≤ ${c.maxOrdersPer30} per 30 days`] : []),
];

/** Normalised constraint slacks (≥ 0 = satisfied). Scales: 1 pp of service, 5% of budget/capacity, 1 order. */
export function slacks(m: PolicyMetrics, c: PolicyConstraints): number[] {
  const out = [(m[c.serviceMetric] - c.minService) / 0.01];
  if (c.maxInventoryValue !== null) out.push((c.maxInventoryValue - m.avgInventoryValue) / (0.05 * c.maxInventoryValue));
  if (c.maxPeakOnHand !== null) out.push((c.maxPeakOnHand - m.peakOnHand) / (0.05 * c.maxPeakOnHand));
  if (c.maxOrdersPer30 !== null) out.push(c.maxOrdersPer30 - m.ordersPer30);
  return out;
}

export function toPolicy(x: number[], b: PolicyBounds, mode: SearchMode, fixedQ: number): { s: number; Q: number } {
  const s = b.s[0] + x[0] * (b.s[1] - b.s[0]);
  const Q = mode === 's' ? fixedQ : b.Q[0] + x[1] * (b.Q[1] - b.Q[0]);
  return { s, Q };
}

export function fromPolicy(s: number, Q: number, b: PolicyBounds, mode: SearchMode): number[] {
  const xs = (s - b.s[0]) / (b.s[1] - b.s[0]);
  return mode === 's' ? [xs] : [xs, (Q - b.Q[0]) / (b.Q[1] - b.Q[0])];
}

export function policyProblem(
  sim: Omit<PolicySimConfig, 'trace'>, bounds: PolicyBounds, cons: PolicyConstraints, mode: SearchMode, fixedQ: number,
): BoProblem<PolicyEvalInfo> & { evaluations: () => number } {
  let count = 0;
  return {
    dim: mode === 's' ? 1 : 2,
    evaluations: () => count,
    evaluate: (x): Evaluation<PolicyEvalInfo> => {
      count++;
      const { s, Q } = toPolicy(x, bounds, mode, fixedQ);
      const r = simulatePolicy({ s, Q }, sim);
      return { objective: r.mean.totalCostPerDay, slacks: slacks(r.mean, cons), info: { s, Q, metrics: r.mean } };
    },
  };
}

/** One perturbation for the optimal-policy sensitivity analysis. */
export interface PolicyPerturbation {
  id: string;
  label: string;
  apply: (sim: Omit<PolicySimConfig, 'trace'>, cons: PolicyConstraints, bounds: PolicyBounds) => { sim: Omit<PolicySimConfig, 'trace'>; cons: PolicyConstraints; bounds: PolicyBounds };
}

const shiftS = (b: PolicyBounds, ds: number): PolicyBounds => ({ ...b, s: [b.s[0] + ds * 0.5, b.s[1] + ds * 1.5] });

/** "What happens to the best policy if…" questions, each answered by re-running the optimiser. */
export function standardPerturbations(dBar: number): PolicyPerturbation[] {
  return [
    { id: 'base', label: 'Current assumptions', apply: (sim, cons, bounds) => ({ sim, cons, bounds }) },
    { id: 'penalty-x2', label: 'Lost-sale penalty ×2', apply: (sim, cons, bounds) => ({ sim: { ...sim, costs: { ...sim.costs, stockoutPenalty: sim.costs.stockoutPenalty * 2 } }, cons, bounds }) },
    { id: 'holding-x2', label: 'Holding cost ×2', apply: (sim, cons, bounds) => ({ sim: { ...sim, costs: { ...sim.costs, holdingPerUnitDay: sim.costs.holdingPerUnitDay * 2 } }, cons, bounds }) },
    { id: 'order-x2', label: 'Order cost ×2', apply: (sim, cons, bounds) => ({ sim: { ...sim, costs: { ...sim.costs, orderCost: sim.costs.orderCost * 2 } }, cons, bounds }) },
    { id: 'sdL-x2', label: 'Lead-time variability σL ×2', apply: (sim, cons, bounds) => ({ sim: { ...sim, leadTime: { ...sim.leadTime, sd: sim.leadTime.sd * 2 } }, cons, bounds: shiftS(bounds, dBar * sim.leadTime.sd * 2) }) },
    { id: 'L+2', label: 'Lead time +2 days', apply: (sim, cons, bounds) => ({ sim: { ...sim, leadTime: { ...sim.leadTime, mean: sim.leadTime.mean + 2 } }, cons, bounds: shiftS(bounds, 2 * dBar) }) },
    { id: 'service-99', label: 'Service target 99%', apply: (sim, cons, bounds) => ({ sim, cons: { ...cons, minService: Math.max(cons.minService, 0.99) }, bounds }) },
  ];
}
