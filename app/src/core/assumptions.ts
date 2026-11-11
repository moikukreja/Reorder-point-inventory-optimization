/**
 * Documented default assumptions for inputs the dataset does not contain.
 * Every value can be changed from the dashboard's control panel. These are only starting points.
 *
 * Units: one period = one day. Demand is in units/day, lead time in days, and costs are in $.
 */
import type { LeadTimeDist } from './stats';

export interface LeadTimeAssumption {
  /** Mean supplier lead time L̄ (days). */
  mean: number;
  /** Lead-time standard deviation σ_L (days). */
  sd: number;
  dist: LeadTimeDist;
}

export interface CostAssumption {
  /** Purchase cost per unit ($), used for the inventory value / working-capital constraint. */
  unitCost: number;
  /** Holding cost per unit per day ($) = unit cost × annual carrying rate / 365. */
  holdingPerUnitDay: number;
  /** Fixed cost per replenishment order ($): transport, receiving, admin. */
  orderCost: number;
  /** Penalty per unit of lost sales ($): lost margin plus goodwill. */
  stockoutPenalty: number;
}

/**
 * Supplier lead time: mean 4 days, sd 1.2 days, gamma-distributed (right-skewed, because late
 * deliveries are more common than early ones). Typical for a regional beverage/grocery distributor.
 */
export const DEFAULT_LEAD_TIME: LeadTimeAssumption = { mean: 4, sd: 1.2, dist: 'gamma' };

/** Carrying rate 25% a year on a $1.50 unit. Order cost $150. Lost-sale penalty $0.75 a unit. */
export const DEFAULT_COSTS: CostAssumption = {
  unitCost: 1.5,
  holdingPerUnitDay: (1.5 * 0.25) / 365,
  orderCost: 150,
  stockoutPenalty: 0.75,
};

/** Target cycle service level (probability of no stockout during a replenishment lead time). */
export const DEFAULT_SERVICE_LEVEL = 0.95;

/** Window of recent history (days) for the baseline's static demand statistics. */
export const BASELINE_WINDOW_DAYS = 365;
