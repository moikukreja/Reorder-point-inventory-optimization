/**
 * Builds the two safety-stock calculations the dashboard compares:
 *   BASELINE: the class notebook method (static history, SS = Z·σ·√L)
 *   AI:       forecast-driven D̄, forecast-error σ_D, and the slide's full formula with σ_L
 */
import { BASELINE_WINDOW_DAYS, type LeadTimeAssumption } from './assumptions';
import type { CurrentForecast } from './forecastNow';
import type { SeriesData } from './features';
import { baselineSafetyStock, forecastMeanOverLeadTime, historicalDemandStats, slideSafetyStock, type SafetyStockResult } from './safetyStock';

export interface RiskComparison {
  baseline: SafetyStockResult & { source: string };
  ai: SafetyStockResult & { source: string };
}

export function compareSafetyStock(s: SeriesData, fc: CurrentForecast, lt: LeadTimeAssumption, serviceLevel: number): RiskComparison {
  const T = s.y.length - 1;
  const hist = historicalDemandStats(s.y, T, BASELINE_WINDOW_DAYS);
  const baseline = baselineSafetyStock({ serviceLevel, dBar: hist.dBar, sdD: hist.sdD, L: lt.mean, sdL: lt.sd });
  const ai = slideSafetyStock({ serviceLevel, dBar: forecastMeanOverLeadTime(fc.ensemble, lt.mean), sdD: fc.residualSd, L: lt.mean, sdL: lt.sd });
  return {
    baseline: { ...baseline, source: `Mean and SD of the last ${hist.n} days of sales; σL ignored (SS = Z·σ·√L)` },
    ai: { ...ai, source: `Ensemble forecast for the next ${lt.mean} days; σ_D = SD of recent out-of-sample forecast errors; slide formula with σL` },
  };
}
