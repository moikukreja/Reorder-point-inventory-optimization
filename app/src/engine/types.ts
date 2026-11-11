/**
 * Serializable types exchanged between the UI thread and the Web Workers.
 * Everything here survives structured cloning: plain objects, arrays and typed arrays.
 */
import type { CostAssumption, LeadTimeAssumption } from '../core/assumptions';
import type { AccuracyMetrics } from '../core/metrics';
import type { IntervalLevel } from '../core/conformal';
import type { RiskComparison } from '../core/riskInputs';
import type { RiskSummary, Histogram } from '../core/monteCarlo';
import type { PolicyMetrics, TraceDay } from '../core/policySim';
import type { ScenarioId } from '../core/scenarios';
import type { Heatmap, ServiceCurvePoint, TornadoBar, InputKey } from '../core/sensitivity';
import type { PolicyBounds, PolicyConstraints, SearchMode } from '../core/policyOptimizer';
import type { SafetyStockResult } from '../core/safetyStock';
import type { ModelName } from '../core/backtest';

export interface Settings {
  serviceLevel: number;
  leadTime: LeadTimeAssumption;
  costs: CostAssumption;
  /** Monte Carlo runs for the lead-time-demand distribution. */
  mcRuns: number;
  /** Replications per policy simulation. */
  simReps: number;
  seed: number;
  scenario: ScenarioId;
  useLstm: boolean;
  /** Block bootstrap (keeps correlated forecast misses) vs independent daily errors. */
  blockBootstrap: boolean;
}

export interface SeriesInfo {
  key: string;
  store: number;
  family: string;
  city: string;
  storeType: string;
}

export interface DatasetInfo {
  source: string;
  generatedAt: string;
  startDate: string;
  historyEnd: string;
  calendarEnd: string;
  series: SeriesInfo[];
  days: number;
  closedDays: number;
}

export interface ForecastBundle {
  key: string;
  info: SeriesInfo;
  originDate: string;
  /** Recent history window for charts. */
  history: { dates: string[]; actual: (number | null)[]; promo: number[]; holiday: number[]; names: (string | null)[] };
  /** Out-of-sample backtest forecasts aligned to the history window (null outside test windows). */
  backtestOverlay: Partial<Record<ModelName, (number | null)[]>>;
  testWindow: { from: string; to: string };
  forecast: {
    dates: string[];
    models: Partial<Record<ModelName, number[]>>;
    ensemble: number[];
    bands: Record<IntervalLevel, { lo: number[]; hi: number[] }>;
    weights: Partial<Record<ModelName, number>>;
    residualSd: number;
    residuals: Float64Array;
    errorPaths: Float64Array[];
    promo: number[];
  };
  leaderboard: {
    models: ModelName[];
    board: Partial<Record<ModelName, AccuracyMetrics>>;
    coverage: Partial<Record<ModelName, Record<IntervalLevel, number>>>;
    lambda: number;
    nDays: number;
  };
  weightsHistory: { date: string; weights: Partial<Record<ModelName, number>> }[];
  ridgeEffects: { feature: string; label: string; beta: number }[];
  eda: { dow: number[]; promoQuartiles: number[]; holiday: [number, number]; monthly: number[]; autocorr7: number };
  residualHistogram: Histogram;
  lstm: { available: boolean; cutoffDate?: string; source?: 'pretrained' | 'browser'; fineTunedTo?: string; note: string };
  uplifts: { promo: number; holiday: number };
  timingMs: number;
}

export interface PolicyRun {
  id: 'baseline' | 'ai-formula' | 'ai-mc';
  label: string;
  s: number;
  Q: number;
  metrics: PolicyMetrics;
  trace: TraceDay[];
}

export interface RiskBundle {
  key: string;
  settingsHash: string;
  comparison: RiskComparison;
  scenario: { id: ScenarioId; label: string; description: string; leadTime: LeadTimeAssumption; demandMultiplier: number; residualMultiplier: number };
  /** The AI formula recomputed under the active scenario. */
  scenarioAi: SafetyStockResult;
  mc: {
    histogram: Histogram;
    leadTimeHistogram: Histogram;
    mean: number;
    sd: number;
    sdNormal: number;
    horizonOverflow: number;
    ai: RiskSummary;
    pStockoutBaseline: number;
    quantiles: { p: number; v: number }[];
  };
  scenarioTable: { id: ScenarioId; label: string; formulaRop: number; mcRop: number; pAtCurrent: number }[];
  policies: PolicyRun[];
  eoq: number;
  sensitivity: { tornado: TornadoBar[]; elasticities: Record<InputKey, number>; heatmap: Heatmap; serviceCurve: ServiceCurvePoint[]; varianceSplit: { demand: number; leadTime: number } };
  optimiserDefaults: { bounds: PolicyBounds; constraints: PolicyConstraints; dBar: number; sdDdlt: number };
  /** Realised L-day forecast-error-sum SD ÷ √L·σ_D (above 1 means misses persist across days). */
  errorPersistence: { ratio: number; L: number; windows: number };
  /** Baseline vs AI replayed on actual demand over the backtest test window (ROP recomputed every 14 days). */
  replay: {
    dates: string[];
    actual: number[];
    originDates: string[];
    Q: number;
    policies: { id: 'baseline' | 'ai-formula' | 'ai-mc'; label: string; sByDay: number[]; metrics: PolicyMetrics; trace: TraceDay[] }[];
  };
  timingMs: number;
}

export interface OptimiseRequest {
  demand: { mean: number[]; residuals: Float64Array; errorPaths?: Float64Array[] };
  leadTime: LeadTimeAssumption;
  costs: CostAssumption;
  horizon: number;
  replications: number;
  seed: number;
  bounds: PolicyBounds;
  constraints: PolicyConstraints;
  mode: SearchMode;
  fixedQ: number;
  budget: number;
  nInit: number;
}

export interface OptimiseIteration {
  iter: number;
  phase: 'init' | 'bo' | 'random' | 'grid';
  s: number;
  Q: number;
  cost: number;
  feasible: boolean;
  bestFeasible: number;
  pFeasible?: number;
  acquisition?: number;
  metrics: PolicyMetrics;
}

export interface SurfaceGrid {
  s: number[];
  q: number[];
  mean: number[][];
  sd: number[][];
  pFeasible: number[][];
  acquisition: number[][];
}

export interface OptimiseResult {
  history: OptimiseIteration[];
  best: OptimiseIteration | null;
  surface: SurfaceGrid | null;
  lengthScales: number[];
  elapsedMs: number;
}

export interface BenchmarkResult {
  grid: { size: number; best: OptimiseIteration | null; bestSoFar: number[]; elapsedMs: number };
  random: { best: OptimiseIteration | null; bestSoFar: number[] };
}

export interface PerturbationRow {
  id: string;
  label: string;
  best: OptimiseIteration | null;
}
