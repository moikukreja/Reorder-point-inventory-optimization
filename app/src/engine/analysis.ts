/**
 * Analysis engine: the orchestration layer between the pure maths in src/core and the UI.
 * It runs inside a Web Worker in the browser and directly in Node for tests, so it has no
 * DOM dependencies. The LSTM loader is injected, because loading differs between browser and Node.
 *
 *  prepare(key)          backtest + leaderboard + current forecast (cached per series and data version)
 *  risk(key, settings)   safety stock (baseline vs AI), Monte Carlo, scenarios, policy simulations,
 *                        historical replay and sensitivity, recomputed whenever a control changes
 *  appendRows(key, rows) new observations: refits everything on the next prepare()
 *  startReplay / replayNext / stopReplay   hold back recent real days and feed them back one at a time
 *  fineTuneLstm / trainLstmFromScratch     continuous learning for the LSTM, in the browser
 *
 * LSTM: two copies are kept per series. The FROZEN model (cut-off before the backtest window) is used
 * only for the leak-free backtest. The LIVE model is fine-tuned on new data and drives the forecast.
 */
import type * as tfTypes from '@tensorflow/tfjs';
import type { SubsetFile } from '../data/schema';
import { buildCalendar, extendCalendar, FEATURE_LABELS, FEATURE_NAMES, indexOf, toSeriesData, TRAIN_START_DATE, dateAt, type Calendar, type SeriesData } from '../core/features';
import { DEFAULT_BACKTEST, originsFor, runBacktest, type BacktestResult } from '../core/backtest';
import { forecastNow, type CurrentForecast } from '../core/forecastNow';
import { buildSamples, createLstmModel, fitScaler, lstmForecast, LSTM_H, LSTM_W, trainLstm, type LstmMeta } from '../core/models/lstm';
import { historicalReplay } from '../core/historicalReplay';
import { validateRows, type NewRow } from '../data/newRows';
import { buildDesign, ridgeFit } from '../core/models/ridge';
import { compareSafetyStock } from '../core/riskInputs';
import { slideSafetyStock } from '../core/safetyStock';
import { histogram, monteCarloRop, simulateDdlt, stockoutProbability, summariseRisk } from '../core/monteCarlo';
import { estimateUplifts, SCENARIOS, type Uplifts } from '../core/scenarios';
import { eoq, simulatePolicy, type DemandSource } from '../core/policySim';
import { elasticities, ropHeatmap, serviceLevelCurve, tornado, varianceSplit } from '../core/sensitivity';
import { defaultBounds, defaultConstraints } from '../core/policyOptimizer';
import { quantile, std } from '../core/stats';
import type { DatasetInfo, ForecastBundle, PolicyRun, RiskBundle, Settings } from './types';


export type LstmHandle = { model: tfTypes.LayersModel; meta: LstmMeta };
export type LstmLoader = (key: string) => Promise<LstmHandle | null>;

interface LstmPair {
  frozen: LstmHandle;
  live: LstmHandle;
  source: 'pretrained' | 'browser';
}

interface Prepared {
  bt: BacktestResult;
  fc: CurrentForecast;
  useLstm: boolean;
  uplifts: Uplifts;
  bundle: ForecastBundle;
  /** Versions of the data, the frozen model and the live model this result was computed with. */
  dv: number;
  fv: number;
  lv: number;
}

export interface FineTuneReport {
  newDays: number;
  samples: number;
  epochs: number;
  ms: number;
  /** Next-7-day forecast total from the LSTM before and after the update. */
  before7: number;
  after7: number;
  fineTunedTo: string;
}

export interface ReplayStatus {
  active: boolean;
  remaining: number;
  total: number;
  nextDate: string | null;
}

/** Days of calendar kept beyond the last observation (forecast horizon 28 days plus headroom). */
const CALENDAR_HEADROOM = 90;
const HISTORY_WINDOW = 140;
const FORECAST_H = 28;
const POLICY_HORIZON = 196;
const POLICY_WARMUP = 14;
const TRACE_DAYS = 120;

export class AnalysisEngine {
  private file: SubsetFile;
  private cal!: Calendar;
  private series = new Map<string, SeriesData>();
  private cache = new Map<string, Prepared>();
  private trainStart = 0;
  private lstms = new Map<string, LstmPair | null>();
  private versions = new Map<string, { dv: number; fv: number; lv: number }>();
  private held = new Map<string, { rows: NewRow[]; total: number }>();

  constructor(file: SubsetFile, private loadLstm: LstmLoader = async () => null) {
    this.file = file;
    this.rebuild();
  }

  private rebuild() {
    const raw = this.file.series.map(toSeriesData);
    const histLen = raw[0].y.length;
    const ext = extendCalendar(buildCalendar(this.file), raw, histLen + CALENDAR_HEADROOM);
    this.cal = ext.cal;
    this.series = new Map(ext.series.map((s) => [s.key, s]));
    this.trainStart = indexOf(this.cal, TRAIN_START_DATE);
    this.cache.clear();
  }

  info(): DatasetInfo {
    const first = [...this.series.values()][0];
    let closed = 0;
    for (const v of first.y) if (Number.isNaN(v)) closed++;
    return {
      source: this.file.source,
      generatedAt: this.file.generatedAt,
      startDate: this.file.startDate,
      historyEnd: dateAt(this.cal, first.y.length - 1),
      calendarEnd: this.file.calendarEnd,
      series: [...this.series.values()].map((s) => ({ key: s.key, store: s.store, family: s.family, city: s.city, storeType: s.storeType })),
      days: first.y.length,
      closedDays: closed,
    };
  }

  private ver(key: string) {
    let v = this.versions.get(key);
    if (!v) this.versions.set(key, (v = { dv: 0, fv: 0, lv: 0 }));
    return v;
  }

  lastDate(key: string): string {
    return dateAt(this.cal, this.get(key).y.length - 1);
  }

  private get(key: string): SeriesData {
    const s = this.series.get(key);
    if (!s) throw new Error(`Unknown series ${key}`);
    return s;
  }

  /** Loads the pre-trained LSTM once per series and makes a live copy for fine-tuning. */
  private async ensureLstm(key: string): Promise<LstmPair | null> {
    if (this.lstms.has(key)) return this.lstms.get(key)!;
    let pair: LstmPair | null = null;
    try {
      const h = await this.loadLstm(key);
      if (h) pair = { frozen: h, live: this.cloneLstm(h), source: 'pretrained' };
    } catch {
      pair = null;
    }
    this.lstms.set(key, pair);
    return pair;
  }

  private cloneLstm(h: LstmHandle): LstmHandle {
    const model = createLstmModel(h.meta.seed);
    model.setWeights(h.model.getWeights());
    return { model, meta: { ...h.meta } };
  }

  async prepare(key: string, useLstm: boolean, onProgress?: (msg: string, frac: number) => void): Promise<ForecastBundle> {
    const s = this.get(key);
    const cal = this.cal;
    const v = { ...this.ver(key) };
    const hit = this.cache.get(key);
    if (hit && hit.useLstm === useLstm && hit.dv === v.dv && hit.fv === v.fv && hit.lv === v.lv) return hit.bundle;
    const t0 = performance.now();
    onProgress?.('Loading models', 0.02);
    const pair = useLstm ? await this.ensureLstm(key) : null;
    const frozenFn = pair ? (T: number) => lstmForecast(pair.frozen.model, pair.frozen.meta, s, cal, T) : undefined;
    const liveFn = pair ? (T: number) => lstmForecast(pair.live.model, pair.live.meta, s, cal, T) : undefined;
    // If only the live LSTM changed, the backtest (which uses the frozen model) is still valid
    const reuse = hit && hit.useLstm === useLstm && hit.dv === v.dv && hit.fv === v.fv;
    const bt = reuse ? hit.bt : runBacktest(s, cal, { ...DEFAULT_BACKTEST, trainStart: this.trainStart }, frozenFn, (d, n) => onProgress?.('Backtesting models on 18 rolling origins', 0.05 + 0.85 * (d / n)));
    onProgress?.('Forecasting the next 28 days', 0.92);
    const fc = forecastNow(s, cal, bt, FORECAST_H, liveFn);
    const T = s.y.length - 1;
    const uplifts = estimateUplifts(s, cal, this.trainStart, T);
    const bundle = this.bundle(key, s, bt, fc, pair, uplifts, performance.now() - t0);
    this.cache.set(key, { bt, fc, useLstm, uplifts, bundle, ...v });
    onProgress?.('Done', 1);
    return bundle;
  }

  /**
   * Appends new observations to a series. Rows must continue day by day from the last date.
   * The calendar grows as needed: holiday flags and oil come from the rows (oil is carried forward).
   */
  appendRows(key: string, rows: NewRow[], fromReplay = false): { appended: number; lastDate: string } {
    if (!fromReplay && this.held.has(key)) throw new Error('A replay is running for this series. Finish or stop it before adding data.');
    const s = this.get(key);
    const v = validateRows(rows.map((r) => ({ ...r, oil: r.oil ?? null })), this.lastDate(key));
    if (v.issues.length) throw new Error(`Row ${v.issues[0].line}: ${v.issues[0].message}`);
    const n0 = s.y.length;
    const needed = n0 + v.rows.length + CALENDAR_HEADROOM;
    if (needed > this.cal.length) {
      const ext = extendCalendar(this.cal, [...this.series.values()], needed);
      this.cal = ext.cal;
      for (const e of ext.series) this.series.set(e.key, { ...this.series.get(e.key)!, promo: e.promo });
    }
    const cur = this.get(key);
    const y = new Float64Array(n0 + v.rows.length);
    y.set(cur.y);
    v.rows.forEach((r, i) => {
      const t = n0 + i;
      y[t] = r.sales === null ? NaN : r.sales;
      cur.promo[t] = r.onpromotion;
      this.cal.holiday[t] = r.holiday;
      if (r.holiday && !this.cal.name[t]) this.cal.name[t] = 'Holiday (added)';
      // A user's new oil price is carried forward (the future is unknown); a replayed day only restores its own value
      if (r.oil) for (let u = t; u < (fromReplay ? t + 1 : this.cal.length); u++) this.cal.oil[u] = r.oil;
    });
    this.series.set(key, { ...cur, y });
    this.ver(key).dv++;
    return { appended: v.rows.length, lastDate: this.lastDate(key) };
  }

  /** Holds back the last n observed days, so they can be fed back one at a time (a live-feed demo on real data). */
  startReplay(key: string, n: number): ReplayStatus {
    if (this.held.has(key)) return this.replayStatus(key);
    const s = this.get(key);
    const T = s.y.length - 1;
    const k = Math.max(1, Math.min(n, 120));
    const rows: NewRow[] = [];
    for (let t = T - k + 1; t <= T; t++) rows.push({ date: dateAt(this.cal, t), sales: Number.isNaN(s.y[t]) ? null : s.y[t], onpromotion: s.promo[t], holiday: this.cal.holiday[t] ? 1 : 0, oil: this.cal.oil[t] });
    this.series.set(key, { ...s, y: s.y.slice(0, T - k + 1) });
    this.held.set(key, { rows, total: k });
    this.ver(key).dv++;
    return this.replayStatus(key);
  }

  replayNext(key: string, k = 1): ReplayStatus & { added: NewRow[] } {
    const h = this.held.get(key);
    if (!h) throw new Error('No replay running.');
    const added = h.rows.splice(0, k);
    if (added.length) this.appendRows(key, added, true);
    if (!h.rows.length) this.held.delete(key);
    return { ...this.replayStatus(key), added };
  }

  /** Ends a replay and restores every held-back day. */
  stopReplay(key: string): ReplayStatus {
    const h = this.held.get(key);
    if (h && h.rows.length) this.appendRows(key, h.rows.splice(0), true);
    this.held.delete(key);
    return this.replayStatus(key);
  }

  replayStatus(key: string): ReplayStatus {
    const h = this.held.get(key);
    return h ? { active: true, remaining: h.rows.length, total: h.total, nextDate: h.rows[0]?.date ?? null } : { active: false, remaining: 0, total: 0, nextDate: null };
  }

  /**
   * Continuous learning: fine-tunes the LIVE LSTM on the most recent windows, including every day
   * since its last update (warm start, a few epochs). The frozen backtest model is untouched.
   */
  async fineTuneLstm(key: string, onProgress?: (msg: string, frac: number) => void, maxEpochs = 10): Promise<FineTuneReport | null> {
    const pair = await this.ensureLstm(key);
    if (!pair) return null;
    const s = this.get(key);
    const T = s.y.length - 1;
    const meta = pair.live.meta;
    const since = meta.fineTunedTo ?? meta.cutoff;
    const newDays = T - since;
    if (newDays <= 0) return { newDays: 0, samples: 0, epochs: 0, ms: 0, before7: 0, after7: 0, fineTunedTo: dateAt(this.cal, since) };
    const oTo = T - LSTM_H;
    const oFrom = Math.max(this.trainStart + LSTM_W - 1, Math.min(since, oTo) - 150);
    const t0 = performance.now();
    const sum7 = () => lstmForecast(pair.live.model, pair.live.meta, s, this.cal, T).slice(0, 7).reduce((a, b) => a + b, 0);
    const before7 = sum7();
    const smp = buildSamples(s, this.cal, oFrom, oTo, meta.mu, meta.sigma);
    let epochs = 0;
    if (smp.origins.length >= 8) {
      const r = await trainLstm(pair.live.model, smp, meta.mu, meta.sigma, {
        epochs: maxEpochs, learningRate: 0.0007, validationShare: 0.15, patience: 4,
        onEpoch: (e) => onProgress?.(`Fine-tuning LSTM (epoch ${e + 1})`, (e + 1) / maxEpochs),
      });
      epochs = r.epochs;
      pair.live.meta = { ...meta, correction: r.correction, fineTunedTo: T };
    } else pair.live.meta = { ...meta, fineTunedTo: T };
    this.ver(key).lv++;
    return { newDays, samples: smp.origins.length, epochs, ms: performance.now() - t0, before7, after7: sum7(), fineTunedTo: dateAt(this.cal, T) };
  }

  /**
   * Trains a new LSTM from scratch in the browser. The cut-off is the first backtest calibration
   * origin (so the backtest stays leak-free), and the live copy is then fine-tuned to the latest data.
   */
  async trainLstmFromScratch(key: string, onProgress?: (msg: string, frac: number) => void, epochs = 60): Promise<{ ms: number; epochs: number; valLoss: number; samples: number }> {
    const s = this.get(key);
    const last = s.y.length - 1;
    const cutoff = originsFor(last, DEFAULT_BACKTEST).cal[0];
    const { mu, sigma } = fitScaler(s, this.trainStart, cutoff);
    const smp = buildSamples(s, this.cal, this.trainStart + LSTM_W - 1, cutoff - LSTM_H, mu, sigma);
    const t0 = performance.now();
    const model = createLstmModel(7);
    const r = await trainLstm(model, smp, mu, sigma, { epochs, learningRate: 0.003, validationShare: 0.15, patience: 8, onEpoch: (e, _l, vl) => onProgress?.(`Training LSTM from scratch: epoch ${e + 1}, validation MSE ${vl.toFixed(3)}`, (e + 1) / epochs) });
    const meta: LstmMeta = { version: 1, key, W: LSTM_W, H: LSTM_H, mu, sigma, correction: r.correction, trainStart: this.trainStart, cutoff, cutoffDate: dateAt(this.cal, cutoff), trainedAt: new Date().toISOString(), epochs: r.epochs, valLoss: r.valLoss, seed: 7 };
    const frozen = { model, meta };
    this.lstms.set(key, { frozen, live: this.cloneLstm(frozen), source: 'browser' });
    const v = this.ver(key);
    v.fv++;
    v.lv++;
    return { ms: performance.now() - t0, epochs: r.epochs, valLoss: r.valLoss, samples: smp.origins.length };
  }

  private bundle(key: string, s: SeriesData, bt: BacktestResult, fc: CurrentForecast, pair: LstmPair | null, uplifts: Uplifts, ms: number): ForecastBundle {
    const cal = this.cal;
    const T = s.y.length - 1;
    const from = T - HISTORY_WINDOW + 1;
    const dates: string[] = [], actual: (number | null)[] = [], promo: number[] = [], hol: number[] = [], names: (string | null)[] = [];
    for (let t = from; t <= T; t++) {
      dates.push(dateAt(cal, t));
      actual.push(Number.isNaN(s.y[t]) ? null : s.y[t]);
      promo.push(s.promo[t]);
      hol.push(cal.holiday[t]);
      names.push(cal.name[t] ?? null);
    }
    // Out-of-sample forecasts placed on the history window
    const overlay: ForecastBundle['backtestOverlay'] = {};
    const origins = [...bt.calOrigins, ...bt.testOrigins];
    for (const m of bt.models) {
      const arr: (number | null)[] = new Array(dates.length).fill(null);
      origins.forEach((o, i) => {
        const f = bt.forecasts[m]![i];
        for (let h = 0; h < f.length; h++) {
          const idx = o + h + 1 - from;
          if (idx >= 0 && idx < arr.length) arr[idx] = f[h];
        }
      });
      overlay[m] = arr;
    }
    // Leaderboard
    const coverage: ForecastBundle['leaderboard']['coverage'] = {};
    for (const m of bt.models) coverage[m] = { '0.8': bt.conformal[m]!.levels['0.8'].coverage!, '0.95': bt.conformal[m]!.levels['0.95'].coverage! };
    // Ridge effects: standardised coefficients (units/day per 1 SD change in the feature)
    const rm = ridgeFit(buildDesign(s.y, s.promo, cal, this.trainStart, T), bt.lambda);
    const ridgeEffects = FEATURE_NAMES.map((f, j) => ({ feature: f, label: FEATURE_LABELS[f], beta: rm.beta[j] }))
      .filter((e) => !e.feature.startsWith('f_'))
      .sort((a, b) => Math.abs(b.beta) - Math.abs(a.beta))
      .slice(0, 12);
    // Exploratory effects on the training window
    const dow = new Array(7).fill(0), dowN = new Array(7).fill(0), mon = new Array(12).fill(0), monN = new Array(12).fill(0);
    let hS = 0, hN = 0, nS = 0, nN = 0;
    const open: number[] = [], pr: number[] = [];
    for (let t = this.trainStart; t <= T; t++) {
      const v = s.y[t];
      if (Number.isNaN(v)) continue;
      dow[cal.dow[t]] += v; dowN[cal.dow[t]]++;
      const m = Number(dateAt(cal, t).slice(5, 7)) - 1;
      mon[m] += v; monN[m]++;
      if (cal.holiday[t]) { hS += v; hN++; } else { nS += v; nN++; }
      open.push(v);
      pr.push(s.promo[t]);
    }
    const order = pr.map((p, i) => [p, i] as const).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const q = [0, 0, 0, 0], qn = [0, 0, 0, 0];
    order.forEach(([, i], r) => { const k = Math.min(3, Math.floor((r / order.length) * 4)); q[k] += open[i]; qn[k]++; });
    let num = 0, d1 = 0;
    const mu = open.reduce((a, b) => a + b, 0) / open.length;
    for (let i = 7; i < open.length; i++) num += (open[i] - mu) * (open[i - 7] - mu);
    for (const v of open) d1 += (v - mu) ** 2;
    const residSorted = Float64Array.from(fc.residuals).sort();
    return {
      key,
      info: { key, store: s.store, family: s.family, city: s.city, storeType: s.storeType },
      originDate: fc.originDate,
      history: { dates, actual, promo, holiday: hol, names },
      backtestOverlay: overlay,
      testWindow: { from: dateAt(cal, bt.testOrigins[0] + 1), to: dateAt(cal, bt.testOrigins.at(-1)! + bt.config.H) },
      forecast: {
        dates: fc.dates,
        models: Object.fromEntries(Object.entries(fc.models).map(([k, v]) => [k, Array.from(v!)])),
        ensemble: Array.from(fc.ensemble),
        bands: { '0.8': { lo: Array.from(fc.bands['0.8'].lo), hi: Array.from(fc.bands['0.8'].hi) }, '0.95': { lo: Array.from(fc.bands['0.95'].lo), hi: Array.from(fc.bands['0.95'].hi) } },
        weights: fc.weights,
        residualSd: fc.residualSd,
        residuals: fc.residuals,
        errorPaths: fc.errorPaths,
        promo: Array.from({ length: FORECAST_H }, (_, h) => s.promo[T + h + 1]),
      },
      leaderboard: { models: bt.models, board: bt.board, coverage, lambda: bt.lambda, nDays: bt.board.snaive!.n },
      weightsHistory: bt.testOrigins.map((o, i) => ({ date: dateAt(cal, o), weights: bt.ensembleWeights[bt.calOrigins.length + i] })),
      ridgeEffects,
      eda: { dow: dow.map((v, i) => v / dowN[i]), promoQuartiles: q.map((v, i) => v / qn[i]), holiday: [nS / nN, hS / hN], monthly: mon.map((v, i) => v / monN[i]), autocorr7: num / d1 },
      residualHistogram: histogram(residSorted, 30, 0.005, 0.995),
      lstm: pair
        ? {
            available: true,
            cutoffDate: pair.frozen.meta.cutoffDate,
            source: pair.source,
            fineTunedTo: pair.live.meta.fineTunedTo !== undefined ? dateAt(this.cal, pair.live.meta.fineTunedTo) : undefined,
            note: `${pair.source === 'pretrained' ? 'Pre-trained at build time' : 'Trained in this browser'} on data up to ${pair.frozen.meta.cutoffDate}; the backtest uses these frozen weights (leak-free). ${pair.live.meta.fineTunedTo !== undefined ? `The live model used for the forecast is fine-tuned up to ${dateAt(this.cal, pair.live.meta.fineTunedTo)}.` : 'The live model has not been fine-tuned on later data yet.'}`,
          }
        : { available: false, note: 'No LSTM for this series yet: classical models only. Use "Train LSTM in this browser" on the Data page.' },
      uplifts,
      timingMs: ms,
    };
  }

  risk(key: string, st: Settings): RiskBundle {
    const t0 = performance.now();
    const p = this.cache.get(key);
    if (!p) throw new Error('prepare() must run before risk()');
    const s = this.get(key);
    const { fc, uplifts } = p;
    const sc = SCENARIOS.find((x) => x.id === st.scenario) ?? SCENARIOS[0];
    const eff = sc.apply(st.leadTime, uplifts);
    // Scenario-aware forecast: the AI sees the demand change coming; the baseline only sees history
    const fcS = { ...fc, ensemble: fc.ensemble.map((v) => v * eff.demandMultiplier), residualSd: fc.residualSd * eff.residualMultiplier };
    const comparison = compareSafetyStock(s, fcS, eff.leadTime, st.serviceLevel);
    const scenarioAi = comparison.ai;
    const paths = st.blockBootstrap ? fc.errorPaths : undefined;

    const mcCfg = { mode: 'bootstrap' as const, seed: st.seed, leadTime: eff.leadTime, forecast: fc.ensemble, residuals: fc.residuals, errorPaths: paths, demandMultiplier: eff.demandMultiplier, residualMultiplier: eff.residualMultiplier };
    const mc = simulateDdlt({ ...mcCfg, n: st.mcRuns });
    const mcN = simulateDdlt({ mode: 'normal', n: Math.min(st.mcRuns, 100_000), seed: st.seed, leadTime: eff.leadTime, dBar: scenarioAi.dBar, sdD: scenarioAi.sdD });
    const ai = summariseRisk(mc, comparison.ai.reorderPoint, st.serviceLevel, comparison.ai.cycleStock);

    // All scenarios at a glance (smaller runs), measured against today's base-case AI ROP
    const baseAi = compareSafetyStock(s, fc, st.leadTime, st.serviceLevel).ai;
    const scenarioTable = SCENARIOS.map((x) => {
      const e = x.apply(st.leadTime, uplifts);
      const r = slideSafetyStock({ serviceLevel: st.serviceLevel, dBar: baseAi.dBar * e.demandMultiplier, sdD: baseAi.sdD * e.residualMultiplier, L: e.leadTime.mean, sdL: e.leadTime.sd });
      const m = simulateDdlt({ ...mcCfg, n: 40_000, leadTime: e.leadTime, demandMultiplier: e.demandMultiplier, residualMultiplier: e.residualMultiplier });
      return { id: x.id, label: x.label, formulaRop: r.reorderPoint, mcRop: monteCarloRop(m.samples, st.serviceLevel), pAtCurrent: stockoutProbability(m.samples, baseAi.reorderPoint) };
    });

    // Policy simulations: the same demand stream for every policy (common random numbers)
    const Q = eoq(comparison.ai.dBar, st.costs.orderCost, st.costs.holdingPerUnitDay);
    const demand: DemandSource = { kind: 'generated', mean: fc.ensemble, residuals: fc.residuals, errorPaths: paths, demandMultiplier: eff.demandMultiplier, residualMultiplier: eff.residualMultiplier };
    const simBase = { horizon: POLICY_HORIZON, warmup: POLICY_WARMUP, replications: st.simReps, seed: st.seed, leadTime: eff.leadTime, costs: st.costs, demand, trace: true };
    const run = (id: PolicyRun['id'], label: string, sVal: number): PolicyRun => {
      const r = simulatePolicy({ s: sVal, Q }, simBase);
      return { id, label, s: sVal, Q, metrics: r.mean, trace: r.trace!.slice(POLICY_WARMUP, POLICY_WARMUP + TRACE_DAYS) };
    };
    const policies = [
      run('baseline', 'Baseline (class notebook)', comparison.baseline.reorderPoint),
      run('ai-formula', 'AI: slide formula', comparison.ai.reorderPoint),
      run('ai-mc', 'AI: Monte Carlo ROP', ai.mcRop),
    ];

    // Sensitivity of the AI reorder point
    const base = { serviceLevel: st.serviceLevel, dBar: scenarioAi.dBar, sdD: scenarioAi.sdD, L: eff.leadTime.mean, sdL: eff.leadTime.sd };
    const Ls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const sdLs = [0, 0.5, 1, 1.5, 2, 2.5, 3];
    const sensitivity = {
      tornado: tornado(base),
      elasticities: elasticities(base),
      heatmap: ropHeatmap(base, Ls, sdLs),
      serviceCurve: serviceLevelCurve(base, st.costs.holdingPerUnitDay, { ...mcCfg, n: 60_000 }),
      varianceSplit: varianceSplit(base),
    };

    // Do forecast misses persist across the lead time?
    const L = Math.max(1, Math.round(eff.leadTime.mean));
    const sums: number[] = [];
    p.bt.actuals.forEach((a, i) => {
      const f = p.bt.forecasts.ensemble![i];
      for (let h = 0; h + L <= a.length; h += L) {
        let e = 0, ok = true;
        for (let k = h; k < h + L; k++) { if (Number.isNaN(a[k])) ok = false; e += a[k] - f[k]; }
        if (ok) sums.push(e);
      }
    });
    const ratio = std(sums, 1) / (Math.sqrt(L) * fc.residualSd);

    const replay = historicalReplay(s, p.bt, st.leadTime, st.serviceLevel, st.costs, Q, { replications: Math.min(st.simReps, 200), seed: st.seed, block: st.blockBootstrap });

    const sortedLt = mc.leadTimes;
    return {
      key,
      settingsHash: JSON.stringify(st),
      comparison,
      scenario: { id: sc.id, label: sc.label, description: sc.description, leadTime: eff.leadTime, demandMultiplier: eff.demandMultiplier, residualMultiplier: eff.residualMultiplier },
      scenarioAi,
      mc: {
        histogram: histogram(mc.samples, 60),
        leadTimeHistogram: histogram(sortedLt, 40, 0.0005, 0.9995),
        mean: mc.mean,
        sd: mc.sd,
        sdNormal: mcN.sd,
        horizonOverflow: mc.horizonOverflow,
        ai,
        pStockoutBaseline: stockoutProbability(mc.samples, comparison.baseline.reorderPoint),
        quantiles: [0.5, 0.8, 0.9, 0.95, 0.975, 0.99].map((q) => ({ p: q, v: quantile(mc.samples, q, true) })),
      },
      scenarioTable,
      policies,
      eoq: Q,
      sensitivity,
      optimiserDefaults: {
        bounds: defaultBounds(comparison.ai.dBar, eff.leadTime.mean, comparison.ai.sdDdlt),
        constraints: defaultConstraints(comparison.ai.dBar, st.costs.unitCost),
        dBar: comparison.ai.dBar,
        sdDdlt: comparison.ai.sdDdlt,
      },
      errorPersistence: { ratio, L, windows: sums.length },
      replay: {
        dates: Array.from({ length: replay.to - replay.from + 1 }, (_, d) => dateAt(this.cal, replay.from + d)),
        actual: replay.actual,
        originDates: replay.origins.map((o) => dateAt(this.cal, o)),
        Q: replay.Q,
        policies: replay.policies,
      },
      timingMs: performance.now() - t0,
    };
  }

  /** Demand model handed to the optimiser worker (it needs no access to the raw data). */
  demandModel(key: string, st: Settings): { mean: number[]; residuals: Float64Array; errorPaths?: Float64Array[] } {
    const p = this.cache.get(key);
    if (!p) throw new Error('prepare() must run first');
    const sc = SCENARIOS.find((x) => x.id === st.scenario) ?? SCENARIOS[0];
    const eff = sc.apply(st.leadTime, p.uplifts);
    return {
      mean: Array.from(p.fc.ensemble, (v) => v * eff.demandMultiplier),
      residuals: p.fc.residuals.map((r) => r * eff.residualMultiplier),
      errorPaths: st.blockBootstrap ? p.fc.errorPaths.map((e) => e.map((r) => r * eff.residualMultiplier)) : undefined,
    };
  }

  /** Lead time after the active scenario is applied (the optimiser must use the same one). */
  scenarioLeadTime(key: string, st: Settings) {
    const p = this.cache.get(key)!;
    const sc = SCENARIOS.find((x) => x.id === st.scenario) ?? SCENARIOS[0];
    return sc.apply(st.leadTime, p.uplifts).leadTime;
  }
}

