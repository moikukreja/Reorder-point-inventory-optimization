/**
 * Golden-fixture tests: the TypeScript models must reproduce the Python reference notebook.
 * The fixtures are derived from the dataset, which is git-ignored, so these tests are skipped
 * when the local data folder is absent.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import { hasLocalData, loadFixtures, loadSubset } from '../../scripts/lib/local-data';
import { buildCalendar, featureRow, FEATURE_NAMES, impute, indexOf, toSeriesData, TRAIN_START_DATE, type Calendar, type SeriesData } from '../../src/core/features';
import { seasonalNaiveForecast } from '../../src/core/models/seasonalNaive';
import { hwFit, hwForecast, hwRun } from '../../src/core/models/holtWinters';
import { buildDesign, ridgeFit, ridgeForecast } from '../../src/core/models/ridge';
import { DEFAULT_BACKTEST, runBacktest } from '../../src/core/backtest';

const run = hasLocalData() ? describe : describe.skip;

run('models vs reference notebook (Store 44 × BEVERAGES)', () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let fx: any, cal: Calendar, all: SeriesData[], s: SeriesData, T: number, TS: number;
  beforeAll(() => {
    fx = loadFixtures();
    const sub = loadSubset();
    cal = buildCalendar(sub);
    all = sub.series.map(toSeriesData);
    s = all.find((x) => x.key === fx.meta.key)!;
    T = fx.meta.origin;
    TS = indexOf(cal, TRAIN_START_DATE);
  });

  it('uses the same training start and feature list', () => {
    expect(TS).toBe(fx.meta.trainStart);
    expect([...FEATURE_NAMES]).toEqual(fx.meta.features);
  });

  it('builds identical feature rows', () => {
    const z = impute(s.y, T);
    for (const [t, ref] of Object.entries(fx.features) as [string, number[]][]) {
      const row = featureRow(Number(t), z, s.promo, cal);
      ref.forEach((v, j) => expect(row[j]).toBeCloseTo(v, 9));
    }
  });

  it('seasonal naive matches exactly', () => {
    expect(Array.from(seasonalNaiveForecast(s.y, T, 14))).toEqual(fx.snaive);
  });

  it('Holt-Winters with fixed parameters matches (forecast and SSE)', () => {
    const [alpha, beta, gamma, phi] = fx.hwFixed.params;
    const st = hwRun(Array.from(s.y.slice(TS, T + 1)), { alpha, beta, gamma, phi });
    expect(st.sse / fx.hwFixed.sse).toBeCloseTo(1, 10);
    hwForecast(st, 14, phi).forEach((v, i) => expect(v).toBeCloseTo(fx.hwFixed.forecast[i], 6));
  });

  it('Holt-Winters fitting reaches the same optimum as SciPy Nelder–Mead', () => {
    const fit = hwFit(Array.from(s.y.slice(TS, T + 1)));
    expect(fit.sse / fx.hwFit.sse).toBeCloseTo(1, 4);
    const p = [fit.params.alpha, fit.params.beta, fit.params.gamma, fit.params.phi];
    p.forEach((v, i) => expect(v).toBeCloseTo(fx.hwFit.params[i], 3));
  });

  it('Ridge (λ = 10) coefficients and recursive forecast match', () => {
    const m = ridgeFit(buildDesign(s.y, s.promo, cal, TS, T), 10);
    expect(m.b0).toBeCloseTo(fx.ridgeFixed.b0, 8);
    m.beta.forEach((v, j) => expect(v).toBeCloseTo(fx.ridgeFixed.beta[j], 6));
    ridgeForecast(m, s.y, s.promo, cal, T, 14).forEach((v, i) => expect(v).toBeCloseTo(fx.ridgeFixed.forecast[i], 4));
  });

  it('backtest leaderboard matches the notebook for all 9 series', () => {
    for (const series of all) {
      const ref = fx.backtest[series.key];
      const bt = runBacktest(series, cal, { ...DEFAULT_BACKTEST, trainStart: TS });
      expect(bt.lambda).toBe(ref.lam);
      expect(bt.maseScale).toBeCloseTo(ref.scale, 6);
      for (const m of ['snaive', 'ridge'] as const) {
        expect(bt.board[m]!.mae).toBeCloseTo(ref.board[m].mae, 4);
        expect(bt.board[m]!.wape).toBeCloseTo(ref.board[m].wape, 6);
      }
      // Holt-Winters goes through an iterative optimiser: allow 0.5% relative difference
      expect(Math.abs(bt.board.holtwinters!.mae / ref.board.holtwinters.mae - 1)).toBeLessThan(0.005);
    }
  });

  it('conformal intervals and coverage match for the default series', () => {
    const bt = runBacktest(s, cal, { ...DEFAULT_BACKTEST, trainStart: TS });
    for (const m of ['snaive', 'ridge'] as const) {
      for (const lv of ['0.8', '0.95'] as const) {
        expect(bt.conformal[m]!.levels[lv].lo).toBeCloseTo(fx.conformal[m][lv].lo, 4);
        expect(bt.conformal[m]!.levels[lv].hi).toBeCloseTo(fx.conformal[m][lv].hi, 4);
        expect(bt.conformal[m]!.levels[lv].coverage).toBeCloseTo(fx.conformal[m][lv].coverage, 10);
      }
    }
  });
});
