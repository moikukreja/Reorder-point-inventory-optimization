/**
 * End-to-end test of the analysis engine that the UI's worker uses (prepare → risk), run in Node.
 * It needs the local dataset and is skipped otherwise.
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import { hasLocalData, MODELS_DIR, SUBSET_PATH } from '../../scripts/lib/local-data';
import { initNodeBackend, loadModel } from '../../scripts/lib/node-tf';
import { AnalysisEngine } from '../../src/engine/analysis';
import { validateSubset } from '../../src/data/schema';
import { modelSlug } from '../../src/core/models/lstm';
import { DEFAULT_SETTINGS } from '../../src/state/defaults';
import type { ForecastBundle, RiskBundle } from '../../src/engine/types';

const run = hasLocalData() ? describe : describe.skip;

run('AnalysisEngine (Store 44 × BEVERAGES)', () => {
  let eng: AnalysisEngine, fb: ForecastBundle, rk: RiskBundle;
  beforeAll(async () => {
    await initNodeBackend();
    eng = new AnalysisEngine(validateSubset(JSON.parse(readFileSync(SUBSET_PATH, 'utf-8'))), async (key) => {
      const dir = join(MODELS_DIR, modelSlug(key));
      if (!existsSync(join(dir, 'model.json'))) return null;
      return { model: await loadModel(dir), meta: JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf-8')) };
    });
    fb = await eng.prepare('44|BEVERAGES', true);
    rk = eng.risk('44|BEVERAGES', DEFAULT_SETTINGS);
  });

  it('describes the dataset', () => {
    const i = eng.info();
    expect(i.series.length).toBe(9);
    expect(i.historyEnd).toBe('2017-08-15');
  });
  it('produces a 28-day forecast with ordered bands and a full leaderboard', () => {
    expect(fb.forecast.ensemble.length).toBe(28);
    fb.forecast.ensemble.forEach((v, h) => {
      expect(fb.forecast.bands['0.95'].lo[h]).toBeLessThanOrEqual(fb.forecast.bands['0.8'].lo[h] + 1e-9);
      expect(fb.forecast.bands['0.8'].lo[h]).toBeLessThanOrEqual(v + 1e-9);
      expect(fb.forecast.bands['0.8'].hi[h]).toBeGreaterThanOrEqual(v - 1e-9);
    });
    expect(fb.leaderboard.models).toEqual(['snaive', 'holtwinters', 'ridge', 'lstm', 'ensemble']);
    expect(fb.lstm.available).toBe(true);
    expect(fb.history.dates.length).toBe(140);
  });
  it('computes baseline and AI safety stock with the slide formula', () => {
    const a = rk.comparison.ai;
    expect(a.safetyStock).toBeCloseTo(a.z * Math.sqrt(a.sdD ** 2 * a.L + a.dBar ** 2 * a.sdL ** 2), 6);
    expect(rk.comparison.baseline.sdL).toBe(0);
    expect(rk.policies.map((p) => p.id)).toEqual(['baseline', 'ai-formula', 'ai-mc']);
    expect(rk.scenarioTable.length).toBe(6);
    expect(rk.mc.histogram.counts.length).toBe(60);
  });
  it('a supplier-delay scenario raises the AI reorder point', () => {
    const d = eng.risk('44|BEVERAGES', { ...DEFAULT_SETTINGS, scenario: 'supplier-delay' });
    expect(d.comparison.ai.reorderPoint).toBeGreaterThan(rk.comparison.ai.reorderPoint);
    expect(d.scenario.leadTime.mean).toBe(DEFAULT_SETTINGS.leadTime.mean + 2);
  });
});
