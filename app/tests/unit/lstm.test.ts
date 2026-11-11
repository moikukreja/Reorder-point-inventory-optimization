/**
 * LSTM sanity tests. They need the local dataset and the pre-trained weights in
 * public/models, and are skipped otherwise.
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import type * as tfTypes from '@tensorflow/tfjs';
import { hasLocalData, loadSubset, MODELS_DIR } from '../../scripts/lib/local-data';
import { initNodeBackend, loadModel } from '../../scripts/lib/node-tf';
import { buildCalendar, indexOf, toSeriesData, type Calendar, type SeriesData } from '../../src/core/features';
import { buildSamples, lstmForecast, LSTM_H, LSTM_W, modelSlug, type LstmMeta } from '../../src/core/models/lstm';

const KEY = '44|BEVERAGES';
const dir = join(MODELS_DIR, modelSlug(KEY));
const run = hasLocalData() && existsSync(join(dir, 'model.json')) ? describe : describe.skip;

run('LSTM (pre-trained, Store 44 × BEVERAGES)', () => {
  let model: tfTypes.LayersModel, meta: LstmMeta, s: SeriesData, cal: Calendar;
  beforeAll(async () => {
    await initNodeBackend();
    const sub = loadSubset();
    cal = buildCalendar(sub);
    s = toSeriesData(sub.series.find((x) => x.key === KEY)!);
    model = await loadModel(dir);
    meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf-8'));
  });

  it('metadata records a leak-free cut-off before the backtest calibration window', () => {
    expect(meta.cutoffDate).toBe('2016-12-06');
    expect(meta.W).toBe(LSTM_W);
    expect(meta.H).toBe(LSTM_H);
  });

  it('produces 14 finite, non-negative forecasts at a plausible level', () => {
    const T = s.y.length - 1;
    const f = lstmForecast(model, meta, s, cal, T);
    expect(f.length).toBe(LSTM_H);
    let recent = 0;
    for (let t = T - 27; t <= T; t++) recent += s.y[t] / 28;
    const avg = f.reduce((a, v) => a + v, 0) / LSTM_H;
    f.forEach((v) => expect(Number.isFinite(v) && v >= 0).toBe(true));
    expect(avg / recent).toBeGreaterThan(0.7);
    expect(avg / recent).toBeLessThan(1.3);
  });

  it('forecasts zero demand on a closed day (25 Dec)', () => {
    const xmas = indexOf(cal, '2016-12-25');
    const f = lstmForecast(model, meta, s, cal, xmas - 3);
    expect(f[2]).toBe(0);
  });

  it('builds sample tensors with the documented shapes and skips windows with closed targets', () => {
    const smp = buildSamples(s, cal, indexOf(cal, '2016-12-01'), indexOf(cal, '2016-12-31'), meta.mu, meta.sigma);
    const n = smp.origins.length;
    expect(n).toBeLessThan(31); // windows that contain 25 Dec or 1 Jan as a target are skipped
    expect(smp.past.length).toBe(n * LSTM_W * 8);
    expect(smp.fut.length).toBe(n * LSTM_H * 8);
    expect(smp.target.length).toBe(n * LSTM_H);
  });
});
