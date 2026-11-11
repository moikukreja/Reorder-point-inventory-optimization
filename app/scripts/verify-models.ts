/**
 * Verification report for Part 1 (data and models):
 *  1. Runs the rolling-origin backtest with all models, including the pre-trained LSTM and the ensemble.
 *  2. Checks the TypeScript results against the Python reference notebook (fixtures).
 *  3. Demonstrates the update path: it fine-tunes the LSTM on the data after its training cut-off
 *     and shows how the 14-day forecast changes.
 *
 * Usage: npm run verify:models
 */
import { join } from 'node:path';
import { existsSync, readFileSync } from 'node:fs';
import type * as tfTypes from '@tensorflow/tfjs';
import { loadFixtures, loadSubset, MODELS_DIR } from './lib/local-data';
import { initNodeBackend, loadModel } from './lib/node-tf';
import { buildCalendar, dateAt, indexOf, toSeriesData, TRAIN_START_DATE } from '../src/core/features';
import { DEFAULT_BACKTEST, MODEL_LABELS, runBacktest, type ModelName } from '../src/core/backtest';
import { buildSamples, lstmForecast, LSTM_H, modelSlug, trainLstm, type LstmMeta } from '../src/core/models/lstm';

const pad = (s: string | number, n: number) => String(s).padStart(n);
const padR = (s: string, n: number) => s.padEnd(n);

async function main() {
  const backend = await initNodeBackend();
  const sub = loadSubset();
  const cal = buildCalendar(sub);
  const fx = loadFixtures();
  const TS = indexOf(cal, TRAIN_START_DATE);
  console.log(`Backend: ${backend}. Training start ${TRAIN_START_DATE} (index ${TS}). History ${sub.startDate} → ${sub.historyEnd}.\n`);

  const wapeTable: Record<string, Partial<Record<ModelName, number>>> = {};
  let maxDiff = 0;
  const loaded: Record<string, { model: tfTypes.LayersModel; meta: LstmMeta }> = {};

  for (const raw of sub.series) {
    const s = toSeriesData(raw);
    const dir = join(MODELS_DIR, modelSlug(raw.key));
    let lstm: ((T: number) => Float64Array) | undefined;
    if (existsSync(join(dir, 'model.json'))) {
      const model = await loadModel(dir);
      const meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf-8')) as LstmMeta;
      loaded[raw.key] = { model, meta };
      lstm = (T) => lstmForecast(model, meta, s, cal, T);
    }
    const bt = runBacktest(s, cal, { ...DEFAULT_BACKTEST, trainStart: TS }, lstm);
    wapeTable[raw.key] = Object.fromEntries(bt.models.map((m) => [m, bt.board[m]!.wape]));
    for (const m of ['snaive', 'holtwinters', 'ridge'] as const) maxDiff = Math.max(maxDiff, Math.abs(bt.board[m]!.wape - fx.backtest[raw.key].board[m].wape));

    if (raw.key === fx.meta.key) {
      console.log(`=== Leaderboard: ${raw.key} | test origins ${dateAt(cal, bt.testOrigins[0])} … ${dateAt(cal, bt.testOrigins.at(-1)!)} | H = 14 | n = ${bt.board.snaive!.n} days`);
      console.log(`Ridge λ chosen on the calibration origins: ${bt.lambda}   (MAE by λ: ${Object.entries(bt.lambdaScores).map(([k, v]) => `${k}→${v.toFixed(0)}`).join(', ')})`);
      console.log(`${padR('Model', 30)}${pad('MAE', 9)}${pad('RMSE', 9)}${pad('WAPE%', 8)}${pad('MASE', 7)}${pad('Bias', 8)}${pad('Cov80', 7)}${pad('Cov95', 7)}`);
      for (const m of bt.models) {
        const b = bt.board[m]!, c = bt.conformal[m]!.levels;
        console.log(`${padR(MODEL_LABELS[m], 30)}${pad(b.mae.toFixed(0), 9)}${pad(b.rmse.toFixed(0), 9)}${pad(b.wape.toFixed(1), 8)}${pad(b.mase.toFixed(3), 7)}${pad(b.bias.toFixed(0), 8)}${pad((c['0.8'].coverage! * 100).toFixed(0) + '%', 7)}${pad((c['0.95'].coverage! * 100).toFixed(0) + '%', 7)}`);
      }
      const w = bt.ensembleWeights.at(-1)!;
      console.log(`Ensemble weights at the last origin: ${Object.entries(w).map(([k, v]) => `${k} ${(v! * 100).toFixed(0)}%`).join(', ')}\n`);
    }
  }

  console.log('=== WAPE % on the 6 test origins, all series (lower is better)');
  const cols: ModelName[] = ['snaive', 'holtwinters', 'ridge', 'lstm', 'ensemble'];
  console.log(padR('Series', 16) + cols.map((c) => pad(c, 13)).join(''));
  for (const [k, row] of Object.entries(wapeTable)) console.log(padR(k, 16) + cols.map((c) => pad(row[c] !== undefined ? row[c]!.toFixed(1) : '–', 13)).join(''));
  const avg = (c: ModelName) => Object.values(wapeTable).reduce((s, r) => s + (r[c] ?? NaN), 0) / Object.keys(wapeTable).length;
  console.log(padR('Mean', 16) + cols.map((c) => pad(Number.isFinite(avg(c)) ? avg(c).toFixed(1) : '–', 13)).join(''));
  console.log(`\nTS vs Python reference (snaive, Holt-Winters, Ridge): max |ΔWAPE| = ${maxDiff.toExponential(2)} percentage points\n`);

  // Update path: fine-tune the default series' LSTM on the data after its cut-off
  const def = loaded[fx.meta.key];
  if (def) {
    const s = toSeriesData(sub.series.find((x) => x.key === fx.meta.key)!);
    const last = s.y.length - 1;
    const before = lstmForecast(def.model, def.meta, s, cal, last);
    const smp = buildSamples(s, cal, Math.max(def.meta.cutoff - 180, TS + 27), last - LSTM_H, def.meta.mu, def.meta.sigma);
    const t0 = Date.now();
    const r = await trainLstm(def.model, smp, def.meta.mu, def.meta.sigma, { epochs: 15, learningRate: 0.0007, validationShare: 0.15, patience: 5 });
    const after = lstmForecast(def.model, { ...def.meta, correction: r.correction }, s, cal, last);
    console.log(`=== LSTM fine-tune on new data (${dateAt(cal, def.meta.cutoff)} → ${dateAt(cal, last)}): ${smp.origins.length} samples, ${r.epochs} epochs, ${((Date.now() - t0) / 1000).toFixed(1)} s`);
    console.log(`Next-14-day forecast from ${dateAt(cal, last)}:`);
    console.log(`  before: ${Array.from(before, (v) => v.toFixed(0)).join(' ')}`);
    console.log(`  after : ${Array.from(after, (v) => v.toFixed(0)).join(' ')}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
