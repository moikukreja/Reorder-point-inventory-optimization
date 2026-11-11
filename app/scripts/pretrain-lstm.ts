/**
 * Build-time LSTM pre-training (the one precomputed artefact in the system).
 *
 * For each series it trains on data up to the first backtest calibration origin, so the
 * dashboard's backtest of the LSTM is leak-free. It writes public/models/<slug>/{model.json,
 * weights.bin, meta.json}. In the browser the model is then fine-tuned on everything after
 * the cut-off, and again whenever new data arrives.
 *
 * Usage: npm run pretrain                 (all series)
 *        npm run pretrain -- 44|BEVERAGES (one series)
 */
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import * as tf from '@tensorflow/tfjs';
import { loadSubset, MODELS_DIR } from './lib/local-data';
import { initNodeBackend, saveModel } from './lib/node-tf';
import { buildCalendar, dateAt, indexOf, toSeriesData, TRAIN_START_DATE } from '../src/core/features';
import { buildSamples, createLstmModel, fitScaler, LSTM_H, LSTM_W, modelSlug, trainLstm, type LstmMeta } from '../src/core/models/lstm';
import { DEFAULT_BACKTEST, originsFor } from '../src/core/backtest';

const SEED = 7;
const EPOCHS = 120;

async function main() {
  const backend = await initNodeBackend();
  const sub = loadSubset();
  const cal = buildCalendar(sub);
  const only = process.argv[2];
  const TS = indexOf(cal, TRAIN_START_DATE);
  console.log(`TF.js ${tf.version.tfjs} backend=${backend} | series=${only ?? 'all'}`);

  for (const raw of sub.series) {
    if (only && raw.key !== only) continue;
    const s = toSeriesData(raw);
    const last = s.y.length - 1;
    const cutoff = originsFor(last, DEFAULT_BACKTEST).cal[0];
    const { mu, sigma } = fitScaler(s, TS, cutoff);
    const smp = buildSamples(s, cal, TS + LSTM_W - 1, cutoff - LSTM_H, mu, sigma);
    // Weight initialisation is seeded. TF.js batch shuffling uses Math.random, so reruns differ slightly.
    const model = createLstmModel(SEED);
    const t0 = Date.now();
    const res = await trainLstm(model, smp, mu, sigma, {
      epochs: EPOCHS, learningRate: 0.003, batchSize: 32, validationShare: 0.15, patience: 12,
      onEpoch: (e, l, vl) => { if (e % 10 === 0) console.log(`  ${raw.key} epoch ${e} loss=${l.toFixed(4)} val=${vl.toFixed(4)}`); },
    });
    const meta: LstmMeta = {
      version: 1, key: raw.key, W: LSTM_W, H: LSTM_H, mu, sigma, correction: res.correction,
      trainStart: TS, cutoff, cutoffDate: dateAt(cal, cutoff), trainedAt: new Date().toISOString(),
      epochs: res.epochs, valLoss: res.valLoss, seed: SEED,
    };
    const dir = join(MODELS_DIR, modelSlug(raw.key));
    await saveModel(model, dir);
    writeFileSync(join(dir, 'meta.json'), JSON.stringify(meta, null, 1));
    console.log(`✔ ${raw.key}: ${smp.origins.length} samples, ${res.epochs} epochs, val MSE ${res.valLoss.toFixed(4)}, correction ${res.correction.toFixed(3)}, ${((Date.now() - t0) / 1000).toFixed(1)} s → ${dir}`);
    model.dispose();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
