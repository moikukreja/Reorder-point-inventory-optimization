/**
 * LSTM demand model (TensorFlow.js). The same code runs at build time (Node, for
 * pre-training) and in the browser's forecast worker (for fine-tuning and inference).
 *
 * Architecture (direct multi-horizon, H = 14):
 *   past window  [W = 28 days × 8]   ──► LSTM(32) ─┐
 *   known future [H days × 8, flat]  ──► Dense(32, relu) ─┴► concat ► Dense(64, relu) ► Dense(H)
 *
 * Past-step features:   scaled log sales, promotions, holiday, event, quake, payday, sin/cos weekday
 * Future-day features:  promotions, holiday, event, quake, payday, sin/cos weekday, closed flag
 * Target: z = (log1p(sales) − μ) / σ for the next H days. Back-transform: expm1(zσ + μ) × c, where
 * c is a smearing correction (Σy / Σŷ on the validation window) that removes log-retransform bias.
 */
import * as tf from '@tensorflow/tfjs';
import { impute, isClosed, type Calendar, type SeriesData } from '../features';

export const LSTM_W = 28;
export const LSTM_H = 14;
const PAST_F = 8;
const FUT_F = 8;

export interface LstmMeta {
  version: 1;
  key: string;
  W: number;
  H: number;
  mu: number;
  sigma: number;
  correction: number;
  trainStart: number;
  /** Last day index whose sales were used in training (inclusive). */
  cutoff: number;
  cutoffDate: string;
  trainedAt: string;
  epochs: number;
  valLoss: number;
  seed: number;
  fineTunedTo?: number;
}

export interface Samples {
  past: Float32Array;
  fut: Float32Array;
  target: Float32Array;
  origins: number[];
}

const promoScale = (p: number) => Math.log1p(p) / 5;

function pastStep(out: Float32Array, o: number, t: number, z: Float64Array, s: SeriesData, cal: Calendar, mu: number, sigma: number) {
  const dow = cal.dow[t];
  out[o] = (Math.log1p(z[t]) - mu) / sigma;
  out[o + 1] = promoScale(s.promo[t]);
  out[o + 2] = cal.holiday[t];
  out[o + 3] = cal.event[t];
  out[o + 4] = cal.quake[t];
  out[o + 5] = cal.payday[t];
  out[o + 6] = Math.sin((2 * Math.PI * dow) / 7);
  out[o + 7] = Math.cos((2 * Math.PI * dow) / 7);
}

function futStep(out: Float32Array, o: number, t: number, s: SeriesData, cal: Calendar) {
  const dow = cal.dow[t];
  out[o] = promoScale(s.promo[t]);
  out[o + 1] = cal.holiday[t];
  out[o + 2] = cal.event[t];
  out[o + 3] = cal.quake[t];
  out[o + 4] = cal.payday[t];
  out[o + 5] = Math.sin((2 * Math.PI * dow) / 7);
  out[o + 6] = Math.cos((2 * Math.PI * dow) / 7);
  out[o + 7] = isClosed(s, cal, t) ? 1 : 0;
}

/** Scaler fitted on open days in [from, to]. */
export function fitScaler(s: SeriesData, from: number, to: number): { mu: number; sigma: number } {
  let n = 0, m = 0, m2 = 0;
  for (let t = from; t <= to; t++) {
    const v = s.y[t];
    if (Number.isNaN(v)) continue;
    const x = Math.log1p(v);
    n++;
    const d = x - m;
    m += d / n;
    m2 += d * (x - m);
  }
  return { mu: m, sigma: Math.sqrt(m2 / n) };
}

/**
 * Builds training samples for origins o ∈ [oFrom, oTo]. Each sample uses past days
 * o−W+1..o and targets o+1..o+H. Samples whose targets include a closed day are skipped.
 */
export function buildSamples(s: SeriesData, cal: Calendar, oFrom: number, oTo: number, mu: number, sigma: number): Samples {
  const z = impute(s.y, oTo + LSTM_H);
  const origins: number[] = [];
  for (let o = oFrom; o <= oTo; o++) {
    let ok = true;
    for (let h = 1; h <= LSTM_H; h++) if (Number.isNaN(s.y[o + h])) ok = false;
    if (ok) origins.push(o);
  }
  const n = origins.length;
  const past = new Float32Array(n * LSTM_W * PAST_F);
  const fut = new Float32Array(n * LSTM_H * FUT_F);
  const target = new Float32Array(n * LSTM_H);
  origins.forEach((o, i) => {
    for (let k = 0; k < LSTM_W; k++) pastStep(past, (i * LSTM_W + k) * PAST_F, o - LSTM_W + 1 + k, z, s, cal, mu, sigma);
    for (let h = 1; h <= LSTM_H; h++) {
      futStep(fut, (i * LSTM_H + h - 1) * FUT_F, o + h, s, cal);
      target[i * LSTM_H + h - 1] = (Math.log1p(s.y[o + h]) - mu) / sigma;
    }
  });
  return { past, fut, target, origins };
}

export function createLstmModel(seed = 7): tf.LayersModel {
  const pastIn = tf.input({ shape: [LSTM_W, PAST_F], name: 'past' });
  const futIn = tf.input({ shape: [LSTM_H * FUT_F], name: 'future' });
  const h = tf.layers
    .lstm({
      units: 32,
      kernelInitializer: tf.initializers.glorotUniform({ seed }),
      // Glorot instead of TF's default orthogonal: the orthogonal initializer logs a console warning
      // for this 32×128 matrix on every model build, including when weights are loaded in the browser
      recurrentInitializer: tf.initializers.glorotUniform({ seed: seed + 1 }),
      name: 'lstm',
    })
    .apply(pastIn) as tf.SymbolicTensor;
  const f = tf.layers.dense({ units: 32, activation: 'relu', kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 2 }), name: 'future_dense' }).apply(futIn) as tf.SymbolicTensor;
  const c = tf.layers.concatenate({ name: 'concat' }).apply([h, f]) as tf.SymbolicTensor;
  const d = tf.layers.dense({ units: 64, activation: 'relu', kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 3 }), name: 'head' }).apply(c) as tf.SymbolicTensor;
  const out = tf.layers.dense({ units: LSTM_H, kernelInitializer: tf.initializers.glorotUniform({ seed: seed + 4 }), name: 'out' }).apply(d) as tf.SymbolicTensor;
  return tf.model({ inputs: [pastIn, futIn], outputs: out, name: 'demand_lstm' });
}

export interface TrainOptions {
  epochs: number;
  learningRate: number;
  batchSize?: number;
  /** Share of the most recent samples held out for early stopping and the smearing correction. */
  validationShare?: number;
  patience?: number;
  onEpoch?: (epoch: number, loss: number, valLoss: number) => void;
  /** Checked every epoch, so the user can cancel. */
  shouldStop?: () => boolean;
}

/**
 * Trains (or fine-tunes) on samples. It keeps the weights with the best validation loss
 * and returns that loss plus the smearing correction computed on the validation window.
 */
export async function trainLstm(model: tf.LayersModel, smp: Samples, mu: number, sigma: number, o: TrainOptions): Promise<{ valLoss: number; epochs: number; correction: number }> {
  const n = smp.origins.length;
  const nVal = Math.max(1, Math.round(n * (o.validationShare ?? 0.15)));
  const nTr = n - nVal;
  const pastT = tf.tensor3d(smp.past, [n, LSTM_W, PAST_F]);
  const futT = tf.tensor2d(smp.fut, [n, LSTM_H * FUT_F]);
  const yT = tf.tensor2d(smp.target, [n, LSTM_H]);
  const [pTr, pVa] = [pastT.slice([0, 0, 0], [nTr, -1, -1]), pastT.slice([nTr, 0, 0], [nVal, -1, -1])];
  const [fTr, fVa] = [futT.slice([0, 0], [nTr, -1]), futT.slice([nTr, 0], [nVal, -1])];
  const [yTr, yVa] = [yT.slice([0, 0], [nTr, -1]), yT.slice([nTr, 0], [nVal, -1])];

  model.compile({ optimizer: tf.train.adam(o.learningRate), loss: 'meanSquaredError' });
  let best = Infinity;
  let bestWeights: tf.Tensor[] | null = null;
  let since = 0;
  let epochsRun = 0;
  await model.fit([pTr, fTr], yTr, {
    epochs: o.epochs,
    batchSize: o.batchSize ?? 32,
    shuffle: true,
    validationData: [[pVa, fVa], yVa],
    verbose: 0,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        epochsRun = epoch + 1;
        const vl = logs?.val_loss ?? Infinity;
        o.onEpoch?.(epoch, logs?.loss ?? NaN, vl);
        if (vl < best - 1e-5) {
          best = vl;
          since = 0;
          bestWeights?.forEach((w) => w.dispose());
          bestWeights = model.getWeights().map((w) => w.clone());
        } else if (++since >= (o.patience ?? 10)) {
          model.stopTraining = true;
        }
        if (o.shouldStop?.()) model.stopTraining = true;
      },
    },
  });
  if (bestWeights) {
    model.setWeights(bestWeights);
    (bestWeights as tf.Tensor[]).forEach((w) => w.dispose());
  }
  // Smearing correction on the validation window: Σ actual / Σ predicted (in units)
  const pred = model.predict([pVa, fVa]) as tf.Tensor;
  const pz = await pred.data();
  const az = await yVa.data();
  let sa = 0, sp = 0;
  for (let i = 0; i < pz.length; i++) {
    sa += Math.expm1(az[i] * sigma + mu);
    sp += Math.expm1(pz[i] * sigma + mu);
  }
  const correction = Math.min(1.1, Math.max(0.9, sa / sp));
  tf.dispose([pastT, futT, yT, pTr, pVa, fTr, fVa, yTr, yVa, pred]);
  return { valLoss: best, epochs: epochsRun, correction };
}

/** H-day forecast from origin T (observed data up to T). Closed days get a forecast of 0. */
export function lstmForecast(model: tf.LayersModel, meta: LstmMeta, s: SeriesData, cal: Calendar, T: number): Float64Array {
  const z = impute(s.y, T);
  const past = new Float32Array(LSTM_W * PAST_F);
  for (let k = 0; k < LSTM_W; k++) pastStep(past, k * PAST_F, T - LSTM_W + 1 + k, z, s, cal, meta.mu, meta.sigma);
  const fut = new Float32Array(LSTM_H * FUT_F);
  for (let h = 1; h <= LSTM_H; h++) futStep(fut, (h - 1) * FUT_F, T + h, s, cal);
  const out = tf.tidy(() => {
    const p = model.predict([tf.tensor3d(past, [1, LSTM_W, PAST_F]), tf.tensor2d(fut, [1, LSTM_H * FUT_F])]) as tf.Tensor;
    return p.dataSync();
  });
  const res = new Float64Array(LSTM_H);
  for (let h = 0; h < LSTM_H; h++) res[h] = isClosed(s, cal, T + h + 1) ? 0 : Math.max(0, Math.expm1(out[h] * meta.sigma + meta.mu) * meta.correction);
  return res;
}

/** Filesystem-safe folder name for a series key, e.g. "44|BEVERAGES" → "44-beverages". */
export const modelSlug = (key: string) => key.toLowerCase().replace(/[^a-z0-9]+/g, '-');
