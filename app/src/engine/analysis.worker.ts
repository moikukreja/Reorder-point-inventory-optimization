/// <reference lib="webworker" />
/**
 * Analysis Web Worker: forecasting, backtests, Monte Carlo, policy simulation and sensitivity
 * all run here, so the UI thread stays responsive.
 * The dataset only exists in this worker's memory, inside the user's browser.
 */
import './worker-shim';
import * as Comlink from 'comlink';
import * as tf from '@tensorflow/tfjs';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import { validateSubset } from '../data/schema';
import { modelSlug, type LstmMeta } from '../core/models/lstm';
import { AnalysisEngine, type LstmHandle } from './analysis';
import type { Settings } from './types';
import { generateDemoData } from '../data/demoData';
import type { NewRow } from '../data/newRows';

let engine: AnalysisEngine | null = null;
let tfReady: Promise<void> | null = null;

/**
 * TF.js on the WebAssembly backend (about 10× faster than plain JS for LSTM training, and no WebGL
 * context needed in a worker). It falls back to the CPU backend if WebAssembly is unavailable.
 */
function ensureTf() {
  tfReady ??= (async () => {
    try {
      setWasmPaths(new URL(`${import.meta.env.BASE_URL}tfjs-wasm/`, self.location.origin).href);
      if (!(await tf.setBackend('wasm'))) throw new Error('wasm unavailable');
    } catch {
      await tf.setBackend('cpu');
    }
    await tf.ready();
  })();
  return tfReady;
}

async function loadLstm(key: string): Promise<LstmHandle | null> {
  await ensureTf();
  const base = new URL(`${import.meta.env.BASE_URL}models/${modelSlug(key)}/`, self.location.origin);
  const res = await fetch(new URL('meta.json', base));
  if (!res.ok) return null;
  const meta = (await res.json()) as LstmMeta;
  const model = await tf.loadLayersModel(new URL('model.json', base).href);
  return { model, meta };
}

function need(): AnalysisEngine {
  if (!engine) throw new Error('No dataset loaded');
  return engine;
}

const api = {
  init(raw: unknown) {
    engine = new AnalysisEngine(validateSubset(raw), loadLstm);
    return engine.info();
  },
  /** Generates the synthetic demo dataset here (no real data involved) and returns it for persistence. */
  demoData() {
    return generateDemoData();
  },
  async prepare(key: string, useLstm: boolean, onProgress?: (msg: string, frac: number) => void) {
    if (!engine) throw new Error('No dataset loaded');
    return engine.prepare(key, useLstm, onProgress);
  },
  risk(key: string, settings: Settings) {
    if (!engine) throw new Error('No dataset loaded');
    return engine.risk(key, settings);
  },
  appendRows(key: string, rows: NewRow[]) {
    return need().appendRows(key, rows);
  },
  lastDate(key: string) {
    return need().lastDate(key);
  },
  startReplay(key: string, n: number) {
    return need().startReplay(key, n);
  },
  replayNext(key: string, k: number) {
    return need().replayNext(key, k);
  },
  stopReplay(key: string) {
    return need().stopReplay(key);
  },
  replayStatus(key: string) {
    return need().replayStatus(key);
  },
  async fineTuneLstm(key: string, onProgress?: (msg: string, frac: number) => void) {
    await ensureTf();
    return need().fineTuneLstm(key, onProgress);
  },
  async trainLstm(key: string, onProgress?: (msg: string, frac: number) => void) {
    await ensureTf();
    return need().trainLstmFromScratch(key, onProgress);
  },
  optimiserInputs(key: string, settings: Settings) {
    if (!engine) throw new Error('No dataset loaded');
    return { demand: engine.demandModel(key, settings), leadTime: engine.scenarioLeadTime(key, settings) };
  },
};

export type AnalysisApi = typeof api;
Comlink.expose(api);
