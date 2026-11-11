/**
 * UI-side handles to the two workers. They are created lazily, once.
 * Long optimiser runs block only the optimiser worker, so analysis and the UI stay responsive.
 */
import * as Comlink from 'comlink';
import type { AnalysisApi } from './analysis.worker';
import type { OptimiseApi } from './optimise.worker';

let analysis: Comlink.Remote<AnalysisApi> | null = null;
let optimiser: { api: Comlink.Remote<OptimiseApi>; worker: Worker } | null = null;

export function analysisWorker(): Comlink.Remote<AnalysisApi> {
  analysis ??= Comlink.wrap<AnalysisApi>(new Worker(new URL('./analysis.worker.ts', import.meta.url), { type: 'module', name: 'analysis' }));
  return analysis;
}

export function optimiserWorker(): Comlink.Remote<OptimiseApi> {
  optimiser ??= (() => {
    const worker = new Worker(new URL('./optimise.worker.ts', import.meta.url), { type: 'module', name: 'optimiser' });
    return { api: Comlink.wrap<OptimiseApi>(worker), worker };
  })();
  return optimiser.api;
}

/**
 * Hard-stops a running optimisation. The synchronous optimiser loop can't receive messages
 * mid-run, so the worker is terminated and a fresh one is created on the next call.
 */
export function stopOptimiser() {
  if (!optimiser) return;
  optimiser.worker.terminate();
  optimiser = null;
}

export { Comlink };
