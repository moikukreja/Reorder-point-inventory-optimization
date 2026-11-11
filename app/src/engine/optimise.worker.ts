/// <reference lib="webworker" />
/**
 * Optimiser Web Worker: Bayesian optimisation of (s, Q), the random and grid benchmarks, and
 * optimal-policy sensitivity. Progress is streamed to the UI after every evaluation.
 * Runs are synchronous inside this worker. To stop one, the UI terminates the worker (client.ts).
 */
import * as Comlink from 'comlink';
import { acquisition, bayesianOptimise, bestRecord, gridSearch, randomSearch, type BoRecord } from '../core/bayesOpt';
import { policyProblem, standardPerturbations, toPolicy, type PolicyEvalInfo } from '../core/policyOptimizer';
import { predictGp } from '../core/gp';
import type { PolicySimConfig } from '../core/policySim';
import type { BenchmarkResult, OptimiseIteration, OptimiseRequest, OptimiseResult, PerturbationRow, SurfaceGrid } from './types';

function simConfig(r: OptimiseRequest): Omit<PolicySimConfig, 'trace'> {
  return {
    horizon: r.horizon, warmup: 14, replications: r.replications, seed: r.seed, leadTime: r.leadTime, costs: r.costs,
    demand: { kind: 'generated', mean: r.demand.mean, residuals: r.demand.residuals, errorPaths: r.demand.errorPaths },
  };
}

function toIter(h: BoRecord<PolicyEvalInfo>): OptimiseIteration {
  return { iter: h.iter, phase: h.phase, s: h.info!.s, Q: h.info!.Q, cost: h.objective, feasible: h.feasible, bestFeasible: h.bestFeasible, pFeasible: h.pFeasible, acquisition: h.acquisition, metrics: h.info!.metrics };
}

const api = {
  optimise(r: OptimiseRequest, onIter?: (it: OptimiseIteration) => void): OptimiseResult {
    const t0 = performance.now();
    const prob = policyProblem(simConfig(r), r.bounds, r.constraints, r.mode, r.fixedQ);
    const res = bayesianOptimise(prob, { budget: r.budget, nInit: r.nInit, seed: r.seed, onIteration: (h) => onIter?.(toIter(h)) });
    // Surrogate surfaces for the charts (posterior over the whole search box)
    const best = res.best ? toIter(res.best) : null;
    const nG = r.mode === 'sQ' ? 32 : 120;
    const sAxis: number[] = [], qAxis: number[] = [];
    const mean: number[][] = [], sd: number[][] = [], pF: number[][] = [], acq: number[][] = [];
    const bestVal = res.best ? res.best.objective : Infinity;
    const qN = r.mode === 'sQ' ? nG : 1;
    for (let j = 0; j < qN; j++) {
      const rowM: number[] = [], rowS: number[] = [], rowP: number[] = [], rowA: number[] = [];
      for (let i = 0; i < nG; i++) {
        const x = r.mode === 'sQ' ? [i / (nG - 1), j / (nG - 1)] : [i / (nG - 1)];
        const p = toPolicy(x, r.bounds, r.mode, r.fixedQ);
        if (j === 0) sAxis.push(p.s);
        if (i === 0) qAxis.push(p.Q);
        const o = predictGp(res.surrogates.objective, x);
        const a = acquisition(res.surrogates, x, bestVal, 0.01 * res.surrogates.objective.yStd);
        rowM.push(o.mean); rowS.push(o.sd); rowP.push(a.pFeas); rowA.push(a.acq);
      }
      mean.push(rowM); sd.push(rowS); pF.push(rowP); acq.push(rowA);
    }
    const surface: SurfaceGrid = { s: sAxis, q: qAxis, mean, sd, pFeasible: pF, acquisition: acq };
    return { history: res.history.map((h) => toIter(h as BoRecord<PolicyEvalInfo>)), best, surface, lengthScales: res.surrogates.objective.hyper.ls, elapsedMs: performance.now() - t0 };
  },

  /** Grid ("manual sweep") and random-search benchmarks with the same simulator. */
  benchmark(r: OptimiseRequest, gridSize: number, onProgress?: (frac: number) => void): BenchmarkResult {
    const sim = simConfig(r);
    const t0 = performance.now();
    const gp = policyProblem(sim, r.bounds, r.constraints, r.mode, r.fixedQ);
    const total = r.mode === 'sQ' ? gridSize * gridSize : gridSize;
    let done = 0;
    const wrapped = { ...gp, evaluate: (x: number[]) => { const e = gp.evaluate(x); done++; if (done % 10 === 0) onProgress?.(done / total); return e; } };
    const grid = gridSearch(wrapped, gridSize);
    const gMs = performance.now() - t0;
    const rnd = randomSearch(policyProblem(sim, r.bounds, r.constraints, r.mode, r.fixedQ), r.budget, r.seed);
    const g = bestRecord(grid) as BoRecord<PolicyEvalInfo> | null;
    const rb = bestRecord(rnd) as BoRecord<PolicyEvalInfo> | null;
    return {
      grid: { size: grid.length, best: g ? toIter(g) : null, bestSoFar: grid.map((h) => h.bestFeasible), elapsedMs: gMs },
      random: { best: rb ? toIter(rb) : null, bestSoFar: rnd.map((h) => h.bestFeasible) },
    };
  },

  /** Re-runs the optimiser under perturbed economics: how the optimal policy moves. */
  perturb(r: OptimiseRequest, budget: number, onProgress?: (frac: number, label: string) => void): PerturbationRow[] {
    const sim = simConfig(r);
    const list = standardPerturbations(r.demand.mean.reduce((a, b) => a + b, 0) / r.demand.mean.length);
    const out: PerturbationRow[] = [];
    list.forEach((p, i) => {
      onProgress?.(i / list.length, p.label);
      const a = p.apply(sim, r.constraints, r.bounds);
      const res = bayesianOptimise(policyProblem(a.sim, a.bounds, a.cons, r.mode, r.fixedQ), { budget, nInit: 6, seed: r.seed + 17 });
      out.push({ id: p.id, label: p.label, best: res.best ? toIter(res.best as BoRecord<PolicyEvalInfo>) : null });
    });
    onProgress?.(1, 'Done');
    return out;
  },
};

export type OptimiseApi = typeof api;
Comlink.expose(api);
