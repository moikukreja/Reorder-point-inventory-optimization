/**
 * Constrained Bayesian optimisation over [0, 1]^d, as a replacement for manual parameter sweeps.
 *
 *   minimise f(x)   subject to   g_k(x) ≥ 0 for every constraint k
 *
 * - One GP models the objective, and one GP per constraint models its slack.
 * - Acquisition is constrained expected improvement: EI(x) × Π_k P(g_k(x) ≥ 0).
 *   EI is measured against the best feasible observation. Until a feasible point exists,
 *   the optimiser maximises the probability of feasibility instead.
 * - Initial design: a Latin hypercube. Each step evaluates the single point that maximises the
 *   acquisition, found by random search over candidates plus Nelder–Mead refinement.
 * - Every evaluation is reported through `onIteration`, so the UI can stream progress from a Web Worker.
 */
import { fitGp, predictGp, type GpHyper, type GpModel } from './gp';
import { nelderMead } from './optim';
import { normCdf, normPdf, Rng } from './stats';

export interface Evaluation<I = unknown> {
  objective: number;
  /** Constraint slacks, already normalised to comparable scales: ≥ 0 means satisfied. */
  slacks: number[];
  info?: I;
}

export interface BoProblem<I = unknown> {
  dim: number;
  evaluate: (x01: number[]) => Evaluation<I>;
}

export interface BoRecord<I = unknown> extends Evaluation<I> {
  iter: number;
  x: number[];
  feasible: boolean;
  phase: 'init' | 'bo' | 'random' | 'grid';
  /** Best feasible objective found so far (Infinity if none yet). */
  bestFeasible: number;
  /** Acquisition value and P(feasible) at the chosen point (BO phase only). */
  acquisition?: number;
  pFeasible?: number;
  /** Surrogate prediction at the chosen point before evaluating it. */
  predicted?: { mean: number; sd: number };
}

export interface BoOptions<I = unknown> {
  budget: number;
  nInit: number;
  seed: number;
  candidates?: number;
  /** Refit GP hyperparameters every k iterations (warm-started in between). */
  refitEvery?: number;
  /** Exploration margin for EI, as a fraction of the objective's SD. */
  xi?: number;
  onIteration?: (r: BoRecord<I>, all: BoRecord<I>[]) => void;
  shouldStop?: () => boolean;
}

export interface Surrogates {
  objective: GpModel;
  constraints: GpModel[];
}

export interface BoResult<I = unknown> {
  history: BoRecord<I>[];
  best: BoRecord<I> | null;
  surrogates: Surrogates;
}

export function latinHypercube(n: number, d: number, rng: Rng): number[][] {
  const pts = Array.from({ length: n }, () => new Array<number>(d));
  for (let j = 0; j < d; j++) {
    const perm = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const k = rng.int(i + 1);
      [perm[i], perm[k]] = [perm[k], perm[i]];
    }
    for (let i = 0; i < n; i++) pts[i][j] = (perm[i] + rng.next()) / n;
  }
  return pts;
}

/** Expected improvement for minimisation. */
export function expectedImprovement(mu: number, sd: number, best: number, xi: number): number {
  if (sd < 1e-12) return Math.max(0, best - mu - xi);
  const z = (best - mu - xi) / sd;
  return (best - mu - xi) * normCdf(z) + sd * normPdf(z);
}

const isFeasible = (sl: number[]) => sl.every((v) => v >= 0);

function record<I>(iter: number, x: number[], e: Evaluation<I>, phase: BoRecord['phase'], prevBest: number): BoRecord<I> {
  const feasible = isFeasible(e.slacks);
  return { ...e, iter, x, feasible, phase, bestFeasible: feasible ? Math.min(prevBest, e.objective) : prevBest };
}

export function fitSurrogates(hist: BoRecord[], prev?: { obj: GpHyper; cons: GpHyper[] }, fixed = false): Surrogates {
  const X = hist.map((h) => h.x);
  const nCons = hist[0].slacks.length;
  return {
    objective: fitGp(X, hist.map((h) => h.objective), { init: prev?.obj, fixed: fixed && !!prev }),
    constraints: Array.from({ length: nCons }, (_, k) => fitGp(X, hist.map((h) => h.slacks[k]), { init: prev?.cons[k], fixed: fixed && !!prev })),
  };
}

/** Constrained-EI acquisition and P(feasible) at x under the current surrogates. */
export function acquisition(s: Surrogates, x: number[], best: number, xiAbs: number): { acq: number; pFeas: number; mean: number; sd: number } {
  let pFeas = 1;
  for (const g of s.constraints) {
    const p = predictGp(g, x);
    pFeas *= normCdf(p.mean / Math.max(p.sd, 1e-9));
  }
  const o = predictGp(s.objective, x);
  const acq = Number.isFinite(best) ? expectedImprovement(o.mean, o.sd, best, xiAbs) * pFeas : pFeas;
  return { acq, pFeas, mean: o.mean, sd: o.sd };
}

export function bayesianOptimise<I>(prob: BoProblem<I>, o: BoOptions<I>): BoResult<I> {
  const rng = new Rng(o.seed);
  const hist: BoRecord<I>[] = [];
  let best = Infinity;
  const push = (r: BoRecord<I>) => {
    hist.push(r);
    best = r.bestFeasible;
    o.onIteration?.(r, hist);
  };
  for (const x of latinHypercube(o.nInit, prob.dim, rng)) {
    if (o.shouldStop?.()) break;
    push(record(hist.length, x, prob.evaluate(x), 'init', best));
  }

  let sur = fitSurrogates(hist);
  const refit = o.refitEvery ?? 3;
  const nCand = o.candidates ?? 1500;
  while (hist.length < o.budget && !o.shouldStop?.()) {
    const it = hist.length - o.nInit;
    if (it > 0) {
      const prev = { obj: sur.objective.hyper, cons: sur.constraints.map((c) => c.hyper) };
      sur = fitSurrogates(hist, prev, it % refit !== 0);
    }
    const xiAbs = (o.xi ?? 0.01) * sur.objective.yStd;
    // Random candidates, then refine the best three with Nelder–Mead (clamped to the unit box)
    const scored: { x: number[]; a: number }[] = [];
    for (let i = 0; i < nCand; i++) {
      const x = Array.from({ length: prob.dim }, () => rng.next());
      scored.push({ x, a: acquisition(sur, x, best, xiAbs).acq });
    }
    scored.sort((a, b) => b.a - a.a);
    let bestX = scored[0].x, bestA = scored[0].a;
    for (const c of scored.slice(0, 3)) {
      const r = nelderMead((x) => {
        if (x.some((v) => v < 0 || v > 1)) return 1e9;
        return -acquisition(sur, x, best, xiAbs).acq;
      }, c.x, { maxiter: 60, xatol: 1e-4, fatol: 1e-12 });
      if (-r.fun > bestA) {
        bestA = -r.fun;
        bestX = r.x;
      }
    }
    // Avoid re-sampling an existing point (it would add no information)
    if (hist.some((h) => h.x.every((v, j) => Math.abs(v - bestX[j]) < 1e-3))) bestX = scored.find((c) => !hist.some((h) => h.x.every((v, j) => Math.abs(v - c.x[j]) < 1e-3)))?.x ?? bestX;
    const a = acquisition(sur, bestX, best, xiAbs);
    const rec = record(hist.length, bestX, prob.evaluate(bestX), 'bo', best);
    rec.acquisition = a.acq;
    rec.pFeasible = a.pFeas;
    rec.predicted = { mean: a.mean, sd: a.sd };
    push(rec);
  }
  sur = fitSurrogates(hist, { obj: sur.objective.hyper, cons: sur.constraints.map((c) => c.hyper) });
  return { history: hist, best: bestRecord(hist), surrogates: sur };
}

export function bestRecord<I>(hist: BoRecord<I>[]): BoRecord<I> | null {
  let b: BoRecord<I> | null = null;
  for (const h of hist) if (h.feasible && (!b || h.objective < b.objective)) b = h;
  return b;
}

/** Benchmark: uniform random search with the same evaluation budget. */
export function randomSearch<I>(prob: BoProblem<I>, budget: number, seed: number): BoResult<I>['history'] {
  const rng = new Rng(seed);
  const hist: BoRecord<I>[] = [];
  let best = Infinity;
  for (let i = 0; i < budget; i++) {
    const x = Array.from({ length: prob.dim }, () => rng.next());
    const r = record(i, x, prob.evaluate(x), 'random', best);
    best = r.bestFeasible;
    hist.push(r);
  }
  return hist;
}

/** Benchmark: exhaustive grid with g points per dimension (the "manual parameter sweep"). */
export function gridSearch<I>(prob: BoProblem<I>, g: number): BoResult<I>['history'] {
  const hist: BoRecord<I>[] = [];
  let best = Infinity;
  const total = g ** prob.dim;
  for (let k = 0; k < total; k++) {
    const x: number[] = [];
    let r = k;
    for (let j = 0; j < prob.dim; j++) {
      x.push((r % g) / (g - 1));
      r = Math.floor(r / g);
    }
    const rec = record(k, x, prob.evaluate(x), 'grid', best);
    best = rec.bestFeasible;
    hist.push(rec);
  }
  return hist;
}
