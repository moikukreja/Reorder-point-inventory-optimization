/**
 * Gaussian-process regression used as the Bayesian optimiser's surrogate model.
 *
 * - Kernel: Matérn-5/2 with automatic relevance determination (one length-scale per input):
 *     k(r) = σf² (1 + √5 r + 5r²/3) exp(−√5 r),   r² = Σ_d ((x_d − x'_d) / ℓ_d)²
 * - Inputs are expected in [0, 1]^d, and outputs are standardised internally.
 * - A noise term σn² handles Monte Carlo noise in the simulated objective.
 * - Hyperparameters maximise the log marginal likelihood (Nelder–Mead, multi-start), using
 *   bounded transforms so they stay in sensible ranges.
 */
import { backSub, cholesky, forwardSub } from './linalg';
import { nelderMead } from './optim';

export interface GpHyper {
  sf: number; // signal SD (standardised units)
  ls: number[]; // length-scales
  sn: number; // noise SD (standardised units)
}

export interface GpModel {
  X: number[][];
  yMean: number;
  yStd: number;
  hyper: GpHyper;
  L: Float64Array;
  alpha: Float64Array;
  logMarginalLikelihood: number;
}

const SQRT5 = Math.sqrt(5);
const LS_MIN = 0.03, LS_MAX = 3, SN_MIN = 1e-4, SN_MAX = 1, SF_MIN = 0.1, SF_MAX = 10;
const sig = (u: number) => 1 / (1 + Math.exp(-u));
const logit = (p: number) => Math.log(p / (1 - p));
const between = (lo: number, hi: number, u: number) => Math.exp(Math.log(lo) + (Math.log(hi) - Math.log(lo)) * sig(u));
const inv = (lo: number, hi: number, v: number) => logit(Math.min(1 - 1e-9, Math.max(1e-9, (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)))));

function unpack(u: number[], d: number): GpHyper {
  return { sf: between(SF_MIN, SF_MAX, u[0]), ls: u.slice(1, 1 + d).map((v) => between(LS_MIN, LS_MAX, v)), sn: between(SN_MIN, SN_MAX, u[1 + d]) };
}
function pack(h: GpHyper): number[] {
  return [inv(SF_MIN, SF_MAX, h.sf), ...h.ls.map((l) => inv(LS_MIN, LS_MAX, l)), inv(SN_MIN, SN_MAX, h.sn)];
}

export function matern52(a: ArrayLike<number>, b: ArrayLike<number>, h: GpHyper): number {
  let r2 = 0;
  for (let i = 0; i < a.length; i++) r2 += ((a[i] - b[i]) / h.ls[i]) ** 2;
  const r = Math.sqrt(r2);
  return h.sf * h.sf * (1 + SQRT5 * r + (5 / 3) * r2) * Math.exp(-SQRT5 * r);
}

function factor(X: number[][], ys: Float64Array, h: GpHyper): { L: Float64Array; alpha: Float64Array; lml: number } | null {
  const n = X.length;
  const K = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      const k = matern52(X[i], X[j], h);
      K[i * n + j] = k;
      K[j * n + i] = k;
    }
    K[i * n + i] += h.sn * h.sn + 1e-8;
  }
  let L: Float64Array;
  try {
    L = cholesky(K, n);
  } catch {
    return null;
  }
  const alpha = backSub(L, n, forwardSub(L, n, ys));
  let quad = 0, logdet = 0;
  for (let i = 0; i < n; i++) {
    quad += ys[i] * alpha[i];
    logdet += Math.log(L[i * n + i]);
  }
  return { L, alpha, lml: -0.5 * quad - logdet - 0.5 * n * Math.log(2 * Math.PI) };
}

export interface GpFitOptions {
  /** Start from these hyperparameters (warm start), otherwise use default multi-starts. */
  init?: GpHyper;
  /** Skip optimisation and use `init` as-is. */
  fixed?: boolean;
  maxiter?: number;
}

export function fitGp(X: number[][], y: number[], opts: GpFitOptions = {}): GpModel {
  const n = y.length, d = X[0].length;
  const yMean = y.reduce((a, b) => a + b, 0) / n;
  const yStd = Math.sqrt(y.reduce((a, b) => a + (b - yMean) ** 2, 0) / Math.max(1, n - 1)) || 1;
  const ys = Float64Array.from(y, (v) => (v - yMean) / yStd);
  const negLml = (u: number[]) => {
    const f = factor(X, ys, unpack(u, d));
    return f ? -f.lml : 1e10;
  };
  let best: GpHyper;
  if (opts.fixed && opts.init) best = opts.init;
  else {
    const starts: GpHyper[] = opts.init
      ? [opts.init]
      : [
          { sf: 1, ls: new Array(d).fill(0.3), sn: 0.05 },
          { sf: 1, ls: new Array(d).fill(0.8), sn: 0.2 },
          { sf: 1.5, ls: new Array(d).fill(0.15), sn: 0.01 },
        ];
    let bestVal = Infinity;
    best = starts[0];
    for (const st of starts) {
      const r = nelderMead(negLml, pack(st), { maxiter: opts.maxiter ?? 250, xatol: 1e-3, fatol: 1e-4 });
      if (r.fun < bestVal) {
        bestVal = r.fun;
        best = unpack(r.x, d);
      }
    }
  }
  const f = factor(X, ys, best) ?? factor(X, ys, { ...best, sn: Math.max(best.sn, 0.1) })!;
  return { X, yMean, yStd, hyper: best, L: f.L, alpha: f.alpha, logMarginalLikelihood: f.lml };
}

/** Posterior mean and SD at x (original output units). */
export function predictGp(m: GpModel, x: ArrayLike<number>): { mean: number; sd: number } {
  const n = m.X.length;
  const ks = new Float64Array(n);
  for (let i = 0; i < n; i++) ks[i] = matern52(x, m.X[i], m.hyper);
  let mu = 0;
  for (let i = 0; i < n; i++) mu += ks[i] * m.alpha[i];
  const v = forwardSub(m.L, n, ks);
  let vv = 0;
  for (let i = 0; i < n; i++) vv += v[i] * v[i];
  const varS = Math.max(1e-12, m.hyper.sf * m.hyper.sf - vv);
  return { mean: m.yMean + m.yStd * mu, sd: m.yStd * Math.sqrt(varS) };
}
