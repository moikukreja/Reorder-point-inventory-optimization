/** Data-free tests for the Gaussian process, Bayesian optimisation and sensitivity analysis. */
import { describe, expect, it } from 'vitest';
import { fitGp, predictGp } from '../../src/core/gp';
import { bayesianOptimise, expectedImprovement, gridSearch, latinHypercube, randomSearch, type BoProblem } from '../../src/core/bayesOpt';
import { elasticities, tornado, varianceSplit } from '../../src/core/sensitivity';
import { slideSafetyStock } from '../../src/core/safetyStock';
import { Rng } from '../../src/core/stats';

/** Branin–Hoo on [0,1]²; global minimum 0.397887 at three points. */
function branin(x: number[]): number {
  const a = 15 * x[0] - 5, b = 15 * x[1];
  return (b - (5.1 / (4 * Math.PI ** 2)) * a * a + (5 / Math.PI) * a - 6) ** 2 + 10 * (1 - 1 / (8 * Math.PI)) * Math.cos(a) + 10;
}
const braninProblem: BoProblem = { dim: 2, evaluate: (x) => ({ objective: branin(x), slacks: [] }) };

describe('Gaussian process', () => {
  const X = [0, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1].map((v) => [v]);
  const f = (x: number) => Math.sin(6 * x) + 0.5 * x;
  const gp = fitGp(X, X.map((x) => f(x[0])));
  it('interpolates the training data with small uncertainty', () => {
    for (const x of X) {
      const p = predictGp(gp, x);
      expect(Math.abs(p.mean - f(x[0]))).toBeLessThan(0.05);
      expect(p.sd).toBeLessThan(0.1);
    }
  });
  it('predicts unseen points and is more uncertain between data points', () => {
    for (const x of [0.07, 0.36, 0.64, 0.93]) expect(Math.abs(predictGp(gp, [x]).mean - f(x))).toBeLessThan(0.15);
    expect(predictGp(gp, [0.36]).sd).toBeGreaterThan(predictGp(gp, [0.29]).sd);
  });
  it('expected improvement is non-negative and grows with uncertainty', () => {
    expect(expectedImprovement(1, 0, 0.5, 0)).toBe(0);
    expect(expectedImprovement(1, 1, 0.5, 0)).toBeGreaterThan(expectedImprovement(1, 0.2, 0.5, 0));
  });
});

describe('design of experiments', () => {
  it('Latin hypercube puts exactly one point in each stratum of every dimension', () => {
    const pts = latinHypercube(10, 3, new Rng(1));
    for (let d = 0; d < 3; d++) expect(new Set(pts.map((p) => Math.floor(p[d] * 10))).size).toBe(10);
  });
  it('grid search covers g^d points including the corners', () => {
    const h = gridSearch(braninProblem, 5);
    expect(h.length).toBe(25);
    expect(h.some((r) => r.x[0] === 0 && r.x[1] === 0)).toBe(true);
    expect(h.some((r) => r.x[0] === 1 && r.x[1] === 1)).toBe(true);
  });
});

describe('Bayesian optimisation', () => {
  it('finds the Branin minimum (0.398) within 40 evaluations and beats random search', () => {
    const bo = [1, 2, 3].map((seed) => bayesianOptimise(braninProblem, { budget: 40, nInit: 8, seed }).best!.objective);
    const rs = [1, 2, 3].map((seed) => Math.min(...randomSearch(braninProblem, 40, seed).map((r) => r.objective)));
    const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
    expect(Math.max(...bo)).toBeLessThan(0.6);
    expect(mean(bo)).toBeLessThan(mean(rs));
  });
  it('respects a constraint: minimise (x − 0.8)² subject to x ≤ 0.5 → x ≈ 0.5', () => {
    const p: BoProblem = { dim: 1, evaluate: (x) => ({ objective: (x[0] - 0.8) ** 2, slacks: [(0.5 - x[0]) / 0.05] }) };
    const r = bayesianOptimise(p, { budget: 20, nInit: 5, seed: 4 });
    expect(r.best!.feasible).toBe(true);
    expect(Math.abs(r.best!.x[0] - 0.5)).toBeLessThan(0.02);
  });
  it('streams every evaluation and reports a non-increasing best-so-far', () => {
    const seen: number[] = [];
    const r = bayesianOptimise(braninProblem, { budget: 15, nInit: 5, seed: 9, onIteration: (rec) => seen.push(rec.bestFeasible) });
    expect(seen.length).toBe(15);
    for (let i = 1; i < seen.length; i++) expect(seen[i]).toBeLessThanOrEqual(seen[i - 1]);
    expect(r.history.filter((h) => h.phase === 'bo').every((h) => h.acquisition !== undefined)).toBe(true);
  });
  it('can be stopped early', () => {
    let n = 0;
    const r = bayesianOptimise(braninProblem, { budget: 40, nInit: 5, seed: 2, onIteration: () => n++, shouldStop: () => n >= 7 });
    expect(r.history.length).toBe(7);
  });
});

describe('sensitivity analysis', () => {
  const base = { serviceLevel: 0.95, dBar: 100, sdD: 20, L: 5, sdL: 1.5 };
  it('tornado bars are sorted by swing and bracket the base ROP', () => {
    const t = tornado(base);
    const rop = slideSafetyStock(base).reorderPoint;
    for (let i = 1; i < t.length; i++) expect(t[i - 1].swing).toBeGreaterThanOrEqual(t[i].swing);
    for (const b of t) expect(Math.min(b.ropLow, b.ropHigh)).toBeLessThan(rop);
  });
  it('analytic elasticities match finite differences', () => {
    const e = elasticities(base);
    const rop = (i: typeof base) => slideSafetyStock(i).reorderPoint;
    for (const k of ['dBar', 'sdD', 'L', 'sdL', 'serviceLevel'] as const) {
      const h = base[k] * 1e-5;
      const fd = ((rop({ ...base, [k]: base[k] + h }) - rop({ ...base, [k]: base[k] - h })) / (2 * h)) * (base[k] / rop(base));
      expect(e[k]).toBeCloseTo(fd, 4);
    }
  });
  it('variance split for the hand example: 91.8% from lead-time uncertainty', () => {
    const v = varianceSplit(base);
    expect(v.demand + v.leadTime).toBeCloseTo(1, 12);
    expect(v.leadTime).toBeCloseTo(22500 / 24500, 10);
  });
});
