/** Data-free tests for statistical primitives. They always run, including in CI without the dataset. */
import { describe, expect, it } from 'vitest';
import { invNormCdf, normCdf, quantile, Rng, sampleLeadTime, mean, std } from '../../src/core/stats';
import { nelderMead } from '../../src/core/optim';
import { solveSPD } from '../../src/core/linalg';

describe('normal distribution', () => {
  it('inverse CDF matches scipy.stats.norm.ppf', () => {
    // Reference values from scipy.stats.norm.ppf
    const ref: [number, number][] = [
      [0.5, 0], [0.8, 0.8416212335729143], [0.9, 1.2815515655446004], [0.95, 1.6448536269514722],
      [0.975, 1.959963984540054], [0.99, 2.3263478740408408], [0.999, 3.090232306167813], [0.01, -2.3263478740408408],
    ];
    for (const [p, z] of ref) expect(invNormCdf(p)).toBeCloseTo(z, 10);
  });
  it('CDF inverts the quantile', () => {
    for (const p of [0.001, 0.05, 0.3, 0.7, 0.95, 0.9999]) expect(normCdf(invNormCdf(p))).toBeCloseTo(p, 12);
  });
});

describe('quantile (NumPy linear / type 7)', () => {
  it('matches np.quantile on a small array', () => {
    const x = [3, 1, 4, 1, 5, 9, 2, 6];
    // np.quantile(x, [0.1, 0.5, 0.975]) = [1.0, 3.5, 8.475]
    expect(quantile(x, 0.1)).toBeCloseTo(1.0, 12);
    expect(quantile(x, 0.5)).toBeCloseTo(3.5, 12);
    expect(quantile(x, 0.975)).toBeCloseTo(8.475, 12);
  });
});

describe('random samplers', () => {
  const N = 200_000;
  it('normal sampler has mean 0 and sd 1', () => {
    const r = new Rng(1);
    const xs = Float64Array.from({ length: N }, () => r.normal());
    expect(Math.abs(mean(xs))).toBeLessThan(0.01);
    expect(std(xs)).toBeCloseTo(1, 2);
  });
  it.each(['gamma', 'lognormal', 'normal'] as const)('%s lead time matches the requested mean and sd within 1%%', (dist) => {
    const r = new Rng(3);
    const xs = Float64Array.from({ length: N }, () => sampleLeadTime(r, 4, 1.2, dist));
    expect(Math.abs(mean(xs) / 4 - 1)).toBeLessThan(0.01);
    expect(Math.abs(std(xs) / 1.2 - 1)).toBeLessThan(0.01);
    expect(xs.reduce((m, v) => Math.min(m, v), Infinity)).toBeGreaterThan(0);
  });
  it('is reproducible for a given seed', () => {
    const a = new Rng(99), b = new Rng(99);
    for (let i = 0; i < 10; i++) expect(a.next()).toBe(b.next());
  });
});

describe('numerical routines', () => {
  it('Nelder–Mead finds the Rosenbrock minimum', () => {
    const f = (x: number[]) => 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2;
    const r = nelderMead(f, [-1.2, 1], { maxiter: 2000, xatol: 1e-8, fatol: 1e-10 });
    expect(r.x[0]).toBeCloseTo(1, 4);
    expect(r.x[1]).toBeCloseTo(1, 4);
  });
  it('Cholesky solve recovers x for an SPD system', () => {
    const A = Float64Array.from([4, 2, 0.6, 2, 5, 1, 0.6, 1, 3]);
    const x = [1, -2, 3];
    const b = [0, 1, 2].map((i) => A[i * 3] * x[0] + A[i * 3 + 1] * x[1] + A[i * 3 + 2] * x[2]);
    const s = solveSPD(A, 3, b);
    s.forEach((v, i) => expect(v).toBeCloseTo(x[i], 12));
  });
});
