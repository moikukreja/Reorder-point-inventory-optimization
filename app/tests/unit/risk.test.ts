/**
 * Data-free tests for safety stock, Monte Carlo and the policy simulation.
 * They include the hand calculation from the implementation plan (§4.4).
 */
import { describe, expect, it } from 'vitest';
import { baselineSafetyStock, forecastMeanOverLeadTime, historicalDemandStats, slideSafetyStock } from '../../src/core/safetyStock';
import { expectedShortage, histogram, monteCarloRop, simulateDdlt, stockoutProbability } from '../../src/core/monteCarlo';
import { eoq, simulatePolicy, type PolicySimConfig } from '../../src/core/policySim';
import { DEFAULT_COSTS } from '../../src/core/assumptions';
import { Rng } from '../../src/core/stats';

const HAND = { serviceLevel: 0.95, dBar: 100, sdD: 20, L: 5, sdL: 1.5 };
const Z95 = 1.6448536269514722; // scipy.stats.norm.ppf(0.95)

describe('slide safety-stock formula: hand calculation', () => {
  it('SS = 1.6449 × √(20²·5 + 100²·1.5²) = 257.46, ROP = 757.46', () => {
    const r = slideSafetyStock(HAND);
    expect(r.z).toBeCloseTo(Z95, 10);
    expect(r.varDemand).toBe(2000);
    expect(r.varLeadTime).toBe(22500);
    expect(r.sdDdlt).toBeCloseTo(Math.sqrt(24500), 12);
    expect(r.safetyStock).toBeCloseTo(257.46, 2);
    expect(r.cycleStock).toBe(500);
    expect(r.reorderPoint).toBeCloseTo(757.46, 2);
  });
  it('the baseline (class notebook) ignores σL: SS = Z·σ·√L = 73.56', () => {
    const r = baselineSafetyStock(HAND);
    expect(r.safetyStock).toBeCloseTo(Z95 * 20 * Math.sqrt(5), 6);
    expect(r.safetyStock).toBeCloseTo(73.56, 2);
  });
  it('reduces to Z·σ_D·√L when σL = 0 and to Z·D̄·σL when σ_D = 0', () => {
    expect(slideSafetyStock({ ...HAND, sdL: 0 }).safetyStock).toBeCloseTo(Z95 * 20 * Math.sqrt(5), 5);
    expect(slideSafetyStock({ ...HAND, sdD: 0 }).safetyStock).toBeCloseTo(Z95 * 100 * 1.5, 4);
  });
  it('rejects invalid inputs', () => {
    expect(() => slideSafetyStock({ ...HAND, serviceLevel: 1 })).toThrow();
    expect(() => slideSafetyStock({ ...HAND, L: 0 })).toThrow();
  });
  it('forecast mean over a fractional lead time weights the last day', () => {
    expect(forecastMeanOverLeadTime([10, 20, 30, 40], 2.5)).toBeCloseTo((10 + 20 + 15) / 2.5, 12);
    expect(forecastMeanOverLeadTime([10, 20, 30, 40], 4)).toBe(25);
  });
  it('historical stats skip closed days', () => {
    const r = historicalDemandStats([1, NaN, 3, 5, NaN, 7], 5, 3);
    expect(r.n).toBe(3);
    expect(r.dBar).toBe(5);
  });
});

describe('Monte Carlo demand during lead time', () => {
  it('fixed lead time + normal demand reproduces the textbook 5% stockout risk', () => {
    const r = simulateDdlt({ mode: 'normal', n: 400_000, seed: 1, leadTime: { mean: 5, sd: 0, dist: 'gamma' }, dBar: 100, sdD: 20 });
    expect(r.mean).toBeCloseTo(500, 0);
    expect(r.sd / (20 * Math.sqrt(5))).toBeCloseTo(1, 2);
    const rop = slideSafetyStock({ ...HAND, sdL: 0 }).reorderPoint;
    expect(Math.abs(stockoutProbability(r.samples, rop) - 0.05)).toBeLessThan(0.002);
  });
  it('gamma lead time: mean D̄L̄ and SD √(σ_D²L + D̄²σL²) match theory; the tail is fatter than normal', () => {
    const r = simulateDdlt({ mode: 'normal', n: 400_000, seed: 2, leadTime: { mean: 5, sd: 1.5, dist: 'gamma' }, dBar: 100, sdD: 20 });
    expect(Math.abs(r.mean / 500 - 1)).toBeLessThan(0.003);
    expect(Math.abs(r.sd / Math.sqrt(24500) - 1)).toBeLessThan(0.01);
    // Reference notebook (independent NumPy RNG): P(stockout at formula ROP) = 0.0628, MC 95% ROP = 780.3
    const p = stockoutProbability(r.samples, 757.4603);
    expect(Math.abs(p - 0.0628)).toBeLessThan(0.003);
    expect(Math.abs(monteCarloRop(r.samples, 0.95) - 780.3)).toBeLessThan(3);
  });
  it('bootstrap mode with a flat forecast and symmetric errors has mean D̄·L̄', () => {
    const rng = new Rng(5);
    const residuals = Float64Array.from({ length: 5000 }, () => 20 * rng.normal());
    const r = simulateDdlt({ mode: 'bootstrap', n: 200_000, seed: 3, leadTime: { mean: 4, sd: 1.2, dist: 'gamma' }, forecast: new Array(28).fill(100), residuals });
    expect(Math.abs(r.mean / 400 - 1)).toBeLessThan(0.005);
    expect(r.horizonOverflow).toBe(0);
  });
  it('block bootstrap keeps persistent forecast misses (wider DDLT than iid errors)', () => {
    // Two error paths: always +20 or always −20. iid draws partly cancel; consecutive draws do not.
    const paths = [new Array(14).fill(20), new Array(14).fill(-20)];
    const pool = [...paths[0], ...paths[1]];
    const cfg = { mode: 'bootstrap' as const, n: 100_000, seed: 8, leadTime: { mean: 4, sd: 0, dist: 'gamma' as const }, forecast: new Array(14).fill(100), residuals: pool };
    const iid = simulateDdlt(cfg);
    const blk = simulateDdlt({ ...cfg, errorPaths: paths });
    expect(iid.sd).toBeCloseTo(40, 0); // √4 × 20
    expect(blk.sd).toBeGreaterThan(70); // ≈ 4 × 20 = 80 when a block spans the whole lead time
  });
  it('stockout probability, expected shortage and histogram are exact on a small sample', () => {
    const s = Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(stockoutProbability(s, 7)).toBe(0.3);
    expect(stockoutProbability(s, 7.5)).toBe(0.3);
    expect(expectedShortage(s, 7)).toBeCloseTo((1 + 2 + 3) / 10, 12);
    const h = histogram(s, 5, 0, 1);
    expect(h.counts.reduce((a, b) => a + b, 0)).toBe(10);
  });
});

describe('(s, Q) inventory policy simulation', () => {
  const base: PolicySimConfig = {
    horizon: 120, replications: 1, seed: 1, warmup: 0, trace: true,
    leadTime: { mean: 5, sd: 0, dist: 'gamma' }, costs: DEFAULT_COSTS,
    demand: { kind: 'actual', values: new Array(120).fill(100) },
  };
  it('conserves inventory: on-hand(t) = on-hand(t−1) + received − served', () => {
    const r = simulatePolicy({ s: 600, Q: 800 }, { ...base, demand: { kind: 'generated', mean: [100, 80, 120, 90, 150, 200, 60], residuals: [-30, 0, 30, 60] }, initialOnHand: 900 });
    let prev = 900;
    for (const d of r.trace!) {
      expect(d.onHand).toBeCloseTo(prev + d.received - d.served, 9);
      expect(d.served + d.lost).toBeCloseTo(d.demand, 9);
      prev = d.onHand;
    }
  });
  it('deterministic demand and lead time: s = D·L never stocks out, s < D·L stocks out every cycle', () => {
    const ok = simulatePolicy({ s: 500, Q: 500 }, { ...base, initialOnHand: 1000 });
    expect(ok.mean.fillRate).toBe(1);
    expect(ok.mean.cycleServiceLevel).toBe(1);
    expect(ok.mean.ordersPer30).toBeCloseTo(6, 0); // 100/day ÷ 500 per order × 30 days
    const short = simulatePolicy({ s: 400, Q: 500 }, { ...base, initialOnHand: 1000, warmup: 20 });
    expect(short.mean.cycleServiceLevel).toBe(0);
    expect(short.mean.fillRate).toBeLessThan(1);
  });
  it('uses common random numbers: the same seed gives identical results', () => {
    const cfg: PolicySimConfig = { ...base, replications: 5, trace: false, leadTime: { mean: 4, sd: 1.2, dist: 'gamma' }, demand: { kind: 'generated', mean: [100], residuals: [-40, -10, 0, 15, 35] } };
    expect(simulatePolicy({ s: 450, Q: 700 }, cfg).mean).toEqual(simulatePolicy({ s: 450, Q: 700 }, cfg).mean);
  });
  it('cycle service level agrees with the Monte Carlo stockout risk (bracketed by the daily-review undershoot)', () => {
    const rng = new Rng(11);
    const residuals = Float64Array.from({ length: 5000 }, () => 20 * rng.normal());
    const lt = { mean: 5, sd: 1.5, dist: 'gamma' as const };
    const s = 757;
    const sim = simulatePolicy({ s, Q: 1500 }, { horizon: 730, replications: 120, seed: 4, warmup: 30, leadTime: lt, costs: DEFAULT_COSTS, demand: { kind: 'generated', mean: [100], residuals } });
    const mc = simulateDdlt({ mode: 'bootstrap', n: 300_000, seed: 9, leadTime: lt, forecast: new Array(40).fill(100), residuals });
    const upper = 1 - stockoutProbability(mc.samples, s);
    const lower = 1 - stockoutProbability(mc.samples, s - 100); // inventory can fall up to one day's demand below s before the order
    expect(sim.mean.cycleServiceLevel).toBeLessThan(upper + 0.01);
    expect(sim.mean.cycleServiceLevel).toBeGreaterThan(lower - 0.01);
  });
  it('EOQ = √(2DK/h)', () => {
    expect(eoq(100, 50, 0.01)).toBeCloseTo(Math.sqrt(1_000_000), 9);
  });
});
