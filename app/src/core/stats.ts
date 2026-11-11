/**
 * Statistical primitives: normal quantile/CDF, descriptive stats, NumPy-compatible quantiles
 * and a seedable random number generator with normal / gamma / lognormal samplers.
 * Everything here is pure and deterministic given a seed, so results can be reproduced.
 */

/** Inverse standard-normal CDF (Acklam's rational approximation + one Halley refinement step, |err| < 1e-12). */
export function invNormCdf(p: number): number {
  if (!(p > 0 && p < 1)) {
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    throw new RangeError(`invNormCdf: p must be in (0,1), got ${p}`);
  }
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  let x: number;
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= 1 - pLow) {
    const q = p - 0.5;
    const r = q * q;
    x = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  // Halley refinement using the exact CDF
  const e = normCdf(x) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp((x * x) / 2);
  return x - u / (1 + (x * u) / 2);
}

/**
 * Complementary error function (Numerical Recipes "erfcc", fractional error < 1.2e-7).
 * Only used by normCdf for |x| ≥ 7, where Φ is within 1e-11 of 0 or 1.
 */
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? r : 2 - r;
}

/** Standard-normal CDF Φ(x). High precision via a series/continued-fraction split. */
export function normCdf(x: number): number {
  // Series expansion is exact to double precision for |x| < 7 when summed far enough.
  if (Math.abs(x) < 7) {
    let sum = x;
    let term = x;
    const x2 = x * x;
    for (let n = 1; n < 500; n++) {
      term *= x2 / (2 * n + 1);
      sum += term;
      if (Math.abs(term) < 1e-17 * Math.abs(sum)) break;
    }
    return 0.5 + (sum * Math.exp(-x2 / 2)) / Math.sqrt(2 * Math.PI);
  }
  return 0.5 * erfc(-x / Math.SQRT2);
}

/** Standard-normal density φ(x). */
export function normPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

export function sum(xs: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s;
}

export function mean(xs: ArrayLike<number>): number {
  return xs.length ? sum(xs) / xs.length : NaN;
}

/** Standard deviation. ddof = 0 matches np.std; ddof = 1 is the sample standard deviation. */
export function std(xs: ArrayLike<number>, ddof = 0): number {
  const n = xs.length;
  if (n - ddof <= 0) return NaN;
  const m = mean(xs);
  let s = 0;
  for (let i = 0; i < n; i++) s += (xs[i] - m) ** 2;
  return Math.sqrt(s / (n - ddof));
}

/** Quantile with NumPy's default 'linear' method (Hyndman–Fan type 7). */
export function quantile(xs: ArrayLike<number>, p: number, sorted = false): number {
  const a = sorted ? xs : Float64Array.from(xs).sort();
  const n = a.length;
  if (!n) return NaN;
  const h = (n - 1) * p;
  const lo = Math.floor(h);
  const hi = Math.min(lo + 1, n - 1);
  return a[lo] + (h - lo) * (a[hi] - a[lo]);
}

/** Drops NaN / null values. */
export function finite(xs: ArrayLike<number | null>): number[] {
  const out: number[] = [];
  for (let i = 0; i < xs.length; i++) {
    const v = xs[i];
    if (v !== null && Number.isFinite(v)) out.push(v);
  }
  return out;
}

/**
 * Seedable PRNG (xoshiro128**), uniform in [0, 1). Fast, good statistical quality,
 * and reproducible across browsers.
 */
export class Rng {
  private s = new Uint32Array(4);
  private spare: number | null = null;

  constructor(seed = 42) {
    // SplitMix32 expands the seed into the 128-bit state
    let x = seed >>> 0;
    for (let i = 0; i < 4; i++) {
      x = (x + 0x9e3779b9) >>> 0;
      let z = x;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
      this.s[i] = (z ^ (z >>> 16)) >>> 0;
    }
  }

  /** Uniform [0, 1). */
  next(): number {
    const s = this.s;
    const result = Math.imul(rotl(Math.imul(s[1], 5) >>> 0, 7), 9) >>> 0;
    const t = (s[1] << 9) >>> 0;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 11);
    return result / 4294967296;
  }

  /** Integer in [0, n). */
  int(n: number): number {
    return Math.floor(this.next() * n);
  }

  /** Standard normal (polar Box–Muller, caches the second variate). */
  normal(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    let u: number, v: number, s: number;
    do {
      u = 2 * this.next() - 1;
      v = 2 * this.next() - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const m = Math.sqrt((-2 * Math.log(s)) / s);
    this.spare = v * m;
    return u * m;
  }

  /** Gamma(shape k, scale θ), using Marsaglia–Tsang (with the boost for k < 1). */
  gamma(k: number, theta: number): number {
    if (k < 1) {
      const u = this.next();
      return this.gamma(1 + k, theta) * Math.pow(u, 1 / k);
    }
    const d = k - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x: number, v: number;
      do {
        x = this.normal();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = this.next();
      if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * theta;
    }
  }
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

/** Lead-time distribution families supported by the dashboard. */
export type LeadTimeDist = 'gamma' | 'lognormal' | 'normal';

/**
 * Draws one lead time (days) with the requested mean and standard deviation.
 * - gamma: shape = (mean/sd)², scale = sd²/mean (right-skewed, positive)
 * - lognormal: parameters chosen to match the mean and sd exactly
 * - normal: truncated at 0 (redrawn), for comparison with the textbook assumption
 */
export function sampleLeadTime(rng: Rng, meanL: number, sdL: number, dist: LeadTimeDist): number {
  if (sdL <= 0) return meanL;
  switch (dist) {
    case 'gamma':
      return rng.gamma((meanL / sdL) ** 2, (sdL * sdL) / meanL);
    case 'lognormal': {
      const s2 = Math.log(1 + (sdL / meanL) ** 2);
      return Math.exp(Math.log(meanL) - s2 / 2 + Math.sqrt(s2) * rng.normal());
    }
    case 'normal': {
      let x: number;
      do x = meanL + sdL * rng.normal();
      while (x <= 0);
      return x;
    }
  }
}
