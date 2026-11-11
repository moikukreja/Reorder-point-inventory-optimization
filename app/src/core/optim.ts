/**
 * Nelder–Mead simplex minimiser. It is a faithful port of SciPy's `_minimize_neldermead`
 * (non-adaptive coefficients, the same initial simplex, the same convergence test), so
 * Holt-Winters parameters fitted here match the Python reference notebook.
 */

export interface NelderMeadOptions {
  maxiter?: number;
  xatol?: number;
  fatol?: number;
}

export interface NelderMeadResult {
  x: number[];
  fun: number;
  iterations: number;
  evaluations: number;
}

export function nelderMead(f: (x: number[]) => number, x0: number[], opts: NelderMeadOptions = {}): NelderMeadResult {
  const { maxiter = 200 * x0.length, xatol = 1e-4, fatol = 1e-4 } = opts;
  const N = x0.length;
  const rho = 1, chi = 2, psi = 0.5, sigma = 0.5;
  const nonzdelt = 0.05, zdelt = 0.00025;

  let sim: number[][] = [x0.slice()];
  for (let k = 0; k < N; k++) {
    const y = x0.slice();
    y[k] = y[k] !== 0 ? (1 + nonzdelt) * y[k] : zdelt;
    sim.push(y);
  }
  let evaluations = 0;
  const call = (x: number[]) => {
    evaluations++;
    return f(x);
  };
  let fsim = sim.map(call);
  const sortSimplex = () => {
    const ind = fsim.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]).map((p) => p[1]);
    sim = ind.map((i) => sim[i]);
    fsim = ind.map((i) => fsim[i]);
  };
  sortSimplex();

  let iterations = 1;
  while (iterations < maxiter) {
    let maxX = 0, maxF = 0;
    for (let j = 1; j <= N; j++) {
      maxF = Math.max(maxF, Math.abs(fsim[0] - fsim[j]));
      for (let i = 0; i < N; i++) maxX = Math.max(maxX, Math.abs(sim[j][i] - sim[0][i]));
    }
    if (maxX <= xatol && maxF <= fatol) break;

    const xbar = new Array<number>(N).fill(0);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) xbar[i] += sim[j][i] / N;
    const worst = sim[N];
    const comb = (a: number, b: number) => xbar.map((xb, i) => a * xb + b * worst[i]);

    const xr = comb(1 + rho, -rho);
    const fxr = call(xr);
    let doShrink = false;
    if (fxr < fsim[0]) {
      const xe = comb(1 + rho * chi, -rho * chi);
      const fxe = call(xe);
      if (fxe < fxr) { sim[N] = xe; fsim[N] = fxe; }
      else { sim[N] = xr; fsim[N] = fxr; }
    } else if (fxr < fsim[N - 1]) {
      sim[N] = xr; fsim[N] = fxr;
    } else if (fxr < fsim[N]) {
      const xc = comb(1 + psi * rho, -psi * rho);
      const fxc = call(xc);
      if (fxc <= fxr) { sim[N] = xc; fsim[N] = fxc; }
      else doShrink = true;
    } else {
      const xcc = comb(1 - psi, psi);
      const fxcc = call(xcc);
      if (fxcc < fsim[N]) { sim[N] = xcc; fsim[N] = fxcc; }
      else doShrink = true;
    }
    if (doShrink) {
      for (let j = 1; j <= N; j++) {
        sim[j] = sim[j].map((v, i) => sim[0][i] + sigma * (v - sim[0][i]));
        fsim[j] = call(sim[j]);
      }
    }
    sortSimplex();
    iterations++;
  }
  return { x: sim[0], fun: fsim[0], iterations, evaluations };
}
