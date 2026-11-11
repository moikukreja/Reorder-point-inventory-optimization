/**
 * Small dense linear algebra used by Ridge regression and the Gaussian process.
 * Matrices are row-major Float64Array with an explicit dimension.
 */

/** In-place Cholesky factorisation of a symmetric positive-definite n×n matrix A = L·Lᵀ. Returns L (lower). */
export function cholesky(A: Float64Array, n: number): Float64Array {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (s <= 0) throw new Error(`cholesky: matrix not positive definite (pivot ${s} at ${i})`);
        L[i * n + i] = Math.sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  return L;
}

/** Solves L·x = b (forward substitution). */
export function forwardSub(L: Float64Array, n: number, b: ArrayLike<number>): Float64Array {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

/** Solves Lᵀ·x = b (back substitution). */
export function backSub(L: Float64Array, n: number, b: ArrayLike<number>): Float64Array {
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

/** Solves A·x = b for SPD A via Cholesky. */
export function solveSPD(A: Float64Array, n: number, b: ArrayLike<number>): Float64Array {
  const L = cholesky(A, n);
  return backSub(L, n, forwardSub(L, n, b));
}
