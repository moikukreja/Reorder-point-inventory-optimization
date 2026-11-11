/**
 * Weekly seasonal naive: ŷ(T+h) = y(T + h − 7·⌈h/7⌉). This is the benchmark every AI model
 * must beat, and the scale used by MASE.
 */
import { impute } from '../features';

export function seasonalNaiveForecast(y: ArrayLike<number>, T: number, H: number): Float64Array {
  const z = impute(y, T);
  const out = new Float64Array(H);
  for (let h = 1; h <= H; h++) out[h - 1] = z[T + h - 7 * Math.ceil(h / 7)];
  return out;
}
