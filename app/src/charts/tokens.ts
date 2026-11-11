/**
 * Chart colour tokens. Colour follows the entity everywhere in the app:
 * AI is always blue, the baseline is always orange, Holt-Winters is always aqua, and so on.
 *
 * The categorical slots [AI, baseline, Holt-Winters, Ridge, LSTM] pass every check of the dataviz
 * validator in dark mode on #101826 (worst adjacent CVD ΔE 8.4, normal-vision ΔE 19.3, all ≥ 3:1).
 * The optimiser chart pairs AI blue with random-search magenta (all-pairs ΔE 15.9).
 * Status colours (ROP amber, stockout red) are reserved and always appear with a label or icon.
 */
export const C = {
  surface: '#101826',
  ink: '#e8edf5',
  ink2: '#aab6c7',
  ink3: '#788599',
  grid: 'rgba(148,163,184,0.10)',
  axis: 'rgba(148,163,184,0.28)',

  actual: '#e8edf5',
  ai: '#3987e5',
  baseline: '#d95926',
  holtwinters: '#199e70',
  ridge: '#c98500',
  lstm: '#d55181',
  snaive: '#8792a3',
  random: '#d55181',
  grid2: '#8792a3',

  rop: '#fab219',
  stockout: '#d03b3b',
  good: '#0ca30c',
} as const;

export const MODEL_COLORS: Record<string, string> = {
  snaive: C.snaive,
  holtwinters: C.holtwinters,
  ridge: C.ridge,
  lstm: C.lstm,
  ensemble: C.ai,
};

/** Sequential blue ramp (light → dark on a dark surface means low → high: dark is low here). */
export const SEQ_BLUE = ['#104281', '#184f95', '#1c5cab', '#256abf', '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef', '#9ec5f4'];

/** Diverging blue ↔ red with a neutral midpoint (dark mode). */
export const DIVERGING = ['#3987e5', '#5598e7', '#383835', '#e66767', '#d03b3b'];

export const withAlpha = (hex: string, a: number) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
};
