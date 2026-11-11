import type { Config } from 'tailwindcss';

// Colours come from CSS custom properties in src/theme.css, so charts and UI share one source.
const v = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: v('bg'),
        surface: v('surface'),
        ink: { DEFAULT: v('ink'), 2: v('ink-2'), 3: v('ink-3') },
        line: v('line'),
        ai: v('ai'),
        base: v('baseline'),
        rop: v('rop'),
        bad: v('bad'),
        good: v('good'),
        warn: v('warn'),
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: { glass: '0 1px 0 0 rgb(255 255 255 / 0.05) inset, 0 10px 30px -12px rgb(0 0 0 / 0.6)' },
    },
  },
  plugins: [],
} satisfies Config;
