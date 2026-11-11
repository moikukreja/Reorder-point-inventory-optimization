/**
 * Base UI building blocks: glass panels, KPI tiles, plain-language "what this means" boxes,
 * controls and tables. Motion is purposeful: panels fade up on mount, and numbers ease to
 * their new value only when they change. Everything respects prefers-reduced-motion.
 */
import { animate, motion, useReducedMotion } from 'framer-motion';
import { Info, Lightbulb, Table2, LineChart as ChartIcon, Loader2 } from 'lucide-react';
import { useEffect, useId, useRef, useState, type ReactNode } from 'react';

export const cx = (...xs: (string | false | null | undefined)[]) => xs.filter(Boolean).join(' ');

export function Panel({ title, subtitle, actions, children, className, delay = 0 }: { title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; delay?: number }) {
  const reduce = useReducedMotion();
  return (
    <motion.section
      initial={reduce ? false : { opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay, ease: 'easeOut' }}
      className={cx('glass p-4 sm:p-5 min-w-0', className)}
    >
      {(title || actions) && (
        <header className="mb-3 flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            {title && <h2 className="text-[15px] font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-[13px] leading-snug text-ink-3">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </motion.section>
  );
}

/** Eases a number to its new value (no animation when reduced motion is requested). */
export function AnimatedNumber({ value, format }: { value: number; format: (v: number) => string }) {
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(value);
  const prev = useRef(value);
  useEffect(() => {
    if (reduce || !Number.isFinite(prev.current) || !Number.isFinite(value)) {
      setShown(value);
      prev.current = value;
      return;
    }
    const c = animate(prev.current, value, { duration: 0.6, ease: 'easeOut', onUpdate: setShown });
    prev.current = value;
    return () => c.stop();
  }, [value, reduce]);
  return <>{format(shown)}</>;
}

export function KpiTile({ label, value, format, sub, accent, icon }: { label: string; value: number; format: (v: number) => string; sub?: ReactNode; accent?: 'ai' | 'base' | 'rop' | 'bad' | 'good'; icon?: ReactNode }) {
  const bar = { ai: 'bg-ai', base: 'bg-base', rop: 'bg-rop', bad: 'bg-bad', good: 'bg-good' }[accent ?? 'ai'];
  return (
    <div className="glass relative overflow-hidden p-4">
      <span className={cx('absolute inset-y-0 left-0 w-[3px]', bar)} aria-hidden />
      <div className="flex items-center gap-2 text-[12px] font-medium uppercase tracking-wide text-ink-3">
        {icon}
        {label}
      </div>
      <div className="num mt-1.5 text-[26px] font-semibold leading-none text-ink">
        <AnimatedNumber value={value} format={format} />
      </div>
      {sub && <div className="mt-2 text-[12.5px] leading-snug text-ink-2">{sub}</div>}
    </div>
  );
}

/** Plain-language interpretation for an operations manager. */
export function Explain({ children, title = 'What this means' }: { children: ReactNode; title?: string }) {
  return (
    <div className="mt-3 flex gap-3 rounded-xl border border-ai/20 bg-ai/[0.06] p-3 text-[13px] leading-relaxed text-ink-2">
      <Lightbulb className="mt-0.5 h-4 w-4 shrink-0 text-ai" aria-hidden />
      <div>
        <span className="font-semibold text-ink">{title}: </span>
        {children}
      </div>
    </div>
  );
}

export function Note({ children, tone = 'info' }: { children: ReactNode; tone?: 'info' | 'warn' }) {
  return (
    <div className={cx('flex gap-2.5 rounded-xl border p-3 text-[12.5px] leading-relaxed', tone === 'warn' ? 'border-warn/30 bg-warn/[0.07] text-ink-2' : 'border-white/10 bg-white/[0.03] text-ink-2')}>
      <Info className={cx('mt-0.5 h-4 w-4 shrink-0', tone === 'warn' ? 'text-warn' : 'text-ink-3')} aria-hidden />
      <div>{children}</div>
    </div>
  );
}

export function Button({ children, onClick, variant = 'primary', disabled, busy, title, type = 'button' }: { children: ReactNode; onClick?: () => void; variant?: 'primary' | 'ghost' | 'danger'; disabled?: boolean; busy?: boolean; title?: string; type?: 'button' | 'submit' }) {
  const styles = {
    primary: 'bg-ai text-white hover:bg-ai/90 shadow-[0_6px_20px_-8px_rgb(57_135_229/0.8)]',
    ghost: 'border border-white/10 bg-white/[0.04] text-ink hover:bg-white/[0.08]',
    danger: 'border border-bad/40 bg-bad/10 text-ink hover:bg-bad/20',
  }[variant];
  return (
    <button type={type} title={title} onClick={onClick} disabled={disabled || busy} className={cx('inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-[13px] font-medium transition disabled:cursor-not-allowed disabled:opacity-50', styles)}>
      {busy && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
      {children}
    </button>
  );
}

export function Segmented<T extends string>({ value, options, onChange, label }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border border-white/10 bg-white/[0.03] p-0.5">
      {options.map((o) => (
        <button key={o.value} role="radio" aria-checked={value === o.value} onClick={() => onChange(o.value)} className={cx('rounded-md px-2.5 py-1 text-[12.5px] transition', value === o.value ? 'bg-white/10 text-ink' : 'text-ink-3 hover:text-ink-2')}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Slider({ label, value, min, max, step, onChange, format, hint }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; format: (v: number) => string; hint?: string }) {
  const id = useId();
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-[12.5px] text-ink-2">{label}</label>
        <span className="num text-[12.5px] text-ink">{format(value)}</span>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} aria-valuetext={format(value)} />
      {hint && <p className="text-[11.5px] leading-snug text-ink-3">{hint}</p>}
    </div>
  );
}

export function NumberField({ label, value, onChange, step = 1, min, suffix, prefix }: { label: string; value: number; onChange: (v: number) => void; step?: number; min?: number; suffix?: string; prefix?: string }) {
  const id = useId();
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="text-[12.5px] text-ink-2">{label}</label>
      <div className="flex items-center rounded-lg border border-white/10 bg-white/[0.03] px-2.5 focus-within:border-ai/60">
        {prefix && <span className="text-[12.5px] text-ink-3">{prefix}</span>}
        <input id={id} type="number" step={step} min={min} value={Number.isFinite(value) ? value : ''} onChange={(e) => e.target.value !== '' && onChange(Number(e.target.value))} className="num w-full bg-transparent px-1.5 py-1.5 text-[13px] text-ink outline-none" />
        {suffix && <span className="text-[12px] text-ink-3">{suffix}</span>}
      </div>
    </div>
  );
}

export function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 text-[12.5px] text-ink-2">
      {label}
      <button role="switch" aria-checked={checked} onClick={() => onChange(!checked)} className={cx('relative h-5 w-9 rounded-full transition', checked ? 'bg-ai' : 'bg-white/15')}>
        <span className={cx('absolute top-0.5 h-4 w-4 rounded-full bg-white transition', checked ? 'left-[18px]' : 'left-0.5')} />
      </button>
    </label>
  );
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'ai' | 'base' | 'good' | 'bad' | 'rop' }) {
  const t = { neutral: 'bg-white/[0.06] text-ink-2 border-white/10', ai: 'bg-ai/15 text-ink border-ai/30', base: 'bg-base/15 text-ink border-base/30', good: 'bg-good/15 text-ink border-good/30', bad: 'bg-bad/15 text-ink border-bad/30', rop: 'bg-rop/15 text-ink border-rop/30' }[tone];
  return <span className={cx('inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11.5px] font-medium', t)}>{children}</span>;
}

export function Progress({ value, label }: { value: number; label: string }) {
  return (
    <div className="space-y-1" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value * 100)} aria-label={label}>
      <div className="flex justify-between text-[12px] text-ink-3"><span>{label}</span><span className="num">{Math.round(value * 100)}%</span></div>
      <div className="h-1.5 overflow-hidden rounded-full bg-white/10"><div className="h-full rounded-full bg-ai transition-[width] duration-300" style={{ width: `${value * 100}%` }} /></div>
    </div>
  );
}

/** A simple accessible data table (also the "table view" alternative of each chart). */
export function DataTable({ columns, rows, caption, maxHeight = 320, highlightRow }: { columns: { key: string; label: string; align?: 'left' | 'right'; num?: boolean }[]; rows: Record<string, ReactNode>[]; caption: string; maxHeight?: number; highlightRow?: number }) {
  return (
    <div className="overflow-auto rounded-xl border border-white/[0.07]" style={{ maxHeight }}>
      <table className="w-full border-collapse text-[12.5px]">
        <caption className="sr-only">{caption}</caption>
        <thead className="sticky top-0 bg-surface/95 backdrop-blur">
          <tr>
            {columns.map((c) => (
              <th key={c.key} scope="col" className={cx('whitespace-nowrap border-b border-white/10 px-3 py-2 font-medium text-ink-3', c.align === 'right' || c.num ? 'text-right' : 'text-left')}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={cx('border-b border-white/[0.04] last:border-0', i === highlightRow && 'bg-ai/[0.08]')}>
              {columns.map((c) => (
                <td key={c.key} className={cx('whitespace-nowrap px-3 py-1.5 text-ink-2', (c.align === 'right' || c.num) && 'text-right', c.num && 'num text-ink')}>{r[c.key]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Chart container with a chart/table toggle, so every visual has a non-colour, screen-reader-friendly alternative. */
export function ChartCard({ title, subtitle, chart, table, actions, children, delay, className }: { title: ReactNode; subtitle?: ReactNode; chart: ReactNode; table?: ReactNode; actions?: ReactNode; children?: ReactNode; delay?: number; className?: string }) {
  const [view, setView] = useState<'chart' | 'table'>('chart');
  return (
    <Panel
      title={title}
      subtitle={subtitle}
      delay={delay}
      className={className}
      actions={
        <>
          {actions}
          {table && (
            <button onClick={() => setView(view === 'chart' ? 'table' : 'chart')} className="inline-flex items-center gap-1.5 rounded-md border border-white/10 px-2 py-1 text-[12px] text-ink-3 hover:text-ink" aria-label={view === 'chart' ? 'Show data table' : 'Show chart'}>
              {view === 'chart' ? <Table2 className="h-3.5 w-3.5" /> : <ChartIcon className="h-3.5 w-3.5" />}
              {view === 'chart' ? 'Table' : 'Chart'}
            </button>
          )}
        </>
      }
    >
      {view === 'chart' ? chart : table}
      {children}
    </Panel>
  );
}

export function Legend({ items }: { items: { color: string; label: string; dashed?: boolean; area?: boolean }[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1.5 text-[12px] text-ink-2" aria-label="Legend">
      {items.map((it) => (
        <li key={it.label} className="flex items-center gap-1.5">
          <span aria-hidden className="inline-block" style={it.area ? { width: 12, height: 10, borderRadius: 3, background: it.color } : { width: 14, height: 0, borderTop: `2px ${it.dashed ? 'dashed' : 'solid'} ${it.color}` }} />
          {it.label}
        </li>
      ))}
    </ul>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="grid place-items-center rounded-xl border border-dashed border-white/10 p-8 text-center text-[13px] text-ink-3">{children}</div>;
}
