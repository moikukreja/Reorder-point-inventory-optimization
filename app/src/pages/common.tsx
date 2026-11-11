import { Loader2 } from 'lucide-react';
import type { ReactNode } from 'react';
import { useApp } from '../state/store';
import type { ForecastBundle, RiskBundle } from '../engine/types';

export function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: ReactNode; children?: ReactNode }) {
  return (
    <header className="mb-5 max-w-4xl">
      <div className="text-[11.5px] font-semibold uppercase tracking-[0.12em] text-ai">{eyebrow}</div>
      <h1 className="mt-1 text-[22px] font-semibold leading-tight text-ink sm:text-[26px]">{title}</h1>
      {children && <p className="mt-2 text-[14px] leading-relaxed text-ink-2">{children}</p>}
    </header>
  );
}

/** Renders children only once the forecast and risk bundles are available. */
export function WithData({ children }: { children: (fb: ForecastBundle, rk: RiskBundle) => ReactNode }) {
  const fb = useApp((s) => s.forecast);
  const rk = useApp((s) => s.risk);
  const progress = useApp((s) => s.progress);
  if (!fb || !rk)
    return (
      <div className="grid h-72 place-items-center text-[13px] text-ink-3">
        <span className="flex items-center gap-2"><Loader2 className="h-4 w-4 animate-spin" />{progress?.msg ?? 'Computing'}…</span>
      </div>
    );
  return <>{children(fb, rk)}</>;
}

export const seriesTitle = (fb: ForecastBundle) => `Store ${fb.info.store} · ${fb.info.family}`;
