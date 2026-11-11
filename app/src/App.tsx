/**
 * App shell: sidebar navigation, top bar (series selector and progress), the page area, and the
 * controls panel. The panel is a sticky column on wide screens and a drawer on smaller ones.
 * The page is mirrored in the URL hash, so views can be linked and tested directly.
 */
import { AnimatePresence, motion } from 'framer-motion';
import { Loader2, Menu, SlidersHorizontal, X } from 'lucide-react';
import { lazy, Suspense, useEffect, useState, type LazyExoticComponent, type ReactNode } from 'react';
import { pageFromHash, useApp, type PageId } from './state/store';
import { NAV } from './ui/nav';
import { Controls } from './ui/Controls';
import { LoadData } from './ui/LoadData';
import { Progress, cx } from './ui/primitives';
import { ControlTower } from './pages/ControlTower';

const pages: Record<PageId, LazyExoticComponent<() => JSX.Element> | (() => JSX.Element)> = {
  tower: ControlTower,
  data: lazy(() => import('./pages/DataFeed').then((m) => ({ default: m.DataFeed }))),
  forecast: lazy(() => import('./pages/Forecast').then((m) => ({ default: m.Forecast }))),
  risk: lazy(() => import('./pages/Risk').then((m) => ({ default: m.Risk }))),
  optimiser: lazy(() => import('./pages/Optimiser').then((m) => ({ default: m.Optimiser }))),
  sensitivity: lazy(() => import('./pages/Sensitivity').then((m) => ({ default: m.Sensitivity }))),
  compare: lazy(() => import('./pages/Compare').then((m) => ({ default: m.Compare }))),
  method: lazy(() => import('./pages/Method').then((m) => ({ default: m.Method }))),
};

function useHashSync() {
  const page = useApp((s) => s.page);
  const setPage = useApp((s) => s.setPage);
  useEffect(() => {
    // The initial page is read from the hash when the store is created; this only follows later changes
    const fromHash = () => {
      const h = pageFromHash();
      if (h && h !== useApp.getState().page) setPage(h);
    };
    window.addEventListener('hashchange', fromHash);
    return () => window.removeEventListener('hashchange', fromHash);
  }, [setPage]);
  useEffect(() => {
    if (window.location.hash !== `#/${page}`) history.replaceState(null, '', `#/${page}`);
  }, [page]);
}

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const page = useApp((s) => s.page);
  const setPage = useApp((s) => s.setPage);
  return (
    <nav aria-label="Main" className="flex h-full flex-col">
      <div className="flex items-center gap-2.5 px-4 pb-5 pt-5">
        <div className="tape h-2.5 w-8 rounded-full" aria-hidden />
        <div className="leading-tight">
          <div className="text-[14px] font-semibold text-ink">Control Tower</div>
          <div className="text-[11.5px] text-ink-3">AI reorder point</div>
        </div>
      </div>
      <ul className="flex-1 space-y-0.5 px-2">
        {NAV.map((n) => {
          const Icon = n.icon;
          const active = page === n.id;
          return (
            <li key={n.id}>
              <a
                href={`#/${n.id}`}
                onClick={(e) => { e.preventDefault(); setPage(n.id); onNavigate?.(); }}
                aria-current={active ? 'page' : undefined}
                className={cx('group relative flex items-center gap-3 rounded-lg px-3 py-2 text-[13.5px] transition', active ? 'bg-white/[0.07] text-ink' : 'text-ink-3 hover:bg-white/[0.04] hover:text-ink-2')}
              >
                {active && <motion.span layoutId="nav-active" className="absolute inset-y-1.5 left-0 w-[3px] rounded-full bg-ai" />}
                <Icon className="h-4 w-4" aria-hidden />
                <span className="flex-1">{n.label}</span>
                {n.step && <span className="num rounded bg-white/[0.06] px-1.5 text-[10.5px] text-ink-3" title={`Exercise step ${n.step}`}>S{n.step}</span>}
              </a>
            </li>
          );
        })}
      </ul>
      <p className="px-4 pb-4 pt-6 text-[11px] leading-snug text-ink-3">All computation runs locally in Web Workers. No data leaves this browser.</p>
    </nav>
  );
}

function TopBar({ onMenu, onControls }: { onMenu: () => void; onControls: () => void }) {
  const dataset = useApp((s) => s.dataset);
  const key = useApp((s) => s.key);
  const select = useApp((s) => s.selectKey);
  const progress = useApp((s) => s.progress);
  const riskBusy = useApp((s) => s.riskBusy);
  const forecast = useApp((s) => s.forecast);
  return (
    <header className="sticky top-0 z-30 border-b border-white/[0.06] bg-bg/75 backdrop-blur-xl">
      <div className="flex items-center gap-3 px-4 py-2.5 sm:px-6">
        <button className="rounded-lg p-2 text-ink-2 hover:bg-white/5 lg:hidden" onClick={onMenu} aria-label="Open navigation"><Menu className="h-5 w-5" /></button>
        <label className="flex min-w-0 items-center gap-2">
          <span className="hidden text-[12px] text-ink-3 sm:inline">Series</span>
          <select value={key ?? ''} onChange={(e) => void select(e.target.value)} className="min-w-0 rounded-lg border border-white/10 bg-surface px-2.5 py-1.5 text-[13px] text-ink outline-none focus:border-ai/60" aria-label="Store and product family">
            {dataset?.series.map((s) => (
              <option key={s.key} value={s.key}>Store {s.store} · {s.family} ({s.city}, type {s.storeType})</option>
            ))}
          </select>
        </label>
        <div className="hidden items-center gap-2 text-[12px] text-ink-3 md:flex">
          {forecast && <span>Data to <span className="num text-ink-2">{forecast.originDate}</span></span>}
          {(progress || riskBusy) && <span className="flex items-center gap-1.5 text-ink-2"><Loader2 className="h-3.5 w-3.5 animate-spin" />{progress ? progress.msg : 'Recalculating'}</span>}
        </div>
        <button onClick={onControls} className="ml-auto inline-flex items-center gap-2 rounded-lg border border-white/10 px-3 py-1.5 text-[13px] text-ink-2 hover:bg-white/5 2xl:hidden">
          <SlidersHorizontal className="h-4 w-4" /> Controls
        </button>
      </div>
      {progress && <div className="h-0.5 bg-white/5"><div className="h-full bg-ai transition-[width] duration-300" style={{ width: `${progress.frac * 100}%` }} /></div>}
    </header>
  );
}

function Drawer({ open, onClose, side, children, label }: { open: boolean; onClose: () => void; side: 'left' | 'right'; children: ReactNode; label: string }) {
  return (
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label={label}>
          <motion.div className="absolute inset-0 bg-black/60" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
          <motion.div
            className={cx('glass-strong absolute inset-y-0 w-[min(340px,88vw)] overflow-y-auto rounded-none', side === 'left' ? 'left-0' : 'right-0 p-5 pt-12')}
            initial={{ x: side === 'left' ? -40 : 40, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: side === 'left' ? -40 : 40, opacity: 0 }}
            transition={{ duration: 0.22 }}
          >
            <button onClick={onClose} className="absolute right-3 top-3 rounded-lg p-1.5 text-ink-3 hover:bg-white/5" aria-label="Close"><X className="h-4 w-4" /></button>
            {children}
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}

export function App() {
  const phase = useApp((s) => s.phase);
  const boot = useApp((s) => s.boot);
  const page = useApp((s) => s.page);
  const progress = useApp((s) => s.progress);
  const controlsOpen = useApp((s) => s.controlsOpen);
  const setControls = useApp((s) => s.setControlsOpen);
  const [navOpen, setNavOpen] = useState(false);
  useHashSync();
  useEffect(() => {
    void boot();
  }, [boot]);

  if (phase === 'boot') return <div className="grid h-full place-items-center text-ink-3"><Loader2 className="h-6 w-6 animate-spin" aria-label="Loading" /></div>;
  if (phase === 'needData' || phase === 'error') return <LoadData />;
  if (phase === 'loading')
    return (
      <div className="mx-auto flex h-full max-w-md flex-col justify-center gap-4 px-6">
        <div className="text-[15px] font-medium text-ink">Preparing models…</div>
        <Progress value={progress?.frac ?? 0} label={progress?.msg ?? 'Starting'} />
        <p className="text-[12.5px] text-ink-3">Backtesting four forecasting models on 18 rolling origins, in a background worker in your browser.</p>
      </div>
    );

  const Page = pages[page];
  return (
    <div className="flex min-h-full">
      <aside className="sticky top-0 hidden h-screen w-[232px] shrink-0 border-r border-white/[0.06] bg-bg/60 backdrop-blur-xl lg:block">
        <Sidebar />
      </aside>
      <Drawer open={navOpen} onClose={() => setNavOpen(false)} side="left" label="Navigation"><Sidebar onNavigate={() => setNavOpen(false)} /></Drawer>
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar onMenu={() => setNavOpen(true)} onControls={() => setControls(true)} />
        <div className="flex min-w-0 flex-1">
          <main className="min-w-0 flex-1 px-4 py-5 sm:px-6 sm:py-6">
            <Suspense fallback={<div className="grid h-64 place-items-center text-ink-3"><Loader2 className="h-5 w-5 animate-spin" /></div>}>
              <Page />
            </Suspense>
          </main>
          <aside className="sticky top-[49px] hidden h-[calc(100vh-49px)] w-[310px] shrink-0 overflow-y-auto border-l border-white/[0.06] bg-bg/40 p-5 backdrop-blur-xl 2xl:block" aria-label="Controls">
            <Controls />
          </aside>
        </div>
      </div>
      <Drawer open={controlsOpen} onClose={() => setControls(false)} side="right" label="Controls"><Controls /></Drawer>
    </div>
  );
}
