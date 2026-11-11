/**
 * Application state (Zustand). It holds the loaded dataset's metadata, the selected series,
 * the user's settings, and the latest results from the workers.
 *
 * Data handling: the dataset stays in the user's browser. It lives in the analysis worker's
 * memory and in this origin's IndexedDB (so a reload doesn't need the file again). It is never
 * uploaded anywhere.
 */
import { create } from 'zustand';
import { del, get, set as idbSet } from 'idb-keyval';
import { analysisWorker, Comlink, optimiserWorker, stopOptimiser } from '../engine/client';
import type { BenchmarkResult, DatasetInfo, ForecastBundle, OptimiseIteration, OptimiseRequest, OptimiseResult, PerturbationRow, RiskBundle, Settings } from '../engine/types';
import type { PolicyConstraints, SearchMode } from '../core/policyOptimizer';
import type { NewRow } from '../data/newRows';
import { DEFAULT_SETTINGS } from './defaults';

export type PageId = 'tower' | 'data' | 'forecast' | 'risk' | 'optimiser' | 'sensitivity' | 'compare' | 'method';

export { DEFAULT_SETTINGS };

const DATA_KEY = 'reorder-dashboard:dataset:v1';
/** Rows the user added per series (form / CSV). Replayed days are never stored: they are already in the file. */
const ADDED_KEY = 'reorder-dashboard:added-rows:v1';
const SETTINGS_KEY = 'reorder-dashboard:settings:v1';

function loadSettings(): Settings {
  try {
    const s = localStorage.getItem(SETTINGS_KEY);
    if (s) return { ...DEFAULT_SETTINGS, ...JSON.parse(s), leadTime: { ...DEFAULT_SETTINGS.leadTime, ...JSON.parse(s).leadTime }, costs: { ...DEFAULT_SETTINGS.costs, ...JSON.parse(s).costs } };
  } catch {
    /* storage unavailable: use defaults */
  }
  return { ...DEFAULT_SETTINGS };
}

function saveSettings(s: Settings) {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
  } catch {
    /* private mode or blocked storage: settings simply aren't remembered */
  }
}

export interface OptimiserState {
  mode: SearchMode;
  budget: number;
  constraints: PolicyConstraints | null; // null → defaults derived from demand
  running: boolean;
  iterations: OptimiseIteration[];
  result: OptimiseResult | null;
  benchmark: BenchmarkResult | null;
  benchmarkProgress: number | null;
  perturb: PerturbationRow[] | null;
  perturbProgress: { frac: number; label: string } | null;
  /** Settings hash and series the result belongs to (a result is marked stale if either changes). */
  forHash: string | null;
}

const emptyOptimiser = (): OptimiserState => ({ mode: 'sQ', budget: 40, constraints: null, running: false, iterations: [], result: null, benchmark: null, benchmarkProgress: null, perturb: null, perturbProgress: null, forHash: null });

/** One entry in the "what changed" feed, created whenever new data or a model update arrives. */
export interface UpdateEntry {
  id: number;
  at: string; // local time
  key: string;
  kind: 'form' | 'csv' | 'replay' | 'lstm' | 'import' | 'reset';
  title: string;
  detail: string;
  before?: Snapshot;
  after?: Snapshot;
}

export interface Snapshot {
  lastDate: string;
  rop: number;
  ss: number;
  mcRop: number;
  next7: number;
  weights: string;
}

export interface ReplayState {
  active: boolean;
  remaining: number;
  total: number;
  nextDate: string | null;
  playing: boolean;
  /** Milliseconds between days while playing (after each recalculation completes). */
  speedMs: number;
}

interface AppState {
  phase: 'boot' | 'needData' | 'loading' | 'ready' | 'error';
  error: string | null;
  page: PageId;
  dataset: DatasetInfo | null;
  key: string | null;
  settings: Settings;
  forecast: ForecastBundle | null;
  progress: { msg: string; frac: number } | null;
  risk: RiskBundle | null;
  riskBusy: boolean;
  opt: OptimiserState;
  controlsOpen: boolean;
  updates: UpdateEntry[];
  replay: ReplayState;
  /** Running LSTM job (fine-tune or training), shown as progress. */
  lstmJob: { msg: string; frac: number } | null;
  /** Number of user-added rows per series (persisted). */
  added: Record<string, NewRow[]>;
  dataBusy: boolean;

  boot: () => Promise<void>;
  loadDataset: (raw: unknown, persist: boolean) => Promise<void>;
  clearData: () => Promise<void>;
  setPage: (p: PageId) => void;
  selectKey: (k: string) => Promise<void>;
  updateSettings: (patch: Partial<Settings>) => void;
  resetSettings: () => void;
  setControlsOpen: (v: boolean) => void;
  setOpt: (patch: Partial<OptimiserState>) => void;
  runOptimiser: () => Promise<void>;
  cancelOptimiser: () => void;
  runBenchmark: (grid: number) => Promise<void>;
  runPerturb: () => Promise<void>;
  loadDemo: () => Promise<void>;
  addRows: (rows: NewRow[], source: 'form' | 'csv' | 'import') => Promise<void>;
  resetAdded: () => Promise<void>;
  startReplay: (days: number) => Promise<void>;
  replayStep: (k?: number) => Promise<void>;
  replayPlay: () => void;
  replayPause: () => void;
  replayStop: () => Promise<void>;
  setReplaySpeed: (ms: number) => void;
  fineTuneNow: () => Promise<void>;
  trainLstmInBrowser: () => Promise<void>;
}

const PAGES: PageId[] = ['tower', 'data', 'forecast', 'risk', 'optimiser', 'sensitivity', 'compare', 'method'];
/** Page from the URL hash (#/forecast), so a reload or a shared link opens the same view. */
export function pageFromHash(): PageId | null {
  const h = (typeof location !== 'undefined' ? location.hash : '').replace('#/', '') as PageId;
  return PAGES.includes(h) ? h : null;
}

let riskTimer: ReturnType<typeof setTimeout> | null = null;
let riskSeq = 0;

export const useApp = create<AppState>((set, getState) => ({
  phase: 'boot',
  error: null,
  page: pageFromHash() ?? 'tower',
  dataset: null,
  key: null,
  settings: loadSettings(),
  forecast: null,
  progress: null,
  risk: null,
  riskBusy: false,
  opt: emptyOptimiser(),
  controlsOpen: false,
  updates: [],
  replay: { active: false, remaining: 0, total: 0, nextDate: null, playing: false, speedMs: 600 },
  lstmJob: null,
  added: {},
  dataBusy: false,
  loadDemo: (...a: unknown[]) => (live.loadDemo as (...x: unknown[]) => never)(...a),
  addRows: (...a: unknown[]) => (live.addRows as (...x: unknown[]) => never)(...a),
  resetAdded: (...a: unknown[]) => (live.resetAdded as (...x: unknown[]) => never)(...a),
  startReplay: (...a: unknown[]) => (live.startReplay as (...x: unknown[]) => never)(...a),
  replayStep: (...a: unknown[]) => (live.replayStep as (...x: unknown[]) => never)(...a),
  replayPlay: (...a: unknown[]) => (live.replayPlay as (...x: unknown[]) => never)(...a),
  replayPause: (...a: unknown[]) => (live.replayPause as (...x: unknown[]) => never)(...a),
  replayStop: (...a: unknown[]) => (live.replayStop as (...x: unknown[]) => never)(...a),
  setReplaySpeed: (...a: unknown[]) => (live.setReplaySpeed as (...x: unknown[]) => never)(...a),
  fineTuneNow: (...a: unknown[]) => (live.fineTuneNow as (...x: unknown[]) => never)(...a),
  trainLstmInBrowser: (...a: unknown[]) => (live.trainLstmInBrowser as (...x: unknown[]) => never)(...a),

  boot: async () => {
    try {
      const raw = await get(DATA_KEY);
      if (raw) return getState().loadDataset(raw, false);
    } catch {
      /* IndexedDB unavailable: ask for the file */
    }
    set({ phase: 'needData' });
  },

  loadDataset: async (raw, persist) => {
    set({ phase: 'loading', error: null, progress: { msg: 'Validating dataset', frac: 0 } });
    try {
      const info = await analysisWorker().init(raw);
      if (persist) {
        try {
          await idbSet(DATA_KEY, raw);
        } catch {
          /* not persisted; still works for this session */
        }
      }
      // Re-apply rows the user added in earlier sessions (stored in this browser only)
      let added: Record<string, NewRow[]> = {};
      try {
        added = ((await get(ADDED_KEY)) as Record<string, NewRow[]> | undefined) ?? {};
      } catch {
        added = {};
      }
      if (!persist) {
        for (const [k, rows] of Object.entries(added)) {
          if (!rows.length || !info.series.some((x) => x.key === k)) continue;
          try {
            await analysisWorker().appendRows(k, rows);
          } catch {
            delete added[k]; // rows no longer fit this dataset
          }
        }
      } else added = {};
      const key = info.series.find((s) => s.key === '44|BEVERAGES')?.key ?? info.series[0].key;
      set({ dataset: info, added, updates: [], replay: { ...getState().replay, active: false, remaining: 0, total: 0, nextDate: null, playing: false } });
      if (persist) await saveAdded({});
      await getState().selectKey(key);
      set({ phase: 'ready' });
    } catch (e) {
      set({ phase: 'error', error: e instanceof Error ? e.message : String(e) });
    }
  },

  clearData: async () => {
    try {
      await del(DATA_KEY);
      await del(ADDED_KEY);
    } catch {
      /* nothing stored */
    }
    set({ phase: 'needData', dataset: null, key: null, forecast: null, risk: null, opt: emptyOptimiser() });
  },

  setPage: (p) => {
    set({ page: p });
    window.scrollTo({ top: 0 });
  },

  selectKey: async (k) => {
    stopOptimiser();
    set({ key: k, forecast: null, risk: null, opt: { ...emptyOptimiser(), mode: getState().opt.mode, budget: getState().opt.budget }, progress: { msg: 'Starting', frac: 0 } });
    const w = analysisWorker();
    const fb = await w.prepare(k, getState().settings.useLstm, Comlink.proxy((msg: string, frac: number) => set({ progress: { msg, frac } })));
    if (getState().key !== k) return;
    set({ forecast: fb, progress: null, replay: { ...getState().replay, ...(await w.replayStatus(k)), playing: false } });
    await computeRisk(k);
    // Bring the live LSTM up to date with every day after its training cut-off (in the background)
    if (fb.lstm.available && !fb.lstm.fineTunedTo) void enqueue(() => fineTune(k, 'Caught up on the days after its training cut-off'));
  },

  updateSettings: (patch) => {
    const prev = getState().settings;
    const next: Settings = { ...prev, ...patch, leadTime: { ...prev.leadTime, ...(patch.leadTime ?? {}) }, costs: { ...prev.costs, ...(patch.costs ?? {}) } };
    set({ settings: next });
    saveSettings(next);
    const k = getState().key;
    if (!k) return;
    if (patch.useLstm !== undefined && patch.useLstm !== prev.useLstm) {
      void getState().selectKey(k);
      return;
    }
    if (riskTimer) clearTimeout(riskTimer);
    set({ riskBusy: true });
    riskTimer = setTimeout(() => void computeRisk(k), 250);
  },

  resetSettings: () => getState().updateSettings({ ...DEFAULT_SETTINGS, leadTime: { ...DEFAULT_SETTINGS.leadTime }, costs: { ...DEFAULT_SETTINGS.costs } }),
  setControlsOpen: (v) => set({ controlsOpen: v }),
  setOpt: (patch) => set({ opt: { ...getState().opt, ...patch } }),

  runOptimiser: async () => {
    const req = await optimiseRequest();
    if (!req) return;
    const hash = optHash();
    set({ opt: { ...getState().opt, running: true, iterations: [], result: null, benchmark: null, perturb: null, forHash: hash } });
    try {
      const res = await optimiserWorker().optimise(req, Comlink.proxy((it: OptimiseIteration) => set({ opt: { ...getState().opt, iterations: [...getState().opt.iterations, it] } })));
      set({ opt: { ...getState().opt, running: false, result: res } });
    } catch {
      // A terminated worker (user pressed Stop) rejects the pending call
      set({ opt: { ...getState().opt, running: false } });
    }
  },

  cancelOptimiser: () => {
    stopOptimiser();
    set({ opt: { ...getState().opt, running: false, benchmarkProgress: null, perturbProgress: null } });
  },

  runBenchmark: async (grid) => {
    const req = await optimiseRequest();
    if (!req) return;
    set({ opt: { ...getState().opt, benchmarkProgress: 0 } });
    try {
      const b = await optimiserWorker().benchmark(req, grid, Comlink.proxy((f: number) => set({ opt: { ...getState().opt, benchmarkProgress: f } })));
      set({ opt: { ...getState().opt, benchmark: b, benchmarkProgress: null } });
    } catch {
      set({ opt: { ...getState().opt, benchmarkProgress: null } });
    }
  },

  runPerturb: async () => {
    const req = await optimiseRequest();
    if (!req) return;
    set({ opt: { ...getState().opt, perturbProgress: { frac: 0, label: 'Starting' } } });
    try {
      const rows = await optimiserWorker().perturb(req, 25, Comlink.proxy((frac: number, label: string) => set({ opt: { ...getState().opt, perturbProgress: { frac, label } } })));
      set({ opt: { ...getState().opt, perturb: rows, perturbProgress: null } });
    } catch {
      set({ opt: { ...getState().opt, perturbProgress: null } });
    }
  },
}));

// ---------------------------------------------------------------- live feed: new data and continuous updating

/** Worker jobs that change data or models run one at a time, in order. */
let queue: Promise<unknown> = Promise.resolve();
function enqueue<T>(job: () => Promise<T>): Promise<T> {
  const next = queue.then(job, job);
  queue = next.catch(() => undefined);
  return next;
}

let updateId = 0;
function log(e: Omit<UpdateEntry, 'id' | 'at'>) {
  const entry: UpdateEntry = { ...e, id: ++updateId, at: new Date().toLocaleTimeString() };
  useApp.setState({ updates: [entry, ...useApp.getState().updates].slice(0, 200) });
}

function snapshot(): Snapshot | undefined {
  const { forecast: f, risk: r } = useApp.getState();
  if (!f || !r) return undefined;
  const w = Object.entries(f.forecast.weights).map(([m, v]) => `${m} ${Math.round((v ?? 0) * 100)}%`).join(', ');
  return { lastDate: f.originDate, rop: r.comparison.ai.reorderPoint, ss: r.comparison.ai.safetyStock, mcRop: r.mc.ai.mcRop, next7: f.forecast.ensemble.slice(0, 7).reduce((a, b) => a + b, 0), weights: w };
}

async function saveAdded(added: Record<string, NewRow[]>) {
  try {
    await idbSet(ADDED_KEY, added);
  } catch {
    /* not persisted: the rows still apply for this session */
  }
}

/** Recomputes forecast and risk after a data or model change. */
async function refresh(k: string) {
  const w = analysisWorker();
  const fb = await w.prepare(k, useApp.getState().settings.useLstm, Comlink.proxy((msg: string, frac: number) => useApp.setState({ progress: { msg, frac } })));
  if (useApp.getState().key !== k) return;
  useApp.setState({ forecast: fb, progress: null });
  await computeRisk(k);
}

/** Fine-tunes the live LSTM, refreshes, and logs what changed. */
async function fineTune(k: string, why: string) {
  if (!useApp.getState().forecast?.lstm.available) return;
  const before = snapshot();
  useApp.setState({ lstmJob: { msg: 'Fine-tuning LSTM', frac: 0 } });
  try {
    const rep = await analysisWorker().fineTuneLstm(k, Comlink.proxy((msg: string, frac: number) => useApp.setState({ lstmJob: { msg, frac } })));
    if (!rep || rep.newDays === 0) return;
    await refresh(k);
    log({ key: k, kind: 'lstm', title: `LSTM fine-tuned on ${rep.newDays} new day${rep.newDays > 1 ? 's' : ''}`, detail: `${why}. ${rep.samples} training windows, ${rep.epochs} epochs, ${(rep.ms / 1000).toFixed(1)} s in this browser. LSTM next-7-day forecast ${Math.round(rep.before7).toLocaleString('en-US')} → ${Math.round(rep.after7).toLocaleString('en-US')} units.`, before, after: snapshot() });
  } finally {
    useApp.setState({ lstmJob: null });
  }
}

/** Days since the live LSTM was last fine-tuned that trigger an automatic update. */
const FINE_TUNE_EVERY = 7;
let daysSinceTune = 0;

async function afterNewData(k: string, n: number, kind: UpdateEntry['kind'], title: string, before: Snapshot | undefined) {
  await refresh(k);
  const after = snapshot();
  const d = before && after ? after.rop - before.rop : 0;
  log({ key: k, kind, title, detail: before && after ? `AI reorder point ${Math.round(before.rop).toLocaleString('en-US')} → ${Math.round(after.rop).toLocaleString('en-US')} (${d >= 0 ? '+' : ''}${Math.round(d).toLocaleString('en-US')}); next-7-day forecast ${Math.round(before.next7).toLocaleString('en-US')} → ${Math.round(after.next7).toLocaleString('en-US')}. Holt-Winters and Ridge refitted; backtest and ensemble weights updated.` : 'Models refitted.', before, after });
  daysSinceTune += n;
  if (daysSinceTune >= FINE_TUNE_EVERY) {
    daysSinceTune = 0;
    await fineTune(k, `Automatic update after ${FINE_TUNE_EVERY}+ new days`);
  }
}

/** Live-feed actions (the store delegates to these; defined here so they can use the helpers above). */
const live: Pick<AppState, 'loadDemo' | 'addRows' | 'resetAdded' | 'startReplay' | 'replayStep' | 'replayPlay' | 'replayPause' | 'replayStop' | 'setReplaySpeed' | 'fineTuneNow' | 'trainLstmInBrowser'> = {
  loadDemo: async () => {
    const raw = await analysisWorker().demoData();
    await useApp.getState().loadDataset(raw, true);
  },

  addRows: (rows, source) =>
    enqueue(async () => {
      const k = useApp.getState().key;
      if (!k || !rows.length) return;
      useApp.setState({ dataBusy: true });
      try {
        const before = snapshot();
        const r = await analysisWorker().appendRows(k, rows);
        const added = { ...useApp.getState().added, [k]: [...(useApp.getState().added[k] ?? []), ...rows] };
        useApp.setState({ added });
        await saveAdded(added);
        await afterNewData(k, r.appended, source, `${r.appended} new day${r.appended > 1 ? 's' : ''} added (${source === 'form' ? 'form' : source === 'csv' ? 'CSV upload' : 'import'}), now up to ${r.lastDate}`, before);
      } finally {
        useApp.setState({ dataBusy: false });
      }
    }),

  resetAdded: () =>
    enqueue(async () => {
      const k = useApp.getState().key;
      await saveAdded({});
      useApp.setState({ added: {} });
      const raw = await get(DATA_KEY);
      if (raw) await useApp.getState().loadDataset(raw, false);
      if (k) log({ key: k, kind: 'reset', title: 'Added data removed', detail: 'The dataset was restored to the loaded file.' });
    }),

  startReplay: (days) =>
    enqueue(async () => {
      const k = useApp.getState().key;
      if (!k) return;
      useApp.setState({ dataBusy: true });
      try {
        const st = await analysisWorker().startReplay(k, days);
        useApp.setState({ replay: { ...useApp.getState().replay, ...st, playing: false } });
        daysSinceTune = 0;
        await refresh(k);
        log({ key: k, kind: 'replay', title: `Replay started: ${st.total} days held back`, detail: `The data now ends on ${useApp.getState().forecast?.originDate}. The held-back days will arrive one at a time, as if they were new.`, after: snapshot() });
      } finally {
        useApp.setState({ dataBusy: false });
      }
    }),

  replayStep: (k = 1) =>
    enqueue(async () => {
      const key = useApp.getState().key;
      if (!key || !useApp.getState().replay.active) return;
      useApp.setState({ dataBusy: true });
      try {
        const before = snapshot();
        const st = await analysisWorker().replayNext(key, k);
        useApp.setState({ replay: { ...useApp.getState().replay, ...st } });
        const day = st.added[st.added.length - 1];
        await afterNewData(key, st.added.length, 'replay', `Day arrived: ${day.date} sold ${day.sales === null ? 'closed' : Math.round(day.sales).toLocaleString('en-US')}${day.onpromotion ? ` (${day.onpromotion} items on promotion)` : ''}`, before);
        if (!st.active) useApp.setState({ replay: { ...useApp.getState().replay, playing: false } });
      } finally {
        useApp.setState({ dataBusy: false });
      }
    }),

  replayPlay: () => {
    if (!useApp.getState().replay.active) return;
    useApp.setState({ replay: { ...useApp.getState().replay, playing: true } });
    const tick = async () => {
      const r = useApp.getState().replay;
      if (!r.playing || !r.active) return;
      await useApp.getState().replayStep(1);
      setTimeout(() => void tick(), useApp.getState().replay.speedMs);
    };
    void tick();
  },

  replayPause: () => useApp.setState({ replay: { ...useApp.getState().replay, playing: false } }),

  replayStop: () =>
    enqueue(async () => {
      const k = useApp.getState().key;
      if (!k) return;
      useApp.setState({ replay: { ...useApp.getState().replay, playing: false }, dataBusy: true });
      try {
        const st = await analysisWorker().stopReplay(k);
        useApp.setState({ replay: { ...useApp.getState().replay, ...st } });
        await refresh(k);
        log({ key: k, kind: 'replay', title: 'Replay finished: all held-back days restored', detail: `Data now ends on ${useApp.getState().forecast?.originDate}.`, after: snapshot() });
      } finally {
        useApp.setState({ dataBusy: false });
      }
    }),

  setReplaySpeed: (ms) => useApp.setState({ replay: { ...useApp.getState().replay, speedMs: ms } }),

  fineTuneNow: () =>
    enqueue(async () => {
      const k = useApp.getState().key;
      if (k) await fineTune(k, 'Requested manually');
    }),

  trainLstmInBrowser: () =>
    enqueue(async () => {
      const k = useApp.getState().key;
      if (!k) return;
      const before = snapshot();
      useApp.setState({ lstmJob: { msg: 'Training LSTM from scratch', frac: 0 } });
      try {
        if (!useApp.getState().settings.useLstm) useApp.setState({ settings: { ...useApp.getState().settings, useLstm: true } });
        const r = await analysisWorker().trainLstm(k, Comlink.proxy((msg: string, frac: number) => useApp.setState({ lstmJob: { msg, frac } })));
        await refresh(k);
        log({ key: k, kind: 'lstm', title: 'LSTM trained from scratch in this browser', detail: `${r.samples} training windows, ${r.epochs} epochs, validation MSE ${r.valLoss.toFixed(3)}, ${(r.ms / 1000).toFixed(0)} s. The backtest re-ran with the new frozen weights.`, before, after: snapshot() });
      } finally {
        useApp.setState({ lstmJob: null });
      }
      await fineTune(k, 'Bringing the new model up to the latest data');
    }),
};

async function computeRisk(k: string) {
  const seq = ++riskSeq;
  useApp.setState({ riskBusy: true });
  const r = await analysisWorker().risk(k, useApp.getState().settings);
  if (seq !== riskSeq || useApp.getState().key !== k) return; // a newer request superseded this one
  useApp.setState({ risk: r, riskBusy: false });
}

/** Identifies the inputs an optimiser result was computed with. */
export function optHash(): string {
  const s = useApp.getState();
  return JSON.stringify([s.key, s.settings, s.opt.mode, s.opt.constraints, s.opt.budget]);
}

async function optimiseRequest(): Promise<OptimiseRequest | null> {
  const s = useApp.getState();
  if (!s.key || !s.risk) return null;
  const inputs = await analysisWorker().optimiserInputs(s.key, s.settings);
  const d = s.risk.optimiserDefaults;
  return {
    demand: inputs.demand,
    leadTime: inputs.leadTime,
    costs: s.settings.costs,
    horizon: 196,
    replications: s.settings.simReps,
    seed: s.settings.seed,
    bounds: d.bounds,
    constraints: s.opt.constraints ?? d.constraints,
    mode: s.opt.mode,
    fixedQ: s.risk.eoq,
    budget: s.opt.budget,
    nInit: s.opt.mode === 'sQ' ? 8 : 4,
  };
}

// Dev-only debugging handle (compiled out of production builds)
if (import.meta.env.DEV && typeof window !== 'undefined') (window as unknown as { __app: typeof useApp }).__app = useApp;
