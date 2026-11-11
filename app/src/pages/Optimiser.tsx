/**
 * Step 3, Bayesian optimisation: choose (s, Q) to minimise expected cost under service, budget,
 * capacity and order-frequency constraints, instead of sweeping parameters by hand.
 * Iterations stream in live from the optimiser worker.
 */
import { lazy, Suspense, useState } from 'react';
import { Box, Play, Square, Timer } from 'lucide-react';
import type { EChartsOption } from 'echarts';
import { optHash, useApp } from '../state/store';
import { EChart, fmtInt, fmtK, fmtMoney, fmtPct, tipRow } from '../charts/EChart';
import { C, SEQ_BLUE, withAlpha } from '../charts/tokens';
import { constraintLabels, type PolicyConstraints } from '../core/policyOptimizer';
import type { OptimiseIteration, OptimiseResult, RiskBundle } from '../engine/types';
import { Badge, Button, ChartCard, DataTable, Explain, KpiTile, Legend, Note, NumberField, Panel, Progress, Segmented, Slider, Toggle } from '../ui/primitives';
import { PageHeader, WithData, seriesTitle } from './common';

const Surface3D = lazy(() => import('../three/Surface3D'));

function ConstraintEditor({ c, defaults, onChange }: { c: PolicyConstraints; defaults: PolicyConstraints; onChange: (c: PolicyConstraints) => void }) {
  return (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <span className="text-[12.5px] text-ink-2">Service constraint</span>
        <Segmented label="Service metric" value={c.serviceMetric} onChange={(v) => onChange({ ...c, serviceMetric: v })} options={[{ value: 'fillRate', label: 'Fill rate' }, { value: 'cycleServiceLevel', label: 'Cycle service level' }]} />
      </div>
      <Slider label={`Minimum ${c.serviceMetric === 'fillRate' ? 'fill rate' : 'cycle service level'}`} value={c.minService} min={0.8} max={0.999} step={0.001} onChange={(v) => onChange({ ...c, minService: v })} format={(v) => fmtPct(v, 1)} />
      <div className="space-y-2">
        <Toggle label="Working-capital budget" checked={c.maxInventoryValue !== null} onChange={(on) => onChange({ ...c, maxInventoryValue: on ? defaults.maxInventoryValue : null })} />
        {c.maxInventoryValue !== null && <NumberField label="Max average inventory value" prefix="$" step={1000} value={Math.round(c.maxInventoryValue)} onChange={(v) => onChange({ ...c, maxInventoryValue: v })} />}
      </div>
      <div className="space-y-2">
        <Toggle label="Warehouse capacity" checked={c.maxPeakOnHand !== null} onChange={(on) => onChange({ ...c, maxPeakOnHand: on ? defaults.maxPeakOnHand : null })} />
        {c.maxPeakOnHand !== null && <NumberField label="Max on-hand (95th pct of peaks)" suffix="units" step={1000} value={Math.round(c.maxPeakOnHand)} onChange={(v) => onChange({ ...c, maxPeakOnHand: v })} />}
      </div>
      <div className="space-y-2">
        <Toggle label="Order-frequency limit" checked={c.maxOrdersPer30 !== null} onChange={(on) => onChange({ ...c, maxOrdersPer30: on ? defaults.maxOrdersPer30 : null })} />
        {c.maxOrdersPer30 !== null && <NumberField label="Max orders per 30 days" step={1} min={1} value={c.maxOrdersPer30} onChange={(v) => onChange({ ...c, maxOrdersPer30: v })} />}
      </div>
    </div>
  );
}

function convergenceOption(its: OptimiseIteration[], bench: ReturnType<typeof useApp.getState>['opt']['benchmark'], budget: number): EChartsOption {
  const bo = its.map((it, i) => [i + 1, Number.isFinite(it.bestFeasible) ? it.bestFeasible : null]);
  const rnd = bench ? bench.random.bestSoFar.map((v, i) => [i + 1, Number.isFinite(v) ? v : null]) : [];
  const gridBest = bench?.grid.best?.cost;
  return {
    grid: { left: 58, right: 16, top: 20, bottom: 40 },
    xAxis: { type: 'value', min: 1, max: Math.max(budget, bo.length), name: 'simulation runs (evaluations)', nameLocation: 'middle', nameGap: 26, splitLine: { show: false } },
    yAxis: { type: 'value', scale: true, name: '$/day', axisLabel: { formatter: (v: number) => `$${Math.round(v)}` } },
    tooltip: { trigger: 'axis', valueFormatter: (v) => (v === null ? 'no feasible point yet' : fmtMoney(Number(v))) },
    series: [
      { name: 'Bayesian optimisation', type: 'line', step: 'end', data: bo, symbol: 'circle', symbolSize: 6, showSymbol: true, lineStyle: { color: C.ai, width: 2.2 }, itemStyle: { color: C.ai, borderColor: C.surface, borderWidth: 2 },
        markLine: gridBest ? { symbol: 'none', silent: true, data: [{ yAxis: gridBest, lineStyle: { color: C.grid2, type: 'dashed', width: 1.5 }, label: { formatter: `Grid optimum (${bench!.grid.size} runs) ${fmtMoney(gridBest)}`, color: C.ink2, position: 'insideStartTop', fontSize: 11 } }] } : undefined },
      ...(rnd.length ? [{ name: 'Random search', type: 'line' as const, step: 'end' as const, data: rnd, symbol: 'none', lineStyle: { color: C.random, width: 1.8 } }] : []),
    ],
  };
}

function surfaceOption(res: OptimiseResult, rk: RiskBundle): EChartsOption {
  const sf = res.surface!;
  const data: [number, number, number][] = [];
  const mask: [number, number, number][] = [];
  let lo = Infinity, hi = -Infinity;
  sf.mean.forEach((row, j) => row.forEach((v, i) => {
    data.push([i, j, v]);
    if (sf.pFeasible[j][i] >= 0.5) { lo = Math.min(lo, v); hi = Math.max(hi, v); } else mask.push([i, j, 1]);
  }));
  const idx = (val: number, axis: number[]) => ((val - axis[0]) / (axis[axis.length - 1] - axis[0])) * (axis.length - 1);
  const pts = res.history.map((h) => ({ value: [idx(h.s, sf.s), idx(h.Q, sf.q), h.cost], feasible: h.feasible, h }));
  return {
    grid: { left: 64, right: 86, top: 14, bottom: 46 },
    xAxis: { type: 'value', min: -0.5, max: sf.s.length - 0.5, name: 'reorder point s (units)', nameLocation: 'middle', nameGap: 28, axisLabel: { formatter: (v: number) => (sf.s[Math.round(v)] !== undefined && Math.abs(v - Math.round(v)) < 1e-6 ? fmtK(sf.s[Math.round(v)]) : '') }, splitLine: { show: false } },
    yAxis: { type: 'value', min: -0.5, max: sf.q.length - 0.5, name: 'order qty Q', axisLabel: { formatter: (v: number) => (sf.q[Math.round(v)] !== undefined && Math.abs(v - Math.round(v)) < 1e-6 ? fmtK(sf.q[Math.round(v)]) : '') }, splitLine: { show: false } },
    visualMap: { min: lo, max: Math.min(hi, lo * 3), dimension: 2, seriesIndex: 0, calculable: false, orient: 'vertical', right: 6, top: 'middle', itemHeight: 160, inRange: { color: SEQ_BLUE }, text: ['higher cost', 'lower'], textStyle: { color: C.ink3, fontSize: 11 }, formatter: (v: unknown) => `$${Math.round(Number(v))}` },
    tooltip: {
      formatter: (p: unknown) => {
        const q = p as { seriesIndex: number; value: number[]; data: { h?: OptimiseIteration } };
        if (q.seriesIndex === 1) return '';
        if (q.seriesIndex === 2 || q.seriesIndex === 3) {
          const h = q.data.h!;
          return `<div style="font-weight:600;margin-bottom:4px">Evaluation #${h.iter + 1} (${h.phase})</div>${tipRow(C.ai, 's', fmtInt(h.s))}${tipRow(C.ai, 'Q', fmtInt(h.Q))}${tipRow(C.ai, 'Cost/day', fmtMoney(h.cost))}${tipRow(C.ai, 'Fill rate', fmtPct(h.metrics.fillRate, 2))}${tipRow(h.feasible ? C.good : C.stockout, 'Constraints', h.feasible ? 'met' : 'violated')}`;
        }
        const [i, j] = q.value;
        return `${tipRow(C.ai, 's', fmtInt(sf.s[i]))}${tipRow(C.ai, 'Q', fmtInt(sf.q[j]))}${tipRow(C.ai, 'Predicted cost', fmtMoney(sf.mean[j][i]))}${tipRow(C.ai, '± uncertainty', fmtMoney(sf.sd[j][i]))}${tipRow(C.rop, 'P(feasible)', fmtPct(sf.pFeasible[j][i], 0))}`;
      },
    },
    series: [
      // Cells are drawn as a custom series (an ECharts heatmap needs category axes, but the
      // evaluated points must sit at exact, fractional positions on value axes)
      { type: 'custom', data, renderItem: cell() as never, encode: { x: 0, y: 1 }, z: 1 },
      { type: 'custom', data: mask, renderItem: cell('rgba(8,12,19,0.72)') as never, encode: { x: 0, y: 1 }, silent: true, z: 2 },
      { type: 'scatter', data: pts.filter((p) => p.feasible), symbolSize: 9, itemStyle: { color: C.ink, borderColor: C.surface, borderWidth: 2 }, z: 5 },
      { type: 'scatter', data: pts.filter((p) => !p.feasible), symbol: 'path://M2,2 L8,8 M8,2 L2,8', symbolSize: 9, itemStyle: { color: 'transparent', borderColor: C.ink3, borderWidth: 2 }, z: 5 },
      ...(res.best ? [{ type: 'scatter' as const, data: [[idx(res.best.s, sf.s), idx(res.best.Q, sf.q)]], symbol: 'path://M5,0 L6.5,3.5 L10,3.8 L7.3,6.1 L8.1,9.6 L5,7.8 L1.9,9.6 L2.7,6.1 L0,3.8 L3.5,3.5 Z', symbolSize: 20, itemStyle: { color: C.rop, borderColor: C.surface, borderWidth: 1.5 }, z: 6, silent: true }] : []),
      { type: 'scatter', data: [[idx(rk.comparison.ai.reorderPoint, sf.s), idx(rk.eoq, sf.q)]], symbol: 'diamond', symbolSize: 12, itemStyle: { color: 'transparent', borderColor: C.rop, borderWidth: 2 }, z: 6, silent: true },
    ],
  };
}

function oneDOption(res: OptimiseResult, rk: RiskBundle): EChartsOption {
  const sf = res.surface!;
  const m = sf.mean[0], sd = sf.sd[0], pf = sf.pFeasible[0];
  return {
    grid: { left: 58, right: 16, top: 16, bottom: 40 },
    xAxis: { type: 'value', min: sf.s[0], max: sf.s.at(-1), name: 'reorder point s (units)', nameLocation: 'middle', nameGap: 26, axisLabel: { formatter: (v: number) => fmtK(v) }, splitLine: { show: false } },
    yAxis: { type: 'value', scale: true, axisLabel: { formatter: (v: number) => `$${Math.round(v)}` } },
    tooltip: { trigger: 'axis', valueFormatter: (v) => fmtMoney(Number(v)) },
    series: [
      { name: 'lo', type: 'line', data: sf.s.map((s, i) => [s, m[i] - 2 * sd[i]]), stack: 'u', symbol: 'none', lineStyle: { opacity: 0 }, silent: true, tooltip: { show: false } },
      { name: '±2σ', type: 'line', data: sf.s.map((s, i) => [s, 4 * sd[i]]), stack: 'u', symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: withAlpha(C.ai, 0.16) }, silent: true, tooltip: { show: false } },
      { name: 'GP mean cost', type: 'line', data: sf.s.map((s, i) => [s, m[i]]), symbol: 'none', lineStyle: { color: C.ai, width: 2 },
        markArea: { silent: true, itemStyle: { color: 'rgba(208,59,59,0.08)' }, data: runs(sf.s, pf).map(([a, b]) => [{ xAxis: a }, { xAxis: b }]) as never },
        markLine: { symbol: 'none', silent: true, data: [{ xAxis: rk.comparison.ai.reorderPoint, lineStyle: { color: C.rop, type: 'dashed' }, label: { formatter: 'Formula ROP', color: C.ink2, fontSize: 11 } }, ...(res.best ? [{ xAxis: res.best.s, lineStyle: { color: C.good, width: 2 }, label: { formatter: 'Optimum s*', color: C.ink, fontSize: 11 } }] : [])] } },
      { name: 'Evaluations', type: 'scatter', data: res.history.map((h) => [h.s, h.cost]), symbolSize: 8, itemStyle: { color: C.ink, borderColor: C.surface, borderWidth: 2 } },
    ],
  };
}

/** renderItem for one surface cell (index-space rectangle), coloured by the visualMap unless a fill is given. */
function cell(fill?: string) {
  return (_: unknown, api: { value: (d: number) => number; coord: (p: number[]) => number[]; visual: (k: string) => string }) => {
    const i = api.value(0), j = api.value(1);
    const a = api.coord([i - 0.5, j - 0.5]), b = api.coord([i + 0.5, j + 0.5]);
    // Colour cells overlap by half a pixel to avoid hairline seams; the translucent mask must not overlap
    const o = fill ? 0 : 0.5;
    return { type: 'rect', shape: { x: a[0], y: b[1], width: b[0] - a[0] + o, height: a[1] - b[1] + o }, style: { fill: fill ?? api.visual('color') } };
  };
}

/** Contiguous s-ranges where P(feasible) < 0.5 (shaded as "violates constraints"). */
function runs(s: number[], pf: number[]): [number, number][] {
  const out: [number, number][] = [];
  let start: number | null = null;
  pf.forEach((p, i) => {
    if (p < 0.5 && start === null) start = s[i];
    if ((p >= 0.5 || i === pf.length - 1) && start !== null) { out.push([start, s[i]]); start = null; }
  });
  return out;
}

export function Optimiser() {
  const opt = useApp((s) => s.opt);
  const setOpt = useApp((s) => s.setOpt);
  const run = useApp((s) => s.runOptimiser);
  const cancel = useApp((s) => s.cancelOptimiser);
  const bench = useApp((s) => s.runBenchmark);
  const settings = useApp((s) => s.settings);
  const [view3d, setView3d] = useState(false);
  // Re-evaluated on every render: settings and optimiser inputs are subscribed above
  const stale = opt.forHash !== null && !opt.running && opt.forHash !== optHash();

  return (
    <WithData>
      {(fb, rk) => {
        const def = rk.optimiserDefaults;
        const cons = opt.constraints ?? def.constraints;
        const res = opt.result;
        const best = res?.best ?? (opt.iterations.length ? [...opt.iterations].filter((i) => i.feasible).sort((a, b) => a.cost - b.cost)[0] ?? null : null);
        const aiP = rk.policies.find((p) => p.id === 'ai-formula')!;
        const baseP = rk.policies.find((p) => p.id === 'baseline')!;
        const evalsTo1 = res && opt.benchmark?.grid.best ? res.history.findIndex((h) => h.bestFeasible <= opt.benchmark!.grid.best!.cost * 1.01) + 1 : 0;
        return (
          <>
            <PageHeader eyebrow="Step 3 · AI-driven sensitivity analysis" title={`Bayesian optimisation: ${seriesTitle(fb)}`}>
              Instead of sweeping reorder points by hand, a Gaussian-process model learns how cost responds to the policy, and picks each next simulation where improvement is most likely while still meeting every constraint (constrained expected improvement).
            </PageHeader>

            <div className="grid gap-4 xl:grid-cols-[320px_1fr]">
              <Panel title="Problem set-up" subtitle="Minimise expected cost/day = holding + ordering + lost-sales penalty">
                <div className="space-y-4">
                  <div className="space-y-1.5">
                    <span className="text-[12.5px] text-ink-2">Decision variables</span>
                    <Segmented label="Search mode" value={opt.mode} onChange={(v) => setOpt({ mode: v })} options={[{ value: 'sQ', label: 'Reorder point s + quantity Q' }, { value: 's', label: 's only (Q = EOQ)' }]} />
                  </div>
                  <Slider label="Evaluation budget" value={opt.budget} min={15} max={80} step={5} onChange={(v) => setOpt({ budget: v })} format={(v) => `${v} runs`} hint={`Each run simulates ${settings.simReps} × 182 days of the (s, Q) policy.`} />
                  <ConstraintEditor c={cons} defaults={def.constraints} onChange={(c) => setOpt({ constraints: c })} />
                  <div className="text-[11.5px] leading-snug text-ink-3">Search box: s ∈ [{fmtInt(def.bounds.s[0])}, {fmtInt(def.bounds.s[1])}]{opt.mode === 'sQ' && <>, Q ∈ [{fmtInt(def.bounds.Q[0])}, {fmtInt(def.bounds.Q[1])}]</>}</div>
                  <div className="flex flex-wrap gap-2">
                    {opt.running ? <Button variant="danger" onClick={cancel}><Square className="h-4 w-4" />Stop</Button> : <Button onClick={() => void run()}><Play className="h-4 w-4" />Run optimisation</Button>}
                    <Button variant="ghost" busy={opt.benchmarkProgress !== null} disabled={opt.running} onClick={() => void bench(opt.mode === 'sQ' ? 25 : 60)}><Timer className="h-4 w-4" />Benchmark vs grid & random</Button>
                  </div>
                  {opt.benchmarkProgress !== null && <Progress value={opt.benchmarkProgress} label={`Grid sweep (${opt.mode === 'sQ' ? '625' : '60'} runs)`} />}
                </div>
              </Panel>

              <div className="grid min-w-0 gap-4">
                <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                  <KpiTile label="Optimal reorder point s*" value={best?.s ?? NaN} format={fmtInt} accent="rop" sub={best ? <>formula ROP {fmtInt(rk.comparison.ai.reorderPoint)}</> : 'run the optimiser'} />
                  <KpiTile label="Optimal order qty Q*" value={best?.Q ?? NaN} format={fmtInt} accent="ai" sub={<>EOQ {fmtInt(rk.eoq)}</>} />
                  <KpiTile label="Expected cost / day" value={best?.cost ?? NaN} format={(v) => fmtMoney(v)} accent="good" sub={<>AI formula + EOQ {fmtMoney(aiP.metrics.totalCostPerDay)} · baseline {fmtMoney(baseP.metrics.totalCostPerDay)}</>} />
                  <KpiTile label="Fill rate at optimum" value={best?.metrics.fillRate ?? NaN} format={(v) => fmtPct(v, 2)} accent="ai" sub={best ? <>cycle SL {fmtPct(best.metrics.cycleServiceLevel)} · {best.metrics.ordersPer30.toFixed(1)} orders/30d</> : '–'} />
                </div>
                {stale && <Note tone="warn">Settings changed since this run. The results below belong to the previous inputs. Run again to update.</Note>}
                {!res && !opt.running && opt.iterations.length === 0 && <Note>Press <b>Run optimisation</b>. Iterations stream in live from a background worker; the page stays responsive.</Note>}
                <ChartCard title="Convergence: best feasible cost found so far" subtitle={opt.running ? `Running… ${opt.iterations.length}/${opt.budget}` : res ? `${res.history.length} evaluations in ${(res.elapsedMs / 1000).toFixed(1)} s` : 'Live as the optimiser runs'}
                  chart={<><EChart ariaLabel="Best feasible cost versus number of evaluations for Bayesian optimisation and benchmarks" height={260} option={convergenceOption(opt.iterations.length ? opt.iterations : res?.history ?? [], opt.benchmark, opt.budget)} />
                    <div className="mt-2"><Legend items={[{ color: C.ai, label: 'Bayesian optimisation' }, ...(opt.benchmark ? [{ color: C.random, label: 'Random search (same budget)' }, { color: C.grid2, label: 'Grid sweep optimum', dashed: true }] : [])]} /></div></>}>
                  {opt.benchmark?.grid.best && res?.best && (
                    <Explain>The exhaustive grid needed <b className="text-ink">{opt.benchmark.grid.size}</b> simulations to find {fmtMoney(opt.benchmark.grid.best.cost)}/day. Bayesian optimisation came within 1% after <b className="text-ink">{evalsTo1 > 0 ? evalsTo1 : '—'}</b> and finished at {fmtMoney(res.best.cost)}/day. Random search with the same budget reached {opt.benchmark.random.best ? fmtMoney(opt.benchmark.random.best.cost) : 'no feasible point'}.</Explain>
                  )}
                </ChartCard>
              </div>
            </div>

            {res?.surface && (
              <div className="mt-4 grid gap-4 xl:grid-cols-[1.4fr_1fr]">
                <ChartCard title={opt.mode === 'sQ' ? 'Learned cost surface (GP posterior mean)' : 'Learned cost curve (GP posterior)'} subtitle={opt.mode === 'sQ' ? 'Darker = cheaper. Shaded = predicted to violate a constraint. ● feasible run · × infeasible · ★ optimum · ◇ formula ROP + EOQ' : 'Shaded: predicted to violate a constraint. Dots: simulations run.'}
                  actions={opt.mode === 'sQ' && <Button variant="ghost" onClick={() => setView3d(!view3d)}><Box className="h-4 w-4" />{view3d ? '2D map' : '3D view'}</Button>}
                  chart={view3d && opt.mode === 'sQ'
                    ? <Suspense fallback={<div className="grid h-[380px] place-items-center text-ink-3">Loading 3D…</div>}><Surface3D surface={res.surface} history={res.history} best={res.best} /></Suspense>
                    : <EChart ariaLabel="Gaussian-process surrogate of cost over reorder point and order quantity" height={380} option={opt.mode === 'sQ' ? surfaceOption(res, rk) : oneDOption(res, rk)} />}>
                  <Explain>The surrogate is what the optimiser believes cost looks like everywhere, learned from {res.history.length} simulations. It samples densely near the cheap, feasible region and barely at all elsewhere: that's the saving over a grid. Length-scales {res.lengthScales.map((l) => l.toFixed(2)).join(' / ')} (on the unit box) show how quickly cost changes with each variable.</Explain>
                </ChartCard>
                <Panel title="Optimum vs formula policies" subtitle="Same simulator, same random demand (common random numbers)">
                  <DataTable caption="Optimised policy compared with formula policies" columns={[{ key: 'k', label: '' }, { key: 'b', label: 'Baseline', num: true }, { key: 'a', label: 'AI formula', num: true }, { key: 'o', label: 'Optimised', num: true }]}
                    rows={[
                      { k: 'Reorder point s', b: fmtInt(baseP.s), a: fmtInt(aiP.s), o: fmtInt(res.best?.s ?? NaN) },
                      { k: 'Order quantity Q', b: fmtInt(baseP.Q), a: fmtInt(aiP.Q), o: fmtInt(res.best?.Q ?? NaN) },
                      { k: 'Fill rate', b: fmtPct(baseP.metrics.fillRate, 2), a: fmtPct(aiP.metrics.fillRate, 2), o: fmtPct(res.best?.metrics.fillRate ?? NaN, 2) },
                      { k: 'Cycle service level', b: fmtPct(baseP.metrics.cycleServiceLevel), a: fmtPct(aiP.metrics.cycleServiceLevel), o: fmtPct(res.best?.metrics.cycleServiceLevel ?? NaN) },
                      { k: 'Avg inventory value', b: fmtMoney(baseP.metrics.avgInventoryValue, 0), a: fmtMoney(aiP.metrics.avgInventoryValue, 0), o: fmtMoney(res.best?.metrics.avgInventoryValue ?? NaN, 0) },
                      { k: 'Orders per 30 days', b: baseP.metrics.ordersPer30.toFixed(1), a: aiP.metrics.ordersPer30.toFixed(1), o: res.best?.metrics.ordersPer30.toFixed(1) ?? '–' },
                      { k: 'Expected cost/day', b: fmtMoney(baseP.metrics.totalCostPerDay), a: fmtMoney(aiP.metrics.totalCostPerDay), o: fmtMoney(res.best?.cost ?? NaN) },
                    ]} />
                  <div className="mt-3 flex flex-wrap gap-1.5">{constraintLabels(cons).map((l) => <Badge key={l}>{l}</Badge>)}</div>
                  <Explain>{res.best ? <>Cost per day is <b className="text-ink">{fmtPct(1 - res.best.cost / baseP.metrics.totalCostPerDay, 0)}</b> lower than the class-notebook policy, and every constraint holds. Most of the gain comes from holding the reorder point high enough to stop lost sales, and ordering in smaller, more frequent batches than EOQ suggests, because EOQ ignores stockouts and warehouse space.</> : 'No feasible policy found inside the search box: relax a constraint or raise the budget.'}</Explain>
                </Panel>
              </div>
            )}

            {(opt.iterations.length > 0 || res) && (
              <Panel className="mt-4" title="Evaluation log" subtitle="Every simulation the optimiser chose to run">
                <DataTable caption="Optimiser evaluations" maxHeight={300} columns={[{ key: 'i', label: '#', num: true }, { key: 'ph', label: 'Phase' }, { key: 's', label: 's', num: true }, { key: 'q', label: 'Q', num: true }, { key: 'c', label: 'Cost/day', num: true }, { key: 'f', label: 'Fill', num: true }, { key: 'ok', label: 'Constraints' }, { key: 'pf', label: 'P(feasible) predicted', num: true }]}
                  rows={(res?.history ?? opt.iterations).map((h) => ({ i: h.iter + 1, ph: h.phase === 'init' ? 'Latin hypercube' : 'Acquisition', s: fmtInt(h.s), q: fmtInt(h.Q), c: fmtMoney(h.cost), f: fmtPct(h.metrics.fillRate, 2), ok: h.feasible ? <Badge tone="good">met</Badge> : <Badge tone="bad">violated</Badge>, pf: h.pFeasible !== undefined ? fmtPct(h.pFeasible, 0) : '–' }))} />
              </Panel>
            )}
          </>
        );
      }}
    </WithData>
  );
}
