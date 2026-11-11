/**
 * Sensitivity of the reorder point. It replaces the class notebook's one-at-a-time sweeps with a
 * tornado chart, analytic elasticities, an L̄ × σL heatmap, the service-level trade-off (with the
 * Monte Carlo reality check), and AI-driven optimal-policy sensitivity (the optimiser re-run).
 */
import { Play } from 'lucide-react';
import { useApp } from '../state/store';
import { EChart, fmtInt, fmtK, fmtMoney, fmtPct, tipRow } from '../charts/EChart';
import { C, SEQ_BLUE } from '../charts/tokens';
import { INPUT_LABELS, type InputKey } from '../core/sensitivity';
import { Button, ChartCard, DataTable, Explain, Legend, Panel, Progress } from '../ui/primitives';
import { PageHeader, WithData, seriesTitle } from './common';

export function Sensitivity() {
  const opt = useApp((s) => s.opt);
  const runPerturb = useApp((s) => s.runPerturb);
  return (
    <WithData>
      {(fb, rk) => {
        const sn = rk.sensitivity;
        const rop = rk.comparison.ai.reorderPoint;
        const t = sn.tornado;
        const sc = sn.serviceCurve;
        const hm = sn.heatmap;
        const el = sn.elasticities;
        const elKeys = (Object.keys(el) as InputKey[]).sort((a, b) => Math.abs(el[b]) - Math.abs(el[a]));
        const baseRow = opt.perturb?.find((r) => r.id === 'base')?.best;
        return (
          <>
            <PageHeader eyebrow="Step 3 · Sensitivity analysis" title={`What moves the reorder point? ${seriesTitle(fb)}`}>
              Every input is varied around today's values, and the Bayesian optimiser is re-run under changed economics. You see which assumptions matter most and how the best policy shifts.
            </PageHeader>

            <div className="grid gap-4 xl:grid-cols-2">
              <ChartCard title="Tornado: reorder point at ±20% on each input" subtitle={`Service level ±2.5 points. Centre line: today's AI ROP ${fmtInt(rop)}`}
                table={<DataTable caption="Tornado data" columns={[{ key: 'k', label: 'Input' }, { key: 'lo', label: 'ROP (low)', num: true }, { key: 'hi', label: 'ROP (high)', num: true }, { key: 'sw', label: 'Swing', num: true }]} rows={t.map((b) => ({ k: b.label, lo: fmtInt(b.ropLow), hi: fmtInt(b.ropHigh), sw: fmtInt(b.swing) }))} />}
                chart={<><EChart ariaLabel="Tornado chart of reorder-point sensitivity to each input" height={280} option={{
                  grid: { left: 170, right: 24, top: 10, bottom: 30 },
                  xAxis: { type: 'value', scale: true, axisLabel: { formatter: (v: number) => fmtK(v) } },
                  yAxis: { type: 'category', inverse: true, data: t.map((b) => b.label), axisLabel: { color: C.ink2 } },
                  tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: (ps: unknown) => { const b = t[(ps as { dataIndex: number }[])[0].dataIndex]; return `<div style="font-weight:600;margin-bottom:4px">${b.label}</div>${tipRow(C.ai, `Low (${b.key === 'serviceLevel' ? fmtPct(b.lowInput) : (b.lowInput).toFixed(2)})`, fmtInt(b.ropLow))}${tipRow(C.baseline, `High (${b.key === 'serviceLevel' ? fmtPct(b.highInput) : (b.highInput).toFixed(2)})`, fmtInt(b.ropHigh))}`; } },
                  series: [
                    { name: 'base', type: 'bar', stack: 't', data: t.map((b) => Math.min(b.ropLow, b.ropHigh)), itemStyle: { color: 'transparent' }, silent: true, barWidth: '56%' },
                    { name: 'Input −', type: 'bar', stack: 't', data: t.map((b) => Math.max(0, rop - Math.min(b.ropLow, b.ropHigh))), itemStyle: { color: C.ai, borderRadius: [4, 0, 0, 4] } },
                    { name: 'Input +', type: 'bar', stack: 't', data: t.map((b) => Math.max(0, Math.max(b.ropLow, b.ropHigh) - rop)), itemStyle: { color: C.baseline, borderRadius: [0, 4, 4, 0] },
                      markLine: { symbol: 'none', silent: true, data: [{ xAxis: rop, lineStyle: { color: C.rop, width: 1.5 }, label: { formatter: 'today', color: C.ink2, fontSize: 11 } }] } },
                  ],
                }} /><div className="mt-2"><Legend items={[{ color: C.ai, label: 'Input lowered', area: true }, { color: C.baseline, label: 'Input raised', area: true }]} /></div></>}>
                <Explain><b className="text-ink">{t[0].label}</b> moves the reorder point most: a 20% change shifts it by about {fmtInt(t[0].swing / 2)} units. This is the assumption to get right first.</Explain>
              </ChartCard>

              <Panel title="Elasticities" subtitle="% change in ROP for a 1% change in each input (exact derivatives of the slide formula)">
                <div className="space-y-2.5">
                  {elKeys.map((k) => (
                    <div key={k} className="grid grid-cols-[170px_1fr_52px] items-center gap-3 text-[13px]">
                      <span className="text-ink-2">{INPUT_LABELS[k]}</span>
                      <div className="h-2.5 rounded-full bg-white/[0.05]"><div className="h-full rounded-full bg-ai" style={{ width: `${Math.min(100, (Math.abs(el[k]) / Math.abs(el[elKeys[0]])) * 100)}%` }} /></div>
                      <span className="num text-right text-ink">{el[k].toFixed(2)}</span>
                    </div>
                  ))}
                </div>
                <Explain>An elasticity of {el.dBar.toFixed(2)} for demand means 1% more demand needs about {el.dBar.toFixed(2)}% more reorder point. The service-level figure is per 1% of the service level itself (0.95 → 0.9595), which is why raising service near 99% gets expensive quickly.</Explain>
              </Panel>
            </div>

            <div className="mt-4 grid gap-4 xl:grid-cols-[1.2fr_1fr]">
              <ChartCard title="Reorder point by lead time and lead-time variability" subtitle="Slide formula at the current service level and demand"
                table={<DataTable caption="ROP heatmap" columns={[{ key: 'y', label: 'σL \\ L̄' }, ...hm.xs.map((x) => ({ key: String(x), label: `${x} d`, num: true }))]} rows={hm.ys.map((y, iy) => ({ y: `${y} d`, ...Object.fromEntries(hm.xs.map((x, ix) => [String(x), fmtK(hm.rop[iy][ix])])) }))} />}
                chart={<EChart ariaLabel="Heatmap of reorder point by mean lead time and lead-time variability" height={300} option={{
                  grid: { left: 52, right: 80, top: 10, bottom: 40 },
                  xAxis: { type: 'category', data: hm.xs.map((x) => `${x}`), name: 'mean lead time L̄ (days)', nameLocation: 'middle', nameGap: 26 },
                  yAxis: { type: 'category', data: hm.ys.map((y) => `${y}`), name: 'σL (days)' },
                  visualMap: { min: Math.min(...hm.rop.flat()), max: Math.max(...hm.rop.flat()), orient: 'vertical', right: 6, top: 'middle', itemHeight: 150, inRange: { color: SEQ_BLUE }, textStyle: { color: C.ink3, fontSize: 11 }, formatter: (v: unknown) => fmtK(Number(v)) },
                  tooltip: { formatter: (p: unknown) => { const v = (p as { value: number[] }).value; return `${tipRow(C.ai, 'L̄', `${hm.xs[v[0]]} d`)}${tipRow(C.ai, 'σL', `${hm.ys[v[1]]} d`)}${tipRow(C.rop, 'ROP', fmtInt(v[2]))}`; } },
                  series: [{ type: 'heatmap', data: hm.rop.flatMap((r, iy) => r.map((v, ix) => [ix, iy, v])), label: { show: true, formatter: (p: unknown) => fmtK((p as { value: number[] }).value[2]), color: C.ink, fontSize: 10 }, itemStyle: { borderColor: C.surface, borderWidth: 2 } }],
                }} />}>
                <Explain>Moving right (longer lead times) and moving up (less reliable deliveries) both raise the buffer. A supplier that is twice as reliable is often worth more than one that is a day faster.</Explain>
              </ChartCard>
              <div className="grid gap-4">
                <ChartCard title="Safety stock vs target service level" subtitle="Slide formula" chart={<EChart ariaLabel="Safety stock required by target service level" height={190} option={{
                  grid: { left: 52, right: 14, top: 12, bottom: 28 },
                  xAxis: { type: 'category', data: sc.map((p) => fmtPct(p.serviceLevel, 1)), axisLabel: { hideOverlap: true } },
                  yAxis: { type: 'value', axisLabel: { formatter: (v: number) => fmtK(v) } },
                  tooltip: { trigger: 'axis', formatter: (ps: unknown) => { const p = sc[(ps as { dataIndex: number }[])[0].dataIndex]; return `${tipRow(C.ai, 'Safety stock', fmtInt(p.safetyStock))}${tipRow(C.ai, 'Z', p.z.toFixed(3))}${tipRow(C.ai, 'Holding cost of SS', `${fmtMoney(p.safetyHoldingPerDay)}/day`)}`; } },
                  series: [{ type: 'line', data: sc.map((p) => p.safetyStock), lineStyle: { color: C.ai, width: 2 }, itemStyle: { color: C.ai }, symbolSize: 7 }],
                }} />} />
                <ChartCard title="Target vs service actually achieved" subtitle="Monte Carlo at each formula ROP · diagonal = perfect calibration" chart={<><EChart ariaLabel="Target service level versus Monte Carlo achieved service level" height={190} option={{
                  grid: { left: 52, right: 14, top: 12, bottom: 28 },
                  xAxis: { type: 'category', data: sc.map((p) => fmtPct(p.serviceLevel, 1)), axisLabel: { hideOverlap: true } },
                  yAxis: { type: 'value', min: 0.7, max: 1, axisLabel: { formatter: (v: number) => fmtPct(v, 0) } },
                  tooltip: { trigger: 'axis', valueFormatter: (v) => fmtPct(Number(v), 1) },
                  series: [
                    { name: 'Target', type: 'line', data: sc.map((p) => p.serviceLevel), symbol: 'none', lineStyle: { color: C.ink3, type: 'dashed', width: 1.5 } },
                    { name: 'Achieved (Monte Carlo)', type: 'line', data: sc.map((p) => p.mcAchieved), lineStyle: { color: C.rop, width: 2 }, itemStyle: { color: C.rop }, symbolSize: 7 },
                  ],
                }} /><div className="mt-1"><Legend items={[{ color: C.ink3, label: 'Target', dashed: true }, { color: C.rop, label: 'Achieved (Monte Carlo)' }]} /></div></>}>
                  <Explain>The gap between the lines is the formula's blind spot (skewed lead times, correlated misses). At a {fmtPct(sc[4].serviceLevel, 0)} target it delivers {fmtPct(sc[4].mcAchieved)}.</Explain>
                </ChartCard>
              </div>
            </div>

            <Panel className="mt-4" title="AI-driven sensitivity: how the optimal policy moves" subtitle="Each row re-runs the Bayesian optimiser (25 simulations) with one assumption changed"
              actions={<Button busy={opt.perturbProgress !== null} disabled={opt.running} onClick={() => void runPerturb()}><Play className="h-4 w-4" />{opt.perturb ? 'Re-run' : 'Run 7 optimisations'}</Button>}>
              {opt.perturbProgress && <div className="mb-3"><Progress value={opt.perturbProgress.frac} label={opt.perturbProgress.label} /></div>}
              {opt.perturb ? (
                <>
                  <DataTable caption="Optimal policy under perturbations" highlightRow={0}
                    columns={[{ key: 'l', label: 'If…' }, { key: 's', label: 'Optimal s*', num: true }, { key: 'ds', label: 'Δs*', num: true }, { key: 'q', label: 'Optimal Q*', num: true }, { key: 'c', label: 'Cost/day', num: true }, { key: 'f', label: 'Fill rate', num: true }]}
                    rows={opt.perturb.map((r) => r.best ? { l: r.label, s: fmtInt(r.best.s), ds: baseRow ? `${r.best.s - baseRow.s >= 0 ? '+' : ''}${fmtInt(r.best.s - baseRow.s)}` : '–', q: fmtInt(r.best.Q), c: fmtMoney(r.best.cost), f: fmtPct(r.best.metrics.fillRate, 2) } : { l: r.label, s: 'infeasible', ds: '', q: '', c: '', f: '' })} />
                  <Explain>Each answer comes from a full constrained optimisation, not a formula tweak. The optimum is fairly flat, so differences under about 2% in cost are within simulation noise. Large moves, such as a longer lead time or higher holding cost, are real.</Explain>
                </>
              ) : (
                <p className="text-[13px] text-ink-3">Runs seven complete optimisations in the background (about 10–20 seconds). The rest of the dashboard stays usable.</p>
              )}
            </Panel>
          </>
        );
      }}
    </WithData>
  );
}
