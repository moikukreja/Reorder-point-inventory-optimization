/**
 * Baseline vs AI: the "enhancement" the exercise objective asks for. The class-notebook reorder
 * point, the AI formula, the Monte Carlo ROP and (if run) the Bayesian-optimised policy are
 * simulated against identical demand.
 */
import { optHash, useApp } from '../state/store';
import { EChart, fmtInt, fmtMoney, fmtPct } from '../charts/EChart';
import { sawtoothOption } from '../charts/builders';
import { C } from '../charts/tokens';
import { Badge, ChartCard, DataTable, Explain, Legend, Panel } from '../ui/primitives';
import type { RiskBundle } from '../engine/types';
import { PageHeader, WithData, seriesTitle } from './common';

interface Row { label: string; color: string; s: number; Q: number; fill: number; csl: number; inv: number; orders: number; cost: number; lost: number }

function metricBars(rows: Row[], pick: (r: Row) => number, fmt: (v: number) => string, label: string, min?: number) {
  return {
    grid: { left: 10, right: 54, top: 6, bottom: 6, containLabel: true },
    xAxis: { type: 'value' as const, show: false, min },
    yAxis: { type: 'category' as const, inverse: true, data: rows.map((r) => r.label), axisLabel: { color: C.ink2, fontSize: 11 } },
    tooltip: { trigger: 'axis' as const, axisPointer: { type: 'shadow' as const }, valueFormatter: (v: unknown) => fmt(Number(v)) },
    series: [{ name: label, type: 'bar' as const, data: rows.map((r) => ({ value: pick(r), itemStyle: { color: r.color, borderRadius: [0, 4, 4, 0] } })), barWidth: '58%', label: { show: true, position: 'right' as const, color: C.ink, fontSize: 11, formatter: (p: { value?: unknown }) => fmt(Number(p.value)) } }],
  };
}

export function Compare() {
  const opt = useApp((s) => s.opt);
  const settings = useApp((s) => s.settings);
  return (
    <WithData>
      {(fb, rk) => {
        const boBest = opt.forHash === optHash() ? opt.result?.best ?? null : null; // ignore stale results
        const P = Object.fromEntries(rk.policies.map((p) => [p.id, p]));
        const rows: Row[] = [
          { label: 'Baseline (notebook)', color: C.baseline, ...pick(P.baseline) },
          { label: 'AI formula', color: '#2a78d6', ...pick(P['ai-formula']) },
          { label: 'AI Monte Carlo', color: C.ai, ...pick(P['ai-mc']) },
          ...(boBest ? [{ label: 'AI optimised (BO)', color: '#86b6ef', s: boBest.s, Q: boBest.Q, fill: boBest.metrics.fillRate, csl: boBest.metrics.cycleServiceLevel, inv: boBest.metrics.avgOnHand, orders: boBest.metrics.ordersPer30, cost: boBest.cost, lost: boBest.metrics.lostUnitsPerDay }] : []),
        ];
        const b = rows[0], best = rows.reduce((m, r) => (r.cost < m.cost ? r : m), rows[0]);
        return (
          <>
            <PageHeader eyebrow="Objective · the enhancement" title={`Baseline vs AI: ${seriesTitle(fb)}`}>
              The same simulated future (demand from the AI forecast plus real forecast errors, random supplier lead times, lost sales) is run under each policy, so differences come from the policy, not from luck.
            </PageHeader>
            <Panel title="What changes between the methods">
              <DataTable caption="Methods compared" columns={[{ key: 'k', label: '' }, { key: 'b', label: 'Baseline (class notebook)' }, { key: 'a', label: 'AI-enhanced' }]} rows={[
                { k: 'Average demand D̄', b: 'Historical mean (last 365 days)', a: 'Forecast for the coming lead time (4-model adaptive ensemble)' },
                { k: 'Demand variability σ_D', b: 'SD of raw daily sales', a: 'SD of out-of-sample forecast errors' },
                { k: 'Lead-time variability σL', b: 'Ignored (SS = Z·σ·√L)', a: 'Included: slide formula, plus Monte Carlo with skewed lead times' },
                { k: 'Updates', b: 'Static', a: 'Recomputed whenever new data arrives' },
                { k: 'Choosing the policy', b: 'Manual one-at-a-time sweeps', a: 'Bayesian optimisation of (s, Q) under constraints' },
              ]} />
            </Panel>

            <div className="mt-4 grid gap-4 md:grid-cols-2 2xl:grid-cols-4">
              <ChartCard title="Expected cost per day" subtitle="Holding + ordering + lost-sales penalty" chart={<EChart ariaLabel="Expected cost per day by policy" height={170} option={metricBars(rows, (r) => r.cost, (v) => fmtMoney(v, 0), 'Cost/day')} />} />
              <ChartCard title="Fill rate" subtitle="Share of demand served from stock" chart={<EChart ariaLabel="Fill rate by policy" height={170} option={metricBars(rows, (r) => r.fill, (v) => fmtPct(v, 2), 'Fill rate', 0.9)} />} />
              <ChartCard title="Cycle service level" subtitle="Replenishment cycles without a stockout" chart={<EChart ariaLabel="Cycle service level by policy" height={170} option={metricBars(rows, (r) => r.csl, (v) => fmtPct(v, 1), 'Cycle SL', 0)} />} />
              <ChartCard title="Average stock on hand" subtitle="Units (working capital tied up)" chart={<EChart ariaLabel="Average stock on hand by policy" height={170} option={metricBars(rows, (r) => r.inv, fmtInt, 'Avg stock')} />} />
            </div>

            <Panel className="mt-4" title="Policy results" subtitle={`${settings.simReps} replications × 182 days per policy, scenario: ${rk.scenario.label}`}>
              <DataTable caption="Policy comparison" highlightRow={rows.indexOf(best)} columns={[{ key: 'l', label: 'Policy' }, { key: 's', label: 's', num: true }, { key: 'q', label: 'Q', num: true }, { key: 'f', label: 'Fill rate', num: true }, { key: 'c', label: 'Cycle SL', num: true }, { key: 'lo', label: 'Lost units/day', num: true }, { key: 'i', label: 'Avg stock', num: true }, { key: 'o', label: 'Orders/30d', num: true }, { key: 'k', label: 'Cost/day', num: true }, { key: 'd', label: 'vs baseline', num: true }]}
                rows={rows.map((r) => ({ l: <span className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: r.color }} />{r.label}{r === best && <Badge tone="good">lowest cost</Badge>}</span>, s: fmtInt(r.s), q: fmtInt(r.Q), f: fmtPct(r.fill, 2), c: fmtPct(r.csl), lo: fmtInt(r.lost), i: fmtInt(r.inv), o: r.orders.toFixed(1), k: fmtMoney(r.cost), d: r === b ? '–' : `${fmtPct(r.cost / b.cost - 1, 0)}` }))} />
              <Explain>
                Against the class-notebook method, <b className="text-ink">{best.label}</b> cuts expected cost by <b className="text-ink">{fmtPct(1 - best.cost / b.cost, 0)}</b> and lifts the fill rate from {fmtPct(b.fill, 2)} to {fmtPct(best.fill, 2)}. Lost sales fall from {fmtInt(b.lost)} to {fmtInt(best.lost)} units a day.
                {!boBest && <> Run the Bayesian optimiser to add the optimised policy to this comparison.</>}
              </Explain>
            </Panel>

            <div className="mt-4 grid gap-4 xl:grid-cols-2">
              <ChartCard title="Baseline: one simulated path" subtitle={`s = ${fmtInt(P.baseline.s)}, Q = ${fmtInt(P.baseline.Q)}`} chart={<EChart ariaLabel="Simulated inventory for the baseline policy" height={240} option={sawtoothOption(P.baseline, C.baseline)} />} />
              <ChartCard title="AI Monte Carlo ROP: the same path" subtitle={`s = ${fmtInt(P['ai-mc'].s)}, Q = ${fmtInt(P['ai-mc'].Q)}`} chart={<EChart ariaLabel="Simulated inventory for the AI Monte Carlo policy" height={240} option={sawtoothOption(P['ai-mc'], C.ai)} />} />
            </div>
            <HistoricalReplay rk={rk} />
          </>
        );
      }}
    </WithData>
  );
}

function pick(p: { s: number; Q: number; metrics: { fillRate: number; cycleServiceLevel: number; avgOnHand: number; ordersPer30: number; totalCostPerDay: number; lostUnitsPerDay: number } }) {
  return { s: p.s, Q: p.Q, fill: p.metrics.fillRate, csl: p.metrics.cycleServiceLevel, inv: p.metrics.avgOnHand, orders: p.metrics.ordersPer30, cost: p.metrics.totalCostPerDay, lost: p.metrics.lostUnitsPerDay };
}

const REPLAY_COLORS = { baseline: C.baseline, 'ai-formula': '#2a78d6', 'ai-mc': C.ai } as const;

/** Both policies re-run over ACTUAL demand, recomputing the reorder point every 14 days with only the data known then. */
function HistoricalReplay({ rk }: { rk: RiskBundle }) {
  const r = rk.replay;
  const base = r.policies[0];
  const best = r.policies.reduce((m, p) => (p.metrics.totalCostPerDay < m.metrics.totalCostPerDay ? p : m), r.policies[0]);
  return (
    <>
      <div className="mt-6 mb-3 max-w-4xl">
        <div className="text-[11.5px] font-semibold uppercase tracking-[0.12em] text-ai">Historical replay · real demand</div>
        <h2 className="mt-1 text-[19px] font-semibold text-ink">What would have happened, {r.dates[0]} → {r.dates.at(-1)}</h2>
        <p className="mt-1.5 text-[13.5px] leading-relaxed text-ink-2">Every 14 days each method recomputed its reorder point using only the data known at that moment. Then the policies were run against the actual daily sales. Only supplier lead times are random, and every method faces the same delays.</p>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <ChartCard title="Reorder point used each day" subtitle="Baseline barely moves; the AI reorder point follows the forecast and recent forecast errors" chart={<><EChart ariaLabel="Reorder point schedules over the replay period" height={240} option={{
          grid: { left: 52, right: 16, top: 14, bottom: 30 },
          xAxis: { type: 'category', data: r.dates, axisLabel: { formatter: (v: string) => v.slice(5), hideOverlap: true } },
          yAxis: { type: 'value', scale: true, axisLabel: { formatter: (v: number) => `${Math.round(v / 1000)}k` } },
          tooltip: { trigger: 'axis', valueFormatter: (v) => fmtInt(Number(v)) },
          series: r.policies.map((p) => ({ name: p.label, type: 'line', step: 'end', data: p.sByDay, symbol: 'none', lineStyle: { color: REPLAY_COLORS[p.id], width: 2, type: p.id === 'baseline' ? 'dashed' : 'solid' } })),
        }} /><div className="mt-2"><Legend items={r.policies.map((p) => ({ color: REPLAY_COLORS[p.id], label: p.label, dashed: p.id === 'baseline' }))} /></div></>} />
        <ChartCard title="Actual demand in the replay window" subtitle="Units/day, as recorded" chart={<EChart ariaLabel="Actual daily demand in the replay window" height={262} option={{
          grid: { left: 52, right: 16, top: 14, bottom: 30 },
          xAxis: { type: 'category', data: r.dates, axisLabel: { formatter: (v: string) => v.slice(5), hideOverlap: true } },
          yAxis: { type: 'value', axisLabel: { formatter: (v: number) => `${Math.round(v / 1000)}k` } },
          tooltip: { trigger: 'axis', valueFormatter: (v) => fmtInt(Number(v)) },
          series: [{ name: 'Actual', type: 'line', data: r.actual, symbol: 'none', lineStyle: { color: C.actual, width: 1.6 } }],
        }} />} />
      </div>
      <Panel className="mt-4" title="Replay results on actual demand" subtitle={`${r.dates.length} days, Q = ${fmtInt(r.Q)} for every method, lost sales`}>
        <DataTable caption="Historical replay results" highlightRow={r.policies.indexOf(best)} columns={[{ key: 'l', label: 'Policy' }, { key: 'f', label: 'Fill rate', num: true }, { key: 'c', label: 'Cycle SL', num: true }, { key: 'lo', label: 'Lost units/day', num: true }, { key: 'i', label: 'Avg stock', num: true }, { key: 'k', label: 'Cost/day', num: true }, { key: 'd', label: 'vs baseline', num: true }]}
          rows={r.policies.map((p) => ({ l: p.label, f: fmtPct(p.metrics.fillRate, 2), c: fmtPct(p.metrics.cycleServiceLevel), lo: fmtInt(p.metrics.lostUnitsPerDay), i: fmtInt(p.metrics.avgOnHand), k: fmtMoney(p.metrics.totalCostPerDay), d: p === base ? '–' : fmtPct(p.metrics.totalCostPerDay / base.metrics.totalCostPerDay - 1, 0) }))} />
        <Explain>On real 2017 demand, <b className="text-ink">{best.label}</b> {best === base ? 'did best in this window.' : <>lost {fmtInt(best.metrics.lostUnitsPerDay)} units a day against {fmtInt(base.metrics.lostUnitsPerDay)} for the baseline, and cost {fmtPct(1 - best.metrics.totalCostPerDay / base.metrics.totalCostPerDay, 0)} less per day.</>} Unlike the forward simulation above, this uses only what actually happened. Add new days (or run the replay on the Data page) and this window moves forward with the data.</Explain>
      </Panel>
    </>
  );
}
