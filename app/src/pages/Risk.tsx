/**
 * Step 2, Dynamic safety stock: the slide formula with live substituted values, the Monte Carlo
 * risk picture (lead-time and demand-during-lead-time distributions), risk scenarios, and
 * inventory simulations against the reorder point.
 */
import { useState } from 'react';
import { useApp } from '../state/store';
import { EChart, fmtInt, fmtMoney, fmtPct } from '../charts/EChart';
import { ddltOption, describeService, sawtoothOption } from '../charts/builders';
import { C, withAlpha } from '../charts/tokens';
import type { SafetyStockResult } from '../core/safetyStock';
import type { ScenarioId } from '../core/scenarios';
import { Badge, ChartCard, DataTable, Explain, Legend, Note, Panel, Segmented } from '../ui/primitives';
import { PageHeader, WithData, seriesTitle } from './common';

function FormulaCard({ r, title, tone, note }: { r: SafetyStockResult & { source: string }; title: string; tone: 'ai' | 'base'; note: string }) {
  const f1 = (v: number) => v.toLocaleString('en-US', { maximumFractionDigits: 1 });
  return (
    <div className={`rounded-xl border p-4 ${tone === 'ai' ? 'border-ai/30 bg-ai/[0.05]' : 'border-base/30 bg-base/[0.05]'}`}>
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-[14px] font-semibold text-ink">{title}</h3>
        <Badge tone={tone}>{tone === 'ai' ? 'AI-enhanced' : 'Class notebook'}</Badge>
      </div>
      <p className="mt-1 text-[12px] text-ink-3">{r.source}</p>
      <div className="num mt-3 space-y-1.5 text-[12.5px] leading-relaxed text-ink-2">
        <div>Z = Φ⁻¹({fmtPct(r.serviceLevel, 1)}) = <b className="text-ink">{r.z.toFixed(4)}</b></div>
        <div>D̄ = <b className="text-ink">{f1(r.dBar)}</b> u/d · σ<sub>D</sub> = <b className="text-ink">{f1(r.sdD)}</b> u/d · L = <b className="text-ink">{r.L}</b> d · σ<sub>L</sub> = <b className="text-ink">{r.sdL}</b> d</div>
        <div className="pt-1">SS = {r.z.toFixed(4)} × √({f1(r.sdD)}² × {r.L} + {f1(r.dBar)}² × {r.sdL}²)</div>
        <div className="pl-7">= {r.z.toFixed(4)} × √({fmtInt(r.varDemand)} + {fmtInt(r.varLeadTime)})</div>
        <div className="pl-7">= {r.z.toFixed(4)} × {f1(r.sdDdlt)} = <b className="text-[15px] text-ink">{fmtInt(r.safetyStock)}</b> units</div>
        <div className="pt-1">ROP = D̄·L + SS = {fmtInt(r.cycleStock)} + {fmtInt(r.safetyStock)} = <b className="text-[15px] text-rop">{fmtInt(r.reorderPoint)}</b> units</div>
      </div>
      <p className="mt-3 text-[12px] leading-relaxed text-ink-3">{note}</p>
    </div>
  );
}

export function Risk() {
  const settings = useApp((s) => s.settings);
  const update = useApp((s) => s.updateSettings);
  const [which, setWhich] = useState<'baseline' | 'ai-formula' | 'ai-mc'>('ai-formula');
  return (
    <WithData>
      {(fb, rk) => {
        const { ai, baseline } = rk.comparison;
        const vs = rk.sensitivity.varianceSplit;
        const lt = rk.mc.leadTimeHistogram;
        const ltTotal = lt.counts.reduce((a, b) => a + b, 0);
        const pol = rk.policies.find((p) => p.id === which)!;
        const polColor = which === 'baseline' ? C.baseline : C.ai;
        return (
          <>
            <PageHeader eyebrow="Step 2 · Dynamic safety stock" title={`Safety stock & risk: ${seriesTitle(fb)}`}>
              The slide's formula combines demand variability with lead-time uncertainty. Monte Carlo then tests what that buffer actually achieves when lead times are skewed and forecast misses run for several days.
              {rk.scenario.id !== 'base' && <> Scenario: <Badge tone="rop">{rk.scenario.label}</Badge></>}
            </PageHeader>

            <Panel title="Safety Stock = Z · √( σ_D² · L + D̄² · σ_L² )" subtitle="Live substitution with the current inputs (units/day and days throughout)">
              <div className="grid gap-4 lg:grid-cols-2">
                <FormulaCard r={baseline} title="Baseline" tone="base" note="The class notebook method: the historical average and raw demand SD, and lead-time variability is ignored (σL = 0). It never changes unless someone re-runs it." />
                <FormulaCard r={ai} title="AI-enhanced (slide formula)" tone="ai" note="D̄ comes from the forecast for the coming lead time, and σ_D is the forecast-error SD (the predictable weekly pattern is removed). σL is included as the slide requires. It is recomputed whenever data or settings change." />
              </div>
              <div className="mt-4">
                <div className="mb-1.5 flex justify-between text-[12px] text-ink-3"><span>Where the lead-time-demand variance comes from</span><span className="num">σ<sub>D</sub>²·L {fmtPct(vs.demand)} · D̄²·σ<sub>L</sub>² {fmtPct(vs.leadTime)}</span></div>
                <div className="flex h-3 overflow-hidden rounded-full" role="img" aria-label={`Demand uncertainty ${fmtPct(vs.demand)}, lead-time uncertainty ${fmtPct(vs.leadTime)}`}>
                  <div style={{ width: `${vs.demand * 100}%`, background: C.ai }} />
                  <div style={{ width: `${vs.leadTime * 100}%`, background: C.rop, marginLeft: 2 }} />
                </div>
                <div className="mt-2"><Legend items={[{ color: C.ai, label: 'Demand uncertainty (σ_D²·L)', area: true }, { color: C.rop, label: 'Lead-time uncertainty (D̄²·σ_L²)', area: true }]} /></div>
              </div>
              <Explain>
                {vs.leadTime > 0.5 ? <>Most of the risk here is the supplier, not the customer: {fmtPct(vs.leadTime, 0)} of the uncertainty comes from late or erratic deliveries. Cutting σL by negotiating delivery windows would shrink the safety stock more than any forecasting improvement.</> : <>Most of the risk here is demand uncertainty ({fmtPct(vs.demand, 0)}). Better forecasts shrink the buffer directly.</>}
              </Explain>
            </Panel>

            <div className="mt-4 grid gap-4 xl:grid-cols-[1.6fr_1fr]">
              <ChartCard title="Monte Carlo: demand during the lead time" subtitle={`${(settings.mcRuns / 1000).toFixed(0)}k scenarios: lead time drawn from a ${rk.scenario.leadTime.dist} distribution, daily demand = forecast + resampled real forecast errors${settings.blockBootstrap ? ' (block bootstrap)' : ''}`}
                table={<DataTable caption="DDLT quantiles" columns={[{ key: 'p', label: 'Quantile' }, { key: 'v', label: 'Units', num: true }]} rows={rk.mc.quantiles.map((q) => ({ p: fmtPct(q.p, 1), v: fmtInt(q.v) }))} />}
                chart={<><EChart ariaLabel="Histogram of simulated demand during lead time with baseline, formula and Monte Carlo reorder points" height={320} option={ddltOption(rk)} /><div className="mt-2"><Legend items={[{ color: withAlpha(C.ai, 0.75), label: 'Covered by the AI ROP', area: true }, { color: C.stockout, label: 'Stockout tail', area: true }, { color: C.baseline, label: 'Baseline ROP', dashed: true }, { color: C.rop, label: 'AI ROP (formula)' }, { color: C.ai, label: 'Monte Carlo ROP', dashed: true }]} /></div></>}>
                <Explain>
                  At the AI formula ROP of <b className="text-ink">{fmtInt(ai.reorderPoint)}</b>, <b className="text-ink">{fmtPct(rk.mc.ai.pStockoutAtFormulaRop)}</b> of simulated replenishment cycles run out before the delivery arrives, so the shelf goes empty {describeService(rk.mc.ai.achievedServiceAtFormulaRop)}. The baseline ROP runs out in <b className="text-ink">{fmtPct(rk.mc.pStockoutBaseline)}</b>. Holding <b className="text-ink">{fmtInt(rk.mc.ai.mcRop)}</b> instead delivers exactly {fmtPct(settings.serviceLevel)}. Expected shortfall per cycle at the formula ROP: {fmtInt(rk.mc.ai.expectedShortageAtFormulaRop)} units.
                </Explain>
              </ChartCard>
              <div className="grid gap-4">
                <ChartCard title="Lead-time distribution" subtitle={`${rk.scenario.leadTime.dist}, mean ${rk.scenario.leadTime.mean} d, σ ${rk.scenario.leadTime.sd} d`} chart={<EChart ariaLabel="Distribution of simulated supplier lead times" height={180} option={{
                  grid: { left: 40, right: 12, top: 8, bottom: 30 },
                  xAxis: { type: 'value', min: lt.edges[0], max: lt.edges.at(-1), name: 'days', nameLocation: 'middle', nameGap: 22, splitLine: { show: false } },
                  yAxis: { type: 'value', axisLabel: { show: false } },
                  tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: (ps: unknown) => { const i = (ps as { dataIndex: number }[])[0].dataIndex; return `${lt.edges[i].toFixed(1)}–${lt.edges[i + 1].toFixed(1)} days: ${((lt.counts[i] / ltTotal) * 100).toFixed(1)}%`; } },
                  series: [{ type: 'bar', barWidth: '94%', data: Array.from(lt.counts, (c, i) => [(lt.edges[i] + lt.edges[i + 1]) / 2, c]), itemStyle: { color: withAlpha(C.rop, 0.8), borderRadius: [3, 3, 0, 0] } }],
                }} />} />
                <Panel title="Formula vs Monte Carlo">
                  <DataTable caption="Formula vs Monte Carlo" columns={[{ key: 'k', label: '' }, { key: 'f', label: 'Formula', num: true }, { key: 'm', label: 'Monte Carlo', num: true }]} rows={[
                    { k: 'Mean demand in lead time', f: fmtInt(ai.cycleStock), m: fmtInt(rk.mc.mean) },
                    { k: 'SD of demand in lead time', f: fmtInt(ai.sdDdlt), m: fmtInt(rk.mc.sd) },
                    { k: `ROP for ${fmtPct(settings.serviceLevel)}`, f: fmtInt(ai.reorderPoint), m: fmtInt(rk.mc.ai.mcRop) },
                    { k: 'Safety stock', f: fmtInt(ai.safetyStock), m: fmtInt(rk.mc.ai.mcSafetyStock) },
                    { k: 'Service actually achieved', f: fmtPct(rk.mc.ai.achievedServiceAtFormulaRop), m: fmtPct(settings.serviceLevel) },
                  ]} />
                  {rk.errorPersistence.ratio > 1.1 && <div className="mt-3"><Note tone="warn">Forecast errors persist across days: the realised {rk.errorPersistence.L}-day error SD is ×{rk.errorPersistence.ratio.toFixed(2)} what independent daily errors imply ({rk.errorPersistence.windows} windows). The formula assumes independence, so it under-states risk.</Note></div>}
                </Panel>
              </div>
            </div>

            <Panel className="mt-4" title="Risk scenarios" subtitle="Each row re-runs the formula and a Monte Carlo. The last column keeps today's base-case AI ROP and asks how often it would fail.">
              <DataTable caption="Risk scenarios" highlightRow={rk.scenarioTable.findIndex((r) => r.id === settings.scenario)}
                columns={[{ key: 'l', label: 'Scenario' }, { key: 'f', label: 'Formula ROP', num: true }, { key: 'm', label: 'Monte Carlo ROP', num: true }, { key: 'p', label: 'P(stockout) with today’s ROP', num: true }, { key: 'a', label: '' }]}
                rows={rk.scenarioTable.map((r) => ({ l: r.label, f: fmtInt(r.formulaRop), m: fmtInt(r.mcRop), p: <span className={r.pAtCurrent > 0.2 ? 'text-[#e66767]' : ''}>{fmtPct(r.pAtCurrent)}</span>, a: r.id === settings.scenario ? <Badge tone="rop">active</Badge> : <button className="text-[12px] text-ai hover:underline" onClick={() => update({ scenario: r.id as ScenarioId })}>Apply</button> }))} />
              <Explain>A two-day supplier delay is the most dangerous single event here: the buffer sized for today would fail in about {fmtPct(rk.scenarioTable.find((r) => r.id === 'supplier-delay')!.pAtCurrent, 0)} of cycles. Applying a scenario updates every page, including the optimiser.</Explain>
            </Panel>

            <ChartCard className="mt-4" title="Inventory simulation against the reorder point" subtitle={`One simulated path, ${pol.trace.length} days, lost sales. Orders of Q = EOQ ${fmtInt(rk.eoq)} units are placed when the inventory position drops to s.`}
              actions={<Segmented label="Policy" value={which} onChange={setWhich} options={[{ value: 'baseline', label: 'Baseline' }, { value: 'ai-formula', label: 'AI formula' }, { value: 'ai-mc', label: 'AI Monte Carlo' }]} />}
              table={<DataTable caption="Policy simulation results" columns={[{ key: 'l', label: 'Policy' }, { key: 's', label: 's', num: true }, { key: 'f', label: 'Fill rate', num: true }, { key: 'c', label: 'Cycle SL', num: true }, { key: 'i', label: 'Avg stock', num: true }, { key: 'o', label: 'Orders/30d', num: true }, { key: 'k', label: 'Cost/day', num: true }]} rows={rk.policies.map((p) => ({ l: p.label, s: fmtInt(p.s), f: fmtPct(p.metrics.fillRate, 2), c: fmtPct(p.metrics.cycleServiceLevel), i: fmtInt(p.metrics.avgOnHand), o: p.metrics.ordersPer30.toFixed(1), k: fmtMoney(p.metrics.totalCostPerDay) }))} />}
              chart={<><EChart ariaLabel={`Simulated on-hand inventory for the ${pol.label} policy`} height={300} option={sawtoothOption(pol, polColor)} />
                <div className="mt-2 flex flex-wrap items-center justify-between gap-3"><Legend items={[{ color: polColor, label: 'On hand' }, { color: C.rop, label: 'Reorder point / order placed', dashed: true }, { color: C.stockout, label: 'Stockout day', area: true }]} />
                  <div className="num flex flex-wrap gap-3 text-[12px] text-ink-2"><span>Fill {fmtPct(pol.metrics.fillRate, 2)}</span><span>Cycle SL {fmtPct(pol.metrics.cycleServiceLevel)}</span><span>Avg stock {fmtInt(pol.metrics.avgOnHand)}</span><span>{fmtMoney(pol.metrics.totalCostPerDay)}/day</span></div></div></>}>
              <Explain>Inventory falls with daily sales and jumps when a delivery arrives. Because stock is reviewed once a day and a day's demand is large, stock often drops well <i>below</i> s before the order goes out. That's one reason the simulated cycle service level ({fmtPct(pol.metrics.cycleServiceLevel)}) is below the formula's promise. The Bayesian optimiser tunes s and Q against exactly this simulation.</Explain>
            </ChartCard>
          </>
        );
      }}
    </WithData>
  );
}
