/**
 * Control Tower: the one-screen answer for an operations manager. When to reorder, how much
 * buffer to hold, what that protection really delivers, and how it compares with the baseline.
 */
import { AlertTriangle, ArrowRight, CheckCircle2, ShieldCheck } from 'lucide-react';
import { optHash, useApp } from '../state/store';
import { EChart, fmtInt, fmtMoney, fmtPct } from '../charts/EChart';
import { ddltOption, describeService, forecastOption } from '../charts/builders';
import { Badge, ChartCard, Explain, KpiTile, Legend, Panel } from '../ui/primitives';
import { C } from '../charts/tokens';
import { PageHeader, WithData, seriesTitle } from './common';
import { UpdateFeed } from '../ui/LiveFeed';

export function ControlTower() {
  const opt = useApp((s) => s.opt);
  const setPage = useApp((s) => s.setPage);
  const settings = useApp((s) => s.settings);
  return (
    <WithData>
      {(fb, rk) => {
        const ai = rk.comparison.ai, base = rk.comparison.baseline;
        const achieved = rk.mc.ai.achievedServiceAtFormulaRop;
        const next7 = fb.forecast.ensemble.slice(0, 7).reduce((a, b) => a + b, 0);
        // Only use an optimiser result computed with the current inputs
        const boBest = opt.forHash === optHash() ? opt.result?.best ?? null : null;
        const shortfall = settings.serviceLevel - achieved;
        const bl = rk.policies.find((p) => p.id === 'baseline')!.metrics;
        const mcp = rk.policies.find((p) => p.id === 'ai-mc')!.metrics;
        return (
          <>
            <PageHeader eyebrow={`Control tower · decision date ${fb.originDate}`} title={seriesTitle(fb)}>
              The reorder decision for {fb.info.family.toLowerCase()} at store {fb.info.store} ({fb.info.city}). It is driven by the AI demand forecast, a dynamic safety stock, and simulation-based optimisation.
              {rk.scenario.id !== 'base' && <> Active scenario: <Badge tone="rop">{rk.scenario.label}</Badge></>}
            </PageHeader>

            <Panel className="relative overflow-hidden">
              <div className="tape absolute inset-x-0 top-0 h-1 opacity-80" aria-hidden />
              <div className="grid gap-5 lg:grid-cols-[1.4fr_1fr]">
                <div>
                  <div className="text-[12px] font-semibold uppercase tracking-wide text-ink-3">Recommendation</div>
                  <p className="mt-2 text-[18px] leading-snug text-ink sm:text-[20px]">
                    Place a replenishment order when the inventory position falls to{' '}
                    <span className="num font-semibold text-rop">{fmtInt(boBest ? boBest.s : rk.mc.ai.mcRop)}</span> units
                    {boBest ? <> and order <span className="num font-semibold text-ink">{fmtInt(boBest.Q)}</span> units each time.</> : '.'}
                  </p>
                  <p className="mt-2 text-[13.5px] leading-relaxed text-ink-2">
                    {boBest
                      ? <>This is the Bayesian-optimised policy: it minimises expected cost ({fmtMoney(boBest.cost)}/day) while meeting your service, budget, capacity and order-frequency constraints.</>
                      : <>This is the Monte Carlo reorder point: the stock level that actually delivers {fmtPct(settings.serviceLevel)} protection once skewed lead times and correlated forecast misses are accounted for. Run the <button className="text-ai underline-offset-2 hover:underline" onClick={() => setPage('optimiser')}>Bayesian optimiser</button> to also choose the order quantity under constraints.</>}
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <Badge tone="rop">Slide formula ROP {fmtInt(ai.reorderPoint)}</Badge>
                    <Badge tone="base">Baseline ROP {fmtInt(base.reorderPoint)}</Badge>
                    <Badge tone="ai">Monte Carlo ROP {fmtInt(rk.mc.ai.mcRop)}</Badge>
                    {boBest && <Badge tone="good">Optimised s* {fmtInt(boBest.s)} · Q* {fmtInt(boBest.Q)}</Badge>}
                  </div>
                </div>
                <div className="space-y-2.5 rounded-xl border border-white/[0.06] bg-white/[0.02] p-4 text-[13px]">
                  <div className="flex items-start gap-2.5">
                    {shortfall > 0.01 ? <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" /> : <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-good" />}
                    <p className="text-ink-2">The slide formula targets <b className="text-ink">{fmtPct(settings.serviceLevel)}</b>. Monte Carlo shows it really delivers <b className="text-ink">{fmtPct(achieved)}</b>{shortfall > 0.01 ? ', so the formula under-protects this item.' : '.'}</p>
                  </div>
                  <div className="flex items-start gap-2.5">
                    <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-ai" />
                    <p className="text-ink-2">{fmtPct(rk.sensitivity.varianceSplit.leadTime, 0)} of the risk comes from <b className="text-ink">supplier lead-time variability</b>, which the baseline method ignores.</p>
                  </div>
                  {rk.errorPersistence.ratio > 1.1 && (
                    <div className="flex items-start gap-2.5">
                      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" />
                      <p className="text-ink-2">Forecast misses persist across days: the {rk.errorPersistence.L}-day error is ×{rk.errorPersistence.ratio.toFixed(2)} wider than the formula assumes. The Monte Carlo block bootstrap captures this.</p>
                    </div>
                  )}
                </div>
              </div>
            </Panel>

            <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
              <KpiTile label="AI reorder point" value={ai.reorderPoint} format={fmtInt} accent="rop" sub={<>D̄·L {fmtInt(ai.cycleStock)} + SS {fmtInt(ai.safetyStock)}</>} />
              <KpiTile label="AI safety stock" value={ai.safetyStock} format={fmtInt} accent="ai" sub={<>Z {ai.z.toFixed(2)} × σ<sub>DDLT</sub> {fmtInt(ai.sdDdlt)}</>} />
              <KpiTile label="Real service at AI ROP" value={achieved} format={(v) => fmtPct(v)} accent={shortfall > 0.01 ? 'bad' : 'good'} sub={<>Stockout {describeService(achieved)}</>} />
              <KpiTile label="Baseline reorder point" value={base.reorderPoint} format={fmtInt} accent="base" sub={<>Real service {fmtPct(1 - rk.mc.pStockoutBaseline)}</>} />
              <KpiTile label="Forecast, next 7 days" value={next7} format={fmtInt} accent="ai" sub={<>units · σ<sub>D</sub> error {fmtInt(fb.forecast.residualSd)}/day</>} />
              <KpiTile label="Cost/day: baseline → AI" value={mcp.totalCostPerDay} format={(v) => fmtMoney(v, 0)} accent="good" sub={<>from {fmtMoney(bl.totalCostPerDay, 0)} ({fmtPct(1 - mcp.totalCostPerDay / bl.totalCostPerDay, 0)} lower, simulated)</>} />
            </div>

            <div className="mt-4 grid gap-4 xl:grid-cols-[1.5fr_1fr]">
              <ChartCard title="Demand: last 8 weeks and the next 28 days" subtitle="AI ensemble forecast with 80% / 95% conformal bands" actions={<button onClick={() => setPage('forecast')} className="inline-flex items-center gap-1 text-[12.5px] text-ai hover:underline">Forecast details <ArrowRight className="h-3.5 w-3.5" /></button>}
                chart={<><EChart ariaLabel="Actual demand for the last 8 weeks and AI forecast for the next 28 days with uncertainty bands" height={280} option={forecastOption(fb, { historyDays: 56, models: ['ensemble'], showBacktest: false, showBands: true })} /><div className="mt-2"><Legend items={[{ color: C.actual, label: 'Actual' }, { color: C.ai, label: 'AI ensemble forecast' }, { color: 'rgba(57,135,229,0.35)', label: '80% / 95% band', area: true }]} /></div></>}>
                <Explain>Weekends sell far more than weekdays, and the forecast knows that. A reorder decision taken on a Wednesday has to cover the weekend peak, which a single historical average hides.</Explain>
              </ChartCard>
              <ChartCard title="Demand during the lead time" subtitle={`${(settings.mcRuns / 1000).toFixed(0)}k Monte Carlo scenarios · red = stockout`} actions={<button onClick={() => setPage('risk')} className="inline-flex items-center gap-1 text-[12.5px] text-ai hover:underline">Risk details <ArrowRight className="h-3.5 w-3.5" /></button>}
                chart={<EChart ariaLabel="Distribution of demand during the supplier lead time, with the AI reorder point" height={280} option={ddltOption(rk, true)} />}>
                <Explain>Each bar is a share of simulated futures. If demand before the delivery arrives exceeds the reorder point (the red tail), shelves run empty. The tail holds <b className="text-ink">{fmtPct(rk.mc.ai.pStockoutAtFormulaRop)}</b> of the futures.</Explain>
              </ChartCard>
            </div>
            <div className="mt-4"><UpdateFeed limit={5} /></div>
          </>
        );
      }}
    </WithData>
  );
}
