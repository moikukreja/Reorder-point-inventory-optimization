/**
 * Step 1, Demand forecasting: forecast vs actual with uncertainty bands, the model
 * leaderboard, adaptive ensemble weights, and the demand drivers the models learned.
 */
import { useState } from 'react';
import { MODEL_LABELS, type ModelName } from '../core/backtest';
import { EChart, fmtInt, fmtPct } from '../charts/EChart';
import { forecastOption } from '../charts/builders';
import { C, MODEL_COLORS, withAlpha } from '../charts/tokens';
import { Badge, ChartCard, DataTable, Explain, Legend, Note, Panel, Segmented, Toggle } from '../ui/primitives';
import { PageHeader, WithData, seriesTitle } from './common';

const DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const SHORT: Record<ModelName, string> = { snaive: 'Seasonal naive', holtwinters: 'Holt-Winters', ridge: 'Ridge', lstm: 'LSTM', ensemble: 'AI ensemble' };

export function Forecast() {
  const [days, setDays] = useState<'56' | '140'>('140');
  const [showModels, setShowModels] = useState(false);
  const [showBacktest, setShowBacktest] = useState(true);
  return (
    <WithData>
      {(fb) => {
        const lb = fb.leaderboard;
        const models = lb.models;
        const best = models.reduce((b, m) => (lb.board[m]!.wape < lb.board[b]!.wape ? m : b), models[0]);
        const sn = lb.board.snaive!;
        const ens = lb.board.ensemble!;
        const chartModels: ModelName[] = showModels ? models.filter((m) => m !== 'snaive') : ['ensemble'];
        const legend = [{ color: C.actual, label: 'Actual' }, ...chartModels.map((m) => ({ color: MODEL_COLORS[m], label: SHORT[m] })), ...(showBacktest ? [{ color: C.ai, label: 'Out-of-sample backtest', dashed: true }] : []), { color: 'rgba(57,135,229,0.35)', label: '80% / 95% band', area: true }];
        return (
          <>
            <PageHeader eyebrow="Step 1 · Demand forecasting" title={`Forecasting ${seriesTitle(fb)}`}>
              Four models are compared on the same rolling-origin backtest: Holt-Winters (time series), Ridge regression on promotions, holidays, paydays and oil, an LSTM, and an adaptive ensemble. Accuracy is measured only on data the models never saw.
            </PageHeader>

            <ChartCard title="Forecast vs actual" subtitle={`Dashed: forecasts made in the backtest before the actuals were known (test window ${fb.testWindow.from} → ${fb.testWindow.to}). Solid: the next 28 days.`}
              actions={<><Segmented label="History shown" value={days} onChange={setDays} options={[{ value: '56', label: '8 wk' }, { value: '140', label: '20 wk' }]} /></>}
              chart={<><EChart ariaLabel="Actual demand, backtest forecasts and 28-day forecast with uncertainty bands" height={380} option={forecastOption(fb, { historyDays: Number(days), models: chartModels, showBacktest, showBands: true })} />
                <div className="mt-3 flex flex-wrap items-center justify-between gap-3"><Legend items={legend} /><div className="flex gap-4"><Toggle label="Backtest" checked={showBacktest} onChange={setShowBacktest} /><Toggle label="Individual models" checked={showModels} onChange={setShowModels} /></div></div></>}>
              <Explain>
                The AI ensemble misses actual daily demand by <b className="text-ink">{fmtPct(ens.wape / 100)}</b> on average (WAPE), against <b className="text-ink">{fmtPct(sn.wape / 100)}</b> for "same day last week". The 80% band contained {fmtPct(lb.coverage.ensemble!['0.8'], 0)} of actual days in the test window. Use the band, not the line, when deciding how much can go wrong.
              </Explain>
            </ChartCard>

            <div className="mt-4 grid gap-4 xl:grid-cols-[1.35fr_1fr]">
              <Panel title="Model leaderboard" subtitle={`${lb.nDays} out-of-sample days (6 origins × 14-day horizon). Lower is better; MASE < 1 beats the seasonal-naive benchmark.`}>
                <DataTable caption="Forecast accuracy by model" highlightRow={models.indexOf(best)}
                  columns={[{ key: 'm', label: 'Model' }, { key: 'mae', label: 'MAE', num: true }, { key: 'rmse', label: 'RMSE', num: true }, { key: 'wape', label: 'WAPE', num: true }, { key: 'mase', label: 'MASE', num: true }, { key: 'bias', label: 'Bias', num: true }, { key: 'c80', label: '80% cov.', num: true }, { key: 'c95', label: '95% cov.', num: true }]}
                  rows={models.map((m) => {
                    const b = lb.board[m]!;
                    return { m: <span className="flex items-center gap-2"><span className="h-[3px] w-3 rounded" style={{ background: MODEL_COLORS[m] }} />{MODEL_LABELS[m]}{m === best && <Badge tone="good">best</Badge>}</span>, mae: fmtInt(b.mae), rmse: fmtInt(b.rmse), wape: `${b.wape.toFixed(1)}%`, mase: b.mase.toFixed(3), bias: `${b.bias > 0 ? '+' : ''}${fmtInt(b.bias)}`, c80: fmtPct(lb.coverage[m]!['0.8'], 0), c95: fmtPct(lb.coverage[m]!['0.95'], 0) };
                  })} />
                <div className="mt-3 space-y-2">
                  <Note>MAE and RMSE are in units/day. WAPE = Σ|error| ÷ Σ actual. Bias = mean(forecast − actual): positive means over-forecasting, which builds excess stock. Ridge λ = {lb.lambda}, chosen on separate calibration origins.</Note>
                  <Note>{fb.lstm.note}</Note>
                </div>
              </Panel>
              <ChartCard title="Adaptive ensemble weights" subtitle="Each model's weight ∝ 1 / recent MAE, re-estimated at every backtest origin"
                table={<DataTable caption="Ensemble weights by origin" columns={[{ key: 'd', label: 'Origin' }, ...(['holtwinters', 'ridge', 'lstm'] as const).map((m) => ({ key: m, label: SHORT[m], num: true }))]} rows={fb.weightsHistory.map((w) => ({ d: w.date, holtwinters: fmtPct(w.weights.holtwinters ?? 0, 0), ridge: fmtPct(w.weights.ridge ?? 0, 0), lstm: w.weights.lstm !== undefined ? fmtPct(w.weights.lstm, 0) : '–' }))} />}
                chart={<><EChart ariaLabel="Ensemble weights by backtest origin" height={250} option={{
                  grid: { left: 44, right: 12, top: 12, bottom: 30 },
                  xAxis: { type: 'category', data: fb.weightsHistory.map((w) => w.date.slice(5)) },
                  yAxis: { type: 'value', max: 1, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } },
                  tooltip: { trigger: 'axis', valueFormatter: (v) => fmtPct(Number(v), 0) },
                  series: (['holtwinters', 'ridge', 'lstm'] as const).filter((m) => fb.weightsHistory[0].weights[m] !== undefined).map((m) => ({ name: SHORT[m], type: 'bar', stack: 'w', data: fb.weightsHistory.map((w) => w.weights[m] ?? 0), itemStyle: { color: MODEL_COLORS[m], borderColor: C.surface, borderWidth: 2, borderRadius: 2 }, barWidth: '55%' })),
                }} /><div className="mt-2"><Legend items={(['holtwinters', 'ridge', 'lstm'] as const).filter((m) => fb.weightsHistory[0].weights[m] !== undefined).map((m) => ({ color: MODEL_COLORS[m], label: SHORT[m], area: true }))} /></div></>}>
                <Explain>No single model wins every fortnight. The ensemble shifts weight to whichever model has been most accurate recently, so it adapts when promotions or the demand level change.</Explain>
              </ChartCard>
            </div>

            <div className="mt-4 grid gap-4 lg:grid-cols-2 2xl:grid-cols-4">
              <ChartCard title="What drives demand" subtitle="Ridge: units/day per 1-SD change in each driver"
                table={<DataTable caption="Ridge coefficients" columns={[{ key: 'f', label: 'Driver' }, { key: 'b', label: 'Effect (units/day)', num: true }]} rows={fb.ridgeEffects.map((e) => ({ f: e.label, b: `${e.beta > 0 ? '+' : ''}${fmtInt(e.beta)}` }))} />}
                chart={<EChart ariaLabel="Standardised Ridge coefficients for the main demand drivers" height={300} option={{
                  grid: { left: 140, right: 16, top: 6, bottom: 24 },
                  xAxis: { type: 'value', axisLabel: { formatter: (v: number) => `${Math.round(v)}` } },
                  yAxis: { type: 'category', inverse: true, data: fb.ridgeEffects.map((e) => e.label), axisLabel: { color: C.ink2, fontSize: 11 } },
                  tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => `${fmtInt(Number(v))} units/day` },
                  series: [{ type: 'bar', data: fb.ridgeEffects.map((e) => ({ value: e.beta, itemStyle: { color: e.beta >= 0 ? C.ai : '#e66767', borderRadius: e.beta >= 0 ? [0, 4, 4, 0] : [4, 0, 0, 4] } })), barWidth: '60%' }],
                }} />} />
              <ChartCard title="Weekly pattern" subtitle="Mean units/day by weekday (since 2014-04)" chart={<EChart ariaLabel="Mean demand by day of week" height={300} option={barOpt(DOW, fb.eda.dow, C.ai)} />} />
              <ChartCard title="Promotion effect" subtitle="Mean units/day by promotion-intensity quartile" chart={<EChart ariaLabel="Mean demand by promotion quartile" height={300} option={barOpt(['Q1 low', 'Q2', 'Q3', 'Q4 high'], fb.eda.promoQuartiles, C.ridge)} />}>
                <Explain>On the most promoted days demand runs <b className="text-ink">×{(fb.eda.promoQuartiles[3] / fb.eda.promoQuartiles[0]).toFixed(2)}</b> the least promoted. That's why planned promotions are a forecast input.</Explain>
              </ChartCard>
              <ChartCard title="Forecast error distribution" subtitle="Recent out-of-sample errors (actual − forecast)" chart={<EChart ariaLabel="Histogram of recent forecast errors" height={300} option={{
                grid: { left: 44, right: 12, top: 12, bottom: 34 },
                xAxis: { type: 'value', min: fb.residualHistogram.edges[0], max: fb.residualHistogram.edges.at(-1), axisLabel: { formatter: (v: number) => `${Math.round(v / 1000)}k` }, splitLine: { show: false } },
                yAxis: { type: 'value' },
                tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
                series: [{ type: 'bar', barWidth: '94%', data: Array.from(fb.residualHistogram.counts, (c, i) => [(fb.residualHistogram.edges[i] + fb.residualHistogram.edges[i + 1]) / 2, c]), itemStyle: { color: withAlpha(C.ai, 0.75), borderRadius: [3, 3, 0, 0] } }],
              }} />}>
                <Explain>σ<sub>D</sub> = {fmtInt(fb.forecast.residualSd)} units/day. This spread of forecast errors, not of raw demand, is what the AI safety stock has to cover.</Explain>
              </ChartCard>
            </div>
          </>
        );
      }}
    </WithData>
  );
}

function barOpt(cats: string[], vals: number[], color: string) {
  return {
    grid: { left: 48, right: 12, top: 12, bottom: 28 },
    xAxis: { type: 'category' as const, data: cats },
    yAxis: { type: 'value' as const, axisLabel: { formatter: (v: number) => `${Math.round(v / 1000)}k` } },
    tooltip: { trigger: 'axis' as const, axisPointer: { type: 'shadow' as const }, valueFormatter: (v: unknown) => `${fmtInt(Number(v))} units/day` },
    series: [{ type: 'bar' as const, data: vals, barWidth: '58%', itemStyle: { color, borderRadius: [4, 4, 0, 0] } }],
  };
}
