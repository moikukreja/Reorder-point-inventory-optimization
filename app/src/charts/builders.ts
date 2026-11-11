/**
 * Reusable ECharts option builders for the dashboard's core visuals.
 * Each builder takes bundle data from the workers and returns a complete option.
 */
import type { EChartsOption } from 'echarts';
import type { ForecastBundle, PolicyRun, RiskBundle } from '../engine/types';
import type { ModelName } from '../core/backtest';
import { C, MODEL_COLORS, withAlpha } from './tokens';
import { fmtInt, fmtK, fmtPct, tipRow } from './EChart';

const MODEL_SHORT: Record<ModelName, string> = { snaive: 'Seasonal naive', holtwinters: 'Holt-Winters', ridge: 'Ridge', lstm: 'LSTM', ensemble: 'AI ensemble' };

export interface ForecastChartOpts {
  historyDays: number;
  models: ModelName[];
  showBacktest: boolean;
  showBands: boolean;
}

/** History + out-of-sample backtest + 28-day forecast with 80% / 95% conformal bands. */
export function forecastOption(fb: ForecastBundle, o: ForecastChartOpts): EChartsOption {
  const hN = Math.min(o.historyDays, fb.history.dates.length);
  const hFrom = fb.history.dates.length - hN;
  const dates = [...fb.history.dates.slice(hFrom), ...fb.forecast.dates];
  const pad = (n: number) => new Array<null>(n).fill(null);
  const fN = fb.forecast.dates.length;
  const actual = [...fb.history.actual.slice(hFrom), ...pad(fN)];
  const promo = [...fb.history.promo.slice(hFrom), ...fb.forecast.promo];
  const names = [...fb.history.names.slice(hFrom), ...pad(fN)];
  const future = (arr: number[]) => [...pad(hN), ...arr];
  const b = fb.forecast.bands;
  const series: EChartsOption['series'] = [];
  if (o.showBands) {
    series.push(
      { name: 'lo95', type: 'line', data: future(b['0.95'].lo), stack: 'b95', lineStyle: { opacity: 0 }, symbol: 'none', silent: true, tooltip: { show: false } },
      { name: '95% band', type: 'line', data: future(b['0.95'].hi.map((v, i) => v - b['0.95'].lo[i])), stack: 'b95', lineStyle: { opacity: 0 }, symbol: 'none', areaStyle: { color: withAlpha(C.ai, 0.13) }, silent: true },
      { name: 'lo80', type: 'line', data: future(b['0.8'].lo), stack: 'b80', lineStyle: { opacity: 0 }, symbol: 'none', silent: true, tooltip: { show: false } },
      { name: '80% band', type: 'line', data: future(b['0.8'].hi.map((v, i) => v - b['0.8'].lo[i])), stack: 'b80', lineStyle: { opacity: 0 }, symbol: 'none', areaStyle: { color: withAlpha(C.ai, 0.22) }, silent: true },
    );
  }
  series.push({
    name: 'Actual', type: 'line', data: actual, symbol: 'circle', symbolSize: 4, showSymbol: false, lineStyle: { color: C.actual, width: 1.6 }, itemStyle: { color: C.actual }, z: 5,
    markLine: { silent: true, symbol: 'none', label: { color: C.ink2, formatter: 'Forecast origin', position: 'insideEndTop', fontSize: 11 }, lineStyle: { color: 'rgba(148,163,184,0.5)', type: 'dashed' }, data: [{ xAxis: fb.originDate }] },
  });
  if (o.showBacktest) {
    for (const m of o.models) {
      const ov = fb.backtestOverlay[m];
      if (!ov) continue;
      series.push({ name: `${MODEL_SHORT[m]} (backtest)`, type: 'line', data: [...ov.slice(hFrom), ...pad(fN)], symbol: 'none', connectNulls: false, lineStyle: { color: MODEL_COLORS[m], width: 1.4, type: 'dashed', opacity: 0.9 } });
    }
  }
  for (const m of o.models) {
    const f = m === 'ensemble' ? fb.forecast.ensemble : fb.forecast.models[m];
    if (!f) continue;
    series.push({ name: MODEL_SHORT[m], type: 'line', data: future(f), symbol: 'circle', symbolSize: 5, showSymbol: m === 'ensemble', lineStyle: { color: MODEL_COLORS[m], width: m === 'ensemble' ? 2.4 : 1.6 }, itemStyle: { color: MODEL_COLORS[m] }, z: m === 'ensemble' ? 6 : 4 });
  }
  return {
    grid: { left: 54, right: 16, top: 26, bottom: 56 },
    xAxis: { type: 'category', data: dates, boundaryGap: false, axisLabel: { formatter: (v: string) => v.slice(5), hideOverlap: true } },
    yAxis: { type: 'value', name: 'units/day', axisLabel: { formatter: (v: number) => fmtK(v) } },
    dataZoom: [{ type: 'inside', start: 0, end: 100 }, { type: 'slider', height: 16, bottom: 8, borderColor: 'transparent', backgroundColor: 'rgba(255,255,255,0.03)', fillerColor: withAlpha(C.ai, 0.15), handleStyle: { color: C.ai }, textStyle: { color: C.ink3 }, dataBackground: { lineStyle: { color: C.ink3 }, areaStyle: { color: 'transparent' } } }],
    tooltip: {
      trigger: 'axis',
      formatter: (ps: unknown) => {
        const arr = ps as { dataIndex: number; seriesName: string; value: number | null; color: string }[];
        if (!arr.length) return '';
        const i = arr[0].dataIndex;
        const isFuture = i >= hN;
        let html = `<div style="font-weight:600;margin-bottom:4px">${dates[i]}${isFuture ? ' · forecast' : ''}</div>`;
        for (const p of arr) {
          if (p.value === null || p.value === undefined || p.seriesName.startsWith('lo')) continue;
          if (p.seriesName.includes('band')) {
            const lv = p.seriesName.startsWith('95') ? '0.95' : '0.8';
            const j = i - hN;
            html += tipRow(withAlpha(C.ai, 0.5), p.seriesName, `${fmtInt(b[lv].lo[j])} – ${fmtInt(b[lv].hi[j])}`);
          } else {
            const color = p.seriesName === 'Actual' ? C.actual : (p.color as string);
            html += tipRow(color, p.seriesName, fmtInt(p.value));
          }
        }
        html += `<div style="margin-top:4px;color:${C.ink3}">Items on promotion: ${promo[i] ?? '–'}${names[i] ? ` · ${names[i]}` : ''}</div>`;
        return html;
      },
    },
    series,
  };
}

/** Histogram of demand during lead time with the stockout tail and the three reorder points. */
export function ddltOption(rk: RiskBundle, compact = false): EChartsOption {
  const h = rk.mc.histogram;
  const total = h.counts.reduce((a, b) => a + b, 0);
  const aiRop = rk.comparison.ai.reorderPoint;
  const data = Array.from(h.counts, (c, i) => {
    const mid = (h.edges[i] + h.edges[i + 1]) / 2;
    return { value: [mid, (c / total) * 100], itemStyle: { color: mid > aiRop ? withAlpha(C.stockout, 0.85) : withAlpha(C.ai, 0.75), borderRadius: [3, 3, 0, 0] } };
  });
  // Label positions are staggered so close reference lines don't overprint each other
  const line = (x: number, color: string, label: string, type: 'solid' | 'dashed', position: 'end' | 'insideStartTop' | 'insideEndTop' = 'end') => ({ xAxis: x, lineStyle: { color, type, width: 2 }, label: { formatter: label, position, color: C.ink, backgroundColor: 'rgba(11,16,24,0.85)', padding: [2, 5], borderRadius: 4, fontSize: 11 } });
  return {
    grid: { left: compact ? 40 : 52, right: 16, top: compact ? 30 : 36, bottom: 34 },
    xAxis: { type: 'value', min: h.edges[0], max: h.edges[h.edges.length - 1], name: 'units demanded during lead time', nameLocation: 'middle', nameGap: 24, axisLabel: { formatter: (v: number) => fmtK(v) }, splitLine: { show: false } },
    yAxis: { type: 'value', name: compact ? '' : '% of scenarios', axisLabel: { formatter: (v: number) => `${v.toFixed(1)}` } },
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'shadow' },
      formatter: (ps: unknown) => {
        const p = (ps as { dataIndex: number }[])[0];
        const i = p.dataIndex;
        return `<div style="font-weight:600;margin-bottom:4px">${fmtInt(h.edges[i])} – ${fmtInt(h.edges[i + 1])} units</div>${tipRow(C.ai, 'Share of scenarios', `${((h.counts[i] / total) * 100).toFixed(2)}%`)}${h.edges[i] > aiRop ? `<div style="color:${C.ink2};margin-top:4px">⚠ Above the AI reorder point: stockout before delivery</div>` : ''}`;
      },
    },
    series: [
      {
        type: 'bar', data, barWidth: '96%',
        markLine: {
          symbol: 'none', silent: true,
          data: compact
            ? [line(aiRop, C.rop, 'AI ROP', 'solid')]
            : [line(rk.comparison.baseline.reorderPoint, C.baseline, 'Baseline', 'dashed', 'insideStartTop'), line(aiRop, C.rop, 'AI formula', 'solid', 'insideEndTop'), line(rk.mc.ai.mcRop, C.ai, 'Monte Carlo', 'dashed', 'end')],
        },
      },
    ],
  };
}

/** On-hand inventory over time for one policy, against its reorder point, with stockout days marked. */
export function sawtoothOption(p: PolicyRun, color: string, dates?: string[]): EChartsOption {
  const x = p.trace.map((_, i) => (dates ? dates[i] : `Day ${i + 1}`));
  const stock = p.trace.map((d) => d.onHand);
  const out = p.trace.map((d, i) => (d.lost > 0 ? [x[i], 0, d.lost] : null)).filter(Boolean) as [string, number, number][];
  const orders = p.trace.map((d, i) => (d.ordered > 0 ? [x[i], d.onHand] : null)).filter(Boolean) as [string, number][];
  return {
    grid: { left: 52, right: 16, top: 24, bottom: 30 },
    xAxis: { type: 'category', data: x, boundaryGap: false, axisLabel: { hideOverlap: true, formatter: (v: string) => (dates ? v.slice(5) : v.replace('Day ', '')) } },
    yAxis: { type: 'value', axisLabel: { formatter: (v: number) => fmtK(v) } },
    tooltip: {
      trigger: 'axis',
      formatter: (ps: unknown) => {
        const i = (ps as { dataIndex: number }[])[0].dataIndex;
        const d = p.trace[i];
        return `<div style="font-weight:600;margin-bottom:4px">${x[i]}</div>${tipRow(color, 'On hand (end of day)', fmtInt(d.onHand))}${tipRow(C.ink3, 'Demand', fmtInt(d.demand))}${d.received ? tipRow(C.good, 'Delivery received', fmtInt(d.received)) : ''}${d.ordered ? tipRow(C.rop, 'Order placed', fmtInt(d.ordered)) : ''}${d.lost ? tipRow(C.stockout, '⚠ Lost sales', fmtInt(d.lost)) : ''}`;
      },
    },
    series: [
      {
        name: 'On hand', type: 'line', step: 'end', data: stock, symbol: 'none', lineStyle: { color, width: 1.8 }, areaStyle: { color: withAlpha(color, 0.12) },
        markLine: { symbol: 'none', silent: true, data: [{ yAxis: p.s, lineStyle: { color: C.rop, type: 'dashed', width: 1.5 }, label: { formatter: `ROP s = ${fmtInt(p.s)}`, color: C.ink, position: 'insideEndTop', fontSize: 11 } }] },
      },
      { name: 'Order placed', type: 'scatter', data: orders, symbol: 'triangle', symbolSize: 9, itemStyle: { color: C.rop, borderColor: C.surface, borderWidth: 2 } },
      { name: 'Stockout', type: 'scatter', data: out, symbol: 'diamond', symbolSize: 10, itemStyle: { color: C.stockout, borderColor: C.surface, borderWidth: 2 } },
    ],
  };
}

export const describeService = (p: number) => (p >= 0.995 ? 'about once in 200+ cycles' : `about once every ${Math.max(1, Math.round(1 / Math.max(1e-6, 1 - p)))} replenishment cycles`);
export const pctOrDash = fmtPct;
