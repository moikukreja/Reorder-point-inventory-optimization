/**
 * Tree-shaken ECharts wrapper with the dashboard's dark theme. Registers only the chart types
 * and components the app uses. It uses canvas rendering, so it stays fast with long time series.
 */
import { memo } from 'react';
import ReactEChartsCore from 'echarts-for-react/lib/core';
import * as echarts from 'echarts/core';
import { BarChart, CustomChart, HeatmapChart, LineChart, ScatterChart } from 'echarts/charts';
import { DataZoomComponent, GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent, MarkPointComponent, TooltipComponent, VisualMapComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { EChartsOption } from 'echarts';
import { C } from './tokens';

echarts.use([LineChart, BarChart, ScatterChart, HeatmapChart, CustomChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent, MarkAreaComponent, MarkPointComponent, DataZoomComponent, VisualMapComponent, CanvasRenderer]);

const font = 'Inter, system-ui, sans-serif';
const mono = '"JetBrains Mono", ui-monospace, monospace';

echarts.registerTheme('control-tower', {
  backgroundColor: 'transparent',
  textStyle: { fontFamily: font, color: C.ink2 },
  color: [C.ai, C.baseline, C.holtwinters, C.ridge, C.lstm],
  grid: { left: 56, right: 18, top: 28, bottom: 36, containLabel: false },
  categoryAxis: {
    axisLine: { lineStyle: { color: C.axis } },
    axisTick: { show: false },
    axisLabel: { color: C.ink3, fontSize: 11 },
    splitLine: { show: false },
  },
  valueAxis: {
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: C.ink3, fontSize: 11, fontFamily: mono },
    splitLine: { lineStyle: { color: C.grid } },
    nameTextStyle: { color: C.ink3, fontSize: 11 },
  },
  timeAxis: {
    axisLine: { lineStyle: { color: C.axis } },
    axisTick: { show: false },
    axisLabel: { color: C.ink3, fontSize: 11 },
    splitLine: { show: false },
  },
  legend: { textStyle: { color: C.ink2, fontSize: 12 }, icon: 'roundRect', itemWidth: 12, itemHeight: 4, itemGap: 16 },
  tooltip: {
    backgroundColor: 'rgba(11,16,24,0.94)',
    borderColor: 'rgba(148,163,184,0.25)',
    borderWidth: 1,
    padding: [8, 10],
    textStyle: { color: C.ink, fontSize: 12, fontFamily: font },
    extraCssText: 'backdrop-filter: blur(8px); border-radius: 10px; box-shadow: 0 8px 24px rgba(0,0,0,.45);',
    axisPointer: { lineStyle: { color: 'rgba(148,163,184,0.45)', width: 1 }, crossStyle: { color: 'rgba(148,163,184,0.45)' } },
  },
  line: { symbolSize: 8, lineStyle: { width: 2 }, smooth: false },
  bar: { itemStyle: { borderRadius: [4, 4, 0, 0] } },
});

export interface EChartProps {
  option: EChartsOption;
  height?: number | string;
  /** Accessible description for screen readers. */
  ariaLabel: string;
  onEvents?: Record<string, (p: unknown) => void>;
}

export const EChart = memo(function EChart({ option, height = 300, ariaLabel, onEvents }: EChartProps) {
  return (
    <div role="img" aria-label={ariaLabel} style={{ height }}>
      <ReactEChartsCore
        echarts={echarts}
        option={{ animationDuration: 500, animationEasing: 'cubicOut', aria: { enabled: true, label: { description: ariaLabel } }, ...option }}
        theme="control-tower"
        notMerge
        lazyUpdate
        style={{ height: '100%', width: '100%' }}
        opts={{ renderer: 'canvas' }}
        onEvents={onEvents}
      />
    </div>
  );
});

export const fmtInt = (v: number) => (Number.isFinite(v) ? Math.round(v).toLocaleString('en-US') : '–');
export const fmtK = (v: number) => {
  if (Math.abs(v) < 1000) return `${Math.round(v)}`;
  const k = v / 1000;
  return `${Math.abs(k) >= 10 || Number.isInteger(k) ? Math.round(k) : k.toFixed(1)}k`;
};
export const fmtPct = (v: number, d = 1) => (Number.isFinite(v) ? `${(v * 100).toFixed(d)}%` : '–');
export const fmtMoney = (v: number, d = 2) => (Number.isFinite(v) ? `$${v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d })}` : '–');

/** Standard tooltip row: a colour swatch, then the label and value in ink (text never wears the series colour). */
export const tipRow = (color: string, label: string, value: string) =>
  `<div style="display:flex;align-items:center;gap:8px;line-height:1.6"><span style="width:10px;height:3px;border-radius:2px;background:${color}"></span><span style="color:${C.ink2}">${label}</span><span style="margin-left:auto;padding-left:14px;font-family:${mono};color:${C.ink}">${value}</span></div>`;
