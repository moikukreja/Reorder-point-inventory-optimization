/**
 * Data & Live Feed: what the dashboard is working from (source, coverage, quality) and the most
 * recent observations. Adding new data arrives in the integration step.
 */
import { Trash2 } from 'lucide-react';
import { useApp } from '../state/store';
import { EChart, fmtInt } from '../charts/EChart';
import { C, withAlpha } from '../charts/tokens';
import { Button, ChartCard, DataTable, Panel } from '../ui/primitives';
import { AddDayForm, AddedData, CsvUpload, LstmPanel, ReplayPanel, UpdateFeed } from '../ui/LiveFeed';
import { PageHeader, WithData, seriesTitle } from './common';

export function DataFeed() {
  const ds = useApp((s) => s.dataset);
  const clear = useApp((s) => s.clearData);
  return (
    <WithData>
      {(fb) => {
        const h = fb.history;
        const rows = h.dates.map((d, i) => ({ date: d, units: h.actual[i] === null ? 'closed' : fmtInt(h.actual[i]!), promo: h.promo[i], holiday: h.names[i] ?? '' })).reverse().slice(0, 42);
        return (
          <>
            <PageHeader eyebrow="Data & live feed" title={seriesTitle(fb)}>
              Daily unit sales with promotions, holidays and oil prices. Add new days here, and the forecasts, safety stock and reorder point update straight away. Everything is computed locally.
            </PageHeader>
            <div className="grid gap-4 lg:grid-cols-3">
              <Panel title="Dataset">
                <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-[13px]">
                  <dt className="text-ink-3">Source</dt><dd className="text-ink-2">{ds?.source}</dd>
                  <dt className="text-ink-3">History</dt><dd className="num text-ink">{ds?.startDate} → {ds?.historyEnd}</dd>
                  <dt className="text-ink-3">Days</dt><dd className="num text-ink">{fmtInt(ds?.days ?? 0)} ({ds?.closedDays} closed)</dd>
                  <dt className="text-ink-3">Series</dt><dd className="text-ink-2">{ds?.series.length} store × family combinations (Quito)</dd>
                  <dt className="text-ink-3">Prepared</dt><dd className="num text-ink-2">{ds?.generatedAt}</dd>
                </dl>
              </Panel>
              <Panel title="Data quality & cleaning" subtitle="Applied in the preparation notebook">
                <ul className="list-disc space-y-1.5 pl-5 text-[13px] leading-relaxed text-ink-2">
                  <li>25 Dec (missing) and 1 Jan (store closed) are treated as <b className="text-ink">closed days</b>: no demand, and excluded from model fitting.</li>
                  <li>Promotion counts are all zero before 2014-04-01, so models train from that date.</li>
                  <li>Oil prices exist only for trading days; they are interpolated over time.</li>
                  <li>Holidays are matched to Quito: national, Pichincha and Quito entries; transferred days are moved.</li>
                </ul>
              </Panel>
              <Panel title="Your data stays here" actions={<Button variant="danger" onClick={() => void clear()}><Trash2 className="h-4 w-4" />Clear local data</Button>}>
                <p className="text-[13px] leading-relaxed text-ink-2">The dataset lives only in this browser (IndexedDB) and in a background worker's memory. The deployed site contains no data. Clearing removes it from this device.</p>
              </Panel>
            </div>

            <div className="mt-4 grid gap-4 lg:grid-cols-2 2xl:grid-cols-4">
              <AddDayForm />
              <CsvUpload />
              <ReplayPanel />
              <LstmPanel />
            </div>
            <div className="mt-4"><AddedData /></div>

            <div className="mt-4 grid gap-4 xl:grid-cols-2">
              <ChartCard title="Units sold, last 20 weeks" subtitle="Daily demand (units/day)" chart={
                <EChart ariaLabel="Daily units sold over the last 20 weeks" height={240} option={{
                  grid: { left: 52, right: 16, top: 16, bottom: 30 },
                  xAxis: { type: 'category', data: h.dates, axisLabel: { formatter: (v: string) => v.slice(5), hideOverlap: true } },
                  yAxis: { type: 'value', axisLabel: { formatter: (v: number) => `${Math.round(v / 1000)}k` } },
                  tooltip: { trigger: 'axis', valueFormatter: (v) => fmtInt(Number(v)) },
                  series: [{ name: 'Units sold', type: 'line', data: h.actual, symbol: 'none', lineStyle: { color: C.actual, width: 1.6 } }],
                }} />} />
              <ChartCard title="Promotions, last 20 weeks" subtitle="Number of items on promotion each day" chart={
                <EChart ariaLabel="Number of items on promotion per day over the last 20 weeks" height={240} option={{
                  grid: { left: 52, right: 16, top: 16, bottom: 30 },
                  xAxis: { type: 'category', data: h.dates, axisLabel: { formatter: (v: string) => v.slice(5), hideOverlap: true } },
                  yAxis: { type: 'value' },
                  tooltip: { trigger: 'axis' },
                  series: [{ name: 'Items on promotion', type: 'bar', data: h.promo, barCategoryGap: '20%', itemStyle: { color: withAlpha(C.ridge, 0.85), borderRadius: [2, 2, 0, 0] } }],
                }} />} />
            </div>

            <div className="mt-4 grid gap-4 xl:grid-cols-[1fr_1fr]">
              <Panel title="Most recent observations" subtitle="Newest first">
                <DataTable caption="Recent daily observations" columns={[{ key: 'date', label: 'Date' }, { key: 'units', label: 'Units sold', num: true }, { key: 'promo', label: 'Items on promo', num: true }, { key: 'holiday', label: 'Holiday / event' }]} rows={rows} maxHeight={380} />
              </Panel>
              <UpdateFeed limit={30} />
            </div>
          </>
        );
      }}
    </WithData>
  );
}
