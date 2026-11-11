/**
 * Live-feed tools: add one day, upload a CSV (with a validation preview), replay held-back real
 * days, keep the LSTM learning, and the "what changed" update feed.
 */
import { Brain, Download, FastForward, Pause, Play, Plus, RotateCcw, Square, Upload } from 'lucide-react';
import { useRef, useState } from 'react';
import { useApp, type UpdateEntry } from '../state/store';
import { rowsFromCsv, rowsToCsv, validateRows, type NewRow, type ValidationResult } from '../data/newRows';
import { fmtInt } from '../charts/EChart';
import { Badge, Button, DataTable, Note, NumberField, Panel, Progress, Slider, Toggle } from './primitives';

const nextDay = (iso: string) => new Date(Date.parse(`${iso}T00:00:00Z`) + 86_400_000).toISOString().slice(0, 10);

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function AddDayForm() {
  const fb = useApp((s) => s.forecast);
  const add = useApp((s) => s.addRows);
  const busy = useApp((s) => s.dataBusy);
  const replay = useApp((s) => s.replay);
  const [sales, setSales] = useState<number>(NaN);
  const [closed, setClosed] = useState(false);
  const [promo, setPromo] = useState(0);
  const [holiday, setHoliday] = useState(false);
  const [oil, setOil] = useState<number>(NaN);
  const [err, setErr] = useState<string | null>(null);
  if (!fb) return null;
  const date = nextDay(fb.originDate);
  const submit = async () => {
    const v = validateRows([{ date, sales: closed ? null : Number.isFinite(sales) ? sales : '', onpromotion: promo, holiday: holiday ? 1 : 0, oil: Number.isFinite(oil) ? oil : null }], fb.originDate);
    if (v.issues.length) return setErr(v.issues[0].message);
    if (!closed && !Number.isFinite(sales)) return setErr('Enter units sold, or tick "closed".');
    setErr(null);
    await add(v.rows, 'form');
    setSales(NaN);
  };
  return (
    <Panel title="Add a day" subtitle={<>Next date: <span className="num text-ink-2">{date}</span> (rows must continue day by day)</>}>
      <div className="grid grid-cols-2 gap-3">
        <NumberField label="Units sold" value={sales} min={0} step={100} onChange={setSales} />
        <NumberField label="Items on promotion" value={promo} min={0} step={1} onChange={(v) => setPromo(Math.max(0, Math.round(v)))} />
        <NumberField label="Oil price (optional)" prefix="$" value={oil} min={0} step={0.5} onChange={setOil} />
        <div className="space-y-2.5 pt-1">
          <Toggle label="Holiday" checked={holiday} onChange={setHoliday} />
          <Toggle label="Store closed" checked={closed} onChange={setClosed} />
        </div>
      </div>
      {err && <div className="mt-3"><Note tone="warn">{err}</Note></div>}
      <div className="mt-3"><Button onClick={() => void submit()} busy={busy} disabled={replay.active}><Plus className="h-4 w-4" />Add {date} and update</Button></div>
      {replay.active && <p className="mt-2 text-[12px] text-ink-3">Finish or stop the replay before adding your own days.</p>}
    </Panel>
  );
}

export function CsvUpload() {
  const fb = useApp((s) => s.forecast);
  const add = useApp((s) => s.addRows);
  const busy = useApp((s) => s.dataBusy);
  const replay = useApp((s) => s.replay);
  const [res, setRes] = useState<ValidationResult | null>(null);
  const [name, setName] = useState('');
  const input = useRef<HTMLInputElement>(null);
  if (!fb) return null;
  const read = async (f: File) => {
    setName(f.name);
    setRes(rowsFromCsv(await f.text(), fb.originDate));
  };
  const template = rowsToCsv([{ date: nextDay(fb.originDate), sales: 9000, onpromotion: 12, holiday: 0, oil: 47.5 }]);
  return (
    <Panel title="Upload new days (CSV)" subtitle="Columns: date, units_sold, onpromotion, holiday, oil (optional)"
      actions={<Button variant="ghost" onClick={() => download('new-days-template.csv', template, 'text/csv')}><Download className="h-4 w-4" />Template</Button>}>
      <input ref={input} type="file" accept=".csv,text/csv" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) void read(f); e.target.value = ''; }} />
      <Button variant="ghost" onClick={() => input.current?.click()} disabled={replay.active}><Upload className="h-4 w-4" />Choose CSV</Button>
      {res && (
        <div className="mt-3 space-y-3">
          <div className="flex flex-wrap gap-2 text-[12.5px]"><Badge>{name}</Badge><Badge tone="good">{res.rows.length} valid row{res.rows.length === 1 ? '' : 's'}</Badge>{res.issues.length > 0 && <Badge tone="bad">{res.issues.length} problem{res.issues.length === 1 ? '' : 's'}</Badge>}</div>
          {res.issues.length > 0 && <Note tone="warn"><ul className="list-disc pl-4">{res.issues.slice(0, 6).map((i) => <li key={i.line}>Line {i.line}: {i.message}</li>)}</ul>{res.issues.length > 6 && <div>…and {res.issues.length - 6} more.</div>}</Note>}
          {res.rows.length > 0 && <DataTable caption="Rows to add" maxHeight={180} columns={[{ key: 'd', label: 'Date' }, { key: 's', label: 'Units', num: true }, { key: 'p', label: 'Promo', num: true }, { key: 'h', label: 'Holiday', num: true }]} rows={res.rows.map((r) => ({ d: r.date, s: r.sales === null ? 'closed' : fmtInt(r.sales), p: r.onpromotion, h: r.holiday }))} />}
          <Button onClick={() => { void add(res.rows, 'csv').then(() => setRes(null)); }} busy={busy} disabled={!res.rows.length || res.issues.length > 0}>Add {res.rows.length} day{res.rows.length === 1 ? '' : 's'} and update</Button>
          {res.issues.length > 0 && <p className="text-[12px] text-ink-3">Fix the problems and upload again. Rows are only added when the whole file is valid.</p>}
        </div>
      )}
    </Panel>
  );
}

export function ReplayPanel() {
  const r = useApp((s) => s.replay);
  const busy = useApp((s) => s.dataBusy);
  const a = useApp.getState();
  const [days, setDays] = useState(28);
  return (
    <Panel title="Live replay on real data" subtitle="Hold back the latest days, then feed them in one at a time as if they were arriving live. Every day refits the models and recomputes safety stock and the reorder point.">
      {!r.active ? (
        <div className="space-y-3">
          <Slider label="Days to hold back" value={days} min={7} max={84} step={7} onChange={setDays} format={(v) => `${v} days`} />
          <Button onClick={() => void a.startReplay(days)} busy={busy}><Play className="h-4 w-4" />Start replay</Button>
        </div>
      ) : (
        <div className="space-y-3">
          <Progress value={(r.total - r.remaining) / r.total} label={`${r.total - r.remaining} of ${r.total} days replayed${r.nextDate ? ` · next ${r.nextDate}` : ''}`} />
          <div className="flex flex-wrap gap-2">
            {r.playing ? <Button variant="ghost" onClick={a.replayPause}><Pause className="h-4 w-4" />Pause</Button> : <Button onClick={a.replayPlay} disabled={busy}><Play className="h-4 w-4" />Play</Button>}
            <Button variant="ghost" onClick={() => void a.replayStep(1)} disabled={busy || r.playing}><FastForward className="h-4 w-4" />Next day</Button>
            <Button variant="ghost" onClick={() => void a.replayStep(7)} disabled={busy || r.playing}>+7 days</Button>
            <Button variant="danger" onClick={() => void a.replayStop()} disabled={busy && !r.playing}><Square className="h-4 w-4" />Stop & restore</Button>
          </div>
          <Slider label="Pause between days" value={r.speedMs} min={0} max={3000} step={100} onChange={a.setReplaySpeed} format={(v) => `${(v / 1000).toFixed(1)} s`} hint="Each day also waits for the recalculation (about 1–2 s) to finish." />
        </div>
      )}
    </Panel>
  );
}

export function LstmPanel() {
  const fb = useApp((s) => s.forecast);
  const job = useApp((s) => s.lstmJob);
  const a = useApp.getState();
  if (!fb) return null;
  return (
    <Panel title="LSTM learning" subtitle="Continuous updating of the deep-learning model, here in your browser">
      <p className="text-[13px] leading-relaxed text-ink-2">{fb.lstm.note}</p>
      <p className="mt-2 text-[12px] text-ink-3">It fine-tunes automatically after every 7 new days. Holt-Winters and Ridge are refitted on every single new day.</p>
      {job && <div className="mt-3"><Progress value={job.frac} label={job.msg} /></div>}
      <div className="mt-3 flex flex-wrap gap-2">
        <Button variant="ghost" onClick={() => void a.fineTuneNow()} disabled={!!job || !fb.lstm.available}><Brain className="h-4 w-4" />Fine-tune now</Button>
        <Button variant="ghost" onClick={() => void a.trainLstmInBrowser()} disabled={!!job}><RotateCcw className="h-4 w-4" />Train LSTM in this browser</Button>
      </div>
      <p className="mt-2 text-[11.5px] text-ink-3">Training from scratch takes about a minute; the dashboard stays usable meanwhile.</p>
    </Panel>
  );
}

export function AddedData() {
  const key = useApp((s) => s.key);
  const added = useApp((s) => (s.key ? s.added[s.key] ?? [] : []));
  const anyAdded = useApp((s) => Object.values(s.added).some((r) => r.length > 0));
  const a = useApp.getState();
  const inputRef = useRef<HTMLInputElement>(null);
  const fb = useApp((s) => s.forecast);
  return (
    <Panel title="Data you added" subtitle="Stored in this browser only. Export it to move it to another device."
      actions={<>
        <Button variant="ghost" disabled={!added.length} onClick={() => download(`added-days-${(key ?? '').replace('|', '-')}.csv`, rowsToCsv(added), 'text/csv')}><Download className="h-4 w-4" />Export CSV</Button>
        <Button variant="ghost" onClick={() => inputRef.current?.click()}><Upload className="h-4 w-4" />Import</Button>
        <Button variant="danger" disabled={!anyAdded} onClick={() => void a.resetAdded()}><RotateCcw className="h-4 w-4" />Remove added data</Button>
      </>}>
      <input ref={inputRef} type="file" accept=".csv,text/csv" className="hidden" onChange={async (e) => { const f = e.target.files?.[0]; e.target.value = ''; if (!f || !fb) return; const v = rowsFromCsv(await f.text(), fb.originDate); if (v.issues.length) alert(`Import failed. Line ${v.issues[0].line}: ${v.issues[0].message}`); else await a.addRows(v.rows, 'import'); }} />
      {added.length ? (
        <DataTable caption="Added rows" maxHeight={220} columns={[{ key: 'd', label: 'Date' }, { key: 's', label: 'Units', num: true }, { key: 'p', label: 'Promo', num: true }, { key: 'h', label: 'Holiday', num: true }, { key: 'o', label: 'Oil', num: true }]} rows={[...added].reverse().map((r: NewRow) => ({ d: r.date, s: r.sales === null ? 'closed' : fmtInt(r.sales), p: r.onpromotion, h: r.holiday, o: r.oil ?? '–' }))} />
      ) : <p className="text-[13px] text-ink-3">Nothing added yet for this series.</p>}
    </Panel>
  );
}

const KIND: Record<UpdateEntry['kind'], { label: string; tone: 'ai' | 'rop' | 'good' | 'neutral' | 'base' }> = {
  form: { label: 'New day', tone: 'ai' }, csv: { label: 'CSV', tone: 'ai' }, import: { label: 'Import', tone: 'ai' },
  replay: { label: 'Live feed', tone: 'rop' }, lstm: { label: 'LSTM', tone: 'good' }, reset: { label: 'Reset', tone: 'neutral' },
};

export function UpdateFeed({ limit = 50, compact = false }: { limit?: number; compact?: boolean }) {
  const updates = useApp((s) => s.updates);
  const key = useApp((s) => s.key);
  const list = updates.filter((u) => u.key === key).slice(0, limit);
  return (
    <Panel title="What changed" subtitle="Every new day or model update, and its effect on the decision">
      {list.length ? (
        <ol className="space-y-2.5" aria-live="polite">
          {list.map((u) => (
            <li key={u.id} className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
              <div className="flex flex-wrap items-center gap-2 text-[12.5px]"><Badge tone={KIND[u.kind].tone}>{KIND[u.kind].label}</Badge><span className="font-medium text-ink">{u.title}</span><span className="ml-auto num text-[11.5px] text-ink-3">{u.at}</span></div>
              {!compact && <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-2">{u.detail}</p>}
              {compact && u.before && u.after && <p className="num mt-1 text-[12px] text-ink-2">ROP {fmtInt(u.before.rop)} → {fmtInt(u.after.rop)}</p>}
            </li>
          ))}
        </ol>
      ) : <p className="text-[13px] text-ink-3">No updates yet. Add a day, upload a CSV or start the replay on the Data page.</p>}
    </Panel>
  );
}
