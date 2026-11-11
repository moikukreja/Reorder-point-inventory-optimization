/**
 * First-run screen (data handling Option A): the deployed site ships no data. The user loads
 * their own prepared file. It is parsed and processed in this browser and kept only in this
 * origin's IndexedDB.
 */
import { FileJson, Lock, Upload } from 'lucide-react';
import { useRef, useState } from 'react';
import { useApp } from '../state/store';
import { Button, Note } from './primitives';

export function LoadData() {
  const load = useApp((s) => s.loadDataset);
  const loadDemo = useApp((s) => s.loadDemo);
  const error = useApp((s) => s.error);
  const [drag, setDrag] = useState(false);
  const [localErr, setLocalErr] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const readFile = async (f: File) => {
    setLocalErr(null);
    try {
      const raw = JSON.parse(await f.text());
      await load(raw, true);
    } catch (e) {
      setLocalErr(e instanceof SyntaxError ? 'That file is not valid JSON.' : e instanceof Error ? e.message : String(e));
    }
  };

  // Dev-server only: the whole body is compiled out of production builds (import.meta.env.DEV is false)
  const loadDev = async () => {
    if (!import.meta.env.DEV) return;
    setLocalErr(null);
    const r = await fetch('/__dev-data/favorita_subset.json');
    if (!r.ok) return setLocalErr(await r.text());
    await load(await r.json(), true);
  };

  return (
    <div className="mx-auto flex min-h-full max-w-2xl flex-col justify-center px-4 py-12">
      <div className="mb-6 flex items-center gap-3">
        <div className="tape h-2 w-10 rounded-full" aria-hidden />
        <span className="text-[12px] font-semibold uppercase tracking-[0.12em] text-ink-3">AI in Operations · Reorder point exercise</span>
      </div>
      <h1 className="text-[28px] font-semibold leading-tight text-ink sm:text-[34px]">Reorder Point Control Tower</h1>
      <p className="mt-3 text-[15px] leading-relaxed text-ink-2">
        Demand forecasting, dynamic safety stock with Monte Carlo simulation, and Bayesian optimisation of the reorder point, all running in your browser.
      </p>

      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) void readFile(f); }}
        className={`glass mt-8 grid place-items-center gap-3 border-2 border-dashed p-10 text-center transition ${drag ? 'border-ai/70 bg-ai/10' : 'border-white/10'}`}
      >
        <FileJson className="h-10 w-10 text-ai" aria-hidden />
        <div className="text-[15px] font-medium text-ink">Drop <span className="num">favorita_subset.json</span> here</div>
        <p className="max-w-md text-[13px] text-ink-3">This file is produced by <span className="num">notebooks/01_prepare_and_validate.ipynb</span> from the Kaggle Store Sales (Favorita) data.</p>
        <input ref={input} type="file" accept="application/json,.json" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) void readFile(f); }} />
        <div className="flex flex-wrap justify-center gap-2">
          <Button onClick={() => input.current?.click()}><Upload className="h-4 w-4" /> Choose file</Button>
          <Button variant="ghost" onClick={() => void loadDemo()}>Explore with synthetic demo data</Button>
          {import.meta.env.DEV && <Button variant="ghost" onClick={() => void loadDev()}>Load local dev dataset</Button>}
        </div>
      </div>

      <p className="mt-3 text-[12px] text-ink-3">No file? The synthetic demo generates realistic, clearly labelled fake data in your browser, so anyone can try every feature.</p>
      {(localErr || error) && <div className="mt-4"><Note tone="warn">{localErr ?? error}</Note></div>}

      <div className="mt-6 flex items-start gap-3 text-[12.5px] leading-relaxed text-ink-3">
        <Lock className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
        <p>
          Your data stays on this device. The file is read by your browser, processed in a background worker, and stored only in this site's local storage, so you don't need to load it again. Nothing is uploaded. This follows the exercise rule "No sharing of data with peers allowed". You can clear it at any time from the Data page.
        </p>
      </div>
    </div>
  );
}
