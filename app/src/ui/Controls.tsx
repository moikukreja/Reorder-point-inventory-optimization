/**
 * Global controls: service level, lead-time assumptions, risk scenario, costs and simulation
 * settings. Changes are debounced in the store and recomputed in the worker.
 */
import { RotateCcw } from 'lucide-react';
import type { ReactNode } from 'react';
import { SCENARIOS, type ScenarioId } from '../core/scenarios';
import type { LeadTimeDist } from '../core/stats';
import { useApp } from '../state/store';
import { Button, NumberField, Segmented, Slider, Toggle } from './primitives';

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <fieldset className="space-y-3.5 border-t border-white/[0.06] pt-4 first:border-0 first:pt-0">
      <legend className="mb-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-3">{title}</legend>
      {children}
    </fieldset>
  );
}

export function Controls() {
  const st = useApp((s) => s.settings);
  const up = useApp((s) => s.updateSettings);
  const reset = useApp((s) => s.resetSettings);
  const busy = useApp((s) => s.riskBusy);
  const carrying = (st.costs.holdingPerUnitDay * 365) / st.costs.unitCost;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-[14px] font-semibold text-ink">Controls</h2>
        <span className="flex items-center gap-1.5 text-[11.5px] text-ink-3" aria-live="polite">
          <span className={`h-1.5 w-1.5 rounded-full ${busy ? 'animate-pulse bg-rop' : 'bg-good'}`} />
          {busy ? 'Recalculating…' : 'Up to date'}
        </span>
      </div>

      <Section title="Service">
        <Slider label="Target service level (cycle)" value={st.serviceLevel} min={0.8} max={0.995} step={0.005} onChange={(v) => up({ serviceLevel: v })} format={(v) => `${(v * 100).toFixed(1)}%`} hint="Probability of not running out while a replenishment is on its way. Sets Z = Φ⁻¹(SL)." />
      </Section>

      <Section title="Supplier lead time">
        <Slider label="Mean lead time L̄" value={st.leadTime.mean} min={1} max={12} step={0.5} onChange={(v) => up({ leadTime: { ...st.leadTime, mean: v } })} format={(v) => `${v.toFixed(1)} days`} />
        <Slider label="Lead-time variability σL" value={st.leadTime.sd} min={0} max={4} step={0.1} onChange={(v) => up({ leadTime: { ...st.leadTime, sd: v } })} format={(v) => `${v.toFixed(1)} days`} hint="An assumption: the dataset has no delivery records. Gamma is right-skewed, so late deliveries are likelier than early ones." />
        <div className="space-y-1.5">
          <span className="text-[12.5px] text-ink-2">Distribution</span>
          <Segmented<LeadTimeDist> label="Lead-time distribution" value={st.leadTime.dist} onChange={(v) => up({ leadTime: { ...st.leadTime, dist: v } })} options={[{ value: 'gamma', label: 'Gamma' }, { value: 'lognormal', label: 'Lognormal' }, { value: 'normal', label: 'Normal' }]} />
        </div>
      </Section>

      <Section title="Risk scenario">
        <label className="block space-y-1.5">
          <span className="text-[12.5px] text-ink-2">Scenario</span>
          <select value={st.scenario} onChange={(e) => up({ scenario: e.target.value as ScenarioId })} className="w-full rounded-lg border border-white/10 bg-surface px-2.5 py-2 text-[13px] text-ink outline-none focus:border-ai/60">
            {SCENARIOS.map((s) => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </select>
        </label>
        <p className="text-[11.5px] leading-snug text-ink-3">{SCENARIOS.find((s) => s.id === st.scenario)?.description}</p>
      </Section>

      <Section title="Costs (assumptions)">
        <div className="grid grid-cols-2 gap-3">
          <NumberField label="Unit cost" prefix="$" step={0.1} min={0.01} value={st.costs.unitCost} onChange={(v) => up({ costs: { ...st.costs, unitCost: v, holdingPerUnitDay: (v * carrying) / 365 } })} />
          <NumberField label="Carrying rate" suffix="%/yr" step={1} min={1} value={Math.round(carrying * 100)} onChange={(v) => up({ costs: { ...st.costs, holdingPerUnitDay: (st.costs.unitCost * (v / 100)) / 365 } })} />
          <NumberField label="Cost per order" prefix="$" step={10} min={0} value={st.costs.orderCost} onChange={(v) => up({ costs: { ...st.costs, orderCost: v } })} />
          <NumberField label="Lost-sale penalty" prefix="$" step={0.05} min={0} value={st.costs.stockoutPenalty} onChange={(v) => up({ costs: { ...st.costs, stockoutPenalty: v } })} />
        </div>
      </Section>

      <Section title="Simulation">
        <Slider label="Monte Carlo runs" value={st.mcRuns} min={20_000} max={400_000} step={20_000} onChange={(v) => up({ mcRuns: v })} format={(v) => `${(v / 1000).toFixed(0)}k`} />
        <Slider label="Policy-simulation replications" value={st.simReps} min={50} max={500} step={50} onChange={(v) => up({ simReps: v })} format={(v) => `${v}`} />
        <NumberField label="Random seed (reproducibility)" value={st.seed} step={1} min={0} onChange={(v) => up({ seed: Math.max(0, Math.round(v)) })} />
        <Toggle label="Block bootstrap (keep correlated misses)" checked={st.blockBootstrap} onChange={(v) => up({ blockBootstrap: v })} />
        <Toggle label="Include LSTM in the ensemble" checked={st.useLstm} onChange={(v) => up({ useLstm: v })} />
      </Section>

      <Button variant="ghost" onClick={reset}>
        <RotateCcw className="h-4 w-4" /> Reset to defaults
      </Button>
    </div>
  );
}
