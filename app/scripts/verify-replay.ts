/** Prints the historical replay (baseline vs AI on actual demand) for a series. Usage: npx tsx scripts/verify-replay.ts [key] */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { MODELS_DIR, SUBSET_PATH } from './lib/local-data';
import { initNodeBackend, loadModel } from './lib/node-tf';
import { AnalysisEngine } from '../src/engine/analysis';
import { validateSubset } from '../src/data/schema';
import { modelSlug } from '../src/core/models/lstm';
import { DEFAULT_SETTINGS } from '../src/state/defaults';

async function main() {
  await initNodeBackend();
  const key = process.argv[2] ?? '44|BEVERAGES';
  const eng = new AnalysisEngine(validateSubset(JSON.parse(readFileSync(SUBSET_PATH, 'utf-8'))), async (k) => {
    const dir = join(MODELS_DIR, modelSlug(k));
    return existsSync(join(dir, 'model.json')) ? { model: await loadModel(dir), meta: JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf-8')) } : null;
  });
  await eng.prepare(key, true);
  const r = eng.risk(key, DEFAULT_SETTINGS).replay;
  console.log(`${key}: ${r.dates[0]} -> ${r.dates.at(-1)} (${r.dates.length} days of actual demand), Q = ${Math.round(r.Q)}`);
  const b = r.policies[0].metrics.totalCostPerDay;
  for (const p of r.policies) {
    const m = p.metrics;
    console.log(`  ${p.label.padEnd(28)} fill ${(m.fillRate * 100).toFixed(2)}%  cycle SL ${(m.cycleServiceLevel * 100).toFixed(1)}%  lost ${Math.round(m.lostUnitsPerDay)}/day  avg stock ${Math.round(m.avgOnHand)}  $${m.totalCostPerDay.toFixed(2)}/day (${((m.totalCostPerDay / b - 1) * 100).toFixed(0)}%)`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
