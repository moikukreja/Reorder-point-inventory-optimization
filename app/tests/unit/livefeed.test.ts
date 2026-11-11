/** Tests for adding new data, CSV parsing, the demo dataset, replay and continuous updating. */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { rowsFromCsv, rowsToCsv, validateRows } from '../../src/data/newRows';
import { generateDemoData } from '../../src/data/demoData';
import { validateSubset } from '../../src/data/schema';
import { AnalysisEngine } from '../../src/engine/analysis';
import { DEFAULT_SETTINGS } from '../../src/state/defaults';
import { hasLocalData, MODELS_DIR, SUBSET_PATH } from '../../scripts/lib/local-data';
import { initNodeBackend, loadModel } from '../../scripts/lib/node-tf';
import { modelSlug } from '../../src/core/models/lstm';

describe('new-row validation and CSV', () => {
  it('accepts consecutive days, closed days and optional oil', () => {
    const v = validateRows([{ date: '2017-08-16', sales: '9000', onpromotion: 3, holiday: 0 }, { date: '2017-08-17', sales: 'closed', onpromotion: '0', holiday: '1', oil: '47.2' }], '2017-08-15');
    expect(v.issues).toEqual([]);
    expect(v.rows[1]).toEqual({ date: '2017-08-17', sales: null, onpromotion: 0, holiday: 1, oil: 47.2 });
  });
  it('rejects gaps, duplicates and bad values with line numbers', () => {
    const v = validateRows([{ date: '2017-08-18', sales: 5, onpromotion: 0, holiday: 0 }, { date: '2017-08-16', sales: -1, onpromotion: 1.5, holiday: 'x' }], '2017-08-15');
    expect(v.rows.length).toBe(0);
    expect(v.issues[0].message).toMatch(/skips 2 day/);
    expect(v.issues[1].message).toMatch(/units sold/);
    expect(v.issues[1].message).toMatch(/whole number/);
    expect(v.issues[1].message).toMatch(/0 or 1/);
  });
  it('round-trips through CSV export and import', () => {
    const rows = validateRows([{ date: '2017-08-16', sales: 100, onpromotion: 2, holiday: 0, oil: 50 }, { date: '2017-08-17', sales: null, onpromotion: 0, holiday: 1 }], '2017-08-15').rows;
    const back = rowsFromCsv(rowsToCsv(rows), '2017-08-15');
    expect(back.issues).toEqual([]);
    expect(back.rows).toEqual(rows);
  });
  it('reports missing columns', () => {
    expect(rowsFromCsv('date,units_sold\n2017-08-16,5', '2017-08-15').issues[0].message).toMatch(/onpromotion/);
  });
});

describe('synthetic demo dataset', () => {
  it('passes schema validation and is labelled synthetic', () => {
    const d = validateSubset(generateDemoData());
    expect(d.source.startsWith('SYNTHETIC')).toBe(true);
    expect(d.series.length).toBe(3);
  });
  it('supports the full analysis pipeline (classical models)', async () => {
    const eng = new AnalysisEngine(generateDemoData());
    const fb = await eng.prepare('DEMO-1|BEVERAGES', true);
    expect(fb.lstm.available).toBe(false);
    expect(fb.leaderboard.board.ridge!.mase).toBeLessThan(1); // the demo has learnable structure
    const rk = eng.risk('DEMO-1|BEVERAGES', DEFAULT_SETTINGS);
    expect(rk.replay.policies.length).toBe(3);
  });
});

const run = hasLocalData() ? describe : describe.skip;
run('continuous updating on real data (Store 44 × BEVERAGES)', () => {
  const KEY = '44|BEVERAGES';
  const mk = async () => {
    await initNodeBackend();
    return new AnalysisEngine(validateSubset(JSON.parse(readFileSync(SUBSET_PATH, 'utf-8'))), async (key) => {
      const dir = join(MODELS_DIR, modelSlug(key));
      return existsSync(join(dir, 'model.json')) ? { model: await loadModel(dir), meta: JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf-8')) } : null;
    });
  };

  it('a new day refits the models and moves the forecast origin', async () => {
    const eng = await mk();
    const a = await eng.prepare(KEY, false);
    const r = eng.appendRows(KEY, [{ date: '2017-08-16', sales: 15000, onpromotion: 40, holiday: 0 }]);
    expect(r.lastDate).toBe('2017-08-16');
    const b = await eng.prepare(KEY, false);
    expect(b.originDate).toBe('2017-08-16');
    expect(b.forecast.dates[0]).toBe('2017-08-17');
    expect(b.forecast.ensemble[0]).not.toBe(a.forecast.ensemble[1]);
    expect(() => eng.appendRows(KEY, [{ date: '2017-08-20', sales: 1, onpromotion: 0, holiday: 0 }])).toThrow(/skips/);
  });

  it('replay holds back real days and restores them exactly', async () => {
    const eng = await mk();
    const full = await eng.prepare(KEY, false);
    const st = eng.startReplay(KEY, 14);
    expect(st.remaining).toBe(14);
    expect(eng.lastDate(KEY)).toBe('2017-08-01');
    const step = eng.replayNext(KEY, 1);
    expect(step.added[0].date).toBe('2017-08-02');
    expect(() => eng.appendRows(KEY, [{ date: '2017-08-03', sales: 1, onpromotion: 0, holiday: 0 }])).toThrow(/replay/);
    eng.stopReplay(KEY);
    const back = await eng.prepare(KEY, false);
    expect(back.originDate).toBe(full.originDate);
    expect(back.forecast.ensemble).toEqual(full.forecast.ensemble);
  });

  it('fine-tunes the live LSTM on new data without touching the frozen backtest model', async () => {
    const eng = await mk();
    const a = await eng.prepare(KEY, true);
    const rep = await eng.fineTuneLstm(KEY, undefined, 4);
    expect(rep!.newDays).toBeGreaterThan(200);
    const b = await eng.prepare(KEY, true);
    expect(b.lstm.fineTunedTo).toBe('2017-08-15');
    expect(b.leaderboard.board.lstm!.mae).toBe(a.leaderboard.board.lstm!.mae); // backtest reused (frozen weights)
    expect(b.forecast.models.lstm).not.toEqual(a.forecast.models.lstm);
  }, 180_000);

  it('historical replay recomputes the AI reorder point every 14 days on actual demand', async () => {
    const eng = await mk();
    await eng.prepare(KEY, false);
    const rk = eng.risk(KEY, DEFAULT_SETTINGS);
    const ai = rk.replay.policies.find((p) => p.id === 'ai-formula')!;
    expect(rk.replay.dates.length).toBe(84);
    expect(new Set(ai.sByDay).size).toBe(6);
    expect(new Set(rk.replay.policies[0].sByDay).size).toBe(6);
  });
});
