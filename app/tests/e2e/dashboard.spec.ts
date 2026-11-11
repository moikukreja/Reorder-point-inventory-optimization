/**
 * End-to-end tests, mapped to the traceability table (docs/TEST_REPORT.md).
 * Every test also fails on any console error, warning or CSP violation (see fixtures.ts).
 */
import { writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { dataTo, expect, goto, loadDemo, test } from './fixtures';

const PAGES: [string, string][] = [
  ['#/tower', 'Recommendation'],
  ['#/data', 'Add a day'],
  ['#/forecast', 'Model leaderboard'],
  ['#/risk', 'Monte Carlo: demand during the lead time'],
  ['#/optimiser', 'Problem set-up'],
  ['#/sensitivity', 'Tornado'],
  ['#/compare', 'Replay results on actual demand'],
  ['#/method', 'Traceability'],
];

test('T1 · first visit ships no data and explains local processing', async ({ page, problems }) => {
  void problems;
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await expect(page.getByText('Drop favorita_subset.json here').or(page.getByText('Recommendation'))).toBeVisible();
  const res = await page.request.get('/favorita_subset.json');
  expect((await res.text()).startsWith('{"schema"')).toBe(false);
});

test('T2 · every page renders with charts, under the production CSP', async ({ page, problems }) => {
  void problems;
  await loadDemo(page);
  for (const [hash, text] of PAGES) {
    await goto(page, hash);
    await expect(page.getByText(text).first()).toBeVisible();
  }
  await goto(page, '#/forecast');
  await expect(page.locator('canvas').first()).toBeVisible();
  await expect(page.getByText('Holt-Winters (time series)').first()).toBeVisible();
  await expect(page.getByText('Ridge regression').first()).toBeVisible();
});

test('T3 · slide formula is shown with substituted values', async ({ page, problems }) => {
  void problems;
  await loadDemo(page);
  await goto(page, '#/risk');
  await expect(page.getByText('Safety Stock = Z · √( σ_D² · L + D̄² · σ_L² )')).toBeVisible();
  await expect(page.getByText(/Z = Φ⁻¹\(95\.0%\) = 1\.6449/).first()).toBeVisible();
});

test('T4 · controls recompute results (service level, scenario)', async ({ page, problems }) => {
  void problems;
  await loadDemo(page);
  const rop = async () => (await page.getByText(/^Slide formula ROP [\d,]+$/).innerText()).replace(/\D/g, '');
  const before = await rop();
  await page.getByRole('button', { name: 'Controls' }).click();
  await page.getByRole('dialog').getByRole('combobox').selectOption('supplier-delay');
  await expect.poll(rop, { timeout: 30_000 }).not.toBe(before);
  await page.getByRole('dialog').getByRole('button', { name: 'Reset to defaults' }).click();
  await expect.poll(rop, { timeout: 30_000 }).toBe(before);
});

test('T5 · adding a day updates forecast, ROP and the update log, and persists across reload', async ({ page, problems }) => {
  void problems;
  await loadDemo(page);
  await goto(page, '#/data');
  const d0 = await dataTo(page);
  await page.getByLabel('Units sold', { exact: true }).fill('7000');
  await page.getByRole('button', { name: /^Add \d{4}-\d{2}-\d{2} and update$/ }).click();
  await expect.poll(() => dataTo(page)).not.toBe(d0);
  await expect(page.getByText(/1 new day added \(form\)/)).toBeVisible();
  const d1 = await dataTo(page);
  await page.reload();
  await expect.poll(() => dataTo(page), { timeout: 60_000 }).toBe(d1);
  await goto(page, '#/data');
  await page.getByRole('button', { name: 'Remove added data' }).click();
  await expect.poll(() => dataTo(page), { timeout: 60_000 }).toBe(d0);
});

test('T6 · CSV upload validates before adding', async ({ page, problems }) => {
  void problems;
  await loadDemo(page);
  await goto(page, '#/data');
  const d0 = await dataTo(page);
  const next = (iso: string, k: number) => new Date(Date.parse(`${iso}T00:00:00Z`) + k * 86_400_000).toISOString().slice(0, 10);
  const dir = mkdtempSync(join(tmpdir(), 'csv-'));
  const bad = join(dir, 'bad.csv');
  writeFileSync(bad, `date,units_sold,onpromotion,holiday\n${next(d0, 2)},5000,3,0\n`);
  await page.locator('input[type=file][accept=".csv,text/csv"]').first().setInputFiles(bad);
  await expect(page.getByText(/skips 1 day/)).toBeVisible();
  const good = join(dir, 'good.csv');
  writeFileSync(good, `date,units_sold,onpromotion,holiday,oil\n${next(d0, 1)},5000,3,0,50\n${next(d0, 2)},closed,0,1,\n${next(d0, 3)},6100,12,0,\n`);
  await page.locator('input[type=file][accept=".csv,text/csv"]').first().setInputFiles(good);
  await page.getByRole('button', { name: 'Add 3 days and update' }).click();
  await expect.poll(() => dataTo(page)).toBe(next(d0, 3));
  await page.getByRole('button', { name: 'Remove added data' }).click();
  await expect.poll(() => dataTo(page), { timeout: 60_000 }).toBe(d0);
});

test('T7 · live replay feeds held-back days one at a time and restores them', async ({ page, problems }) => {
  void problems;
  await loadDemo(page);
  await goto(page, '#/data');
  const d0 = await dataTo(page);
  await page.getByRole('button', { name: 'Start replay' }).click();
  await expect(page.getByText(/0 of 28 days replayed/)).toBeVisible();
  await expect.poll(() => dataTo(page), { timeout: 30_000 }).not.toBe(d0);
  expect((await dataTo(page)) < d0).toBe(true);
  await page.getByRole('button', { name: 'Next day' }).click();
  await expect(page.getByText(/1 of 28 days replayed/)).toBeVisible();
  await expect(page.getByText(/Day arrived:/).first()).toBeVisible();
  await page.getByRole('button', { name: '+7 days' }).click();
  await expect(page.getByText(/8 of 28 days replayed/)).toBeVisible();
  await expect(page.getByText(/LSTM fine-tuned|AI reorder point .* → /).first()).toBeVisible();
  await page.getByRole('button', { name: /Stop & restore/ }).click();
  await expect.poll(() => dataTo(page), { timeout: 60_000 }).toBe(d0);
});

test('T8 · Bayesian optimiser streams, finds a feasible policy, and never blocks the UI', async ({ page, problems }) => {
  void problems;
  await loadDemo(page);
  await goto(page, '#/optimiser');
  await page.evaluate(() => {
    (window as unknown as { __long: number[] }).__long = [];
    new PerformanceObserver((l) => l.getEntries().forEach((e) => (window as unknown as { __long: number[] }).__long.push(e.duration))).observe({ type: 'longtask', buffered: false });
  });
  await page.getByRole('button', { name: 'Run optimisation' }).click();
  await expect(page.getByText(/Running… \d+\/40/)).toBeVisible();
  await expect(page.getByText('Learned cost surface (GP posterior mean)')).toBeVisible({ timeout: 90_000 });
  await expect(page.getByText(/\d+ evaluations in [\d.]+ s/)).toBeVisible();
  const longest = await page.evaluate(() => Math.max(0, ...(window as unknown as { __long: number[] }).__long));
  expect(longest, 'longest main-thread task during optimisation (ms)').toBeLessThan(250);
  await page.getByRole('button', { name: '3D view' }).click();
  await expect(page.getByRole('img', { name: /3D surface/ })).toBeVisible();
});

test('T9 · sensitivity re-runs the optimiser under perturbations', async ({ page, problems }) => {
  void problems;
  await loadDemo(page);
  await goto(page, '#/sensitivity');
  await page.getByRole('button', { name: 'Run 7 optimisations' }).click();
  await expect(page.getByText('Lead time +2 days')).toBeVisible({ timeout: 120_000 });
});

test('T10 · phone layout has no horizontal scrolling', async ({ page, problems }) => {
  void problems;
  await page.setViewportSize({ width: 375, height: 812 });
  await loadDemo(page);
  for (const [hash] of PAGES) {
    await goto(page, hash);
    await page.waitForTimeout(400);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow, `horizontal overflow on ${hash}`).toBeLessThanOrEqual(0);
  }
});
