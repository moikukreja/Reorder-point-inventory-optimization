/**
 * Shared e2e fixture:
 *  - injects the production Content-Security-Policy from netlify.toml into every response, so the
 *    tests prove the app works under the real deployed security policy
 *  - collects console errors/warnings, page errors and CSP violations; each test fails if any appear
 */
import { test as base, expect, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const toml = readFileSync(resolve(here, '../../../netlify.toml'), 'utf-8');
export const CSP = /Content-Security-Policy = "([^"]+)"/.exec(toml)![1];

export const test = base.extend<{ problems: string[] }>({
  problems: async ({ page, context }, use) => {
    const problems: string[] = [];
    await context.route('**/*', async (route) => {
      const res = await route.fetch();
      await route.fulfill({ response: res, headers: { ...res.headers(), 'content-security-policy': CSP } });
    });
    page.on('console', (m) => {
      if (m.type() === 'error' || m.type() === 'warning') problems.push(`${m.type()}: ${m.text()}`);
    });
    page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
    await use(problems);
    expect(problems, 'console errors / warnings / CSP violations').toEqual([]);
  },
});

export { expect };

/** Fresh start: clears this origin's storage and loads the synthetic demo dataset. */
export async function loadDemo(page: Page) {
  await page.goto('/');
  await page.evaluate(async () => {
    localStorage.clear();
    await new Promise<void>((r) => { const q = indexedDB.deleteDatabase('keyval-store'); q.onsuccess = q.onerror = q.onblocked = () => r(); });
  });
  await page.reload();
  await page.getByRole('button', { name: 'Explore with synthetic demo data' }).click();
  await expect(page.getByRole('heading', { level: 1, name: 'Store 101 · BEVERAGES' })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(/^Slide formula ROP [\d,]+$/)).toBeVisible({ timeout: 60_000 });
}

export async function goto(page: Page, hash: string) {
  await page.evaluate((h) => (location.hash = h), hash);
}

/** Reads the current data end date from the top bar. */
export async function dataTo(page: Page): Promise<string> {
  const t = await page.locator('header').first().innerText();
  return /Data to\s*(\d{4}-\d{2}-\d{2})/.exec(t)?.[1] ?? '';
}
