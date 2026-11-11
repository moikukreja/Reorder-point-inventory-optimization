import { defineConfig, devices } from '@playwright/test';

/**
 * End-to-end tests against the PRODUCTION build (`vite preview`), with the CSP from netlify.toml
 * injected into every response (see tests/e2e/fixtures.ts). They use the synthetic demo dataset,
 * so they need no course data. Run: npm run build && npm run test:e2e
 * Set BASE_URL to test a deployed site, e.g. BASE_URL=https://your-site.netlify.app npm run test:e2e
 */
export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 120_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: { baseURL: process.env.BASE_URL ?? 'http://localhost:4173', ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 },
    // Headless CI has no GPU: opt in to software WebGL for the 3D view (otherwise Chromium logs a deprecation warning)
    launchOptions: { args: ['--enable-unsafe-swiftshader'] } },
  webServer: process.env.BASE_URL ? undefined : { command: 'npm run preview', port: 4173, reuseExistingServer: true, timeout: 60_000 },
});
