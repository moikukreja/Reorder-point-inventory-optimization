import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

/**
 * DEV ONLY: serves the git-ignored local subset at /__dev-data/favorita_subset.json, so the
 * dashboard can be tested without picking the file each time. `apply: 'serve'` keeps it out of
 * production builds, so the deployed site never contains or serves the data.
 */
function devData(): Plugin {
  const file = resolve(__dirname, '../data/derived/favorita_subset.json');
  return {
    name: 'dev-local-data',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__dev-data/favorita_subset.json', (_req, res) => {
        if (!existsSync(file)) {
          res.statusCode = 404;
          res.end('Run notebooks/01_prepare_and_validate.ipynb first.');
          return;
        }
        res.setHeader('Content-Type', 'application/json');
        res.end(readFileSync(file));
      });
    },
  };
}

// Static build for Netlify. All modelling runs in Web Workers in the browser.
export default defineConfig({
  plugins: [react(), devData()],
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    sourcemap: false,
    chunkSizeWarningLimit: 1800,
    rollupOptions: {
      output: {
        manualChunks: { echarts: ['echarts', 'echarts-for-react'], react: ['react', 'react-dom', 'framer-motion', 'zustand'] },
      },
    },
  },
});
