// Copies the TF.js WebAssembly binaries into public/ so the analysis worker can load them (run before dev/build).
import { copyFileSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, '../node_modules/@tensorflow/tfjs-backend-wasm/dist');
const dst = join(here, '../public/tfjs-wasm');
mkdirSync(dst, { recursive: true });
for (const f of readdirSync(src).filter((f) => f.endsWith('.wasm'))) copyFileSync(join(src, f), join(dst, f));
console.log('copied TF.js wasm binaries to public/tfjs-wasm');
