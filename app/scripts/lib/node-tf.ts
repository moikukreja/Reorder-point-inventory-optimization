/**
 * TensorFlow.js setup for Node build scripts. It uses the WebAssembly backend (much faster than
 * the pure-JS CPU backend, with no native build step), and falls back to CPU if WASM is unavailable.
 * It also provides file save/load helpers that write the standard TF.js layers-model format the
 * browser loads.
 */
import * as tf from '@tensorflow/tfjs';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, sep } from 'node:path';

export async function initNodeBackend(): Promise<string> {
  try {
    const req = createRequire(import.meta.url);
    const dist = dirname(req.resolve('@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm'));
    setWasmPaths(dist.split(sep).join('/') + '/');
    await tf.setBackend('wasm');
  } catch {
    await tf.setBackend('cpu');
  }
  await tf.ready();
  return tf.getBackend();
}

/** Writes model.json + weights.bin (the TF.js layers-model format) into dir. */
export async function saveModel(model: tf.LayersModel, dir: string): Promise<void> {
  mkdirSync(dir, { recursive: true });
  await model.save(
    tf.io.withSaveHandler(async (art) => {
      const data = tf.io.CompositeArrayBuffer.join(art.weightData!);
      writeFileSync(join(dir, 'weights.bin'), Buffer.from(data));
      const json = {
        format: 'layers-model',
        generatedBy: `TensorFlow.js tfjs-layers v${tf.version.tfjs}`,
        convertedBy: null,
        modelTopology: art.modelTopology,
        weightsManifest: [{ paths: ['weights.bin'], weights: art.weightSpecs }],
      };
      writeFileSync(join(dir, 'model.json'), JSON.stringify(json));
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON', weightDataBytes: data.byteLength } };
    }),
  );
}

/** Loads a saved model from dir in Node. */
export async function loadModel(dir: string): Promise<tf.LayersModel> {
  const json = JSON.parse(readFileSync(join(dir, 'model.json'), 'utf-8'));
  const buf = readFileSync(join(dir, 'weights.bin'));
  const weightData = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return tf.loadLayersModel(tf.io.fromMemory({ modelTopology: json.modelTopology, weightSpecs: json.weightsManifest[0].weights, weightData }));
}
