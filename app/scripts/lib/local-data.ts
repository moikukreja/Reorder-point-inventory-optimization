/**
 * Node-only helpers that read the git-ignored local data folder (../data/derived).
 * They are used by tests and build scripts, never by the deployed app.
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateSubset, type SubsetFile } from '../../src/data/schema';

const here = dirname(fileURLToPath(import.meta.url));
export const DATA_DIR = resolve(here, '../../../data/derived');
export const SUBSET_PATH = resolve(DATA_DIR, 'favorita_subset.json');
export const FIXTURE_PATH = resolve(DATA_DIR, 'fixtures/reference.json');
export const MODELS_DIR = resolve(here, '../../public/models');

export const hasLocalData = () => existsSync(SUBSET_PATH) && existsSync(FIXTURE_PATH);

export function loadSubset(): SubsetFile {
  return validateSubset(JSON.parse(readFileSync(SUBSET_PATH, 'utf-8')));
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function loadFixtures(): any {
  return JSON.parse(readFileSync(FIXTURE_PATH, 'utf-8'));
}
