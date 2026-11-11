/** Default dashboard settings (kept separate from the store so Node tests can import them). */
import { DEFAULT_COSTS, DEFAULT_LEAD_TIME, DEFAULT_SERVICE_LEVEL } from '../core/assumptions';
import type { Settings } from '../engine/types';

export const DEFAULT_SETTINGS: Settings = {
  serviceLevel: DEFAULT_SERVICE_LEVEL,
  leadTime: { ...DEFAULT_LEAD_TIME },
  costs: { ...DEFAULT_COSTS },
  mcRuns: 100_000,
  simReps: 200,
  seed: 42,
  scenario: 'base',
  useLstm: true,
  blockBootstrap: true,
};
