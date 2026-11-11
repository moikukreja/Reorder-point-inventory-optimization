import { BookOpen, Database, Gauge, Scale, ShieldAlert, SlidersHorizontal, Sparkles, TrendingUp, type LucideIcon } from 'lucide-react';
import type { PageId } from '../state/store';

export const NAV: { id: PageId; label: string; short: string; icon: LucideIcon; step?: string }[] = [
  { id: 'tower', label: 'Control Tower', short: 'Tower', icon: Gauge },
  { id: 'data', label: 'Data & Live Feed', short: 'Data', icon: Database },
  { id: 'forecast', label: 'Demand Forecast', short: 'Forecast', icon: TrendingUp, step: '1' },
  { id: 'risk', label: 'Safety Stock & Risk', short: 'Risk', icon: ShieldAlert, step: '2' },
  { id: 'optimiser', label: 'Bayesian Optimiser', short: 'Optimise', icon: Sparkles, step: '3' },
  { id: 'sensitivity', label: 'Sensitivity', short: 'Sensitivity', icon: SlidersHorizontal, step: '3' },
  { id: 'compare', label: 'Baseline vs AI', short: 'Compare', icon: Scale },
  { id: 'method', label: 'Method & Assumptions', short: 'Method', icon: BookOpen },
];
