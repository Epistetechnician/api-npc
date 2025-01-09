import { BaseMetrics } from './metrics';

export interface PerpetualMetrics extends BaseMetrics {
  funding_rate: number;
  open_interest: number;
  volume_24h: number;
  long_positions: number;
  short_positions: number;
  liquidations_24h: number;
} 