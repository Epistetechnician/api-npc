export interface QueryConfig {
  query: string;
  parameters?: Record<string, any>;
}

export interface PerpetualMetrics {
  symbol: string;
  timestamp: string;
  mark_price: number;
  funding_rate: number;
  open_interest: number;
  volume_24h: number;
  daily_volume: number;
  price_change_24h: number;
  total_supply: number;
  market_cap: number;
  liquidity: number;
  spot_price: number;
  spot_volume_24h: number;
  txns_24h: number;
  holder_count: number | null;
  long_positions: number;
}

export interface StrategyConfig {
  threshold?: number;
  interval?: number;
} 