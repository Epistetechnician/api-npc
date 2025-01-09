// Birdeye Types
export interface BirdeyeTokenInfo {
  success: boolean;
  data: {
    mintAddress: string;
    owner: string;
    supply: string;
    holderCount: number;
    tokenInfo: {
      name: string;
      symbol: string;
      decimals: number;
    };
  };
}

export interface BirdeyeApiResponse {
  data: {
    data: BirdeyeTokenInfo
    [key: string]: unknown
  }
  success?: boolean
  status?: number
}

export interface BirdeyeTokenData {
  price: number;
  volume24h: number;
  priceChange24h: number;
  marketCap: number;
  totalSupply: number;
  holderCount: number;
  liquidity: number;
}

export interface BirdeyeTokenOverview {
  success: boolean;
  data: {
    price: number;
    volume24h: number;
    priceChange24h: number;
    marketCap: number;
    liquidity: number;
  };
}

// HyperLiquid Types
export interface HyperliquidAssetCtx {
  dayNtlVlm: string
  funding: string
  impactPxs: [string, string]
  markPx: string
  midPx: string
  openInterest: string
  oraclePx: string
  premium: string
  prevDayPx: string
}

export interface HyperliquidMetaAndAssetCtxResponse {
  meta: {
    universe: Array<{
      name: string
      szDecimals: number
      maxLeverage: number
      onlyIsolated: boolean
    }>
  }
  assetCtxs: Record<string, HyperliquidAssetCtx>
}

export interface HyperliquidMarketData {
  funding_rate: number
  mark_price: number
  open_interest: number
  volume_24h: number
  oracle_price?: number
  premium?: number
  price_24h_ago?: number
  impact_prices?: number[]
  mid_price?: number
}

export interface HyperliquidRawData {
  name: string
  funding: string
  markPrice: string
  openInterest: string
  volume24h: string
  oraclePrice?: string
  premium?: string
  price24hAgo?: string
  impactPrices?: string[]
  midPrice?: string
}

// DexScreener Types
export interface DexScreenerData {
  price: number
  volume24h: number
  liquidity: number
  priceChange24h: number
}

// Raw Data Types
export interface RawData {
  timestamp: string
  hyperliquid: Record<string, any>
  dexscreener: Record<string, any>
  birdeye: Record<string, any>
  solscan: Record<string, any>
}

// Metric Types
export interface MetricData {
  symbol: string
  timestamp: string
  mark_price: number
  funding_rate: number
  open_interest: number
  volume_24h: number
  price_change_24h: number
  total_supply: number
  market_cap: number
  liquidity: number
  spot_price: number
  spot_volume_24h: number
  txns_24h: number
  holder_count?: number
} 