export enum DataProvider {
  DEFILLAMA = 'defillama',
  DUNE = 'dune',
  BITQUERY = 'bitquery',
  FOOTPRINT = 'footprint',
  THEGRAPH = 'thegraph',
  HYPERLIQUID = 'hyperliquid'
}

export enum AssetType {
  TOKEN = 'token',
  NFT = 'nft',
  PERPETUAL = 'perpetual',
  SYNTHETIC = 'synthetic',
  OPTION = 'option'
}

export enum Environment {
  TEST = 'test',
  PRODUCTION = 'production'
}

export enum ChainName {
  ETHEREUM = 'ethereum',
  SOLANA = 'solana',
  ARBITRUM = 'arbitrum',
  OPTIMISM = 'optimism',
  BASE = 'base',
  POLYGON = 'polygon'
}

export interface DataSource {
  id: number;
  provider: DataProvider;
  version: string;
  config?: Record<string, any>;
  rate_limit?: number;
  priority?: number;
  is_active: boolean;
  created_at: Date;
  updated_at: Date;
} 