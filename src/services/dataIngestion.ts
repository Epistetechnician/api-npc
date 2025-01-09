import { supabase } from '@/lib/supabaseClient';
import { formatNumber, formatCurrency } from '@/utils/formatters';

export interface MarketData {
  price: number;
  volume: number;
  liquidity: number;
  mark_price?: number;
  funding_rate?: number;
  open_interest?: number;
  volume_24h?: number;
  priceChange24h?: number;
  totalSupply?: number;
  marketCap?: number;
  volume24h?: number;
  holderCount?: number;
  formatted?: {
    price: string;
    volume: string;
    liquidity: string;
  };
}

export interface PerpetualMetrics {
  symbol: string;
  timestamp: string;
  mark_price: number;
  funding_rate: number;
  open_interest: number;
  volume_24h: number;
  price_change_24h: number;
  total_supply: number;
  market_cap: number;
  liquidity: number;
  spot_price: number;
  spot_volume_24h: number;
  txns_24h: number;
  holder_count: number;
  daily_volume: number;
  long_positions: number;
}

export class DataIngestionService {
  private readonly supabase = supabase;

  async testSupabaseConnection(): Promise<boolean> {
    try {
      const { error } = await this.supabase.from('health').select('*');
      return !error;
    } catch {
      return false;
    }
  }

  async fetchBirdeyeData(address: string): Promise<MarketData> {
    try {
      const data = {
        price: 0,
        volume: 0,
        liquidity: 0,
        priceChange24h: 0,
        totalSupply: 0,
        marketCap: 0,
        volume24h: 0,
        holderCount: 0
      };

      return {
        ...data,
        formatted: {
          price: formatCurrency(data.price),
          volume: formatNumber(data.volume),
          liquidity: formatNumber(data.liquidity)
        }
      };
    } catch (error) {
      console.error('Error fetching Birdeye data:', error);
      throw error;
    }
  }

  async fetchDexScreenerData(address: string): Promise<MarketData> {
    try {
      const data = {
        price: 0,
        volume: 0,
        liquidity: 0,
        volume24h: 0
      };

      return {
        ...data,
        formatted: {
          price: formatCurrency(data.price),
          volume: formatNumber(data.volume),
          liquidity: formatNumber(data.liquidity)
        }
      };
    } catch (error) {
      console.error('Error fetching DexScreener data:', error);
      throw error;
    }
  }

  async fetchHyperliquidData(): Promise<MarketData> {
    try {
      const data = {
        price: 0,
        volume: 0,
        liquidity: 0,
        mark_price: 0,
        funding_rate: 0,
        open_interest: 0,
        volume_24h: 0
      };

      return {
        ...data,
        formatted: {
          price: formatCurrency(data.price),
          volume: formatNumber(data.volume),
          liquidity: formatNumber(data.liquidity)
        }
      };
    } catch (error) {
      console.error('Error fetching Hyperliquid data:', error);
      throw error;
    }
  }

  async fetchCombinedMarketData(_symbol: string, address: string) {
    try {
      const [birdeye, hyperliquid] = await Promise.all([
        this.fetchBirdeyeData(address),
        this.fetchHyperliquidData()
      ]);
      return { birdeye, hyperliquid };
    } catch (error) {
      console.error('Error fetching combined market data:', error);
      throw error;
    }
  }

  async ingestMetrics(metrics: PerpetualMetrics[]) {
    try {
      const { error } = await this.supabase
        .from('perpetual_metrics')
        .insert(metrics);
      
      if (error) throw error;
    } catch (error) {
      console.error('Error ingesting metrics:', error);
      throw error;
    }
  }
} 