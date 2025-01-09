import { createClient } from '@supabase/supabase-js'
import { TokenMetrics, DexPair } from '@/types/metrics'
import type { Database } from '@/types/supabase'
import { TOKEN_ADDRESSES } from '@/config/tokens'
import { withRetry } from '@/utils/retryUtils'

const supabase = createClient<Database>(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)

export const loadTokenMetrics = async (symbol: string): Promise<TokenMetrics> => {
  return withRetry(async () => {
    try {
      // Get the latest metrics for the symbol
      const { data, error } = await supabase
        .from('perpetual_metrics')
        .select('*')
        .eq('symbol', symbol)
        .order('timestamp', { ascending: false })
        .limit(1)
        .single()

      if (error) throw error

      if (!data) {
        throw new Error(`No metrics found for ${symbol}`)
      }

      // Transform the data to match TokenMetrics interface
      return {
        symbol: data.symbol,
        price: data.mark_price,
        volume24h: data.volume_24h,
        priceChange24h: data.price_change_24h,
        marketCap: data.market_cap,
        totalSupply: data.total_supply,
        holderCount: data.holder_count || 0,
        liquidity: data.liquidity
      }
    } catch (error) {
      console.error('Error loading token metrics:', error)
      // Return default values if there's an error
      return {
        symbol,
        price: 0,
        volume24h: 0,
        priceChange24h: 0,
        marketCap: 0,
        totalSupply: 0,
        holderCount: 0,
        liquidity: 0
      }
    }
  })
}

export const loadDexPairs = async (symbol: string): Promise<DexPair[]> => {
  return withRetry(async () => {
    try {
      const address = TOKEN_ADDRESSES[symbol as keyof typeof TOKEN_ADDRESSES]
      if (!address) {
        throw new Error(`No address found for symbol ${symbol}`)
      }

      // Get DEX pairs from Supabase
      const { data, error } = await supabase
        .from('dex_pairs')
        .select('*')
        .or(`base_token.eq.${address},quote_token.eq.${address}`)
        .order('liquidity', { ascending: false })
        .limit(5)

      if (error) throw error

      // Transform the data to match DexPair interface
      return (data || []).map(pair => ({
        address: pair.address,
        baseToken: pair.base_token_symbol,
        quoteToken: pair.quote_token_symbol,
        price: pair.price,
        volume24h: pair.volume_24h,
        liquidity: pair.liquidity,
        priceChange24h: pair.price_change_24h
      }))
    } catch (error) {
      console.error('Error loading DEX pairs:', error)
      return []
    }
  })
} 