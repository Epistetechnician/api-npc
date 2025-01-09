export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      perpetual_metrics: {
        Row: {
          id: number
          created_at: string
          updated_at: string
          symbol: string
          timestamp: string
          funding_rate: number
          volume_24h?: number
          open_interest: number
          mark_price: number
          spot_price: number
          spot_volume_24h: number
          liquidity: number
          market_cap: number | null
          total_supply: number | null
          price_change_24h: number
          txns_24h: number
          holder_count: number | null
        }
        Insert: Omit<Database['public']['Tables']['perpetual_metrics']['Row'], 'id'>
        Update: Partial<Database['public']['Tables']['perpetual_metrics']['Row']>
      }
    }
  }
} 