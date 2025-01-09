import type { StrategyConfig, PerpetualMetrics } from '@/types'

export interface ArbitrageOpportunity {
  symbol: string
  fundingRate: number
  expectedReturn: number
  timestamp: string
}

export class FundingArbitrageStrategy {
  constructor(private readonly config: StrategyConfig) {}

  analyze(metrics: PerpetualMetrics): ArbitrageOpportunity | null {
    const { threshold = 0.1 } = this.config
    
    if (Math.abs(metrics.funding_rate) > threshold) {
      return {
        symbol: metrics.symbol,
        fundingRate: metrics.funding_rate,
        expectedReturn: metrics.funding_rate * metrics.open_interest,
        timestamp: new Date().toISOString()
      }
    }
    
    return null
  }
} 