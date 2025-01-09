'use client'

import React, { useEffect, useState } from 'react'
import { createClient } from '@supabase/supabase-js'
import type { Database } from '@/types/supabase'
import { XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart, CartesianGrid } from 'recharts'
import { format, parseISO } from 'date-fns'
import { getCachedHolderCount, setCachedHolderCount } from '@/utils/cache'
import { TOKEN_ADDRESSES } from '@/config/tokens'

// Add TokenSymbol type if not already defined
type TokenSymbol = keyof typeof TOKEN_ADDRESSES;

// Update PerpMetric interface
interface PerpMetric {
  id: number;
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
  created_at: string;
  updated_at: string;
}

// Add Birdeye API types
interface BirdeyeResponse {
  success: boolean
  data: {
    holderCount: number;
    // Add other Birdeye fields as needed
  }
}

// Create a properly typed Supabase client
const supabase = createClient<Database>(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_KEY!
)

// Add this type for our chart metrics
type MetricType = 'spot_price' | 'spot_volume_24h' | 'liquidity' | 'mark_price' | 'funding_rate' | 'open_interest' | 'market_cap' | 'txns_24h';

interface MetricOption {
  value: MetricType;
  label: string;
  format: (value: number) => string;
  color: string;
}

const metricOptions: MetricOption[] = [
  { 
    value: 'spot_price', 
    label: 'Price', 
    format: (v) => `$${v.toFixed(4)}`,
    color: '#8B5CF6' // Purple
  },
  { 
    value: 'spot_volume_24h', 
    label: 'Volume 24h', 
    format: (v) => `$${(v/1e6).toFixed(2)}M`,
    color: '#10B981' // Green
  },
  { 
    value: 'liquidity', 
    label: 'Liquidity', 
    format: (v) => `$${(v/1e6).toFixed(2)}M`,
    color: '#3B82F6' // Blue
  },
  { 
    value: 'mark_price', 
    label: 'Mark Price', 
    format: (v) => `$${v.toFixed(4)}`,
    color: '#EC4899' // Pink
  },
  { 
    value: 'funding_rate', 
    label: 'Funding Rate', 
    format: (v) => `${(v * 100).toFixed(4)}%`,
    color: '#F59E0B' // Yellow
  },
  { 
    value: 'open_interest', 
    label: 'Open Interest', 
    format: (v) => `$${(v/1e6).toFixed(2)}M`,
    color: '#6366F1' // Indigo
  },
  { 
    value: 'market_cap', 
    label: 'Market Cap', 
    format: (v) => `$${(v/1e6).toFixed(2)}M`,
    color: '#14B8A6' // Teal
  },
  { 
    value: 'txns_24h', 
    label: 'Transactions', 
    format: (v) => v.toLocaleString(),
    color: '#F472B6' // Pink
  }
];

// Add TimeRange type at the top with other type definitions
type TimeRange = '1H' | '4H' | '12H' | '24H';

// Add error boundary component at the top level
class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean }> {
  constructor(props: { children: React.ReactNode }) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Dashboard Error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-900 text-white p-8">
          <div className="p-4 bg-red-500/20 rounded-lg">
            <h2 className="text-xl font-bold mb-2">Something went wrong</h2>
            <button
              onClick={() => this.setState({ hasError: false })}
              className="px-4 py-2 bg-red-500/30 hover:bg-red-500/40 rounded"
            >
              Try again
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

// Add Birdeye API configuration
const BIRDEYE_API_KEY = process.env.NEXT_PUBLIC_BIRDEYE_API_KEY!
const BIRDEYE_API_URL = 'https://public-api.birdeye.so/v1'

// Add function to fetch holder count
async function fetchHolderCount(tokenAddress: string): Promise<number> {
  // Check cache first
  const cached = getCachedHolderCount(tokenAddress)
  if (cached !== null) {
    return cached
  }

  try {
    const response = await fetch(
      `${BIRDEYE_API_URL}/token/holder_count?address=${tokenAddress}`,
      {
        headers: {
          'X-API-KEY': BIRDEYE_API_KEY,
          'Accept': 'application/json',
        }
      }
    )

    if (!response.ok) {
      throw new Error(`Birdeye API error: ${response.statusText}`)
    }

    const data: BirdeyeResponse = await response.json()
    const holderCount = data.success ? data.data.holderCount : 0
    
    // Cache the result
    setCachedHolderCount(tokenAddress, holderCount)
    
    return holderCount
  } catch (error) {
    console.error('Error fetching holder count:', error)
    return 0
  }
}

// Update the fetch function to handle the data properly
const fetchMetrics = async () => {
  try {
    const supabase = createClient<Database>(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    );

    const { data, error } = await supabase
      .from('perpetual_metrics')
      .select('*')
      .order('timestamp', { ascending: false })
      .limit(100);

    if (error) throw error;

    // Ensure all numeric fields are properly typed
    return data.map(metric => ({
      ...metric,
      mark_price: Number(metric.mark_price),
      funding_rate: Number(metric.funding_rate),
      open_interest: Number(metric.open_interest),
      volume_24h: Number(metric.volume_24h),
      price_change_24h: Number(metric.price_change_24h),
      total_supply: Number(metric.total_supply),
      market_cap: Number(metric.market_cap),
      liquidity: Number(metric.liquidity),
      spot_price: Number(metric.spot_price),
      spot_volume_24h: Number(metric.spot_volume_24h),
      txns_24h: Number(metric.txns_24h),
      holder_count: metric.holder_count ? Number(metric.holder_count) : null
    }));
  } catch (error) {
    console.error('Error fetching metrics:', error);
    return [];
  }
};

export default function PerpMetricsPageWrapper() {
  return (
    <ErrorBoundary>
      <PerpMetricsPage />
    </ErrorBoundary>
  )
}

export function PerpMetricsPage() {
  const [metrics, setMetrics] = useState<PerpMetric[]>([])
  const [historicalData, setHistoricalData] = useState<PerpMetric[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedToken, setSelectedToken] = useState<string>('')
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('spot_price')
  const [timeRange, setTimeRange] = useState<TimeRange>('24H')
  const [timeToUpdate, setTimeToUpdate] = useState('')

  // Add real-time subscription
  useEffect(() => {
    let mounted = true
    const channel = supabase
      .channel('perpetual_metrics_changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'perpetual_metrics'
        },
        () => {
          if (mounted) {
            fetchMetrics()
          }
        }
      )
      .subscribe()

    return () => {
      mounted = false
      supabase.removeChannel(channel)
    }
  }, [])

  // Update fetchMetrics to handle errors properly
  async function fetchMetrics() {
    try {
      setLoading(true)
      setError(null)

      const { data: latest, error: latestError } = await supabase
        .from('perpetual_metrics')
        .select('*')
        .order('timestamp', { ascending: false })
        .limit(100) // Limit to recent data for better performance

      if (latestError) throw latestError

      if (latest) {
        // Group by symbol and get latest entry for each
        const latestBySymbol = latest.reduce<Record<string, PerpMetric>>((acc, curr) => {
          if (!acc[curr.symbol] || new Date(curr.timestamp) > new Date(acc[curr.symbol].timestamp)) {
            acc[curr.symbol] = curr
          }
          return acc
        }, {})

        const sortedTokens = Object.values(latestBySymbol).sort((a, b) => {
          if (!a.market_cap && !b.market_cap) return 0
          if (!a.market_cap) return 1
          if (!b.market_cap) return -1
          return b.market_cap - a.market_cap
        })

        setMetrics(sortedTokens)

        // Set initial selected token if none selected
        if (!selectedToken && sortedTokens.length > 0) {
          setSelectedToken(sortedTokens[0].symbol)
        }
      }

      // Fetch historical data if token is selected
      if (selectedToken) {
        await fetchHistoricalData()
      }
    } catch (err) {
      console.error('Error fetching metrics:', err)
      setError(err instanceof Error ? err.message : 'An error occurred while fetching data')
    } finally {
      setLoading(false)
    }
  }

  // Separate function for historical data
  async function fetchHistoricalData() {
    try {
      const timeRangeHours = {
        '1H': 1,
        '4H': 4,
        '12H': 12,
        '24H': 24
      }[timeRange]

      const rangeStart = new Date()
      rangeStart.setHours(rangeStart.getHours() - timeRangeHours)

      const { data: historical, error: historicalError } = await supabase
        .from('perpetual_metrics')
        .select('*')
        .eq('symbol', selectedToken)
        .gte('timestamp', rangeStart.toISOString())
        .order('timestamp', { ascending: true })

      if (historicalError) throw historicalError

      if (historical) {
        // Filter duplicates and ensure proper sorting
        const uniqueHistorical = historical.reduce<Record<string, PerpMetric>>((acc, curr) => {
          const timestamp = new Date(curr.timestamp).getTime().toString()
          if (!acc[timestamp] || curr.id > acc[timestamp].id) {
            acc[timestamp] = curr
          }
          return acc
        }, {})

        setHistoricalData(Object.values(uniqueHistorical).sort((a, b) => 
          new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
        ))
      }
    } catch (err) {
      console.error('Error fetching historical data:', err)
      setError(err instanceof Error ? err.message : 'An error occurred while fetching historical data')
    }
  }

  // Add effect for fetching data
  useEffect(() => {
    fetchMetrics()
    const interval = setInterval(fetchMetrics, 60000) // Refresh every minute
    return () => clearInterval(interval)
  }, [])

  // Add effect for fetching historical data when token or time range changes
  useEffect(() => {
    if (selectedToken) {
      fetchHistoricalData()
    }
  }, [selectedToken, timeRange])

  // Add countdown effect
  useEffect(() => {
    const updateTimer = () => {
      const now = new Date()
      const next = new Date(now)
      next.setMinutes(now.getMinutes() + 1, 0, 0)
      
      const diff = next.getTime() - now.getTime()
      if (diff <= 0) {
        fetchMetrics()
        return
      }
      
      const seconds = Math.floor(diff / 1000)
      setTimeToUpdate(`${Math.floor(seconds / 60)}m ${seconds % 60}s`)
    }

    updateTimer() // Initial call
    const timer = setInterval(updateTimer, 1000)

    return () => clearInterval(timer)
  }, [metrics])

  // Add effect to update holder counts
  useEffect(() => {
    const updateHolderCounts = async () => {
      if (!metrics.length) return;
      
      const updatedMetrics = await Promise.all(
        metrics.map(async (metric) => {
          if (!TOKEN_ADDRESSES[metric.symbol as TokenSymbol]) return metric;
          
          const holderCount = await fetchHolderCount(TOKEN_ADDRESSES[metric.symbol as TokenSymbol]);
          return { ...metric, holder_count: holderCount };
        })
      );
      
      setMetrics(updatedMetrics);
    };

    updateHolderCounts();
  }, [metrics]);

  if (error) {
    return (
      <div className="p-4 text-red-400">
        <h2 className="text-lg font-bold mb-2">Error</h2>
        <p>{error}</p>
        <button 
          onClick={fetchMetrics}
          className="mt-4 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 rounded"
        >
          Retry
        </button>
      </div>
    )
  }

  // Get the current metric option
  const currentMetric = metricOptions.find(m => m.value === selectedMetric)!;

  // Add loading indicator
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white p-8">
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
          <p className="text-gray-400">Loading metrics...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">Perpetual Metrics Dashboard</h1>
        <div className="text-right">
          <div className="text-sm text-gray-400">Next update in</div>
          <div className="text-xl font-mono text-purple-400">{timeToUpdate}</div>
        </div>
      </div>

      {/* Historical Chart */}
      <div className="mb-8 bg-gray-800/50 backdrop-blur-sm p-6 rounded-lg shadow-lg">
        <div className="flex justify-between items-center mb-6">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-bold text-purple-400">
              {selectedToken} - {currentMetric.label} History
            </h2>
            <select 
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value as MetricType)}
              className="bg-gray-700/50 text-white rounded-lg px-4 py-2 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              {metricOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <select 
            value={selectedToken}
            onChange={(e) => setSelectedToken(e.target.value)}
            className="bg-gray-700/50 text-white rounded-lg px-4 py-2 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
          >
            {metrics.map((m) => (
              <option key={m.symbol} value={m.symbol}>{m.symbol}</option>
            ))}
          </select>
        </div>
        
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={historicalData}>
              <defs>
                <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={currentMetric.color} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={currentMetric.color} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid 
                strokeDasharray="3 3" 
                vertical={false}
                stroke="#374151"
              />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(time: string) => format(parseISO(time), 'HH:mm')}
                stroke="#9CA3AF"
                tick={{ fill: '#9CA3AF' }}
                axisLine={{ stroke: '#4B5563' }}
                tickLine={{ stroke: '#4B5563' }}
              />
              <YAxis 
                domain={['auto', 'auto']}
                stroke="#9CA3AF"
                tick={{ fill: '#9CA3AF' }}
                axisLine={{ stroke: '#4B5563' }}
                tickLine={{ stroke: '#4B5563' }}
                tickFormatter={(value) => {
                  if (selectedMetric === 'funding_rate') {
                    return `${(value * 100).toFixed(2)}%`
                  }
                  if (['spot_volume_24h', 'liquidity', 'open_interest', 'market_cap'].includes(selectedMetric)) {
                    return `$${(value/1e6).toFixed(1)}M`
                  }
                  if (selectedMetric === 'txns_24h') {
                    return value.toLocaleString()
                  }
                  return `$${value.toFixed(4)}`
                }}
              />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length && payload[0].value != null) {
                    return (
                      <div className="bg-gray-800 border border-gray-700 p-4 rounded-lg shadow-lg">
                        <p className="text-gray-400 mb-2">
                          {format(parseISO(label), 'MMM dd, HH:mm')}
                        </p>
                        <p className="text-white font-medium">
                          {currentMetric.label}: {currentMetric.format(Number(payload[0].value))}
                        </p>
                      </div>
                    )
                  }
                  return null
                }}
              />
              <Area 
                type="monotone" 
                dataKey={selectedMetric}
                stroke={currentMetric.color}
                strokeWidth={2}
                fill="url(#colorGradient)"
                dot={false}
                activeDot={{
                  r: 6,
                  stroke: currentMetric.color,
                  strokeWidth: 2,
                  fill: '#1F2937'
                }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        
        {/* Add time range selector */}
        <div className="flex justify-center gap-4 mt-4">
          {(['1H', '4H', '12H', '24H'] as const).map((range) => (
            <button
              key={range}
              className={`px-4 py-2 rounded-lg transition-all ${
                timeRange === range 
                  ? 'bg-purple-600 text-white' 
                  : 'bg-gray-700/50 text-gray-400 hover:bg-gray-700'
              }`}
              onClick={() => setTimeRange(range)}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Current Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {metrics.map((metric) => (
          <div 
            key={metric.symbol} 
            className={`bg-gray-800 rounded-lg p-6 shadow-lg hover:shadow-xl transition-all ${
              (metric.spot_price === 0 && metric.symbol !== 'BRETT') ? 'opacity-70' : ''
            } ${selectedToken === metric.symbol ? 'ring-2 ring-purple-500' : ''}`}
            onClick={() => (metric.spot_price > 0 || metric.symbol === 'BRETT') && setSelectedToken(metric.symbol)}
            role="button"
            tabIndex={0}
          >
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold text-purple-400">{metric.symbol}</h2>
              {metric.price_change_24h !== 0 || metric.symbol === 'BRETT' ? (
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  metric.price_change_24h > 0 ? 'bg-green-500/20 text-green-400' :
                  metric.price_change_24h < 0 ? 'bg-red-500/20 text-red-400' :
                  'bg-gray-500/20 text-gray-400'
                }`}>
                  {metric.price_change_24h > 0 ? '↑' : '↓'} 
                  {Math.abs(metric.price_change_24h).toFixed(2)}%
                </span>
              ) : (
                <span className="px-3 py-1 rounded-full text-sm font-medium bg-gray-500/20 text-gray-400">
                  Inactive
                </span>
              )}
            </div>

            <div className="space-y-6">
              {/* Spot Market */}
              <div>
                <h3 className="text-sm font-semibold text-purple-400 mb-2">Spot Market</h3>
                <div className="grid grid-cols-2 gap-2">
                  <span className="text-gray-400">Price</span>
                  <span className="text-right font-mono">
                    {metric.spot_price > 0 ? `$${metric.spot_price.toFixed(4)}` : '-'}
                  </span>
                  <span className="text-gray-400">Volume 24h</span>
                  <span className="text-right font-mono">
                    {metric.spot_volume_24h > 0 ? `$${(metric.spot_volume_24h/1e6).toFixed(2)}M` : '-'}
                  </span>
                  <span className="text-gray-400">Liquidity</span>
                  <span className="text-right font-mono">
                    {metric.liquidity > 0 ? `$${(metric.liquidity/1e6).toFixed(2)}M` : '-'}
                  </span>
                </div>
              </div>

              {/* Perpetual Market */}
              <div>
                <h3 className="text-sm font-semibold text-purple-400 mb-2">Perpetual Market</h3>
                <div className="grid grid-cols-2 gap-2">
                  <span className="text-gray-400">Mark Price</span>
                  <span className="text-right font-mono">${metric.mark_price.toFixed(4)}</span>
                  <span className="text-gray-400">Funding Rate</span>
                  <span className="text-right font-mono">{(metric.funding_rate * 100).toFixed(4)}%</span>
                  <span className="text-gray-400">Open Interest</span>
                  <span className="text-right font-mono">${(metric.open_interest/1e6).toFixed(2)}M</span>
                </div>
              </div>

              {/* Market Stats */}
              <div>
                <h3 className="text-sm font-semibold text-purple-400 mb-2">Market Stats</h3>
                <div className="grid grid-cols-2 gap-2">
                  <span className="text-gray-400">Market Cap</span>
                  <span className="text-right font-mono">${(metric.market_cap/1e6).toFixed(2)}M</span>
                  <span className="text-gray-400">24h Transactions</span>
                  <span className="text-right font-mono">{metric.txns_24h.toLocaleString()}</span>
                  <span className="text-gray-400">Holders</span>
                  <span className="text-right font-mono">
                    {metric.holder_count !== null 
                      ? metric.holder_count.toLocaleString() 
                      : '-'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
} 