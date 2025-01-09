'use client'

import React, { useEffect, useState, useMemo } from 'react'
import { createClient } from '@supabase/supabase-js'
import type { Database } from '@/types/supabase'
import { XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart, CartesianGrid } from 'recharts'
import { format, parseISO } from 'date-fns'
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline'

// Update interfaces for our new data structure
interface FundingSnapshot {
  timestamp: string;
  token: string;
  current_funding_rate: number;
  predicted_funding_rate: number;
  mark_price: number;
  open_interest: number;
  notional_open_interest: number;
  volume_24h: number;
  avg_24h_funding_rate: number;
  exchange: string;
  metadata: {
    funding_difference: number;
  };
}

interface MarketMetrics {
  token: string;
  latest_snapshot: FundingSnapshot;
  historical_data: FundingSnapshot[];
}

// Create Supabase client
const supabase = createClient<Database>(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_KEY!
)

// Add this custom hook for controlled input
const useDebounceValue = (value: string, delay: number) => {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

export default function PerpMetricsPage() {
  const [markets, setMarkets] = useState<MarketMetrics[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy, setSortBy] = useState<'oi' | 'volume' | 'current_funding' | 'predicted' | 'opportunity'>('oi')
  const [recentSearches, setRecentSearches] = useState<string[]>([])
  const [showSearchHistory, setShowSearchHistory] = useState(false)
  const [inputValue, setInputValue] = useState('');
  const debouncedSearchQuery = useDebounceValue(inputValue, 300);

  // Update useEffect to use debouncedSearchQuery
  useEffect(() => {
    setSearchQuery(debouncedSearchQuery);
  }, [debouncedSearchQuery]);

  // Update search input handler for better responsiveness
  const handleSearchInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const query = e.target.value;
    setInputValue(query);
    setShowSearchHistory(query.length > 0);
  };

  // Handle search history
  const handleSearch = (query: string) => {
    setSearchQuery(query)
    if (query && !recentSearches.includes(query)) {
      setRecentSearches(prev => [query, ...prev.slice(0, 4)])
    }
    setShowSearchHistory(false)
  }

  // Sort and filter markets
  const sortedAndFilteredMarkets = useMemo(() => {
    let filtered = markets.filter(market => 
      market.token.toLowerCase().includes(searchQuery.toLowerCase())
    )

    return filtered.sort((a, b) => {
      switch (sortBy) {
        case 'volume':
          return (b.latest_snapshot.volume_24h || 0) - (a.latest_snapshot.volume_24h || 0)
        case 'current_funding':
          return Math.abs(b.latest_snapshot.current_funding_rate || 0) - Math.abs(a.latest_snapshot.current_funding_rate || 0)
        case 'predicted':
          return Math.abs(b.latest_snapshot.predicted_funding_rate || 0) - Math.abs(a.latest_snapshot.predicted_funding_rate || 0)
        case 'opportunity':
          const oppA = Math.abs((a.latest_snapshot.predicted_funding_rate || 0) - (a.latest_snapshot.current_funding_rate || 0))
          const oppB = Math.abs((b.latest_snapshot.predicted_funding_rate || 0) - (b.latest_snapshot.current_funding_rate || 0))
          return oppB - oppA
        case 'oi':
        default:
          return (b.latest_snapshot.notional_open_interest || 0) - (a.latest_snapshot.notional_open_interest || 0)
      }
    })
  }, [markets, searchQuery, sortBy])

  useEffect(() => {
    async function fetchMarketData() {
      try {
        setLoading(true)
        
        // Get the latest snapshot for each token
        const { data: latestSnapshots, error: snapshotError } = await supabase
          .from('funding_rate_snapshots')
          .select('*')
          .order('timestamp', { ascending: false })
          .limit(1000) // Adjust based on your needs

        if (snapshotError) throw snapshotError

        // Group by token and get latest data
        const tokenGroups = latestSnapshots.reduce((acc: { [key: string]: FundingSnapshot[] }, snapshot) => {
          if (!acc[snapshot.token]) {
            acc[snapshot.token] = []
          }
          acc[snapshot.token].push(snapshot)
          return acc
        }, {})

        // Get historical data for each token (last 24 hours)
        const marketMetrics: MarketMetrics[] = await Promise.all(
          Object.entries(tokenGroups).map(async ([token, snapshots]) => {
            const { data: historicalData, error: historyError } = await supabase
              .from('funding_rate_snapshots')
              .select('*')
              .eq('token', token)
              .gte('timestamp', new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString())
              .order('timestamp', { ascending: true })

            if (historyError) throw historyError

            return {
              token,
              latest_snapshot: snapshots[0],
              historical_data: historicalData
            }
          })
        )

        // Sort markets by notional open interest
        const sortedMarkets = marketMetrics.sort((a, b) => 
          (b.latest_snapshot.notional_open_interest || 0) - (a.latest_snapshot.notional_open_interest || 0)
        )

        setMarkets(sortedMarkets)
        setLoading(false)
      } catch (err) {
        console.error('Error fetching market data:', err)
        setError(err instanceof Error ? err.message : 'Failed to fetch market data')
        setLoading(false)
      }
    }

    fetchMarketData()
  }, [])

  // Update the filter buttons section
  const sortOptions = [
    { value: 'oi', label: 'Open Interest' },
    { value: 'volume', label: 'Volume' },
    { value: 'current_funding', label: '|Current Funding|' },
    { value: 'predicted', label: '|Predicted Funding|' },
    { value: 'opportunity', label: 'Funding Opportunities' }
  ]

  if (loading) return <div className="p-4">Loading market data...</div>
  if (error) return <div className="p-4 text-red-500">Error: {error}</div>

  return (
    <div className="p-4 max-w-7xl mx-auto">
      <div className="mb-6 space-y-4">
        <h1 className="text-2xl font-bold">Perpetual Markets Overview</h1>
        
        {/* Search and Sort Section */}
        <div className="flex flex-col md:flex-row gap-4 items-start md:items-center">
          {/* Enhanced Search Bar */}
          <div className="relative w-full md:w-96">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              value={inputValue}
              onChange={handleSearchInput}
              onFocus={() => setShowSearchHistory(inputValue.length > 0)}
              placeholder="Search markets..."
              className="block w-full pl-10 pr-3 py-2.5 border border-gray-600 rounded-lg 
                       bg-gray-800 text-white placeholder-gray-400
                       focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent
                       transition-all duration-200 ease-in-out
                       hover:border-gray-500
                       text-base leading-relaxed
                       selection:bg-purple-500/20
                       font-medium
                       shadow-sm
                       backdrop-blur-sm
                       focus:shadow-lg focus:shadow-purple-500/10"
              autoComplete="off"
              spellCheck="false"
              aria-label="Search markets"
            />
            <div className="absolute inset-y-0 right-0 flex items-center pr-3">
              {inputValue && (
                <button
                  onClick={() => {
                    setInputValue('');
                    setSearchQuery('');
                    setShowSearchHistory(false);
                  }}
                  className="p-1 hover:bg-gray-700/50 rounded-full transition-colors duration-200
                           text-gray-400 hover:text-gray-300 focus:outline-none focus:ring-2 
                           focus:ring-purple-500"
                  aria-label="Clear search"
                >
                  <span className="text-lg">×</span>
                </button>
              )}
            </div>

            {/* Improved Search History Dropdown */}
            {showSearchHistory && recentSearches.length > 0 && (
              <div className="absolute z-10 w-full mt-2 bg-gray-800/95 backdrop-blur-sm
                            border border-gray-700 rounded-lg shadow-lg
                            transform transition-all duration-200 ease-out
                            animate-in fade-in slide-in-from-top-2">
                {recentSearches.map((query, index) => (
                  <button
                    key={index}
                    onClick={() => {
                      setInputValue(query);
                      handleSearch(query);
                    }}
                    className="w-full px-4 py-2.5 text-left text-gray-300 
                             hover:bg-gray-700/50 first:rounded-t-lg last:rounded-b-lg 
                             flex items-center gap-2 transition-colors duration-150
                             group focus:outline-none focus:bg-gray-700/70"
                  >
                    <span className="text-gray-400 group-hover:text-purple-400 transition-colors duration-150">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </span>
                    <span className="truncate">{query}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Enhanced Sort Options */}
          <div className="flex flex-wrap gap-2 mb-4">
            {sortOptions.map((option) => (
              <button
                key={option.value}
                onClick={() => setSortBy(option.value as typeof sortBy)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors
                  ${sortBy === option.value
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-700/50 text-gray-400 hover:bg-gray-700/70'
                  }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* Results count */}
        <div className="text-sm text-gray-400">
          Showing {sortedAndFilteredMarkets.length} of {markets.length} markets
        </div>
      </div>
      
      {/* Markets Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {sortedAndFilteredMarkets.length > 0 ? (
          sortedAndFilteredMarkets.map((market) => (
            <div key={market.token} className="bg-gray-800 rounded-lg p-4">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold">{market.token}</h2>
                <span className="text-sm text-gray-400">
                  {format(parseISO(market.latest_snapshot.timestamp), 'MMM d, HH:mm')}
                </span>
              </div>

              {/* Price Chart */}
              <div className="h-32 mb-4">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={market.historical_data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp"
                      tickFormatter={(timestamp) => format(parseISO(timestamp), 'HH:mm')}
                    />
                    <YAxis />
                    <Tooltip 
                      labelFormatter={(timestamp) => format(parseISO(timestamp as string), 'MMM d, HH:mm')}
                      formatter={(value: number) => [`$${value.toFixed(4)}`, 'Price']}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="mark_price" 
                      stroke="#8884d8" 
                      fill="#8884d8" 
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Market Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h3 className="text-sm font-semibold text-purple-400 mb-2">Market Overview</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <span className="text-gray-400">Mark Price</span>
                    <span className="text-right font-mono">
                      ${market.latest_snapshot.mark_price.toFixed(4)}
                    </span>
                    <span className="text-gray-400">Funding Rate</span>
                    <span className="text-right font-mono">
                      {(market.latest_snapshot.current_funding_rate * 100).toFixed(4)}%
                    </span>
                    <span className="text-gray-400">Predicted Rate</span>
                    <span className="text-right font-mono">
                      {(market.latest_snapshot.predicted_funding_rate * 100).toFixed(4)}%
                    </span>
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-semibold text-purple-400 mb-2">Volume & OI</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <span className="text-gray-400">24h Volume</span>
                    <span className="text-right font-mono">
                      ${(market.latest_snapshot.volume_24h/1e6).toFixed(2)}M
                    </span>
                    <span className="text-gray-400">Open Interest</span>
                    <span className="text-right font-mono">
                      ${(market.latest_snapshot.notional_open_interest/1e6).toFixed(2)}M
                    </span>
                    <span className="text-gray-400">Avg 24h Rate</span>
                    <span className="text-right font-mono">
                      {(market.latest_snapshot.avg_24h_funding_rate * 100).toFixed(4)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="col-span-full text-center py-8 text-gray-400">
            No markets found matching "{searchQuery}"
          </div>
        )}
      </div>

      {/* Add Funding Highlights */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-green-900/20 rounded-lg p-4 border border-green-500/20">
          <h3 className="text-green-400 font-semibold mb-2">Highest Positive Funding</h3>
          {sortedAndFilteredMarkets
            .filter(m => m.latest_snapshot.current_funding_rate > 0)
            .slice(0, 3)
            .map(m => (
              <div key={m.token} className="flex justify-between items-center py-1">
                <span className="text-white">{m.token}</span>
                <span className="text-green-400">
                  {(m.latest_snapshot.current_funding_rate * 100).toFixed(4)}%
                </span>
              </div>
            ))}
        </div>

        <div className="bg-red-900/20 rounded-lg p-4 border border-red-500/20">
          <h3 className="text-red-400 font-semibold mb-2">Highest Negative Funding</h3>
          {sortedAndFilteredMarkets
            .filter(m => m.latest_snapshot.current_funding_rate < 0)
            .slice(0, 3)
            .map(m => (
              <div key={m.token} className="flex justify-between items-center py-1">
                <span className="text-white">{m.token}</span>
                <span className="text-red-400">
                  {(m.latest_snapshot.current_funding_rate * 100).toFixed(4)}%
                </span>
              </div>
            ))}
        </div>

        <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-500/20">
          <h3 className="text-blue-400 font-semibold mb-2">Largest Funding Opportunities</h3>
          {sortedAndFilteredMarkets
            .sort((a, b) => {
              const oppA = Math.abs(a.latest_snapshot.predicted_funding_rate - a.latest_snapshot.current_funding_rate)
              const oppB = Math.abs(b.latest_snapshot.predicted_funding_rate - b.latest_snapshot.current_funding_rate)
              return oppB - oppA
            })
            .slice(0, 3)
            .map(m => (
              <div key={m.token} className="flex justify-between items-center py-1">
                <span className="text-white">{m.token}</span>
                <div className="text-right">
                  <div className="text-gray-400 text-sm">Current: 
                    <span className={m.latest_snapshot.current_funding_rate > 0 ? 'text-green-400' : 'text-red-400'}>
                      {' '}{(m.latest_snapshot.current_funding_rate * 100).toFixed(4)}%
                    </span>
                  </div>
                  <div className="text-gray-400 text-sm">Predicted: 
                    <span className={m.latest_snapshot.predicted_funding_rate > 0 ? 'text-green-400' : 'text-red-400'}>
                      {' '}{(m.latest_snapshot.predicted_funding_rate * 100).toFixed(4)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Add Predicted Funding Rate Highlights */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-indigo-900/20 rounded-lg p-4 border border-indigo-500/20">
          <h3 className="text-indigo-400 font-semibold mb-2">Highest Predicted Funding</h3>
          {sortedAndFilteredMarkets
            .sort((a, b) => b.latest_snapshot.predicted_funding_rate - a.latest_snapshot.predicted_funding_rate)
            .slice(0, 5)
            .map(m => (
              <div key={m.token} className="flex justify-between items-center py-1">
                <span className="text-white">{m.token}</span>
                <div className="text-right">
                  <span className="text-indigo-400">
                    {(m.latest_snapshot.predicted_funding_rate * 100).toFixed(4)}%
                  </span>
                  <span className="text-gray-400 text-sm ml-2">
                    (Current: {(m.latest_snapshot.current_funding_rate * 100).toFixed(4)}%)
                  </span>
                </div>
              </div>
            ))}
        </div>

        <div className="bg-violet-900/20 rounded-lg p-4 border border-violet-500/20">
          <h3 className="text-violet-400 font-semibold mb-2">Lowest Predicted Funding</h3>
          {sortedAndFilteredMarkets
            .sort((a, b) => a.latest_snapshot.predicted_funding_rate - b.latest_snapshot.predicted_funding_rate)
            .slice(0, 5)
            .map(m => (
              <div key={m.token} className="flex justify-between items-center py-1">
                <span className="text-white">{m.token}</span>
                <div className="text-right">
                  <span className="text-violet-400">
                    {(m.latest_snapshot.predicted_funding_rate * 100).toFixed(4)}%
                  </span>
                  <span className="text-gray-400 text-sm ml-2">
                    (Current: {(m.latest_snapshot.current_funding_rate * 100).toFixed(4)}%)
                  </span>
                </div>
              </div>
            ))}
        </div>

        <div className="bg-fuchsia-900/20 rounded-lg p-4 border border-fuchsia-500/20">
          <h3 className="text-fuchsia-400 font-semibold mb-2">Largest Predicted Changes</h3>
          {sortedAndFilteredMarkets
            .sort((a, b) => {
              const changeA = Math.abs(a.latest_snapshot.predicted_funding_rate - a.latest_snapshot.current_funding_rate)
              const changeB = Math.abs(b.latest_snapshot.predicted_funding_rate - b.latest_snapshot.current_funding_rate)
              return changeB - changeA
            })
            .slice(0, 5)
            .map(m => (
              <div key={m.token} className="flex justify-between items-center py-1">
                <span className="text-white">{m.token}</span>
                <div className="text-right">
                  <div className="text-sm">
                    <span className="text-gray-400">Pred: </span>
                    <span className={m.latest_snapshot.predicted_funding_rate > 0 ? 'text-green-400' : 'text-red-400'}>
                      {(m.latest_snapshot.predicted_funding_rate * 100).toFixed(4)}%
                    </span>
                  </div>
                  <div className="text-sm">
                    <span className="text-gray-400">Curr: </span>
                    <span className={m.latest_snapshot.current_funding_rate > 0 ? 'text-green-400' : 'text-red-400'}>
                      {(m.latest_snapshot.current_funding_rate * 100).toFixed(4)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  )
} 