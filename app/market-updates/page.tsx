'use client'

import { useEffect, useState } from 'react'
import Image from 'next/image'
import { format } from 'date-fns'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import Link from 'next/link'
import type { MarketUpdate } from '../../types/market'
import { useRouter } from 'next/navigation'

export default function MarketUpdatesPage() {
  const router = useRouter()
  const [marketUpdates, setMarketUpdates] = useState<MarketUpdate[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function loadMarketUpdates() {
      try {
        const response = await fetch('/market-updates/latest.json')
        if (!response.ok) {
          throw new Error('Failed to load market updates')
        }
        const data = await response.json()
        setMarketUpdates([data])
      } catch (error) {
        console.error('Error loading market updates:', error)
        setError('Failed to load market data. Please try again later.')
      } finally {
        setLoading(false)
      }
    }

    loadMarketUpdates()
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="animate-pulse">Loading market updates...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center gap-4">
        <div className="text-red-400">{error}</div>
        <button 
          onClick={() => router.refresh()}
          className="px-4 py-2 bg-gray-800 rounded hover:bg-gray-700"
        >
          Retry
        </button>
      </div>
    )
  }

  return (
    <main className="min-h-screen bg-gray-900 text-white p-8">
      {/* Navigation Bar */}
      <nav className="max-w-7xl mx-auto mb-8 flex justify-between items-center">
        <Link 
          href="/"
          className="inline-flex items-center px-4 py-2 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors"
        >
          <span className="mr-2">←</span> Back to Terminal
        </Link>
        <Link
          href="/perp-metrics"
          className="px-4 py-2 bg-purple-600 rounded-lg hover:bg-purple-700 transition-colors"
        >
          View Detailed Metrics →
        </Link>
      </nav>

      {marketUpdates.map((update, index) => (
        <div key={index} className="max-w-7xl mx-auto space-y-8">
          {/* Existing content */}
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-4">{update.title}</h1>
            <p className="text-gray-400">{update.summary}</p>
          </div>

          {/* Enhanced Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gray-800 rounded-lg p-6 hover:bg-gray-700 transition-colors">
              <h3 className="text-gray-400 mb-2">Total Markets</h3>
              <p className="text-3xl font-bold">{update.metrics.total_markets}</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-6 hover:bg-gray-700 transition-colors">
              <h3 className="text-gray-400 mb-2">24h Volume</h3>
              <p className="text-3xl font-bold">${(update.metrics.total_volume / 1e6).toFixed(2)}M</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-6 hover:bg-gray-700 transition-colors">
              <h3 className="text-gray-400 mb-2">Open Interest</h3>
              <p className="text-3xl font-bold">${(update.metrics.total_oi / 1e6).toFixed(2)}M</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-6 hover:bg-gray-700 transition-colors">
              <h3 className="text-gray-400 mb-2">Avg Funding Rate</h3>
              <p className="text-3xl font-bold">{update.metrics.avg_funding.toFixed(4)}%</p>
            </div>
          </div>

          {/* Market Data Section */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div className="bg-indigo-900/20 rounded-lg p-6 border border-indigo-500/20">
              <h3 className="text-indigo-400 font-semibold mb-4">Highest Funding Rate</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Token</span>
                  <span className="font-mono">{update.market_data.highest_funding.token}</span>
                </div>
                <div className="flex justify-between">
                  <span>Rate</span>
                  <span className="font-mono">{(update.market_data.highest_funding.rate * 100).toFixed(4)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Volume</span>
                  <span className="font-mono">${(update.market_data.highest_funding.volume / 1e6).toFixed(2)}M</span>
                </div>
              </div>
            </div>
            
            {/* Add similar cards for most_active and largest_oi */}
          </div>

          {/* Existing charts and highlights sections */}
        </div>
      ))}
    </main>
  )
} 