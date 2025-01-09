'use client'

import { useEffect, useState } from 'react'
import Image from 'next/image'
import { format } from 'date-fns'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

interface MarketUpdate {
  timestamp: string
  title: string
  summary: string
  metrics: {
    total_markets: number
    total_volume: number
    total_oi: number
    avg_funding: number
  }
  charts: {
    funding_distribution: string
    top_markets: string
    funding_volume: string
  }
  highlights: Array<{
    title: string
    content: string
  }>
}
  const [marketUpdates, setMarketUpdates] = useState<MarketUpdate[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function loadMarketUpdates() {
      try {
        // Load the latest market update
        const response = await fetch('/market-updates/latest.json')
        const data = await response.json()
        setMarketUpdates([data])
      } catch (error) {
        console.error('Error loading market updates:', error)
      } finally {
        setLoading(false)
      }
    }

    loadMarketUpdates()
  }, [])

  if (loading) {
    return <div>Loading market updates...</div>
  }

    <main className="min-h-screen bg-gray-900 text-white p-8">
      {marketUpdates.map((update, index) => (
        <div key={index} className="max-w-7xl mx-auto space-y-8">
          {/* Header */}
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-4">{update.title}</h1>
            <p className="text-gray-400">{update.summary}</p>
          </div>

          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-gray-400 mb-2">Total Markets</h3>
              <p className="text-3xl font-bold">{update.metrics.total_markets}</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-gray-400 mb-2">24h Volume</h3>
              <p className="text-3xl font-bold">${(update.metrics.total_volume / 1e6).toFixed(2)}M</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-gray-400 mb-2">Open Interest</h3>
              <p className="text-3xl font-bold">${(update.metrics.total_oi / 1e6).toFixed(2)}M</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-gray-400 mb-2">Avg Funding Rate</h3>
              <p className="text-3xl font-bold">{update.metrics.avg_funding.toFixed(4)}%</p>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold mb-4">Funding Rate Distribution</h3>
              <Image
                src={update.charts.funding_distribution}
                alt="Funding Rate Distribution"
                width={500}
                height={300}
                className="w-full"
                priority
              />
            </div>
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold mb-4">Top Markets by Open Interest</h3>
              <Image
                src={update.charts.top_markets}
                alt="Top Markets"
                width={500}
                height={300}
                className="w-full"
                priority
              />
            </div>
          </div>

          {/* Highlights */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {update.highlights.map((highlight, i) => (
              <div key={i} className="bg-gray-800 rounded-lg p-6">
                <h3 className="text-gray-400 mb-2">{highlight.title}</h3>
                <p className="text-xl font-bold">{highlight.content}</p>
              </div>
            ))}
          </div>

          {/* Timestamp */}
          <div className="text-center text-gray-400">
            Last updated: {format(new Date(update.timestamp), 'PPpp')}
          </div>
        </div>
      ))}