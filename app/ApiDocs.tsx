'use client'

import React, { useState } from 'react'
import { Tabs, TabPanel } from './components/Tabs'
import CodeBlock from './components/CodeBlock'
import { Section } from './components/Section'
import { SideNav } from './components/SideNav'

// Define types for the content structure
interface EndpointParam {
  name: string
  type: string
  required: boolean
  description: string
}

interface EndpointResponse {
  description: string
  example: string
}

interface Endpoint {
  name: string
  method: string
  path: string
  description: string
  params?: EndpointParam[]
  response?: EndpointResponse
  example: string
  rateLimit?: string
}

interface DataSourceContent {
  title: string
  description: string
  endpoints: Endpoint[]
  features?: string[]
  authentication?: {
    type: string
    details: string
    example: string
  }
  sdkExample?: {
    title: string
    description: string
    code: string
  }
}

interface Feature {
  title: string
  description: string
  details: string[]
}

interface OverviewContent {
  title: string
  description: string
  features: Feature[]
  quickStart: {
    title: string
    steps: {
      title: string
      code: string
    }[]
  }
  supportedPlatforms: {
    name: string
    features: string[]
    rateLimit: string
  }[]
}

interface GuideStep {
  title: string
  content: string
  code?: string
}

interface GuideContent {
  title: string
  description: string
  steps: GuideStep[]
}

interface RateLimitInfo {
  tier: string
  limits: {
    requests: string
    dataPoints: string
    concurrent: string
  }
}

interface AuthProvider {
  name: string
  authType: string
  details: string
  example: {
    code: string
    description: string
  }[]
  rateLimit: string
  headers: {
    name: string
    value: string
    required: boolean
  }[]
  bestPractices: string[]
}

interface AuthenticationContent {
  title: string
  description: string
  providers: AuthProvider[]
  securityBestPractices: {
    title: string
    description: string
    example?: string
  }[]
}

interface RateLimitsContent {
  title: string
  description: string
  providers: Record<string, RateLimitInfo>
}

interface GuidesContent {
  title: string
  description: string
  guides: {
    [key: string]: {
      title: string
      description: string
      steps: GuideStep[]
    }
  }
}

interface ContentType {
  overview: OverviewContent
  defillama: DataSourceContent
  dune: DataSourceContent
  footprint: DataSourceContent
  flipside: DataSourceContent
  messari: DataSourceContent
  bitquery: DataSourceContent
  subgraphs: DataSourceContent
  dex: DataSourceContent
  lending: DataSourceContent
  metrics: DataSourceContent
  authentication: AuthenticationContent
  rateLimits: RateLimitsContent
  guides: GuidesContent
  liquidStaking: DataSourceContent
  protocols: DataSourceContent
  tvl: DataSourceContent
  bridges: DataSourceContent
  perpetuals: DataSourceContent
  synthetics: DataSourceContent
  nfts: DataSourceContent
  eigenlayer: DataSourceContent
  'best-practices': GuideContent
  hypurrscan: DataSourceContent
  [key: string]: OverviewContent | DataSourceContent | AuthenticationContent | RateLimitsContent | GuidesContent | GuideContent
}

// First, let's define the Provider type if not already defined
interface Provider {
  id: string;
  name: string;
  description?: string;
}

const ApiDocs = () => {
  const [activeTab, setActiveTab] = useState('overview')
  const [selectedProviderId, setSelectedProviderId] = useState<string>('')

  // Add providers array
  const providers: Provider[] = [
    { id: 'defillama', name: 'DefiLlama' },
    { id: 'dune', name: 'Dune Analytics' },
    { id: 'footprint', name: 'Footprint Analytics' },
    { id: 'flipside', name: 'Flipside Crypto' },
    { id: 'messari', name: 'Messari' },
    { id: 'bitquery', name: 'BitQuery' },
    { id: 'subgraphs', name: 'The Graph' },
    { id: 'hypurrscan', name: 'Hypurrscan', description: 'Hyperliquid L1 Analytics & Data' }
  ]

  // Handle provider change
  const handleProviderChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedProviderId(event.target.value)
  }

  // Handle navigation
  const handleNavigation = (sectionId: string) => {
    setActiveTab(sectionId)
  }

  // Expanded sections to include all documentation sources
  const sections = [
    {
      title: 'Getting Started',
      items: [
        { id: 'overview', label: 'Overview' },
        { id: 'authentication', label: 'Authentication' },
        { id: 'best-practices', label: 'Best Practices' }
      ]
    },
    {
      title: 'Data Sources',
      items: [
        { id: 'defillama', label: 'DefiLlama' },
        { id: 'dune', label: 'Dune Analytics' },
        { id: 'footprint', label: 'Footprint Analytics' },
        { id: 'flipside', label: 'Flipside Crypto' },
        { id: 'messari', label: 'Messari' },
        { id: 'bitquery', label: 'BitQuery' },
        { id: 'subgraphs', label: 'The Graph' },
        { id: 'hypurrscan', label: 'Hypurrscan', description: 'Hyperliquid L1 Analytics & Data' }
      ]
    },
    {
      title: 'DeFi Data',
      items: [
        { id: 'protocols', label: 'Protocols' },
        { id: 'tvl', label: 'TVL' },
        { id: 'dex', label: 'DEX' },
        { id: 'lending', label: 'Lending' },
        { id: 'bridges', label: 'Bridges' },
        { id: 'perpetuals', label: 'Perpetuals' },
        { id: 'synthetics', label: 'Synthetics' }
      ]
    },
    {
      title: 'Analytics',
      items: [
        { id: 'metrics', label: 'Chain Metrics' },
        { id: 'nfts', label: 'NFT Analytics' },
        { id: 'eigenlayer', label: 'EigenLayer' }
      ]
    }
  ]

  // Enhanced content object with comprehensive documentation from all sources
  const content: ContentType = {
    overview: {
      title: 'Blockchain Data API Documentation',
      description: `Our unified blockchain data API provides enterprise-grade access to comprehensive DeFi, NFT, and blockchain analytics 
        through a single, standardized interface. By aggregating data from industry-leading providers, we offer real-time insights, 
        historical data, and advanced analytics capabilities for institutional investors and DeFi applications.`,
      features: [
        {
          title: 'Multi-Platform Data Aggregation',
          description: 'Unified access to DefiLlama, Dune Analytics, Bitquery, The Graph, and Footprint Analytics through a single API',
          details: [
            'Real-time protocol TVL and metrics',
            'Cross-chain transaction monitoring',
            'DeFi protocol analytics',
            'Market intelligence and trends'
          ]
        },
        {
          title: 'Real-Time Data Streaming',
          description: 'WebSocket-based streaming for instant market updates and blockchain events',
          details: [
            'Live DEX trading data',
            'Real-time TVL changes',
            'Protocol metrics streaming',
            'Custom event notifications'
          ]
        },
        {
          title: 'Advanced Analytics',
          description: 'Comprehensive analytics and data processing capabilities',
          details: [
            'Custom SQL queries via Dune Analytics',
            'GraphQL support for flexible data fetching',
            'Historical trend analysis',
            'Cross-protocol correlation metrics'
          ]
        },
        {
          title: 'Enterprise Features',
          description: 'Production-ready infrastructure for institutional needs',
          details: [
            'High-availability architecture',
            'Rate limiting and quota management',
            'Data validation and sanitization',
            'Comprehensive error handling'
          ]
        },
        {
          title: 'Integration Support',
          description: 'Multiple integration options for different use cases',
          details: [
            'RESTful API endpoints',
            'WebSocket streams',
            'GraphQL queries',
            'SDK libraries for major languages'
          ]
        }
      ],
      quickStart: {
        title: 'Quick Start Guide',
        steps: [
          {
            title: 'Authentication Setup',
            code: `const client = new BlockchainAPI({
  defiLlama: {
    apiKey: process.env.DEFILLAMA_KEY,
    rateLimit: 300  // requests per 5 minutes
  },
  dune: {
    apiKey: process.env.DUNE_KEY,
    rateLimit: 100  // requests per day
  },
  bitquery: {
    apiKey: process.env.BITQUERY_KEY,
    rateLimit: 100  // requests per day
  }
});`
          },
          {
            title: 'Fetching Protocol Data',
            code: `// Get comprehensive protocol metrics
const protocolData = await client.protocols.getData('aave', {
  metrics: ['tvl', 'volume24h', 'fees24h'],
  timeRange: '7d',
  chains: ['ethereum', 'polygon']
});`
          },
          {
            title: 'Real-Time Data Streaming',
            code: `// Stream DEX trading activity
client.dex.streamTrades({
  protocol: 'uniswap-v3',
  pairs: ['ETH/USDC', 'ETH/USDT'],
  onData: (trade) => {
    console.log('New trade:', trade);
  },
  onError: (error) => {
    console.error('Stream error:', error);
  }
});`
          }
        ]
      },
      supportedPlatforms: [
        {
          name: 'DefiLlama',
          features: ['TVL tracking', 'Protocol metrics', 'Token prices'],
          rateLimit: '300 requests/5min (free tier)'
        },
        {
          name: 'Dune Analytics',
          features: ['Custom SQL queries', 'Historical data', 'Cross-chain analytics'],
          rateLimit: '100 queries/day (free tier)'
        },
        {
          name: 'Bitquery',
          features: ['Real-time data', 'GraphQL API', 'DEX analytics'],
          rateLimit: '100 requests/day (free tier)'
        },
        {
          name: 'The Graph',
          features: ['Indexed blockchain data', 'Protocol-specific subgraphs', 'Custom queries'],
          rateLimit: '1000 queries/day (free tier)'
        },
        {
          name: 'Footprint Analytics',
          features: ['Market intelligence', 'DeFi metrics', 'NFT analytics'],
          rateLimit: '1000 requests/day (free tier)'
        }
      ]
    },
    defillama: {
      title: 'DefiLlama Integration',
      description: `Access comprehensive DeFi protocol data through DefiLlama's API. Get real-time TVL metrics, 
        token prices, yield farming opportunities, and historical data across multiple chains and protocols.`,
      features: [
        'Real-time TVL tracking across chains',
        'Historical protocol metrics',
        'Token price aggregation',
        'Yield farming analytics',
        'Cross-chain protocol data'
      ],
      authentication: {
        type: 'API Key',
        details: 'API key required for authenticated endpoints, passed via x-api-key header',
        example: `curl -H "x-api-key: YOUR_API_KEY" https://api.llama.fi/protocol/aave`
      },
      sdkExample: {
        title: 'SDK Integration',
        description: 'Use the official DefiLlama SDK for easier integration',
        code: `import { DefiLlamaClient } from '@defillama/sdk';

const client = new DefiLlamaClient({
  apiKey: process.env.DEFILLAMA_API_KEY
});

// Get protocol TVL and metrics
const protocolData = await client.getProtocolData('aave');

// Get historical TVL data
const historicalTVL = await client.getHistoricalTVL('aave', {
  timestamp_start: '2023-01-01',
  timestamp_end: '2023-12-31'
});

// Get token prices
const prices = await client.getCurrentPrices([
  'ethereum:0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9'
]);`
      },
      endpoints: [
        {
          name: 'Get Protocol TVL',
          method: 'GET',
          path: '/protocol/{protocol}',
          description: 'Returns comprehensive TVL and key metrics for a specific protocol',
          params: [
            {
              name: 'protocol',
              type: 'string',
              required: true,
              description: 'Protocol slug (e.g., aave, compound)'
            }
          ],
          response: {
            description: 'Detailed protocol metrics including TVL, volume, and chain breakdown',
            example: `{
  "name": "Aave",
  "tvl": 3941797982.49,
  "chainTvls": {
    "ethereum": 2547383901.23,
    "polygon": 893514081.26,
    "avalanche": 500900000.00
  },
  "change_1h": 0.12,
  "change_24h": -1.53,
  "change_7d": 2.81
}`
          },
          example: `curl "https://api.llama.fi/protocol/aave"`,
          rateLimit: '300 requests/5min'
        },
        {
          name: 'Get Token Prices',
          method: 'GET',
          path: '/prices/current/{chain}:{token}',
          description: 'Get current token prices across multiple chains',
          params: [
            {
              name: 'chain',
              type: 'string',
              required: true,
              description: 'Chain identifier (e.g., ethereum, polygon)'
            },
            {
              name: 'token',
              type: 'string',
              required: true,
              description: 'Token contract address'
            }
          ],
          response: {
            description: 'Current token price and metadata',
            example: `{
  "coins": {
    "ethereum:0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": {
      "price": 92.34,
      "symbol": "AAVE",
      "timestamp": 1678234567,
      "confidence": 0.99
    }
  }
}`
          },
          example: `curl "https://coins.llama.fi/prices/current/ethereum:0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9"`,
          rateLimit: '300 requests/5min'
        },
        {
          name: 'Get Historical TVL',
          method: 'GET',
          path: '/tvl/{protocol}',
          description: 'Get historical TVL data for a protocol',
          params: [
            {
              name: 'protocol',
              type: 'string',
              required: true,
              description: 'Protocol slug'
            },
            {
              name: 'start',
              type: 'number',
              required: false,
              description: 'Start timestamp (Unix)'
            },
            {
              name: 'end',
              type: 'number',
              required: false,
              description: 'End timestamp (Unix)'
            }
          ],
          response: {
            description: 'Historical TVL data points',
            example: `[
  {
    "date": 1678234567,
    "tvl": 3941797982.49
  },
  {
    "date": 1678148167,
    "tvl": 3938567123.78
  }
]`
          },
          example: `curl "https://api.llama.fi/tvl/aave"`,
          rateLimit: '300 requests/5min'
        },
        {
          name: 'Get Yield Pools',
          method: 'GET',
          path: '/pools',
          description: 'Get yield farming pool data across protocols',
          params: [
            {
              name: 'protocol',
              type: 'string',
              required: false,
              description: 'Filter by protocol'
            },
            {
              name: 'chain',
              type: 'string',
              required: false,
              description: 'Filter by chain'
            }
          ],
          response: {
            description: 'List of yield pools with APY and TVL data',
            example: `{
  "data": [
    {
      "pool": "USDC-ETH",
      "protocol": "Aave",
      "chain": "ethereum",
      "apy": 3.45,
      "tvlUsd": 125789456.78,
      "apyBase": 2.1,
      "apyReward": 1.35
    }
  ]
}`
          },
          example: `curl "https://yields.llama.fi/pools"`,
          rateLimit: '300 requests/5min'
        }
      ]
    },
    dune: {
      title: 'Dune Analytics Integration',
      description: `Access comprehensive blockchain analytics through Dune's SQL-based query platform. 
        Create, execute, and retrieve results from custom SQL queries across multiple blockchain networks.`,
      features: [
        'Custom SQL query execution',
        'Pre-built analytics dashboards',
        'Cross-chain data analysis',
        'Real-time query results',
        'Historical blockchain data'
      ],
      authentication: {
        type: 'API Key',
        details: 'Authentication requires an API key passed in the X-Dune-API-Key header',
        example: `curl -H "X-Dune-API-Key: YOUR_API_KEY" https://api.dune.com/api/v1/query/1234/results`
      },
      sdkExample: {
        title: 'SDK Integration',
        description: 'Use the official Dune SDK for easier query execution and result handling',
        code: `import { DuneClient } from '@dune/client';

const client = new DuneClient({
  apiKey: process.env.DUNE_API_KEY
});

// Execute a saved query
const queryResult = await client.refresh(1234567);

// Get query results
const results = await client.getResult(1234567);

// Execute a query with parameters
const parameterizedResult = await client.refresh(1234567, {
  params: {
    protocol: 'uniswap',
    timeframe: '7d',
    chain: 'ethereum'
  }
});`
      },
      endpoints: [
        {
          name: 'Execute Query',
          method: 'POST',
          path: '/api/v1/query/{query_id}/execute',
          description: 'Execute a saved Dune query with optional parameters',
          params: [
            {
              name: 'query_id',
              type: 'number',
              required: true,
              description: 'ID of the saved query to execute'
            },
            {
              name: 'params',
              type: 'object',
              required: false,
              description: 'Query parameters object'
            }
          ],
          response: {
            description: 'Query execution status and metadata',
            example: `{
  "execution_id": "01H6QXPNB6BCQYPZN8RJ6SV4F5",
  "state": "QUERY_STATE_EXECUTING",
  "submitted_at": "2023-07-20T10:30:45Z",
  "expires_at": "2023-07-20T11:30:45Z"
}`
          },
          example: `curl -X POST "https://api.dune.com/api/v1/query/1234567/execute" \\
  -H "X-Dune-API-Key: YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"params": {"protocol": "uniswap"}}'`,
          rateLimit: '100 queries/day'
        },
        {
          name: 'Get Query Results',
          method: 'GET',
          path: '/api/v1/query/{query_id}/results',
          description: 'Retrieve results from a previously executed query',
          params: [
            {
              name: 'query_id',
              type: 'number',
              required: true,
              description: 'ID of the executed query'
            }
          ],
          response: {
            description: 'Query results with metadata',
            example: `{
  "execution_id": "01H6QXPNB6BCQYPZN8RJ6SV4F5",
  "state": "QUERY_STATE_COMPLETED",
  "submitted_at": "2023-07-20T10:30:45Z",
  "expires_at": "2023-07-20T11:30:45Z",
  "result": {
    "rows": [
      {
        "protocol": "uniswap",
        "volume_24h": 1234567.89,
        "tvl": 987654321.0,
        "unique_users": 5432
      }
    ],
    "metadata": {
      "column_names": ["protocol", "volume_24h", "tvl", "unique_users"],
      "result_set_bytes": 1234,
      "total_row_count": 1
    }
  }
}`
          },
          example: `curl "https://api.dune.com/api/v1/query/1234567/results" \\
  -H "X-Dune-API-Key: YOUR_API_KEY"`,
          rateLimit: '100 queries/day'
        },
        {
          name: 'Get Query Status',
          method: 'GET',
          path: '/api/v1/query/{query_id}/status',
          description: 'Check the execution status of a query',
          params: [
            {
              name: 'query_id',
              type: 'number',
              required: true,
              description: 'ID of the query to check'
            }
          ],
          response: {
            description: 'Current query execution status',
            example: `{
  "execution_id": "01H6QXPNB6BCQYPZN8RJ6SV4F5",
  "state": "QUERY_STATE_COMPLETED",
  "submitted_at": "2023-07-20T10:30:45Z",
  "expires_at": "2023-07-20T11:30:45Z"
}`
          },
          example: `curl "https://api.dune.com/api/v1/query/1234567/status" \\
  -H "X-Dune-API-Key: YOUR_API_KEY"`,
          rateLimit: '100 queries/day'
        },
        {
          name: 'Get Query Metadata',
          method: 'GET',
          path: '/api/v1/query/{query_id}',
          description: 'Get metadata about a saved query',
          params: [
            {
              name: 'query_id',
              type: 'number',
              required: true,
              description: 'ID of the query'
            }
          ],
          response: {
            description: 'Query metadata and parameters',
            example: `{
  "query_id": 1234567,
  "name": "Uniswap Protocol Metrics",
  "description": "Key metrics for Uniswap protocol",
  "parameters": [
    {
      "name": "protocol",
      "type": "text",
      "default": "uniswap"
    },
    {
      "name": "timeframe",
      "type": "text",
      "default": "7d"
    }
  ],
  "created_at": "2023-01-01T00:00:00Z",
  "updated_at": "2023-07-20T10:30:45Z",
  "user": {
    "id": 12345,
    "name": "DuneAnalyst"
  }
}`
          },
          example: `curl "https://api.dune.com/api/v1/query/1234567" \\
  -H "X-Dune-API-Key: YOUR_API_KEY"`,
          rateLimit: '100 queries/day'
        }
      ]
    },
    footprint: {
      title: 'Footprint Analytics Integration',
      description: `Access comprehensive blockchain market intelligence and analytics through Footprint's API. 
        Get detailed protocol metrics, market trends, and cross-chain analytics with historical and real-time data.`,
      features: [
        'Real-time market intelligence',
        'Cross-chain protocol analytics',
        'NFT market tracking',
        'DeFi protocol metrics',
        'Historical trend analysis'
      ],
      authentication: {
        type: 'Bearer Token',
        details: 'Authentication requires a Bearer token in the Authorization header',
        example: `curl -H "Authorization: Bearer YOUR_TOKEN" https://api.footprint.network/api/v1/protocol/aave/metrics`
      },
      sdkExample: {
        title: 'SDK Integration',
        description: 'Use the Footprint SDK for easier data access and analysis',
        code: `import { FootprintClient } from '@footprint/sdk';

const client = new FootprintClient({
  apiToken: process.env.FOOTPRINT_API_TOKEN
});

// Get protocol metrics
const protocolMetrics = await client.getProtocolMetrics('aave', {
  timeframe: '7d',
  metrics: ['tvl', 'volume', 'users']
});

// Get market overview
const marketData = await client.getMarketOverview({
  chains: ['ethereum', 'bsc', 'polygon'],
  timeRange: '24h'
});

// Stream real-time updates
client.streamMetrics({
  protocols: ['aave', 'compound'],
  metrics: ['tvl', 'volume'],
  onData: (update) => {
    console.log('Metric update:', update);
  }
});`
      },
      endpoints: [
        {
          name: 'Get Protocol Metrics',
          method: 'GET',
          path: '/api/v1/protocol/{protocol}/metrics',
          description: 'Get comprehensive protocol metrics and analytics',
          params: [
            {
              name: 'protocol',
              type: 'string',
              required: true,
              description: 'Protocol identifier'
            },
            {
              name: 'timeframe',
              type: 'string',
              required: false,
              description: 'Time range for metrics (e.g., 24h, 7d, 30d)'
            },
            {
              name: 'metrics',
              type: 'string[]',
              required: false,
              description: 'Specific metrics to retrieve'
            }
          ],
          response: {
            description: 'Detailed protocol metrics and analytics',
            example: `{
  "protocol": "aave",
  "timestamp": "2023-07-20T10:30:45Z",
  "metrics": {
    "tvl": 5678901234.56,
    "volume_24h": 123456789.01,
    "unique_users": 12345,
    "active_markets": 25,
    "total_borrowed": 3456789012.34,
    "total_supplied": 6789012345.67,
    "apy_ranges": {
      "supply": {
        "min": 1.2,
        "max": 8.5
      },
      "borrow": {
        "min": 3.4,
        "max": 12.6
      }
    }
  },
  "chain_breakdown": {
    "ethereum": {
      "tvl": 3456789012.34,
      "volume_24h": 89012345.67
    },
    "polygon": {
      "tvl": 1234567890.12,
      "volume_24h": 34567890.12
    }
  }
}`
          },
          example: `curl "https://api.footprint.network/api/v1/protocol/aave/metrics?timeframe=7d" \\
  -H "Authorization: Bearer YOUR_TOKEN"`,
          rateLimit: '100 requests/minute'
        },
        {
          name: 'Get Market Overview',
          method: 'GET',
          path: '/api/v1/market/overview',
          description: 'Get comprehensive market overview and trends',
          params: [
            {
              name: 'chains',
              type: 'string[]',
              required: false,
              description: 'Filter by specific chains'
            },
            {
              name: 'protocols',
              type: 'string[]',
              required: false,
              description: 'Filter by specific protocols'
            }
          ],
          response: {
            description: 'Market overview with key metrics and trends',
            example: `{
  "timestamp": "2023-07-20T10:30:45Z",
  "total_tvl": 123456789012.34,
  "total_volume_24h": 9876543210.12,
  "active_users_24h": 123456,
  "top_protocols": [
    {
      "name": "aave",
      "tvl": 5678901234.56,
      "volume_24h": 123456789.01,
      "market_share": 12.34
    }
  ],
  "chain_metrics": {
    "ethereum": {
      "tvl": 45678901234.56,
      "volume_24h": 3456789012.34,
      "active_users": 45678
    }
  },
  "defi_trends": {
    "lending": {
      "tvl": 34567890123.45,
      "growth_24h": 2.34
    },
    "dex": {
      "tvl": 23456789012.34,
      "growth_24h": 1.23
    }
  }
}`
          },
          example: `curl "https://api.footprint.network/api/v1/market/overview?chains=ethereum,polygon" \\
  -H "Authorization: Bearer YOUR_TOKEN"`,
          rateLimit: '100 requests/minute'
        }
      ]
    },
    flipside: {
      title: 'Flipside Crypto Integration',
      description: `Access comprehensive blockchain analytics through Flipside's SQL-based query engine. 
        Execute custom SQL queries, analyze historical data, and track real-time metrics across multiple chains.`,
      features: [
        'Custom SQL query execution',
        'Cross-chain analytics',
        'Historical blockchain data',
        'Real-time metrics tracking',
        'Query result caching'
      ],
      authentication: {
        type: 'API Key',
        details: 'Authentication requires an API key passed in the x-api-key header',
        example: `curl -H "x-api-key: YOUR_API_KEY" https://api.flipsidecrypto.com/v1/query`
      },
      sdkExample: {
        title: 'SDK Integration',
        description: 'Use the Flipside SDK for easier query execution and data analysis',
        code: `import { FlipsideClient } from '@flipsidecrypto/sdk';

const client = new FlipsideClient({
  apiKey: process.env.FLIPSIDE_API_KEY
});

// Execute a SQL query
const result = await client.query({
  sql: \`
    SELECT 
      date_trunc('day', block_timestamp) as date,
      count(distinct tx_sender) as unique_users,
      sum(amount) as total_volume
    FROM ethereum.core.fact_transactions
    WHERE block_timestamp >= current_date - 7
    GROUP BY 1
    ORDER BY 1
  \`,
  ttlMinutes: 15
});

// Get cached query results
const cachedResult = await client.getQueryResult(queryId);

// Stream query results
client.streamQuery({
  sql: 'SELECT * FROM ethereum.core.fact_transactions LIMIT 1000',
  onData: (row) => {
    console.log('New transaction:', row);
  },
  onError: (error) => {
    console.error('Query error:', error);
  }
});`
      },
      endpoints: [
        {
          name: 'Execute Query',
          method: 'POST',
          path: '/v1/query',
          description: 'Execute SQL queries against Flipside datasets',
          params: [
            {
              name: 'sql',
              type: 'string',
              required: true,
              description: 'SQL query to execute'
            },
            {
              name: 'ttlMinutes',
              type: 'number',
              required: false,
              description: 'Cache duration in minutes'
            },
            {
              name: 'cached',
              type: 'boolean',
              required: false,
              description: 'Whether to use cached results if available'
            }
          ],
          response: {
            description: 'Query execution results and metadata',
            example: `{
  "query_id": "01H6QXPNB6BCQYPZN8RJ6SV4F5",
  "status": "success",
  "results": [
    {
      "date": "2023-07-20",
      "unique_users": 123456,
      "total_volume": 987654321.12
    }
  ],
  "metadata": {
    "row_count": 1,
    "execution_time_ms": 1234,
    "cached": true,
    "cache_expires_at": "2023-07-20T11:30:45Z"
  }
}`
          },
          example: `curl -X POST "https://api.flipsidecrypto.com/v1/query" \\
  -H "x-api-key: YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "sql": "SELECT * FROM ethereum.core.fact_transactions LIMIT 10",
    "ttlMinutes": 15
  }'`,
          rateLimit: '100 queries/hour'
        },
        {
          name: 'Get Query Status',
          method: 'GET',
          path: '/v1/query/{query_id}/status',
          description: 'Check the execution status of a query',
          params: [
            {
              name: 'query_id',
              type: 'string',
              required: true,
              description: 'ID of the query to check'
            }
          ],
          response: {
            description: 'Current query execution status',
            example: `{
  "query_id": "01H6QXPNB6BCQYPZN8RJ6SV4F5",
  "status": "running",
  "submitted_at": "2023-07-20T10:30:45Z",
  "started_at": "2023-07-20T10:30:46Z",
  "progress": {
    "percentage": 65,
    "current_step": "executing",
    "total_rows_processed": 1234567
  }
}`
          },
          example: `curl "https://api.flipsidecrypto.com/v1/query/01H6QXPNB6BCQYPZN8RJ6SV4F5/status" \\
  -H "x-api-key: YOUR_API_KEY"`,
          rateLimit: '100 requests/minute'
        },
        {
          name: 'Get Query Results',
          method: 'GET',
          path: '/v1/query/{query_id}/results',
          description: 'Retrieve results from a completed query',
          params: [
            {
              name: 'query_id',
              type: 'string',
              required: true,
              description: 'ID of the completed query'
            },
            {
              name: 'format',
              type: 'string',
              required: false,
              description: 'Response format (json, csv)'
            }
          ],
          response: {
            description: 'Query results and metadata',
            example: `{
  "query_id": "01H6QXPNB6BCQYPZN8RJ6SV4F5",
  "status": "success",
  "results": [
    {
      "block_number": 17654321,
      "tx_hash": "0x1234...",
      "from_address": "0xabcd...",
      "to_address": "0xefgh...",
      "value": 1.23,
      "gas_used": 21000
    }
  ],
  "metadata": {
    "column_names": ["block_number", "tx_hash", "from_address", "to_address", "value", "gas_used"],
    "row_count": 1,
    "cached": true
  }
}`
          },
          example: `curl "https://api.flipsidecrypto.com/v1/query/01H6QXPNB6BCQYPZN8RJ6SV4F5/results" \\
  -H "x-api-key: YOUR_API_KEY"`,
          rateLimit: '100 requests/minute'
        }
      ]
    },
    messari: {
      title: 'Messari Integration',
      description: `Access comprehensive crypto asset and market data through Messari's API. 
        Get detailed metrics, research, and analytics for assets, markets, and protocols across the crypto ecosystem.`,
      features: [
        'Real-time asset metrics',
        'Protocol analytics',
        'Market data aggregation',
        'On-chain metrics',
        'Research and analytics'
      ],
      authentication: {
        type: 'API Key',
        details: 'Authentication requires an API key passed in the x-messari-api-key header',
        example: `curl -H "x-messari-api-key: YOUR_API_KEY" https://data.messari.io/api/v2/assets/bitcoin/metrics`
      },
      sdkExample: {
        title: 'SDK Integration',
        description: 'Use the Messari SDK for easier data access and analysis',
        code: `import { MessariClient } from '@messari/sdk';

const client = new MessariClient({
  apiKey: process.env.MESSARI_API_KEY
});

// Get asset metrics
const bitcoinMetrics = await client.getAssetMetrics('bitcoin', {
  fields: ['market_data', 'market_cap', 'volume']
});

// Get protocol data
const aaveData = await client.getProtocolData('aave', {
  metrics: ['tvl', 'volume', 'revenue']
});

// Get market data
const marketData = await client.getMarketData({
  assets: ['bitcoin', 'ethereum'],
  metrics: ['price', 'volume_24h', 'market_cap']
});`
      },
      endpoints: [
        {
          name: 'Get Asset Metrics',
          method: 'GET',
          path: '/v2/assets/{asset}/metrics',
          description: 'Get comprehensive metrics for a specific asset',
          params: [
            {
              name: 'asset',
              type: 'string',
              required: true,
              description: 'Asset slug (e.g., bitcoin, ethereum)'
            },
            {
              name: 'fields',
              type: 'string[]',
              required: false,
              description: 'Specific metrics to retrieve'
            }
          ],
          response: {
            description: 'Detailed asset metrics and market data',
            example: `{
  "status": {
    "elapsed": 0.231,
    "timestamp": "2023-07-20T10:30:45Z"
  },
  "data": {
    "id": "1e31218a-e44e-4285-820c-8282ee222035",
    "symbol": "BTC",
    "name": "Bitcoin",
    "market_data": {
      "price_usd": 29876.54,
      "volume_last_24_hours": 12345678901.23,
      "real_volume_last_24_hours": 9876543210.12,
      "volume_last_24_hours_overstatement_multiple": 1.23,
      "percent_change_usd_last_24_hours": 2.34
    },
    "marketcap": {
      "current_marketcap_usd": 567890123456.78,
      "y_2050_marketcap_usd": 1234567890123.45,
      "y_plus10_marketcap_usd": 987654321098.76,
      "liquid_marketcap_usd": 543210987654.32
    },
    "supply": {
      "y_2050": 21000000,
      "y_plus10": 19500000,
      "liquid": 19250000,
      "circulating": 19150000
    }
  }
}`
          },
          example: `curl "https://data.messari.io/api/v2/assets/bitcoin/metrics" \\
  -H "x-messari-api-key: YOUR_API_KEY"`,
          rateLimit: '50 requests/minute'
        },
        {
          name: 'Get Protocol Data',
          method: 'GET',
          path: '/v2/protocols/{protocol}',
          description: 'Get detailed protocol analytics and metrics',
          params: [
            {
              name: 'protocol',
              type: 'string',
              required: true,
              description: 'Protocol slug (e.g., aave, compound)'
            },
            {
              name: 'metrics',
              type: 'string[]',
              required: false,
              description: 'Specific metrics to retrieve'
            }
          ],
          response: {
            description: 'Comprehensive protocol data and metrics',
            example: `{
  "status": {
    "elapsed": 0.189,
    "timestamp": "2023-07-20T10:30:45Z"
  },
  "data": {
    "id": "b63c5700-f867-4a91-b9d9-d24d95098329",
    "name": "Aave",
    "symbol": "AAVE",
    "category": "Lending",
    "metrics": {
      "tvl": 5678901234.56,
      "total_volume_24h": 234567890.12,
      "total_revenue_24h": 123456.78,
      "unique_users_24h": 5678,
      "active_markets": 45,
      "chains": ["ethereum", "polygon", "avalanche"]
    },
    "market_data": {
      "token_price_usd": 89.12,
      "market_cap_usd": 1234567890.12,
      "volume_24h": 45678901.23
    }
  }
}`
          },
          example: `curl "https://data.messari.io/api/v2/protocols/aave" \\
  -H "x-messari-api-key: YOUR_API_KEY"`,
          rateLimit: '50 requests/minute'
        }
      ]
    },
    bitquery: {
      title: 'BitQuery Integration',
      description: `Access real-time blockchain data through BitQuery's GraphQL API. 
        Query transactions, smart contracts, DEX trades, and more across multiple blockchain networks with flexible GraphQL queries.`,
      features: [
        'Real-time blockchain data',
        'Cross-chain analytics',
        'DEX trading data',
        'Smart contract analysis',
        'Token transfers and metrics'
      ],
      authentication: {
        type: 'API Key',
        details: 'Authentication requires an API key passed in the X-API-KEY header',
        example: `curl -H "X-API-KEY: YOUR_API_KEY" https://graphql.bitquery.io`
      },
      sdkExample: {
        title: 'SDK Integration',
        description: 'Use the BitQuery SDK for easier GraphQL query execution',
        code: `import { BitQueryClient } from '@bitquery/sdk';

const client = new BitQueryClient({
  apiKey: process.env.BITQUERY_API_KEY
});

// Query DEX trades
const trades = await client.query({
  query: \`{
    ethereum {
      dexTrades(
        options: {limit: 10}
        exchangeName: {is: "Uniswap"}
      ) {
        transaction {
          hash
          gasPrice
          gasValue
        }
        block {
          timestamp {
            time
          }
          height
        }
        buyAmount
        buyAmountInUsd: buyAmount(in: USD)
        buyCurrency {
          symbol
          address
        }
        sellAmount
        sellAmountInUsd: sellAmount(in: USD)
        sellCurrency {
          symbol
          address
        }
      }
    }
  }\`
});

// Stream real-time transfers
const subscription = client.subscribe({
  query: \`subscription {
    ethereum {
      transfers(
        options: {limit: 10}
        amount: {gt: 100}
      ) {
        transaction {
          hash
        }
        amount
        currency {
          symbol
        }
        sender {
          address
        }
        receiver {
          address
        }
      }
    }
  }\`,
  onData: (data) => {
    console.log('New transfer:', data);
  }
});`
      },
      endpoints: [
        {
          name: 'Execute GraphQL Query',
          method: 'POST',
          path: '/graphql',
          description: 'Execute GraphQL queries for blockchain data',
          params: [
            {
              name: 'query',
              type: 'string',
              required: true,
              description: 'GraphQL query string'
            },
            {
              name: 'variables',
              type: 'object',
              required: false,
              description: 'Query variables'
            }
          ],
          response: {
            description: 'Query results based on GraphQL schema',
            example: `{
  "data": {
    "ethereum": {
      "dexTrades": [
        {
          "transaction": {
            "hash": "0x1234...",
            "gasPrice": 25000000000,
            "gasValue": 0.002345
          },
          "block": {
            "timestamp": {
              "time": "2023-07-20T10:30:45Z"
            },
            "height": 17654321
          },
          "buyAmount": 1.23,
          "buyAmountInUsd": 2345.67,
          "buyCurrency": {
            "symbol": "ETH",
            "address": "0x0000..."
          },
          "sellAmount": 2500.0,
          "sellAmountInUsd": 2345.67,
          "sellCurrency": {
            "symbol": "USDC",
            "address": "0xa0b86..."
          }
        }
      ]
    }
  }
}`
          },
          example: `curl "https://graphql.bitquery.io" \\
  -H "X-API-KEY: YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "{
      ethereum {
        dexTrades(
          options: {limit: 10}
          exchangeName: {is: \\"Uniswap\\"}
        ) {
          transaction {
            hash
            gasPrice
          }
          buyAmount
          sellAmount
        }
      }
    }"
  }'`,
          rateLimit: '100 requests/minute'
        },
        {
          name: 'WebSocket Subscription',
          method: 'WS',
          path: '/graphql',
          description: 'Subscribe to real-time blockchain events',
          params: [
            {
              name: 'query',
              type: 'string',
              required: true,
              description: 'GraphQL subscription query'
            },
            {
              name: 'variables',
              type: 'object',
              required: false,
              description: 'Subscription variables'
            }
          ],
          response: {
            description: 'Real-time event stream based on subscription',
            example: `{
  "data": {
    "ethereum": {
      "transfers": [
        {
          "transaction": {
            "hash": "0x5678..."
          },
          "amount": 150.0,
          "currency": {
            "symbol": "USDT"
          },
          "sender": {
            "address": "0xabcd..."
          },
          "receiver": {
            "address": "0xefgh..."
          }
        }
      ]
    }
  }
}`
          },
          example: `wscat -c wss://graphql.bitquery.io \\
  -H "X-API-KEY: YOUR_API_KEY" \\
  -x '{
    "query": "subscription {
      ethereum {
        transfers(options: {limit: 10}) {
          transaction { hash }
          amount
          currency { symbol }
        }
      }
    }"
  }'`,
          rateLimit: '10 concurrent subscriptions'
        }
      ]
    },
    subgraphs: {
      title: 'The Graph Integration',
      description: `Access indexed blockchain data through The Graph's decentralized GraphQL API network. 
        Query protocol-specific subgraphs for real-time and historical blockchain data with flexible GraphQL queries.`,
      features: [
        'Protocol-specific subgraphs',
        'Real-time blockchain data',
        'Custom GraphQL queries',
        'Historical data access',
        'Cross-protocol analytics'
      ],
      authentication: {
        type: 'API Key',
        details: 'Authentication required for hosted service, passed via Authorization header',
        example: `curl -H "Authorization: Bearer YOUR_API_KEY" https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3`
      },
      sdkExample: {
        title: 'SDK Integration',
        description: 'Use The Graph Client for easier GraphQL query execution',
        code: `import { createClient } from '@urql/core';

const client = createClient({
  url: 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
  fetchOptions: {
    headers: {
      Authorization: \`Bearer \${process.env.GRAPH_API_KEY}\`
    }
  }
});

// Query pool data
const poolData = await client.query({
  query: \`{
    pools(
      first: 5
      orderBy: totalValueLockedUSD
      orderDirection: desc
    ) {
      id
      token0 {
        symbol
        decimals
      }
      token1 {
        symbol
        decimals
      }
      totalValueLockedUSD
      volumeUSD
      feeTier
    }
  }\`
}).toPromise();

// Query historical data
const historicalData = await client.query({
  query: \`{
    poolDayDatas(
      first: 30
      orderBy: date
      orderDirection: desc
      where: { pool: "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8" }
    ) {
      date
      volumeUSD
      tvlUSD
      feesUSD
    }
  }\`
}).toPromise();

// Stream real-time updates
const subscription = client.subscription({
  query: \`subscription {
    swaps(first: 5, orderBy: timestamp, orderDirection: desc) {
      timestamp
      pool {
        token0 { symbol }
        token1 { symbol }
      }
      amount0
      amount1
      amountUSD
    }
  }\`
}).subscribe(
  result => console.log('New swap:', result),
  error => console.error('Subscription error:', error)
);`
      },
      endpoints: [
        {
          name: 'Query Subgraph',
          method: 'POST',
          path: '/subgraphs/name/{owner}/{name}',
          description: 'Execute GraphQL queries against a specific subgraph',
          params: [
            {
              name: 'owner',
              type: 'string',
              required: true,
              description: 'Subgraph owner (e.g., uniswap)'
            },
            {
              name: 'name',
              type: 'string',
              required: true,
              description: 'Subgraph name (e.g., uniswap-v3)'
            },
            {
              name: 'query',
              type: 'string',
              required: true,
              description: 'GraphQL query string'
            }
          ],
          response: {
            description: 'Query results based on subgraph schema',
            example: `{
  "data": {
    "pools": [
      {
        "id": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
        "token0": {
          "symbol": "USDC",
          "decimals": "6"
        },
        "token1": {
          "symbol": "ETH",
          "decimals": "18"
        },
        "totalValueLockedUSD": "123456789.12",
        "volumeUSD": "9876543.21",
        "feeTier": "3000"
      }
    ]
  }
}`
          },
          example: `curl "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{
    "query": "{
      pools(
        first: 5
        orderBy: totalValueLockedUSD
        orderDirection: desc
      ) {
        id
        token0 { symbol decimals }
        token1 { symbol decimals }
        totalValueLockedUSD
        volumeUSD
        feeTier
      }
    }"
  }'`,
          rateLimit: '1000 queries/day'
        },
        {
          name: 'Stream Updates',
          method: 'WS',
          path: '/subgraphs/name/{owner}/{name}',
          description: 'Subscribe to real-time updates via WebSocket',
          params: [
            {
              name: 'owner',
              type: 'string',
              required: true,
              description: 'Subgraph owner'
            },
            {
              name: 'name',
              type: 'string',
              required: true,
              description: 'Subgraph name'
            },
            {
              name: 'subscription',
              type: 'string',
              required: true,
              description: 'GraphQL subscription query'
            }
          ],
          response: {
            description: 'Real-time updates based on subscription',
            example: `{
  "data": {
    "swaps": [
      {
        "timestamp": "1627384567",
        "pool": {
          "token0": { "symbol": "USDC" },
          "token1": { "symbol": "ETH" }
        },
        "amount0": "1000.00",
        "amount1": "0.5",
        "amountUSD": "1500.00"
      }
    ]
  }
}`
          },
          example: `wscat -c wss://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3 \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -x '{
    "query": "subscription {
      swaps(first: 5, orderBy: timestamp, orderDirection: desc) {
        timestamp
        pool {
          token0 { symbol }
          token1 { symbol }
        }
        amount0
        amount1
        amountUSD
      }
    }"
  }'`,
          rateLimit: '10 concurrent subscriptions'
        }
      ]
    },
    dex: {
      title: 'DEX Analytics',
      description: 'Decentralized Exchange metrics and trading data',
      endpoints: [
        {
          name: 'Get DEX Pairs',
          method: 'GET',
          path: '/v1/dex/pairs/{chain}',
          example: `curl "https://api.dune.com/api/v1/dex/pairs/ethereum"`,
          description: 'Get active trading pairs and liquidity'
        },
        {
          name: 'Get Trading Volume',
          method: 'GET',
          path: '/v1/dex/volume/{dex}',
          example: `curl "https://api.llama.fi/dex/volume/uniswap"`,
          description: 'Get trading volume metrics'
        }
      ]
    },
    lending: {
      title: 'Lending Protocol Analytics',
      description: 'Comprehensive lending protocol data and metrics',
      endpoints: [
        {
          name: 'Get Lending Markets',
          method: 'GET',
          path: '/v1/lending/markets',
          example: `curl "https://api.llama.fi/lending/markets"`,
          description: 'Get active lending markets and rates'
        },
        {
          name: 'Get Protocol Stats',
          method: 'GET',
          path: '/v1/lending/protocol/{protocol}',
          example: `curl "https://api.llama.fi/lending/protocol/aave"`,
          description: 'Get detailed protocol statistics'
        }
      ]
    },
    metrics: {
      title: 'Chain Metrics',
      description: 'Blockchain network statistics and metrics',
      endpoints: [
        {
          name: 'Get Chain Stats',
          method: 'GET',
          path: '/v1/metrics/chain/{chain}',
          example: `curl "https://api.llama.fi/metrics/chain/ethereum"`,
          description: 'Get comprehensive chain statistics'
        }
      ]
    },
    authentication: {
      title: 'Authentication & Security',
      description: `Secure access to our blockchain data APIs through multiple authentication methods. 
        Each data provider requires specific authentication credentials and follows different security protocols.`,
      providers: [
        {
          name: 'DefiLlama',
          authType: 'API Key',
          details: 'Access DefiLlama data using an API key in request headers',
          example: [
            {
              description: 'HTTP Request with API Key',
              code: `const response = await fetch('https://api.llama.fi/protocol/aave', {
  headers: {
    'x-api-key': process.env.DEFILLAMA_API_KEY
  }
});`
            },
            {
              description: 'SDK Integration',
              code: `import { DefiLlamaClient } from '@defillama/sdk';

const client = new DefiLlamaClient({
  apiKey: process.env.DEFILLAMA_API_KEY,
  timeout: 5000
});`
            }
          ],
          rateLimit: '300 requests per 5 minutes',
          headers: [
            {
              name: 'x-api-key',
              value: 'YOUR_API_KEY',
              required: true
            }
          ],
          bestPractices: [
            'Store API keys in environment variables',
            'Implement request retry logic with exponential backoff',
            'Monitor rate limits through response headers'
          ]
        },
        {
          name: 'Dune Analytics',
          authType: 'Bearer Token',
          details: 'Authenticate to Dune Analytics using OAuth2 Bearer tokens',
          example: [
            {
              description: 'Bearer Token Authentication',
              code: `const response = await fetch('https://api.dune.com/api/v1/query/1234/results', {
  headers: {
    'Authorization': \`Bearer \${process.env.DUNE_API_TOKEN}\`
  }
});`
            },
            {
              description: 'SDK Usage',
              code: `import { DuneClient } from '@dune/client';

const client = new DuneClient({
  apiToken: process.env.DUNE_API_TOKEN,
  baseUrl: 'https://api.dune.com'
});`
            }
          ],
          rateLimit: '100 queries per day',
          headers: [
            {
              name: 'Authorization',
              value: 'Bearer YOUR_TOKEN',
              required: true
            }
          ],
          bestPractices: [
            'Rotate tokens periodically',
            'Use separate tokens for different environments',
            'Implement token refresh logic'
          ]
        },
        {
          name: 'The Graph',
          authType: 'GraphQL API Key',
          details: 'Query The Graph using API keys for authenticated subgraph access',
          example: [
            {
              description: 'GraphQL Query with Authentication',
              code: `const response = await fetch('https://api.thegraph.com/subgraphs/name/aave/protocol-v2', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': \`Bearer \${process.env.GRAPH_API_KEY}\`
  },
  body: JSON.stringify({
    query: \`{
      markets(first: 5) {
        id
        inputToken { symbol }
        outputToken { symbol }
      }
    }\`
  })
});`
            }
          ],
          rateLimit: '1000 queries per day',
          headers: [
            {
              name: 'Authorization',
              value: 'Bearer YOUR_KEY',
              required: true
            }
          ],
          bestPractices: [
            'Cache frequently accessed queries',
            'Use query cost estimation',
            'Implement query depth limiting'
          ]
        }
      ],
      securityBestPractices: [
        {
          title: 'Environment Variables',
          description: 'Store all API keys and tokens in environment variables',
          example: `// .env.local
DEFILLAMA_API_KEY=your_api_key
DUNE_API_TOKEN=your_token
GRAPH_API_KEY=your_key

// Next.js API Route
export default function handler(req, res) {
  const apiKey = process.env.DEFILLAMA_API_KEY;
  // Use apiKey in requests
}`
        },
        {
          title: 'Rate Limit Handling',
          description: 'Implement proper rate limit handling and backoff strategies',
          example: `async function fetchWithRetry(url, options, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(url, options);
      if (response.status === 429) {
        const backoff = Math.pow(2, i) * 1000;
        await new Promise(resolve => setTimeout(resolve, backoff));
        continue;
      }
      return response;
    } catch (error) {
      if (i === maxRetries - 1) throw error;
    }
  }
}`
        },
        {
          title: 'Request Validation',
          description: 'Validate and sanitize all API requests',
          example: `function validateApiRequest(req) {
  if (!req.headers['x-api-key']) {
    throw new Error('Missing API key');
  }
  
  // Validate other parameters
  const { protocol, timeRange } = req.query;
  if (!protocol || !timeRange) {
    throw new Error('Missing required parameters');
  }
}`
        }
      ]
    },
    rateLimits: {
      title: 'Rate Limits',
      description: 'Understanding rate limits across different providers',
      providers: {
        dune: {
          tier: 'Standard',
          limits: {
            requests: '60/minute',
            dataPoints: '1M/month',
            concurrent: '3 queries'
          }
        },
        footprint: {
          tier: 'Basic',
          limits: {
            requests: '100/minute',
            dataPoints: '500K/month',
            concurrent: '5 queries'
          }
        }
        // ... add other providers
      }
    },
    guides: {
      title: 'Integration Guides',
      description: 'Step-by-step guides for integrating with our APIs',
      guides: {
        'best-practices': {
          title: 'API Best Practices',
          description: 'Guidelines for optimal API usage',
          steps: [
            {
              title: 'Rate Limiting',
              content: 'Implement proper rate limiting and backoff strategies',
              code: `const rateLimit = new RateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 60 // limit each IP to 60 requests per windowMs
});`
            },
            {
              title: 'Error Handling',
              content: 'Implement comprehensive error handling',
              code: `try {
  const response = await client.query(queryId);
} catch (error) {
  if (error.status === 429) {
    // Handle rate limit
    await delay(1000);
    return retry(queryId);
  }
  // Handle other errors
}`
            }
          ]
        },
        gettingStarted: {
          title: 'Getting Started Guide',
          description: 'Quick start guide for integrating the APIs',
          steps: [
            {
              title: '1. Install Dependencies',
              content: 'Install required packages using npm or yarn',
              code: `npm install @defillama/sdk @dune/client @footprint/sdk`
            },
            {
              title: '2. Configure Authentication',
              content: 'Set up your API keys and authentication',
              code: `const client = new DuneClient(process.env.DUNE_API_KEY)`
            }
          ]
        },
        sdkSetup: {
          title: 'SDK Setup Guide',
          description: 'How to set up and configure the SDKs',
          steps: [
            {
              title: '1. Initialize SDK',
              content: 'Initialize the SDK with your credentials',
              code: `import { DuneClient } from '@dune/client'
              
const client = new DuneClient({
  apiKey: process.env.DUNE_API_KEY
})`
            }
          ]
        }
      }
    },
    liquidStaking: {
      title: 'Liquid Staking Analytics',
      description: 'Access liquid staking protocol data and metrics',
      endpoints: [
        {
          name: 'Get Staking Pools',
          method: 'GET',
          path: '/v1/staking/pools',
          example: `curl "https://api.llama.fi/staking/pools"`,
          description: 'Get active staking pools and APRs'
        }
        // ... add more endpoints
      ]
    },
    protocols: {
      title: 'Protocol Analytics',
      description: 'Comprehensive protocol-level analytics and metrics',
      endpoints: [
        {
          name: 'Get Protocol Overview',
          method: 'GET',
          path: '/v1/protocols/{protocol}',
          example: `curl "https://api.llama.fi/protocols/aave"`,
          description: 'Get comprehensive protocol metrics and analytics'
        },
        {
          name: 'Get Protocol TVL History',
          method: 'GET',
          path: '/v1/protocols/{protocol}/tvl/history',
          example: `curl "https://api.llama.fi/protocols/aave/tvl/history"`,
          description: 'Get historical TVL data for a protocol'
        }
      ]
    },
    tvl: {
      title: 'Total Value Locked (TVL) Analytics',
      description: 'Track TVL across protocols and chains',
      endpoints: [
        {
          name: 'Get Chain TVL',
          method: 'GET',
          path: '/v1/chains/{chain}/tvl',
          example: `curl "https://api.llama.fi/chains/ethereum/tvl"`,
          description: 'Get TVL for a specific blockchain'
        },
        {
          name: 'Get Protocol TVL Breakdown',
          method: 'GET',
          path: '/v1/protocols/{protocol}/tvl/breakdown',
          example: `curl "https://api.llama.fi/protocols/aave/tvl/breakdown"`,
          description: 'Get detailed TVL breakdown by chain and pool'
        }
      ]
    },
    bridges: {
      title: 'Cross-Chain Bridge Analytics',
      description: 'Track cross-chain bridge activity and volume',
      endpoints: [
        {
          name: 'Get Bridge Volume',
          method: 'GET',
          path: '/v1/bridges/{bridge}/volume',
          example: `curl "https://api.llama.fi/bridges/multichain/volume"`,
          description: 'Get bridge transfer volume statistics'
        },
        {
          name: 'Get Bridge TVL',
          method: 'GET',
          path: '/v1/bridges/{bridge}/tvl',
          example: `curl "https://api.llama.fi/bridges/multichain/tvl"`,
          description: 'Get total value locked in bridge contracts'
        }
      ]
    },
    perpetuals: {
      title: 'Perpetual DEX Analytics',
      description: 'Track perpetual futures trading activity',
      endpoints: [
        {
          name: 'Get Trading Volume',
          method: 'GET',
          path: '/v1/perpetuals/{protocol}/volume',
          example: `curl "https://api.llama.fi/perpetuals/gmx/volume"`,
          description: 'Get perpetual trading volume metrics'
        },
        {
          name: 'Get Open Interest',
          method: 'GET',
          path: '/v1/perpetuals/{protocol}/open-interest',
          example: `curl "https://api.llama.fi/perpetuals/gmx/open-interest"`,
          description: 'Get current open interest statistics'
        }
      ]
    },
    synthetics: {
      title: 'Synthetic Assets Analytics',
      description: 'Track synthetic asset protocols and markets',
      endpoints: [
        {
          name: 'Get Synthetic Assets',
          method: 'GET',
          path: '/v1/synthetics/{protocol}/assets',
          example: `curl "https://api.llama.fi/synthetics/synthetix/assets"`,
          description: 'Get list of synthetic assets and their data'
        },
        {
          name: 'Get Market Data',
          method: 'GET',
          path: '/v1/synthetics/{protocol}/markets',
          example: `curl "https://api.llama.fi/synthetics/synthetix/markets"`,
          description: 'Get synthetic asset market statistics'
        }
      ]
    },
    nfts: {
      title: 'NFT Market Analytics',
      description: 'Track NFT collections and marketplace activity',
      endpoints: [
        {
          name: 'Get Collection Stats',
          method: 'GET',
          path: '/v1/nfts/collection/{collection}',
          example: `curl "https://api.llama.fi/nfts/collection/cryptopunks"`,
          description: 'Get NFT collection statistics'
        },
        {
          name: 'Get Marketplace Volume',
          method: 'GET',
          path: '/v1/nfts/marketplace/{marketplace}',
          example: `curl "https://api.llama.fi/nfts/marketplace/opensea"`,
          description: 'Get marketplace trading volume'
        }
      ]
    },
    eigenlayer: {
      title: 'EigenLayer Analytics',
      description: 'Track EigenLayer protocol metrics and restaking',
      endpoints: [
        {
          name: 'Get Restaking Stats',
          method: 'GET',
          path: '/v1/eigenlayer/restaking',
          example: `curl "https://api.llama.fi/eigenlayer/restaking"`,
          description: 'Get restaking statistics and TVL'
        },
        {
          name: 'Get Operator Metrics',
          method: 'GET',
          path: '/v1/eigenlayer/operators',
          example: `curl "https://api.llama.fi/eigenlayer/operators"`,
          description: 'Get operator performance metrics'
        }
      ]
    },
    'best-practices': {
      title: 'API Best Practices',
      description: 'Comprehensive guidelines and recommendations for optimal API integration and usage',
      steps: [
        {
          title: 'Rate Limiting and Backoff',
          content: 'Implement intelligent rate limiting and exponential backoff to avoid service disruptions',
          code: `class RateLimitHandler {
  private requestCounts: Map<string, number> = new Map();
  private readonly limits: { [key: string]: number } = {
    'defillama': 300,  // 300 requests per 5 minutes
    'dune': 100,      // 100 requests per day
    'graph': 1000     // 1000 queries per day
  };

  async executeRequest(provider: string, request: () => Promise<any>) {
    const currentCount = this.requestCounts.get(provider) || 0;
    
    if (currentCount >= this.limits[provider]) {
      const backoff = Math.pow(2, currentCount - this.limits[provider]) * 1000;
      await new Promise(resolve => setTimeout(resolve, backoff));
    }

    try {
      const result = await request();
      this.requestCounts.set(provider, currentCount + 1);
      return result;
    } catch (error) {
      if (error.status === 429) { // Rate limit exceeded
        await this.handleRateLimit(provider);
        return this.executeRequest(provider, request);
      }
      throw error;
    }
  }

  private async handleRateLimit(provider: string) {
    const backoff = Math.pow(2, this.requestCounts.get(provider) || 1) * 1000;
    await new Promise(resolve => setTimeout(resolve, backoff));
  }
}`
        },
        {
          title: 'Caching Strategy',
          content: 'Implement efficient caching to reduce API calls and improve performance',
          code: `class DataCache {
  private cache: Map<string, { data: any, timestamp: number }> = new Map();
  private readonly TTL: { [key: string]: number } = {
    'tvl': 5 * 60 * 1000,       // 5 minutes for TVL data
    'protocol': 15 * 60 * 1000, // 15 minutes for protocol data
    'historical': 60 * 60 * 1000 // 1 hour for historical data
  };

  async getData(key: string, type: string, fetchFn: () => Promise<any>) {
    const cached = this.cache.get(key);
    const now = Date.now();

    if (cached && now - cached.timestamp < this.TTL[type]) {
      return cached.data;
    }

    const data = await fetchFn();
    this.cache.set(key, { data, timestamp: now });
    return data;
  }

  invalidate(pattern: RegExp) {
    for (const key of this.cache.keys()) {
      if (pattern.test(key)) {
        this.cache.delete(key);
      }
    }
}`
        },
        {
          title: 'Error Handling',
          content: 'Implement comprehensive error handling and recovery strategies',
          code: `class APIErrorHandler {
  private static readonly ERROR_MESSAGES = {
    400: 'Invalid request parameters',
    401: 'Authentication failed',
    403: 'Access forbidden',
    429: 'Rate limit exceeded',
    500: 'Internal server error',
    503: 'Service unavailable'
  };

  private static readonly RETRYABLE_ERRORS = [429, 503];
  private static readonly MAX_RETRIES = 3;

  async handleRequest<T>(
    request: () => Promise<T>,
    options: { 
      retryable?: boolean,
      maxRetries?: number 
    } = {}
  ): Promise<T> {
    const retries = options.maxRetries || this.MAX_RETRIES;
    
    for (let attempt = 0; attempt < retries; attempt++) {
      try {
        return await request();
      } catch (error) {
        if (!this.shouldRetry(error, attempt, options.retryable)) {
          throw this.enhanceError(error);
        }
        await this.wait(attempt);
      }
    }

  private shouldRetry(error: any, attempt: number, retryable?: boolean): boolean {
    return (
      retryable !== false &&
      attempt < this.MAX_RETRIES &&
      this.RETRYABLE_ERRORS.includes(error.status)
    );
  }

  private enhanceError(error: any): Error {
    const message = this.ERROR_MESSAGES[error.status] || error.message;
    return new Error(\`API Error: \${message} (Status: \${error.status})\`);
  }

  private wait(attempt: number): Promise<void> {
    const delay = Math.pow(2, attempt) * 1000;
    return new Promise(resolve => setTimeout(resolve, delay));
  }
}`
        },
        {
          title: 'Data Validation',
          content: 'Implement robust data validation for requests and responses',
          code: `import { z } from 'zod';

// Define validation schemas
const ProtocolSchema = z.object({
  name: z.string(),
  tvl: z.number().positive(),
  chainTvls: z.record(z.string(), z.number()),
  change_24h: z.number().optional(),
  volume_24h: z.number().optional()
});

class DataValidator {
  private schemas: Map<string, z.ZodSchema> = new Map();

  constructor() {
    this.schemas.set('protocol', ProtocolSchema);
    // Add more schemas as needed
  }

  validate<T>(data: unknown, schemaKey: string): T {
    const schema = this.schemas.get(schemaKey);
    if (!schema) {
      throw new Error(\`No schema found for key: \${schemaKey}\`);
    }

    try {
      return schema.parse(data) as T;
    } catch (error) {
      throw new Error(\`Validation error: \${error.message}\`);
    }
  }

  validateRequest(endpoint: string, params: any) {
    // Add request validation logic
    const requiredParams = this.getRequiredParams(endpoint);
    const missingParams = requiredParams.filter(param => !params[param]);
    
    if (missingParams.length > 0) {
      throw new Error(\`Missing required parameters: \${missingParams.join(', ')}\`);
    }
  }

  private getRequiredParams(endpoint: string): string[] {
    // Define required parameters for each endpoint
    const paramMap: Record<string, string[]> = {
      '/protocol': ['name'],
      '/tvl': ['protocol', 'timeRange'],
      // Add more endpoints
    };
    return paramMap[endpoint] || [];
  }
}`
        },
        {
          title: 'Request Optimization',
          content: 'Optimize API requests to minimize latency and improve efficiency',
          code: `class RequestOptimizer {
  private batchQueue: Map<string, Set<string>> = new Map();
  private batchTimeout: number = 50; // ms

  async queueRequest<T>(
    type: string,
    id: string,
    batchFetcher: (ids: string[]) => Promise<Record<string, T>>
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      // Add to queue
      if (!this.batchQueue.has(type)) {
        this.batchQueue.set(type, new Set());
        
        // Schedule batch processing
        setTimeout(async () => {
          try {
            const ids = Array.from(this.batchQueue.get(type) || []);
            const results = await batchFetcher(ids);
            
            // Resolve all queued promises
            for (const id of ids) {
              resolve(results[id]);
            }
            
            this.batchQueue.delete(type);
          } catch (error) {
            reject(error);
          }
        }, this.batchTimeout);
      }
      
      this.batchQueue.get(type)?.add(id);
    });
  }

  // Example usage:
  async getTokenPrices(tokens: string[]): Promise<Record<string, number>> {
    const prices: Record<string, number> = {};
    
    await Promise.all(
      tokens.map(token =>
        this.queueRequest('prices', token, async (batchTokens) => {
          const response = await fetch(\`/prices?tokens=\${batchTokens.join(',')}\`);
          return response.json();
        })
      )
    );

    return prices;
  }
}`
        }
      ]
    },
    hypurrscan: {
      title: 'Hypurrscan API Integration',
      description: 'Access comprehensive Hyperliquid L1 data through Hypurrscan\'s API. Get real-time trading metrics, positions, liquidations, and historical data across all markets.',
      features: [
        'Real-time market data and trading metrics',
        'Historical position and liquidation data',
        'Market depth and orderbook analytics',
        'Trader statistics and leaderboards',
        'Cross-market analytics and correlations',
        'WebSocket support for live updates'
      ],
      endpoints: [
        {
          name: 'Market Overview',
          method: 'GET',
          path: '/v1/hyperliquid/markets',
          description: 'Get current market data for all trading pairs',
          response: {
            description: 'Returns detailed market information including prices, volumes, and funding rates',
            example: `{
  "markets": [
    {
      "symbol": "BTC-USDC",
      "price": "65432.50",
      "volume24h": "1234567.89",
      "openInterest": "98765.43",
      "fundingRate": "0.0001"
    }
  ]
}`
          },
          example: 'curl https://api.hypurrscan.io/v1/hyperliquid/markets'
        },
        {
          name: 'Token Holders',
          method: 'GET',
          path: '/v1/hyperliquid/holders/{token}',
          params: [
            {
              name: 'token',
              type: 'string',
              required: true,
              description: 'Token symbol'
            }
          ],
          description: 'Fetch current holders for a specific token',
          example: "curl -X 'GET' \ 'https://api.hypurrscan.io/holders/PURR' \ -H 'accept: application/json' "
        },
        {
          name: 'Trader Positions',
          method: 'GET',
          path: '/v1/hyperliquid/positions/{address}',
          params: [
            {
              name: 'address',
              type: 'string',
              required: true,
              description: 'Trader wallet address'
            }
          ],
          description: 'Fetch current positions for a specific trader',
          example: 'curl https://api.hypurrscan.io/v1/hyperliquid/positions/0x1234...'
        },
        {
          name: 'Recent Liquidations',
          method: 'GET',
          path: '/v1/hyperliquid/liquidations',
          params: [
            {
              name: 'timeframe',
              type: 'string',
              required: false,
              description: 'Time range (1h, 24h, 7d)'
            }
          ],
          description: 'Get recent liquidation events across all markets',
          example: 'curl https://api.hypurrscan.io/v1/hyperliquid/liquidations?timeframe=24h'
        }
      ],
      authentication: {
        type: 'API Key',
        details: 'API key required for enhanced rate limits and additional endpoints',
        example: 'curl -H "X-API-Key: your_api_key" https://api.hypurrscan.io/v1/hyperliquid/markets'
      },
      sdkExample: {
        title: 'Hypurrscan SDK Usage',
        description: 'Example implementation using the Hypurrscan TypeScript SDK',
        code: `import { HypurrscanClient } from '@hypurrscan/sdk';

const client = new HypurrscanClient({
  apiKey: process.env.HYPURRSCAN_API_KEY
});

// Fetch market data
const markets = await client.markets.getAll();

// Stream live trades
client.ws.trades({
  symbols: ['BTC-USDC', 'ETH-USDC'],
  onData: (trade) => {
    console.log('New trade:', trade);
  }
});`
      }
    }
  }

  return (
    <div className="flex min-h-screen">
      <SideNav 
        sections={sections}
        activeSection={activeTab}
        onSectionChange={handleNavigation}
      />

      <div className="flex-1 p-8">
        <div className="mb-6">
          <select
            className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500"
            value={selectedProviderId}
            onChange={handleProviderChange}
          >
            <option value="">Select a provider</option>
            {providers.map((provider) => (
              <option key={provider.id} value={provider.id}>
                {provider.name}
              </option>
            ))}
          </select>
        </div>

        <Tabs>
          {sections.flatMap(section => 
            section.items.map(item => (
              <TabPanel 
                key={item.id} 
                id={item.id}
                activeTab={activeTab}
              >
                {content[item.id] && (
                  <Section title={content[item.id].title}>
                    <p className="text-gray-300 mb-6">{content[item.id].description}</p>
                    
                    {/* Handle Overview Content */}
                    {(content[item.id] as OverviewContent).features && item.id === 'overview' && (
                      <>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
                          {(content[item.id] as OverviewContent).features.map((feature, i) => (
                            <div key={i} className="bg-gray-800/50 rounded-lg p-6">
                              <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                              <p className="text-gray-300 mb-4">{feature.description}</p>
                              <ul className="list-disc pl-6">
                                {feature.details.map((detail, j) => (
                                  <li key={j} className="text-gray-300 mb-2">{detail}</li>
                                ))}
                              </ul>
                            </div>
                          ))}
                        </div>

                        <div className="mb-12">
                          <h3 className="text-2xl font-semibold mb-6">{(content[item.id] as OverviewContent).quickStart.title}</h3>
                          {(content[item.id] as OverviewContent).quickStart.steps.map((step, i) => (
                            <div key={i} className="mb-8">
                              <h4 className="text-xl font-semibold mb-4">{step.title}</h4>
                              <CodeBlock
                                language="typescript"
                                code={step.code}
                              />
                            </div>
                          ))}
                        </div>

                        <div>
                          <h3 className="text-2xl font-semibold mb-6">Supported Platforms</h3>
                          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {(content[item.id] as OverviewContent).supportedPlatforms.map((platform, i) => (
                              <div key={i} className="bg-gray-800/50 rounded-lg p-6">
                                <h4 className="text-xl font-semibold mb-3">{platform.name}</h4>
                                <ul className="list-disc pl-6 mb-4">
                                  {platform.features.map((feature, j) => (
                                    <li key={j} className="text-gray-300 mb-2">{feature}</li>
                                  ))}
                                </ul>
                                <p className="text-sm text-gray-400">{platform.rateLimit}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      </>
                    )}

                    {/* Handle Authentication Content */}
                    {(content[item.id] as AuthenticationContent).providers && item.id === 'authentication' && (
                      <>
                        <div className="space-y-12 mb-12">
                          {(content[item.id] as AuthenticationContent).providers.map((provider, i) => (
                            <div key={i} className="bg-gray-800/50 rounded-lg p-6">
                              <h3 className="text-2xl font-semibold mb-4">{provider.name}</h3>
                              <div className="mb-6">
                                <h4 className="text-xl font-semibold mb-3">Authentication Type</h4>
                                <p className="text-gray-300">{provider.authType}</p>
                                <p className="text-gray-300 mt-2">{provider.details}</p>
                              </div>

                              <div className="mb-6">
                                <h4 className="text-xl font-semibold mb-3">Required Headers</h4>
                                <div className="bg-gray-900/50 rounded p-4">
                                  {provider.headers.map((header, j) => (
                                    <div key={j} className="mb-2">
                                      <code className="text-purple-400">{header.name}</code>
                                      <span className="text-gray-300">: {header.value}</span>
                                      {header.required && <span className="text-red-400 ml-2">(Required)</span>}
                                    </div>
                                  ))}
                                </div>
                              </div>

                              <div className="mb-6">
                                <h4 className="text-xl font-semibold mb-3">Examples</h4>
                                {provider.example.map((ex, j) => (
                                  <div key={j} className="mb-4">
                                    <p className="text-gray-300 mb-2">{ex.description}</p>
                                    <CodeBlock
                                      language="typescript"
                                      code={ex.code}
                                    />
                                  </div>
                                ))}
                              </div>

                              <div className="mb-6">
                                <h4 className="text-xl font-semibold mb-3">Rate Limits</h4>
                                <p className="text-gray-300">{provider.rateLimit}</p>
                              </div>

                              <div>
                                <h4 className="text-xl font-semibold mb-3">Best Practices</h4>
                                <ul className="list-disc pl-6">
                                  {provider.bestPractices.map((practice, j) => (
                                    <li key={j} className="text-gray-300 mb-2">{practice}</li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          ))}
                        </div>

                        <div>
                          <h3 className="text-2xl font-semibold mb-6">Security Best Practices</h3>
                          <div className="space-y-8">
                            {(content[item.id] as AuthenticationContent).securityBestPractices.map((practice, i) => (
                              <div key={i} className="bg-gray-800/50 rounded-lg p-6">
                                <h4 className="text-xl font-semibold mb-3">{practice.title}</h4>
                                <p className="text-gray-300 mb-4">{practice.description}</p>
                                {practice.example && (
                                  <CodeBlock
                                    language="typescript"
                                    code={practice.example}
                                  />
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      </>
                    )}

                    {/* Keep existing DataSourceContent handling */}
                    {(content[item.id] as DataSourceContent).endpoints && (
                      <>
                        {content[item.id] && (
                          <p className="text-gray-300 mb-8 text-lg">{content[item.id].description}</p>
                        )}

                        {((content[item.id] as DataSourceContent)?.features?.length ?? 0) > 0 && (
                          <div className="mb-8">
                            <h3 className="text-xl font-semibold mb-4">Features</h3>
                            <ul className="list-disc pl-6">
                              {(content[item.id] as DataSourceContent).features?.map((feature, i) => (
                                <li key={i} className="text-gray-300 mb-2">{feature}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {(content[item.id] as DataSourceContent).authentication && (
                          <div className="mb-8">
                            <h3 className="text-xl font-semibold mb-4">Authentication</h3>
                            <div className="bg-gray-800/50 rounded-lg p-6">
                              <p className="text-gray-300 mb-4">
                                {(content[item.id] as DataSourceContent).authentication?.details}
                              </p>
                              <CodeBlock
                                language="bash"
                                code={(content[item.id] as DataSourceContent).authentication?.example || ''}
                              />
                            </div>
                          </div>
                        )}

                        {(content[item.id] as DataSourceContent).sdkExample && (
                          <div className="mb-8">
                            <h3 className="text-xl font-semibold mb-4">
                              {(content[item.id] as DataSourceContent).sdkExample?.title || 'SDK Example'}
                            </h3>
                            <p className="text-gray-300 mb-4">
                              {(content[item.id] as DataSourceContent).sdkExample?.description}
                            </p>
                            <CodeBlock
                              language="typescript"
                              code={(content[item.id] as DataSourceContent).sdkExample?.code || ''}
                            />
                          </div>
                        )}

                        <h3 className="text-xl font-semibold mb-6">API Endpoints</h3>
                        {(content[item.id] as DataSourceContent).endpoints?.map((endpoint, index) => (
                          <div key={index} className="mb-12 bg-gray-800/50 rounded-lg p-6">
                            <h4 className="text-2xl font-semibold mb-4">
                              {endpoint.method} {endpoint.path}
                            </h4>
                            <p className="text-gray-300 mb-6">{endpoint.description}</p>

                            {endpoint.params && (
                              <div className="mb-6">
                                <h5 className="text-lg font-semibold mb-3">Parameters</h5>
                                <div className="bg-gray-900/50 rounded p-4">
                                  {endpoint.params.map((param, i) => (
                                    <div key={i} className="mb-3">
                                      <code className="text-purple-400">{param.name}</code>
                                      <span className="text-gray-300"> ({param.type})</span>
                                      {param.required && <span className="text-red-400 ml-2">(Required)</span>}
                                      <p className="text-gray-300 mt-1">{param.description}</p>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {endpoint.response && (
                              <div className="mb-6">
                                <h5 className="text-lg font-semibold mb-3">Response</h5>
                                <p className="text-gray-300 mb-3">{endpoint.response.description}</p>
                                <CodeBlock
                                  language="json"
                                  code={endpoint.response.example}
                                />
                              </div>
                            )}

                            <div className="mb-4">
                              <h5 className="text-lg font-semibold mb-3">Example Request</h5>
                              <CodeBlock
                                language="bash"
                                code={endpoint.example}
                              />
                            </div>

                            {endpoint.rateLimit && (
                              <div className="mt-4 text-sm text-gray-400">
                                Rate Limit: {endpoint.rateLimit}
                              </div>
                            )}
                          </div>
                        ))}
                      </>
                    )}

                    {/* Handle Best Practices Content */}
                    {(content[item.id] as GuideContent).steps && item.id === 'best-practices' && (
                      <div className="space-y-12">
                        {(content[item.id] as GuideContent).steps.map((step, i) => (
                          <div key={i} className="bg-gray-800/50 rounded-lg p-6">
                            <h3 className="text-2xl font-semibold mb-4">{step.title}</h3>
                            <p className="text-gray-300 mb-6">{step.content}</p>
                            {step.code && (
                              <div className="bg-gray-900/50 rounded-lg">
                                <CodeBlock
                                  language="typescript"
                                  code={step.code}
                                />
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </Section>
                )}
              </TabPanel>
            ))
          )}
        </Tabs>
      </div>
    </div>
  )
}

export default ApiDocs 

