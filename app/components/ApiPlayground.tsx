import React, { useState } from 'react'

interface ApiCommand {
  category: string
  name: string
  description: string
  curl: string
}

// Organized API commands from ApiDocs.tsx
const API_COMMANDS: Record<string, ApiCommand[]> = {
  "DeFi Analytics": [
    {
      category: "DeFi Analytics",
      name: "Protocol TVL",
      description: "Returns TVL and key metrics for a specific protocol",
      curl: 'curl "https://api.llama.fi/protocol/aave"'
    },
    {
      category: "DeFi Analytics",
      name: "Token Prices",
      description: "Get current token prices across multiple chains",
      curl: 'curl "https://coins.llama.fi/prices/current/ethereum:USDC"'
    },
    {
      category: "DeFi Analytics",
      name: "Historical TVL",
      description: "Get historical TVL data for a protocol",
      curl: 'curl "https://api.llama.fi/tvl/aave"'
    }
  ],
  "Dune Analytics": [
    {
      category: "Dune Analytics",
      name: "Execute Query",
      description: "Execute a saved Dune query",
      curl: 'curl -X POST "https://api.dune.com/api/v1/query/2030582/execute" -H "x-dune-api-key: YOUR_API_KEY"'
    },
    {
      category: "Dune Analytics",
      name: "Query Results",
      description: "Get results from a previously executed query",
      curl: 'curl "https://api.dune.com/api/v1/query/2030582/results" -H "x-dune-api-key: YOUR_API_KEY"'
    }
  ],
  "BitQuery": [
    {
      category: "BitQuery",
      name: "GraphQL Query",
      description: "Query blockchain data using GraphQL",
      curl: `curl "https://graphql.bitquery.io" \\
-H "X-API-KEY: YOUR_API_KEY" \\
-d '{
  "query": "{
    ethereum {
      transfers(limit: 10) {
        amount
        currency { symbol }
        sender { address }
        receiver { address }
      }
    }
  }"
}'`
    }
  ],
  "The Graph": [
    {
      category: "The Graph",
      name: "Query Subgraph",
      description: "Query indexed blockchain data through GraphQL",
      curl: `curl "https://api.thegraph.com/subgraphs/name/aave/protocol-v2" \\
-H "Content-Type: application/json" \\
-d '{
  "query": "{
    markets(first: 5) {
      id
      inputToken {
        id
        symbol
      }
      outputToken {
        id
        symbol
      }
    }
  }"
}'`
    }
  ],
  "NFT Analytics": [
    {
      category: "NFT Analytics",
      name: "Collection Stats",
      description: "Get NFT collection statistics",
      curl: 'curl "https://api.llama.fi/nfts/collection/cryptopunks"'
    },
    {
      category: "NFT Analytics",
      name: "Marketplace Volume",
      description: "Get marketplace trading volume",
      curl: 'curl "https://api.llama.fi/nfts/marketplace/opensea"'
    }
  ]
}

export default function ApiPlayground() {
  const [selectedCategory, setSelectedCategory] = useState(Object.keys(API_COMMANDS)[0])
  const [copiedIndex, _setCopiedIndex] = useState<number | null>(null)

  return (
    <div className="w-full max-w-4xl mx-auto p-6">
      <div className="flex items-center gap-4 mb-6">
        <label className="font-medium">Category:</label>
        <select 
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          className="bg-gray-700 rounded px-3 py-1.5"
        >
          {Object.keys(API_COMMANDS).map(category => (
            <option key={category} value={category}>{category}</option>
          ))}
        </select>
      </div>

      <div className="grid gap-4">
        {API_COMMANDS[selectedCategory].map((command, index) => (
          <div key={index} className="bg-gray-800 rounded-lg p-4">
            <div className="flex justify-between items-start mb-2">
              <div>
                <h3 className="font-medium text-lg">{command.name}</h3>
                <p className="text-gray-400 text-sm">{command.description}</p>
              </div>
              
            </div>
            <pre className="bg-gray-900 p-3 rounded text-sm font-mono overflow-x-auto whitespace-pre-wrap">
              {command.curl}
            </pre>
            {copiedIndex === index && (
              <div className="text-green-400 text-sm mt-2">
                Copied to clipboard!
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}


