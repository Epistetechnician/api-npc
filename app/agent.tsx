import React, { useState, useRef, useEffect } from 'react'
import { Terminal as LucideTerminal, Send, X, Maximize2, Minimize2 } from 'lucide-react'
import { 
  aggregateProtocolData,
  trackBlockchainMetrics,
  formatNumber
} from '../src/utils/metrics'
import { 
  handleNaturalLanguageQuery,
  type ExtendedQueryConfig,
  type QueryResult
} from '../src/utils/naturalLanguageQuery'
import { validateConfig } from '../config/env'

interface HistoryEntry {
  type: 'system' | 'user' | 'error' | 'ascii' | 'success' | 'chart' | 'link' | 
        'metric' | 'analytics' | 'protocol' | 'defi' | 'database' | 'table' | 
        'warning' | 'info';
  content: string;
  data?: any;
  analytics?: Array<{
    metric: string;
    value: number;
    change?: string;
  }>;
  tableData?: {
    columns: Array<{
      name: string;
      type: string;
    }>;
    rows: any[];
    summary?: {
      total: number;
      filtered: number;
    };
  };
  metadata?: {
    type: 'api-response' | 'nft_sales' | 'navigation';
    timestamp?: string;
    destination?: string;
  };
}

interface CabalState {
  isInitialized: boolean
  currentCabal: string | null
  creationStep: number
  cabals: string[]
  agents: {
    id: string
    name: string
    personality: AgentPersonality
    role: string
    metrics: AgentMetrics
  }[]
  creationData: {
    name?: string
    description?: string
    purpose?: string
    theme?: string
    governanceSettings?: {
      votingPeriod: number
      quorum: number
      executionDelay: number
    }
  }
  treasury?: {
    daoTokens: string
    pumpTokens: string
  }
}

interface CommandContext {
  args: string[];
  options?: Record<string, any>;
  state: CabalState;
  setState: (state: CabalState) => void;
  apiKeys?: {
    dune?: string;
    flipside?: string;
  };
}

type CommandResult = HistoryEntry;

interface CommandHandler {
  description: string;
  handler: (context: CommandContext) => Promise<CommandResult> | CommandResult;
}

type CommandFunction = CommandHandler['handler'];

interface Commands {
  [key: string]: CommandHandler;
}

// Add these type definitions at the top with other interfaces
interface CommandSuggestion {
  command: string
  description: string
}

// Add new interfaces for enhanced functionality
interface AgentPersonality {
  archetype: string;
  traits: Record<string, number>;
  preferences: string[];
  goals: string[];
}

interface AgentMetrics {
  messagesProcessed: number;
  actionsExecuted: number;
  proposalsCreated: number;
  votesParticipated: number;
}

// Add these new interfaces near the top with other interfaces
interface ApiEndpoint {
  name: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  endpoint: string;
  description: string;
  exampleParams?: Record<string, any>;
  exampleResponse?: any;
}

// Add this near the top of the file with other interfaces
interface ApiConfig {
  baseUrl: string;
  headers: Record<string, string>;
}

// Update the API_ENDPOINTS configuration
const getApiConfig = (): ApiConfig => {
  const duneApiKey = process.env.NEXT_PUBLIC_DUNE_API_KEY;
  if (!duneApiKey) {
    throw new Error('NEXT_PUBLIC_DUNE_API_KEY is not configured');
  }

  return {
    baseUrl: 'https://api.dune.com/api/v1',
    headers: {
      'x-dune-api-key': duneApiKey,
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    }
  };
};

const API_ENDPOINTS: ApiEndpoint[] = [
  {
    name: 'getDuneQuery',
    method: 'GET',
    endpoint: '/query/:query_id/results',
    description: 'Execute a Dune Analytics query',
    exampleParams: {
      query_id: '1234567'
    },
    exampleResponse: {
      execution_id: "01GX0P4K3SN6NZV2QSNF4Q4AHH",
      state: "QUERY_STATE_COMPLETED",
      data: {
        rows: [],
        metadata: {}
      }
    }
  },
  {
    name: 'getProtocols',
    method: 'GET',
    endpoint: '/protocols',
    description: 'Get all DeFi protocols data',
    exampleResponse: {
      protocols: [
        { name: 'Uniswap', tvl: 1000000000, volume24h: 500000000 }
      ]
    }
  },
  // ... other endpoints ...
];

const ASCII_LOGO = `
:::= === :::====  :::===== :::==== :::===== :::====  :::======= 
:::===== :::  === :::      :::==== :::      :::  === ::: === ===
======== =======  ===        ===   ======   =======  === === ===
=== ==== ===      ===        ===   ===      === ===  ===     ===
===  === ===       =======   ===   ======== ===  === ===     ===
`

const STARTUP_MESSAGES = [
  "INITIALIZING API DATASPHERE...",
  "QUANTUM NETWORK SYNCHRONIZING...",
  "CRYPTOGRAPHIC HANDSHAKE IN PROGRESS...",
  "NEURAL INTERFACE CALIBRATING...",
  "DIMENSIONAL PROTOCOL ENGAGING..."
]

// Update the getApiKeys function to handle errors gracefully
function getApiKeysWithFallback() {
  try {
    const dune = process.env.NEXT_PUBLIC_DUNE_API_KEY;
    const flipside = process.env.NEXT_PUBLIC_FLIPSIDE_API_KEY;
    
    // Return empty strings if keys are missing instead of throwing
    return {
      dune: dune || '',
      flipside: flipside || ''
    };
  } catch (error) {
    console.warn('API keys not configured:', error);
    return {
      dune: '',
      flipside: ''
    };
  }
}

// Add these interfaces near the top with other interfaces
interface ApiResponse {
  success: boolean;
  data: any;
  error?: string;
}

// Add the processCurlCommand function before the CabalTerminal component
const processCurlCommand = async (curlCommand: string): Promise<ApiResponse> => {
  try {
    const urlMatch = curlCommand.match(/'https?:\/\/[^']+'/);
    const headerMatch = curlCommand.match(/-H\s+'([^']+)'/g);
    
    if (!urlMatch) {
      throw new Error('No valid URL found in curl command');
    }

    const url = urlMatch[0].replace(/'/g, '');
    const headers: Record<string, string> = {};
    
    // Process all headers
    headerMatch?.forEach(match => {
      const headerParts = match.match(/-H\s+'([^:]+):\s*([^']+)'/);
      if (headerParts) {
        const [_, key, value] = headerParts;
        headers[key] = value.trim();
      }
    });

    // Add CORS proxy for external APIs
    const proxyUrl = process.env.NEXT_PUBLIC_CORS_PROXY || 'https://cors-anywhere.herokuapp.com/';
    const finalUrl = url.startsWith('http') ? `${proxyUrl}${url}` : url;

    // Make the request
    const response = await fetch(finalUrl, {
      method: 'GET',
      headers: {
        ...headers,
        'Accept': 'application/json',
        'Origin': window.location.origin
      },
      mode: 'cors'
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    const data = await response.json();
    return { success: true, data };

  } catch (error) {
    return {
      success: false,
      data: null,
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    };
  }
};

// Update the data mapping with proper type annotations
const formatTableData = (data: Record<string, unknown>[]) => {
  return {
    columns: Object.keys(data[0] || {}),
    rows: data.map((item: Record<string, unknown>) => Object.values(item)),
    summary: {
      total: data.length,
      timestamp: new Date().toISOString()
    }
  };
};

// Keep your MetricData interface definition


const CabalTerminal = () => {
  const [history, setHistory] = useState<HistoryEntry[]>([
    { type: 'ascii', content: ASCII_LOGO }
  ])
  const [input, setInput] = useState('')
  const [isMaximized, setIsMaximized] = useState(true)
  const [networkStatus, setNetworkStatus] = useState('CONNECTING')
  const [cabalState, setCabalState] = useState<CabalState>({
    isInitialized: false,
    currentCabal: null,
    creationStep: 0,
    cabals: [],
    agents: [],
    creationData: {}
  })
  const [suggestions, setSuggestions] = useState<CommandSuggestion[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  
  const terminalRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Initialize the terminal with startup sequence
  useEffect(() => {
    const initializeTerminal = async () => {
      for (const message of STARTUP_MESSAGES) {
        await new Promise(resolve => setTimeout(resolve, 800))
        setHistory(prev => [...prev, { type: 'system', content: message }])
      }
      
      await new Promise(resolve => setTimeout(resolve, 1000))
      setNetworkStatus('CONNECTED')

      // Check API keys
      const apiKeys = getApiKeysWithFallback();
      const missingKeys: string[] = [];
      if (!apiKeys.dune) missingKeys.push('DUNE');
      if (!apiKeys.flipside) missingKeys.push('FLIPSIDE');

      setHistory(prev => [
        ...prev,
        { type: 'success' as const, content: 'SYSTEM INITIALIZED' },
        { type: 'system' as const, content: 'Welcome to aicaba! terminal v1.0.0' },
        ...(missingKeys.length > 0 ? [{
          type: 'warning' as const,
          content: `Note: ${missingKeys.join(', ')} API keys not configured. Some features may be limited.`
        }] : []),
        { type: 'system' as const, content: 'Type "help" for available commands' }
      ] as HistoryEntry[]);
      
      setCabalState(prev => ({ ...prev, isInitialized: true }))
    }

    initializeTerminal()
  }, [])

  // Add environment validation on mount
  useEffect(() => {
    const validateEnvironment = () => {
      if (!validateConfig()) {
        setHistory(prev => [
          ...prev,
          {
            type: 'error',
            content: 'Missing required environment variables. Please check your .env.local file.'
          }
        ]);
      }
    };

    validateEnvironment();
  }, []);


  const commands: Commands = {
    'connect': {
      description: 'Connect to a cabal',
      handler: (context: CommandContext): CommandResult => {
        const cabalName = context.args[0];
        if (!cabalName) {
          return {
            type: 'error',
            content: 'Please specify a cabal name'
          };
        }
        return {
          type: 'system',
          content: 'Connection functionality temporarily disabled'
        };
      }
    },

    'analyze-protocol': {
      description: 'Analyze a protocol',
      handler: async (context: CommandContext): Promise<CommandResult> => {
        const query = context.args.join(' ');
        if (!query) {
          return {
            type: 'error',
            content: 'Please specify a protocol to analyze'
          };
        }

        try {
          const data = await aggregateProtocolData(query);
          return {
            type: 'protocol',
            content: `Analysis for ${query}`,
            protocol: {
              name: query,
              ...data
            }
          };
        } catch (error) {
          return {
            type: 'error',
            content: `Failed to analyze protocol: ${error instanceof Error ? error.message : 'Unknown error'}`
          };
        }
      }
    },

    'track-metrics': {
      description: 'Track blockchain metrics',
      handler: async (context: CommandContext): Promise<CommandResult> => {
        const [metric] = context.args;
        if (!metric) {
          return {
            type: 'error',
            content: 'Please specify metrics to track'
          };
        }

        try {
          const metricData = await trackBlockchainMetrics(metric, []);
          return {
            type: 'metric',
            content: `Tracking ${metric}`,
            metrics: metricData.map(m => ({
              label: m.name,
              value: String(m.value),
              change: m.change,
              trend: m.trend
            }))
          };
        } catch (error) {
          return {
            type: 'error',
            content: `Failed to track metrics: ${error instanceof Error ? error.message : 'Unknown error'}`
          };
        }
      }
    },

    'help': {
      description: 'Show available commands',
      handler: (): CommandResult => {
        const commandList = Object.entries(commands)
          .map(([name, cmd]) => `  - ${name}: ${cmd.description}`)
          .join('\n');

        return {
          type: 'system',
          content: `Available commands:\n${commandList}`
        };
      }
    },

    'markets': {
      description: 'View market updates and analytics',
      handler: (): CommandResult => {
        window.location.href = '/market-updates'
        return {
          type: 'success',
          content: `
            🚀 Redirecting to market updates dashboard...
            
            You'll find:
            - Real-time market metrics
            - Funding rate analysis
            - Top market opportunities
            - Volume and OI trends
            
            Type 'help' for more commands.
          `,
          metadata: {
            type: 'navigation',
            destination: 'market-updates'
          }
        }
      }
    },

    'analyze': {
      description: 'Analyze market data',
      handler: async (context: CommandContext): Promise<CommandResult> => {
        const [metric] = context.args;
        if (!metric) {
          return {
            type: 'error',
            content: 'Please specify a metric to analyze'
          };
        }

        try {
          return {
            type: 'chart',
            content: `Analysis of ${metric}`
          };
        } catch (error) {
          return {
            type: 'error',
            content: `Failed to analyze metric: ${error instanceof Error ? error.message : 'Unknown error'}`
          };
        }
      }
    },

    'query-defi': {
      description: 'Query DeFi protocol data',
      handler: async (context: CommandContext): Promise<CommandResult> => {
        const query = context.args.join(' ');
        if (!query) {
          return {
            type: 'error',
            content: 'Please specify a query'
          };
        }

        try {
          const queryConfig: ExtendedQueryConfig = {
            query,
            ...context.apiKeys
          };
          const result = await handleNaturalLanguageQuery(query, queryConfig);
          return {
            type: 'defi',
            content: result.response,
            data: result.data
          };
        } catch (error) {
          return {
            type: 'error',
            content: `Query failed: ${error instanceof Error ? error.message : 'Unknown error'}`
          };
        }
      }
    },

    'test-endpoint': {
      description: 'Test an API endpoint',
      handler: async (context: CommandContext): Promise<CommandResult> => {
        const [endpoint] = context.args;
        if (!endpoint) {
          return {
            type: 'error',
            content: 'Please specify an endpoint to test'
          };
        }

        try {
          return {
            type: 'success',
            content: 'Endpoint test successful',
            metadata: {
              type: 'api-response',
              timestamp: new Date().toISOString()
            }
          };
        } catch (error) {
          return {
            type: 'error',
            content: `Endpoint test failed: ${error instanceof Error ? error.message : 'Unknown error'}`
          };
        }
      }
    },

    'get-my-perps': {
      description: 'View perpetual metrics dashboard',
      handler: async (): Promise<CommandResult> => {
        try {
          window.open('/perp-metrics', '_blank')
          return {
            type: 'success',
            content: 'Opening perpetual metrics dashboard in new tab...'
          };
        } catch (error) {
          return {
            type: 'error',
            content: `Failed to open perpetual metrics dashboard: ${error instanceof Error ? error.message : 'Unknown error'}`
          };
        }
      }
    }
  };

  // Add clear command to commands object
  const clearCommand = (): HistoryEntry => {
    setHistory([{ type: 'ascii', content: ASCII_LOGO }]);
    return {
      type: 'system',
      content: 'Terminal cleared'
    };
  }

  // Update handleCommand function
  const handleCommand = async (cmd: string) => {
    const args = cmd.trim().split(' ')
    const commandName = args[0].toLowerCase()
    
    setHistory(prev => [...prev, { type: 'user', content: cmd }])
    
    if (commandName === 'clear') {
      clearCommand()
      return
    }
    
    const command = commands[commandName]
    
    if (command) {
      const context: CommandContext = {
        args: args.slice(1),
        state: cabalState,
        setState: setCabalState,
        apiKeys: getApiKeysWithFallback()
      }
      
      try {
        const result = await command.handler(context)
        setHistory(prev => [...prev, result])
      } catch (error) {
        setHistory(prev => [...prev, {
          type: 'error',
          content: error instanceof Error ? error.message : 'An unknown error occurred'
        }])
      }
    } else {
      setHistory(prev => [...prev, {
        type: 'error',
        content: `Command not found: ${args[0]}`
      }])
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim()) {
      handleCommand(input)
      setInput('')
    }
  }

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [history])

  // Update click handler to be more selective
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      // Only focus if clicking inside the terminal container
      if (terminalRef.current?.contains(e.target as Node)) {
        inputRef.current?.focus()
      }
    }

    // Add click handler to terminal container only
    terminalRef.current?.addEventListener('click', handleClick)
    
    return () => {
      terminalRef.current?.removeEventListener('click', handleClick)
    }
  }, [])

  // Add new handler for terminal container clicks
  const handleTerminalClick = (e: React.MouseEvent) => {
    // Prevent losing focus when selecting text
    if (window.getSelection()?.toString()) {
      return;
    }
    
    // Keep focus on input unless clicking on a specific interactive element
    const target = e.target as HTMLElement;
    if (!target.closest('button') && !target.closest('a')) {
      inputRef.current?.focus();
    }
  }

  // Add command suggestions logic
  const getCommandSuggestions = (input: string): CommandSuggestion[] => {
    const commandList = Object.entries(commands).map(([name, cmd]) => ({
      command: name,
      description: typeof cmd === 'object' ? cmd.description : 'No description available'
    }))

    if (!input) return commandList
    return commandList.filter(cmd => 
      cmd.command.toLowerCase().startsWith(input.toLowerCase())
    )
  }

  // Update input handling
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setInput(value)
    setSuggestions(getCommandSuggestions(value))
    setShowSuggestions(true)
  }

  // Add suggestion selection
  const handleSuggestionClick = (command: string) => {
    setInput(command)
    setShowSuggestions(false)
    inputRef.current?.focus()
  }

  // Update the renderHistoryEntry function
  const renderHistoryEntry = (entry: HistoryEntry): JSX.Element | undefined => {
    switch (entry.type) {
      case 'chart':
        return (
          <div className="my-2 p-4 bg-gray-800/50 rounded-lg">
            <h3 className="text-sm font-medium mb-2">{entry.content}</h3>
            {/* Add your preferred charting library component here */}
            <div className="h-64 w-full">
              {/* Chart component using entry.data */}
            </div>
          </div>
        );
        
      case 'metric':
        return (
          <div className="my-2 p-4 bg-gray-800/50 rounded-lg">
            <h3 className="text-sm font-medium mb-2">{entry.content}</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {entry.metrics?.map((metric, i) => (
                <div key={i} className="flex flex-col">
                  <span className="text-gray-400 text-sm">{metric.label}</span>
                  <span className="text-xl font-mono">{metric.value}</span>
                  {metric.change && (
                    <span className={`text-sm ${
                      metric.trend === 'up' ? 'text-green-400' :
                      metric.trend === 'down' ? 'text-red-400' :
                      'text-gray-400'
                    }`}>
                      {metric.change}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        );

      case 'link':
        return (
          <div className="my-2 p-4 bg-gray-800/50 rounded-lg">
            <h3 className="text-sm font-medium mb-2">{entry.content}</h3>
            <div className="flex flex-col gap-2">
              {entry.links?.map((link, i) => (
                <a 
                  key={i}
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer" 
                  className="text-blue-400 hover:text-blue-300 transition-colors"
                >
                  {link.title}
                </a>
              ))}
            </div>
          </div>
        );

      case 'analytics':
        return (
          <div className="my-2 p-4 bg-gray-800/50 rounded-lg">
            <h3 className="text-sm font-medium mb-4 text-purple-400">{entry.content}</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {entry.analytics?.metrics.map((metric, i) => (
                <div key={i} className="bg-gray-900/50 p-4 rounded-lg border border-gray-700/50">
                  {/* Token Header */}
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-lg font-bold text-white">{metric.name}</span>
                    <span className={`text-sm px-2 py-1 rounded ${
                      metric.value.change > 0 ? 'bg-green-500/20 text-green-400' :
                      metric.value.change < 0 ? 'bg-red-500/20 text-red-400' :
                      'bg-gray-500/20 text-gray-400'
                    }`}>
                      {metric.value.change > 0 ? '↑' : metric.value.change < 0 ? '↓' : '→'} 
                      {Math.abs(metric.value.change).toFixed(2)}%
                    </span>
                  </div>

                  {/* Market Data */}
                  <div className="space-y-4">
                    {/* Spot Market */}
                    <div className="space-y-2">
                      <div className="text-sm font-medium text-purple-400">Spot Market</div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="text-gray-400">Price:</div>
                        <div className="text-right font-mono text-white">${metric.value.price.toFixed(4)}</div>
                        <div className="text-gray-400">Volume:</div>
                        <div className="text-right font-mono text-white">${formatNumber(metric.value.volume)}</div>
                        <div className="text-gray-400">Liquidity:</div>
                        <div className="text-right font-mono text-white">${formatNumber(metric.value.liquidity)}</div>
                      </div>
                    </div>

                    {/* Perp Market */}
                    <div className="space-y-2 border-t border-gray-700/50 pt-2">
                      <div className="text-sm font-medium text-purple-400">Perpetual Market</div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="text-gray-400">Mark Price:</div>
                        <div className="text-right font-mono text-white">${metric.value.mark_price.toFixed(4)}</div>
                        <div className="text-gray-400">Funding Rate:</div>
                        <div className="text-right font-mono text-white">{(metric.value.funding_rate * 100).toFixed(4)}%</div>
                        <div className="text-gray-400">Volume:</div>
                        <div className="text-right font-mono text-white">${formatNumber(metric.value.perp_volume_24h)}</div>
                        <div className="text-gray-400">Open Interest:</div>
                        <div className="text-right font-mono text-white">${formatNumber(metric.value.open_interest)}</div>
                      </div>
                    </div>

                    {/* Market Stats */}
                    <div className="space-y-2 border-t border-gray-700/50 pt-2">
                      <div className="text-sm font-medium text-purple-400">Market Stats</div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="text-gray-400">Market Cap:</div>
                        <div className="text-right font-mono text-white">${formatNumber(metric.value.market_cap)}</div>
                        <div className="text-gray-400">Transactions:</div>
                        <div className="text-right font-mono text-white">{formatNumber(metric.value.txns_24h)}</div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );

      case 'protocol':
        return (
          <div className="my-2 p-4 bg-gray-800/50 rounded-lg">
            <h3 className="text-lg font-medium mb-2">{entry.protocol?.name}</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>TVL: ${formatNumber(entry.protocol?.tvl || 0)}</div>
              <div>24h Volume: ${formatNumber(entry.protocol?.volume24h || 0)}</div>
              <div>24h Fees: ${formatNumber(entry.protocol?.fees24h || 0)}</div>
              <div>24h Users: {formatNumber(entry.protocol?.users24h || 0)}</div>
            </div>
            <div className="mt-2">
              <span className="text-sm text-gray-400">Chains: </span>
              {entry.protocol?.chains?.join(', ')}
            </div>
          </div>
        );

      case 'success':
        // First handle NFT sales case
        if (entry.metadata?.type === 'nft_sales') {
          return (
            <div className="my-4 space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h3 className="text-xl font-bold text-purple-400 mb-4">NFT Sales Analysis</h3>
                <pre className="whitespace-pre-wrap font-mono text-sm text-gray-300" suppressHydrationWarning>
                  {entry.content}
                </pre>
              </div>
            </div>
          );
        }

        // Then handle API response case
        if (entry.metadata?.type === 'api-response') {
          return (
            <div className="my-2 p-4 bg-gray-800/50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-400 font-medium">{entry.content}</span>
                {entry.metadata?.timestamp && (
                  <span className="text-gray-500 text-sm">
                    {new Date(entry.metadata.timestamp).toLocaleTimeString()}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
                {entry.data && (
                  <>
                    <div>
                      <span className="text-gray-400">Endpoint:</span>
                      <span className="ml-2 text-green-400">{entry.data.endpoint}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Method:</span>
                      <span className="ml-2 text-blue-400">{entry.data.method}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Status:</span>
                      <span className={`ml-2 ${
                        entry.data.status >= 200 && entry.data.status < 300 
                          ? 'text-green-400' 
                          : 'text-red-400'
                      }`}>
                        {entry.data.status}
                      </span>
                    </div>
                  </>
                )}
              </div>
              {entry.data?.params && Object.keys(entry.data.params).length > 0 && (
                <div className="mb-4">
                  <div className="text-gray-400 mb-1">Parameters:</div>
                  <pre className="bg-black/30 p-2 rounded">
                    {JSON.stringify(entry.data.params, null, 2)}
                  </pre>
                </div>
              )}
              {entry.data?.response && (
                <div>
                  <div className="text-gray-400 mb-1">Response:</div>
                  <pre className="bg-black/30 p-2 rounded overflow-auto max-h-96">
                    {JSON.stringify(entry.data.response, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          );
        }

        // Default success case
        return (
          <pre className="whitespace-pre-wrap font-mono" suppressHydrationWarning>
            {entry.content}
          </pre>
        );

      case 'database':
        if (!entry.tableData) return undefined;
        return (
          <div className="my-4">
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-purple-400">Database Operation</h3>
                {entry.tableData.summary && (
                  <span className="text-sm text-gray-400">
                    {entry.tableData.summary.total} records • 
                    {entry.tableData.summary.timestamp && 
                      new Date(entry.tableData.summary.timestamp).toLocaleTimeString()}
                  </span>
                )}
              </div>
              
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-700">
                  <thead>
                    <tr>
                      {entry.tableData.columns.map((col, i) => (
                        <th key={i} className="px-4 py-2 text-left text-gray-400">
                          {col.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-700">
                    {entry.tableData.rows.map((row, i) => (
                      <tr key={i}>
                        {Object.values(row).map((cell: any, j) => (
                          <td key={j} className="px-4 py-2 text-gray-300">
                            {typeof cell === 'number' ? 
                              new Intl.NumberFormat().format(cell) : 
                              String(cell)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        );

      // ... existing cases ...
    }
    return undefined;
  };

  // Update the table rendering function
  const renderTable = (data: any[]): CommandResult => {
    if (!Array.isArray(data) || !data.length) {
      return {
        type: 'error',
        content: 'No data to display'
      };
    }

    const tableData: TableData = {
      columns: Object.keys(data[0]).map((col: string) => ({
        name: col,
        type: typeof data[0][col]
      })),
      rows: data,
      summary: {
        total: data.length,
        filtered: data.length,
        timestamp: new Date().toISOString()
      }
    };

    return {
      type: 'table',
      content: 'Query Results:',
      tableData
    };
  }

  // Update the analytics rendering function
  const renderAnalytics = (metrics: AnalyticsMetric[]): CommandResult => {
    return {
      type: 'analytics',
      content: 'Analytics Results:',
      analytics: metrics.map(metric => ({
        metric: metric.metric,
        value: metric.value,
        change: metric.change
      }))
    };
  }

  // Add at the top with other interfaces
  interface ExtendedQueryConfig {
    query: string;
    dune?: string;
    flipside?: string;
  }

  interface QueryResult {
    data: any;
    response: string;
  }

  interface TableData {
    columns: Array<{
      name: string;
      type: string;
    }>;
    rows: any[];
    summary: {
      total: number;
      filtered: number;
      timestamp?: string;
    };
  }

  interface AnalyticsMetric {
    metric: string;
    value: number;
    change?: string;
    trend?: 'up' | 'down' | 'neutral';
  }

  return (
    <div
      className={`relative overflow-hidden rounded-lg border border-gray-700 bg-black/90 backdrop-blur-sm transition-all duration-300 ${
        isMaximized ? 'h-[80vh]' : 'h-[600px]'
      }`}
      onClick={handleTerminalClick}
    >
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800/50 border-b border-gray-700/50">
        <div className="flex items-center gap-4">
          <div className="flex items-center space-x-2">
            <LucideTerminal className="h-4 w-4 text-purple-400" />
            <span className="text-sm font-medium text-gray-200">AI Cabal Terminal</span>
            <span className={`text-xs ${
              networkStatus === 'CONNECTED' ? 'text-green-400' : 'text-yellow-400'
            }`}>
              [{networkStatus}]
            </span>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsMaximized(!isMaximized)}
            className="rounded p-1.5 hover:bg-gray-700/50"
          >
            {isMaximized ? (
              <Minimize2 className="h-4 w-4 text-gray-400" />
            ) : (
              <Maximize2 className="h-4 w-4 text-gray-400" />
            )}
          </button>
          <button
            onClick={clearCommand}
            className="rounded p-1.5 hover:bg-gray-700/50"
          >
            <X className="h-4 w-4 text-gray-400" />
          </button>
        </div>
      </div>

      {/* Terminal Body */}
      <div
        ref={terminalRef}
        className="h-[calc(100%-8rem)] overflow-y-auto p-4 font-mono text-sm select-text"
        onMouseDown={(e) => {
          // Allow text selection while preventing focus loss
          if (e.target !== inputRef.current) {
            e.preventDefault();
          }
        }}
      >
        {history.map((entry, i) => (
          <div
            key={i}
            className={`mb-2 ${
              entry.type === 'ascii' ? 'text-blue-400' :
              entry.type === 'user' ? 'text-green-400' :
              entry.type === 'error' ? 'text-red-400' :
              entry.type === 'success' ? 'text-emerald-400' :
              'text-gray-300'
            }`}
          >
            {entry.type === 'user' ? '> ' : ''}
            {renderHistoryEntry(entry) || (
              <pre className="whitespace-pre-wrap font-mono">
                {entry.content}
              </pre>
            )}
          </div>
        ))}
      </div>

      {/* Updated Input Area */}
      <form
        onSubmit={handleSubmit}
        className="absolute bottom-0 left-0 right-0 border-t border-gray-700/50 bg-gray-900/50"
        onClick={(e) => e.stopPropagation()} // Prevent terminal click handler
      >
        <div className="relative">
          {showSuggestions && suggestions.length > 0 && (
            <div className="absolute bottom-full left-0 w-full max-h-48 overflow-y-auto bg-gray-900/95 border border-gray-700 rounded-t-lg">
              {suggestions.map((suggestion, index) => (
                <div
                  key={index}
                  onClick={() => handleSuggestionClick(suggestion.command)}
                  className="px-4 py-2 hover:bg-gray-800/50 cursor-pointer flex justify-between"
                >
                  <span className="text-green-400">{suggestion.command}</span>
                  <span className="text-gray-500 text-sm">{suggestion.description}</span>
                </div>
              ))}
            </div>
          )}
          <div className="flex items-center space-x-2 p-3">
            <span className="text-green-400 font-mono">{'>'}</span>
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={handleInputChange}
              onFocus={() => setShowSuggestions(true)}
              onBlur={() => {
                // Delay hiding suggestions to allow for clicks
                setTimeout(() => setShowSuggestions(false), 200)
              }}
              className="flex-1 bg-transparent font-mono text-gray-200 outline-none"
              placeholder="Type a command or use tab for suggestions..."
              autoFocus
            />
            <button
              type="submit"
              className="rounded p-1.5 hover:bg-gray-700/50 text-gray-400"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </div>
      </form>
    </div>
  )
}
export default CabalTerminal

