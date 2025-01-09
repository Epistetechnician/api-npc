import type { QueryConfig } from '@/types'

export interface ExtendedQueryConfig extends QueryConfig {
  type?: 'track' | 'analyze' | 'visualize';
  apiKey?: string;
}

export interface QueryResult {
  data: any
  metadata: {
    timestamp: string
    source: string
  }
}

export function handleNaturalLanguageQuery(
  query: string,
  config?: ExtendedQueryConfig
): Promise<QueryResult> {
  const fullConfig: ExtendedQueryConfig = {
    query,
    parameters: {},
    ...config
  }
  
  return Promise.resolve({
    data: {},
    metadata: {
      timestamp: new Date().toISOString(),
      source: 'default'
    }
  })
} 