import { Flipside, QueryResultSet } from '@flipsidecrypto/sdk'

export class FlipsideClient {
  private sdk: Flipside

  constructor(apiKey: string) {
    this.sdk = new Flipside(
      apiKey,
      'https://api.flipsidecrypto.com'
    )
  }

  async runQuery(query: string): Promise<QueryResultSet> {
    try {
      const result = await this.sdk.query.run({
        sql: query,
        timeoutMinutes: 5,
      })
      
      return result
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Query failed')
    }
  }
} 