interface DuneResponse {
  success: boolean;
  data: any;
  error?: string;
}

export class DuneClient {
  private baseUrl = 'https://api.dune.com/api/v1';

  constructor(private apiKey: string) {}

  async getMetrics(queryId: string): Promise<DuneResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/query/${queryId}`, {
        headers: {
          'x-dune-api-key': this.apiKey
        }
      });

      if (!response.ok) {
        throw new Error(`Dune API error: ${response.statusText}`);
      }

      const data = await response.json();
      return {
        success: true,
        data
      };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async getTokenMetrics(symbol: string): Promise<DuneResponse> {
    // Replace with your actual query ID for token metrics
    return this.getMetrics(`token-metrics-${symbol}`);
  }
} 