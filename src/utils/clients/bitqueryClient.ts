import { BaseApiClient } from './baseClient';

export class BitqueryClient extends BaseApiClient {
  async query(query: string) {
    return this.request('/api/query', {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
  }
} 