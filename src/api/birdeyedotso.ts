import axios from 'axios';
import { env } from '@/config/env';

const birdeyedotso = axios.create({
  baseURL: 'https://public-api.birdeye.so/v1',
  headers: {
    'x-api-key': env.BIRDEYE_API_KEY,
    'Accept': 'application/json',
    'Content-Type': 'application/json'
  }
});

// Add response interceptor for better error handling
birdeyedotso.interceptors.response.use(
  response => response,
  error => {
    console.error('Birdeye API error:', {
      status: error.response?.status,
      data: error.response?.data,
      config: {
        url: error.config?.url,
        method: error.config?.method
      }
    });
    return Promise.reject(error);
  }
);

export default birdeyedotso; 