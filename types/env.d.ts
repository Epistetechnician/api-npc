declare namespace NodeJS {
  interface ProcessEnv {
    NEXT_PUBLIC_DUNE_API_KEY: string;
    NEXT_PUBLIC_FLIPSIDE_API_KEY: string;
    NODE_ENV: 'development' | 'production' | 'test';
    // Add other environment variables as needed
  }
} 