declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NEXT_PUBLIC_DUNE_API_KEY: string;
      NEXT_PUBLIC_FLIPSIDE_API_KEY: string;
      DUNE_API_KEY?: string;
      FLIPSIDE_API_KEY?: string;
      NEXT_PUBLIC_SUPABASE_URL?: string;
      NEXT_PUBLIC_SUPABASE_KEY?: string;
    }
  }
}

export function getApiKeys() {
  return {
    dune: process.env.DUNE_API_KEY,
    flipside: process.env.FLIPSIDE_API_KEY,
    supabaseUrl: process.env.NEXT_PUBLIC_SUPABASE_URL,
    supabaseKey: process.env.NEXT_PUBLIC_SUPABASE_KEY
  };
}

export function validateEnvironmentVariables() {
  const required = [
    'NEXT_PUBLIC_DUNE_API_KEY',
    'NEXT_PUBLIC_FLIPSIDE_API_KEY'
  ] as const;

  const missing = required.filter(key => !process.env[key]);

  if (missing.length > 0) {
    throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
  }

  return true;
}

// This exports the type declarations to be used across the app
export {} 