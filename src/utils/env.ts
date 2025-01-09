export function getApiKeys() {
  return {
    dune: process.env.DUNE_API_KEY,
    flipside: process.env.FLIPSIDE_API_KEY,
    supabaseUrl: process.env.NEXT_PUBLIC_SUPABASE_URL,
    supabaseKey: process.env.NEXT_PUBLIC_SUPABASE_KEY
  };
} 