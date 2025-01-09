/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  eslint: {
    // Only run ESLint on build in production
    ignoreDuringBuilds: process.env.NODE_ENV !== 'production',
  },
  typescript: {
    // Only run type checking on build in production
    ignoreBuildErrors: process.env.NODE_ENV !== 'production',
  },
  // Optimize for Vercel deployment
  output: 'standalone',
  // Handle environment variables
  env: {
    NEXT_PUBLIC_DUNE_API_KEY: process.env.NEXT_PUBLIC_DUNE_API_KEY,
    NEXT_PUBLIC_FLIPSIDE_API_KEY: process.env.NEXT_PUBLIC_FLIPSIDE_API_KEY,
  },
  // Add proper CORS headers
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Credentials', value: 'true' },
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET,POST,OPTIONS' },
          { key: 'Access-Control-Allow-Headers', value: 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version' },
        ]
      }
    ]
  }
}

module.exports = nextConfig 