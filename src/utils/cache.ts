const CACHE_DURATION = 5 * 60 * 1000 // 5 minutes

interface CacheEntry {
  value: number
  timestamp: number
}

const holderCountCache = new Map<string, CacheEntry>()

export function getCachedHolderCount(address: string): number | null {
  const cached = holderCountCache.get(address)
  if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
    return cached.value
  }
  return null
}

export function setCachedHolderCount(address: string, count: number): void {
  holderCountCache.set(address, {
    value: count,
    timestamp: Date.now()
  })
} 