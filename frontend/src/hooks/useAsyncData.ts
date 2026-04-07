import { useState, useEffect, useCallback, useRef } from "react"

interface UseAsyncDataResult<T> {
  data: T | null
  loading: boolean
  error: string | null
  refresh: () => void
}

export function useAsyncData<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = [],
): UseAsyncDataResult<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await fetcherRef.current()
      setData(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }, deps)

  useEffect(() => {
    load()
  }, [load])

  return { data, loading, error, refresh: load }
}
