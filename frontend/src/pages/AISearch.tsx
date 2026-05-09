import { useState, useEffect, useRef, useCallback } from "react"
import { Search, Sparkles, ArrowLeft, ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { SeverityBadge } from "@/components/ui/SeverityBadge"
import { Skeleton } from "@/components/ui/skeleton"
import { searchAlerts, findSimilarAlerts, getDetectionClasses, getCameras, API_BASE } from "@/lib/api"
import type { Severity } from "@/types"

const TIME_RANGES = [
  { label: "Last Hour", value: "1h" },
  { label: "Today", value: "24h" },
  { label: "Last 7 Days", value: "7d" },
  { label: "Last 30 Days", value: "30d" },
]

const PAGE_SIZE = 40

function formatDateTime(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })
}

export function AISearch() {
  const [cameras, setCameras] = useState<{ id: string; name: string }[]>([])
  const [detectionClasses, setDetectionClasses] = useState<string[]>([])

  const [cameraId, setCameraId] = useState("")
  const [timeRange, setTimeRange] = useState("30d")
  const [detectionClass, setDetectionClass] = useState("")
  const [severity, setSeverity] = useState("")
  const [sortBy, setSortBy] = useState<"relevance" | "recent">("relevance")
  const [query, setQuery] = useState("")
  const [debouncedQuery, setDebouncedQuery] = useState("")

  const [results, setResults] = useState<any[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)

  // "Find similar" mode
  const [similarMode, setSimilarMode] = useState(false)
  const [similarSourceId, setSimilarSourceId] = useState("")

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Load cameras and detection classes on mount
  useEffect(() => {
    getCameras().then((data) => {
      const list = Array.isArray(data) ? data : []
      setCameras(list.map((c: any) => ({ id: c.id, name: c.name || c.id })))
    }).catch(() => {})
    getDetectionClasses().then(setDetectionClasses).catch(() => {})
  }, [])

  // Debounce query
  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setDebouncedQuery(query), 300)
    return () => { if (timerRef.current) clearTimeout(timerRef.current) }
  }, [query])

  // Search when filters or debounced query change
  const doSearch = useCallback(async () => {
    const hasFilter = debouncedQuery.trim() || cameraId || detectionClass || severity
    if (!hasFilter) {
      setResults([])
      setTotal(0)
      setHasSearched(false)
      return
    }
    setLoading(true)
    setHasSearched(true)
    setSimilarMode(false)
    try {
      const data = await searchAlerts({
        q: debouncedQuery.trim() || undefined,
        cameraId: cameraId || undefined,
        severity: severity || undefined,
        detectionClass: detectionClass || undefined,
        timeRange: timeRange || undefined,
        sort: sortBy,
        limit: PAGE_SIZE,
        offset,
      })
      setResults(data.results)
      setTotal(data.total)
    } catch {
      setResults([])
      setTotal(0)
    } finally {
      setLoading(false)
    }
  }, [debouncedQuery, cameraId, severity, detectionClass, timeRange, sortBy, offset])

  useEffect(() => { doSearch() }, [doSearch])

  // Reset offset when filters change
  useEffect(() => { setOffset(0) }, [debouncedQuery, cameraId, severity, detectionClass, timeRange, sortBy])

  const handleFindSimilar = async (alertId: string) => {
    setLoading(true)
    setSimilarMode(true)
    setSimilarSourceId(alertId)
    try {
      const data = await findSimilarAlerts(alertId)
      setResults(data)
      setTotal(data.length)
    } catch {
      setResults([])
      setTotal(0)
    } finally {
      setLoading(false)
    }
  }

  const exitSimilarMode = () => {
    setSimilarMode(false)
    setSimilarSourceId("")
    doSearch()
  }

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))
  const currentPage = Math.floor(offset / PAGE_SIZE)

  const selectClass =
    "px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white text-[var(--color-text-primary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer"

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-[var(--color-text-primary)]">AI Search</h1>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Search across all camera alerts and detections
        </p>
      </div>

      {/* Filter row */}
      <div className="flex flex-wrap items-center gap-3">
        <select value={cameraId} onChange={(e) => setCameraId(e.target.value)} className={selectClass}>
          <option value="">All Cameras</option>
          {cameras.map((c) => (
            <option key={c.id} value={c.id}>{c.name}</option>
          ))}
        </select>

        <select value={timeRange} onChange={(e) => setTimeRange(e.target.value)} className={selectClass}>
          {TIME_RANGES.map((t) => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>

        {detectionClasses.length > 0 && (
          <select value={detectionClass} onChange={(e) => setDetectionClass(e.target.value)} className={selectClass}>
            <option value="">All Detection Classes</option>
            {detectionClasses.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        )}

        <select value={severity} onChange={(e) => setSeverity(e.target.value)} className={selectClass}>
          <option value="">All Severities</option>
          <option value="P1">P1 - Critical</option>
          <option value="P2">P2 - High</option>
          <option value="P3">P3 - Medium</option>
          <option value="P4">P4 - Low</option>
        </select>

        <select value={sortBy} onChange={(e) => setSortBy(e.target.value as any)} className={selectClass}>
          <option value="relevance">Sort: Relevance</option>
          <option value="recent">Sort: Most Recent</option>
        </select>
      </div>

      {/* Text search bar */}
      <div className="relative max-w-lg">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--color-text-tertiary)]" />
        <input
          type="text"
          placeholder="Search by rule, zone, camera, description... e.g., 'Missing Helmet'"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full pl-9 pr-3 py-1.5 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
        />
      </div>

      {/* Similar mode banner */}
      {similarMode && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-[var(--radius-md)] bg-[var(--color-bg-tertiary)] border">
          <Button variant="ghost" size="sm" className="gap-1 text-xs" onClick={exitSimilarMode}>
            <ArrowLeft size={14} /> Back to search
          </Button>
          <span className="text-xs text-[var(--color-text-secondary)]">
            Showing alerts similar to <span className="font-mono font-medium">{similarSourceId}</span>
          </span>
        </div>
      )}

      {/* Results */}
      {loading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-64 rounded-[var(--radius-md)]" />
          ))}
        </div>
      ) : hasSearched || similarMode ? (
        <>
          <div className="flex items-center justify-between">
            <p className="text-sm text-[var(--color-text-secondary)]">
              <span className="font-semibold text-[var(--color-text-primary)]">{total}</span> result{total !== 1 ? "s" : ""}
            </p>
            {!similarMode && total > PAGE_SIZE && (
              <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
                <span>Page {currentPage + 1} of {totalPages}</span>
                <Button variant="ghost" size="sm" className="h-7 w-7 p-0" disabled={currentPage === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>
                  <ChevronLeft size={14} />
                </Button>
                <Button variant="ghost" size="sm" className="h-7 w-7 p-0" disabled={currentPage >= totalPages - 1} onClick={() => setOffset(offset + PAGE_SIZE)}>
                  <ChevronRight size={14} />
                </Button>
              </div>
            )}
          </div>

          {results.length > 0 ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
              {results.map((r) => (
                <ResultCard key={r.id} result={r} onFindSimilar={handleFindSimilar} />
              ))}
            </div>
          ) : (
            <p className="text-center text-sm text-[var(--color-text-tertiary)] py-12">
              No results match the selected filters.
            </p>
          )}
        </>
      ) : (
        <div className="flex flex-col items-center justify-center py-20 text-center text-[var(--color-text-tertiary)]">
          <Sparkles className="w-10 h-10 mb-3" />
          <p className="text-sm max-w-md">
            Search across all camera alerts. Use the filters above or type a search query.
          </p>
        </div>
      )}
    </div>
  )
}

function ResultCard({ result, onFindSimilar }: { result: any; onFindSimilar: (id: string) => void }) {
  const hasSnapshot = !!result.snapshotUrl
  const detectedClasses: string[] = result.detectedClasses || []

  return (
    <Card className="flex flex-col gap-2">
      {/* Snapshot or placeholder */}
      <div className="aspect-[4/3] rounded-[var(--radius-sm)] bg-[var(--color-bg-tertiary)] border overflow-hidden">
        {hasSnapshot ? (
          <img
            src={`${API_BASE}${result.snapshotUrl}`}
            alt={result.rule}
            className="w-full h-full object-cover"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <span className="text-xs text-[var(--color-text-tertiary)]">No snapshot</span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-1.5">
        <SeverityBadge severity={result.severity as Severity} />
        <span className="text-xs text-[var(--color-text-tertiary)] tabular-nums">
          {Math.round(result.confidence * 100)}%
        </span>
      </div>

      <div className="space-y-0.5 min-w-0">
        <p className="text-xs font-medium text-[var(--color-text-primary)] truncate" title={result.rule}>
          {result.rule}
        </p>
        <p className="text-xs text-[var(--color-text-secondary)] truncate">
          {result.cameraName}
        </p>
        <p className="text-xs text-[var(--color-text-tertiary)]">
          {formatDateTime(result.timestamp)}
        </p>
        {result.zone && result.zone !== "Unknown" && (
          <p className="text-xs text-[var(--color-text-tertiary)]">{result.zone}</p>
        )}
      </div>

      {detectedClasses.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {detectedClasses.slice(0, 3).map((c) => (
            <span key={c} className="px-1.5 py-0.5 text-[10px] bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] rounded">
              {c}
            </span>
          ))}
        </div>
      )}

      <Button
        variant="ghost"
        size="sm"
        className="mt-auto self-start text-xs"
        onClick={() => onFindSimilar(result.id)}
      >
        Find similar
      </Button>
    </Card>
  )
}
