import { useState, useMemo, useEffect, useRef } from "react"
import { Search, Sparkles } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { toast } from "sonner"

const CAMERAS = [
  "All Cameras",
  "Gate 1 Entry",
  "Gate 2 Exit",
  "Welding Bay",
  "Assembly Line",
  "Loading Dock",
  "Testing Area",
]

const TIME_RANGES = ["Last Hour", "Today", "Last 7 Days", "Last 30 Days"]

const DETECTION_CLASSES = [
  "All Classes",
  "Person",
  "Vehicle",
  "PPE Violation",
  "Fire/Smoke",
  "Phone Usage",
]

function randomFrom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]
}

function generateMockResults() {
  const cameras = CAMERAS.slice(1)
  const classes = DETECTION_CLASSES.slice(1)
  const now = new Date()

  return Array.from({ length: 16 }, (_, i) => {
    const ts = new Date(now.getTime() - Math.random() * 8 * 3600 * 1000)
    return {
      id: `result-${i}`,
      camera: randomFrom(cameras),
      detectionClass: randomFrom(classes),
      similarity: Math.floor(Math.random() * 36) + 60, // 60-95
      timestamp: ts.toISOString(),
    }
  })
}

const MOCK_RESULTS = generateMockResults()

export function AISearch() {
  const [camera, setCamera] = useState("All Cameras")
  const [timeRange, setTimeRange] = useState("Last Hour")
  const [detectionClass, setDetectionClass] = useState("All Classes")
  const [query, setQuery] = useState("")
  const [debouncedQuery, setDebouncedQuery] = useState("")
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      setDebouncedQuery(query)
    }, 300)
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [query])

  const hasInteracted =
    camera !== "All Cameras" ||
    detectionClass !== "All Classes" ||
    debouncedQuery.trim().length > 0

  const filtered = useMemo(() => {
    if (!hasInteracted) return []
    return MOCK_RESULTS.filter((r) => {
      if (camera !== "All Cameras" && r.camera !== camera) return false
      if (detectionClass !== "All Classes" && r.detectionClass !== detectionClass)
        return false
      return true
    })
  }, [camera, detectionClass, hasInteracted])

  const selectClass =
    "px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white text-[var(--color-text-primary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer"

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-[var(--color-text-primary)]">
          AI Search
        </h1>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Search across all camera footage
        </p>
      </div>

      {/* Filter row */}
      <div className="flex flex-wrap items-center gap-3">
        <select
          value={camera}
          onChange={(e) => setCamera(e.target.value)}
          className={selectClass}
        >
          {CAMERAS.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>

        <select
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
          className={selectClass}
        >
          {TIME_RANGES.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>

        <select
          value={detectionClass}
          onChange={(e) => setDetectionClass(e.target.value)}
          className={selectClass}
        >
          {DETECTION_CLASSES.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </div>

      {/* Text search bar */}
      <div className="relative max-w-lg">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--color-text-tertiary)]" />
        <input
          type="text"
          placeholder="Describe what you're looking for... e.g., 'person near forklift'"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full pl-9 pr-3 py-1.5 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
        />
      </div>

      {/* Results */}
      {hasInteracted ? (
        <>
          <p className="text-sm text-[var(--color-text-secondary)]">
            <span className="font-semibold text-[var(--color-text-primary)]">
              {filtered.length}
            </span>{" "}
            results
          </p>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
            {filtered.map((r) => (
              <ResultCard key={r.id} result={r} />
            ))}
          </div>

          {filtered.length === 0 && (
            <p className="text-center text-sm text-[var(--color-text-tertiary)] py-12">
              No results match the selected filters.
            </p>
          )}
        </>
      ) : (
        <div className="flex flex-col items-center justify-center py-20 text-center text-[var(--color-text-tertiary)]">
          <Sparkles className="w-10 h-10 mb-3" />
          <p className="text-sm max-w-md">
            Search across all camera footage. Use the filters above or describe
            what you're looking for.
          </p>
        </div>
      )}
    </div>
  )
}

function ResultCard({
  result,
}: {
  result: {
    id: string
    camera: string
    detectionClass: string
    similarity: number
    timestamp: string
  }
}) {
  const ts = new Date(result.timestamp)
  const timeStr = ts.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  })

  let matchVariant: "success" | "warning" | "default"
  let matchLabel: string
  if (result.similarity > 85) {
    matchVariant = "success"
    matchLabel = "High match"
  } else if (result.similarity >= 60) {
    matchVariant = "warning"
    matchLabel = "Possible match"
  } else {
    matchVariant = "default"
    matchLabel = "Low match"
  }

  return (
    <Card className="flex flex-col gap-2">
      {/* Placeholder image */}
      <div className="aspect-[4/3] rounded-[var(--radius-sm)] bg-[var(--color-bg-tertiary)] border flex items-center justify-center overflow-hidden">
        <svg
          width="160"
          height="120"
          viewBox="0 0 160 120"
          className="w-full h-full"
        >
          <rect width="160" height="120" fill="#e5e5e5" />
          <text
            x="80"
            y="60"
            textAnchor="middle"
            dominantBaseline="central"
            fontSize="12"
            fill="#a3a3a3"
            fontFamily="sans-serif"
          >
            Detection
          </text>
        </svg>
      </div>

      <Badge variant={matchVariant} className="self-start">
        {matchLabel}
      </Badge>

      <div className="space-y-0.5">
        <p className="text-xs text-[var(--color-text-secondary)] truncate">
          {result.camera}
        </p>
        <p className="text-xs font-mono text-[var(--color-text-tertiary)]">
          {timeStr}
        </p>
        <p className="text-xs text-[var(--color-text-secondary)]">
          {result.detectionClass}
        </p>
      </div>

      <Button
        variant="ghost"
        size="sm"
        className="mt-auto self-start"
        onClick={() => toast("Finding similar detections...")}
      >
        Find similar
      </Button>
    </Card>
  )
}
