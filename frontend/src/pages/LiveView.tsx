import { useState, useEffect } from "react"
import { useSearchParams } from "react-router-dom"
import {
  Maximize2,
  AlertTriangle,
  Brain,
  WifiOff,
  Grid2x2,
  Grid3x3,
  ChevronLeft,
  ChevronRight,
  FileText,
  Clock,
  Armchair,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { API_BASE, STREAM_BASE, getCameraOccupancy, getCameras as fetchCameras, getToken } from "@/lib/api"
import { useAlertStore } from "@/stores/alertStore"
import { useAlertConnection } from "@/components/AlertProvider"

interface CameraInfo {
  id: string
  name: string
  zone: string
  rules: string[]
  status: string
  runtime_status: string
  enabled: boolean
  detectionsCount: number
  capabilities?: string[]
}

interface OccupancyChair {
  id: string
  name: string
  status: "occupied" | "empty"
  occupied: boolean
  sampleSeconds: number
  reportReady: boolean
  emptyEvents: number
  longAbsenceEvents: number
  emptyForSeconds: number
}

interface OccupancyReport {
  cameraId: string
  title: string
  sessionSeconds: number
  sampleSeconds: number
  reportReady: boolean
  workHours: { start: string; end: string }
  excludedWindows: { label: string; start: string; end: string }[]
  gracePeriodSeconds: number
  chairs: OccupancyChair[]
}

const formatDuration = (seconds: number) => {
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  return `${hours}h ${minutes % 60}m`
}

export function LiveView() {
  const [searchParams, setSearchParams] = useSearchParams()
  const focusedCamId = searchParams.get("cam")
  const [gridCols, setGridCols] = useState(3)
  const [cameras, setCameras] = useState<CameraInfo[]>([])
  const [camerasLoading, setCamerasLoading] = useState(true)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [vlmResult, setVlmResult] = useState<{ text: string; timestamp: string; elapsed: number } | null>(null)
  const [occupancyReport, setOccupancyReport] = useState<OccupancyReport | null>(null)
  const [page, setPage] = useState(0)
  const connected = useAlertConnection((s) => s.connected)
  const alerts = useAlertStore((s) => s.alerts)

  // Page size matches the visible grid (2×2 = 4, 3×3 = 9) so no slots are wasted.
  const pageSize = gridCols * gridCols
  const focusCamera = (camId: string) => setSearchParams({ cam: camId })
  const unfocusCamera = () => setSearchParams({})
  const totalPages = Math.ceil(cameras.length / pageSize)
  const pagedCameras = cameras.slice(page * pageSize, (page + 1) * pageSize)
  const displayedCameras = focusedCamId ? cameras.filter((c) => c.id === focusedCamId) : pagedCameras
  const focusedCamera = focusedCamId ? cameras.find((c) => c.id === focusedCamId) : null
  const showOccupancyPanel = Boolean(
    focusedCamera?.capabilities?.includes("office_occupancy") && occupancyReport
  )
  const occupancyChairs = occupancyReport?.chairs ?? []
  const occupiedChairCount = occupancyChairs.filter((chair) => chair.occupied).length
  const emptyChairCount = occupancyChairs.length - occupiedChairCount

  // Clamp page when camera count or grid size shrinks the pagination
  useEffect(() => {
    if (totalPages > 0 && page > totalPages - 1) setPage(totalPages - 1)
  }, [totalPages, page])

  // Fetch cameras on mount + poll so newly-added cameras appear without a reload
  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const data = await fetchCameras()
        if (!cancelled) {
          setCameras(data)
          setCameraError(null)
        }
      } catch (err) {
        if (!cancelled) {
          setCameraError(err instanceof Error ? err.message : "Could not load cameras")
        }
      } finally {
        if (!cancelled) setCamerasLoading(false)
      }
    }
    void load()
    const interval = setInterval(load, 10000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [])

  useEffect(() => {
    if (!focusedCamera?.capabilities?.includes("office_occupancy")) {
      setOccupancyReport(null)
      return
    }

    let cancelled = false
    const load = async () => {
      try {
        const report = await getCameraOccupancy(focusedCamera.id)
        if (!cancelled) setOccupancyReport(report)
      } catch {
        if (!cancelled) setOccupancyReport(null)
      }
    }
    void load()
    const interval = setInterval(load, 3000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [focusedCamera?.id, focusedCamera?.capabilities])

  // Poll VLM result (backend returns { cam_id: { text, timestamp, elapsed } })
  useEffect(() => {
    const token = getToken()
    const headers: Record<string, string> = {}
    if (token) headers["Authorization"] = `Bearer ${token}`
    const interval = setInterval(() => {
      fetch(`${API_BASE}/api/vlm/latest`, { headers })
        .then((r) => r.json())
        .then((data) => {
          // Find the most recent VLM result across all cameras
          let latest: { text: string; timestamp: string; elapsed: number } | null = null
          for (const camId of Object.keys(data)) {
            const entry = data[camId]
            if (entry && entry.text && (!latest || entry.timestamp > latest.timestamp)) {
              latest = entry
            }
          }
          if (latest) setVlmResult(latest)
        })
        .catch(() => {})
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  const activeAlerts = alerts.filter((a) => a.status === "active")

  const isCameraLive = (cam: CameraInfo) => cam.enabled && cam.status === "online" && cam.runtime_status === "running"
  const statusLabel = (cam: CameraInfo) => {
    if (!cam.enabled) return "Disabled"
    if (cam.runtime_status === "awaiting_model_install") return "Waiting for model"
    if (cam.runtime_status === "starting") return "Starting"
    if (cam.status === "online" && cam.runtime_status === "running") return "Live"
    return "Offline"
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">

        {/* Connection lost banner */}
        {!connected && (
          <div className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white text-xs font-medium">
            <WifiOff size={14} />
            Connection to backend lost. Retrying...
          </div>
        )}

        {/* Layout toolbar */}
        <div className="flex items-center gap-2 px-4 py-1.5 bg-[var(--color-bg-secondary)]">
          <span className="text-xs font-medium text-neutral-600 dark:text-neutral-400 mr-1">Grid</span>
          {focusedCamId && (
            <button
              onClick={unfocusCamera}
              className="inline-flex items-center justify-center rounded-md p-1.5 text-neutral-500 hover:bg-neutral-100 dark:hover:bg-neutral-800 hover:text-neutral-700 dark:hover:text-neutral-300 transition-colors cursor-pointer"
              title="Exit fullscreen"
            >
              <Maximize2 size={15} />
            </button>
          )}
          {[
            { cols: 2, icon: Grid2x2 },
            { cols: 3, icon: Grid3x3 },
          ].map(({ cols, icon: Icon }) => (
            <button
              key={cols}
              onClick={() => { setGridCols(cols); if (focusedCamId) unfocusCamera() }}
              className={cn(
                "inline-flex items-center justify-center rounded-md p-1.5 transition-colors cursor-pointer",
                gridCols === cols && !focusedCamId
                  ? "bg-neutral-200 dark:bg-neutral-700 text-neutral-900 dark:text-white"
                  : "text-neutral-500 hover:bg-neutral-100 dark:hover:bg-neutral-800 hover:text-neutral-700 dark:hover:text-neutral-300"
              )}
              title={`${cols}×${cols} grid`}
            >
              <Icon size={15} />
            </button>
          ))}

          {/* Pagination */}
          {!focusedCamId && totalPages > 1 && (
            <div className="flex items-center gap-1 ml-auto">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="inline-flex items-center justify-center rounded-md p-1.5 text-neutral-500 hover:bg-neutral-100 dark:hover:bg-neutral-800 hover:text-neutral-700 dark:hover:text-neutral-300 transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-default"
              >
                <ChevronLeft size={15} />
              </button>
              <span className="text-xs text-neutral-500 min-w-[60px] text-center">
                {page + 1} / {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="inline-flex items-center justify-center rounded-md p-1.5 text-neutral-500 hover:bg-neutral-100 dark:hover:bg-neutral-800 hover:text-neutral-700 dark:hover:text-neutral-300 transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-default"
              >
                <ChevronRight size={15} />
              </button>
            </div>
          )}
        </div>

        {/* Camera grid */}
        <div className="flex-1 overflow-auto p-4 bg-[var(--color-bg-secondary)]">
          {camerasLoading ? (
            <div
              className="grid gap-3 h-full"
              style={{ gridTemplateColumns: `repeat(${gridCols}, minmax(0, 1fr))` }}
            >
              {Array.from({ length: gridCols * 2 }).map((_, i) => (
                <div key={i} className="rounded-[var(--radius-lg)] overflow-hidden border border-neutral-800 bg-neutral-900 flex flex-col animate-pulse">
                  <div className="flex-1 min-h-[200px] bg-neutral-800/50" />
                  <div className="bg-neutral-900 border-t border-neutral-800 px-3 py-2">
                    <div className="flex gap-2">
                      <div className="h-4 w-20 bg-neutral-800 rounded" />
                      <div className="h-4 w-16 bg-neutral-800 rounded" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : cameraError ? (
            <div className="h-full min-h-[360px] flex items-center justify-center">
              <div className="max-w-md rounded-[var(--radius-lg)] border border-[var(--color-border-default)] bg-[var(--color-bg-primary)] p-6 text-center shadow-sm">
                <WifiOff className="mx-auto mb-3 h-7 w-7 text-[var(--color-critical)]" />
                <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Live view could not load cameras</h2>
                <p className="mt-2 text-sm text-[var(--color-text-secondary)]">{cameraError}</p>
                <button
                  type="button"
                  onClick={() => {
                    setCamerasLoading(true)
                    setCameraError(null)
                    fetchCameras()
                      .then((data) => setCameras(data))
                      .catch((err) => setCameraError(err instanceof Error ? err.message : "Could not load cameras"))
                      .finally(() => setCamerasLoading(false))
                  }}
                  className="mt-4 inline-flex items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-text-primary)] px-3 py-2 text-sm font-medium text-[var(--color-bg-primary)]"
                >
                  Retry
                </button>
              </div>
            </div>
          ) : cameras.length === 0 ? (
            <div className="h-full min-h-[360px] flex items-center justify-center">
              <div className="max-w-md rounded-[var(--radius-lg)] border border-[var(--color-border-default)] bg-[var(--color-bg-primary)] p-6 text-center shadow-sm">
                <h2 className="text-base font-semibold text-[var(--color-text-primary)]">No cameras configured</h2>
                <p className="mt-2 text-sm text-[var(--color-text-secondary)]">Add or enable a camera to show live streams here.</p>
              </div>
            </div>
          ) : (
            <div
              className="grid gap-3 content-start"
              style={{ gridTemplateColumns: `repeat(${focusedCamId ? 1 : gridCols}, minmax(0, 1fr))` }}
            >
              {displayedCameras.map((cam) => (
                (() => {
                  const live = isCameraLive(cam)
                  return (
                <div
                  key={cam.id}
                  className={cn(
                    "group relative rounded-[var(--radius-lg)] overflow-hidden border border-neutral-800 bg-neutral-900 flex flex-col",
                    !focusedCamId && "cursor-pointer hover:border-neutral-600 transition-colors"
                  )}
                  onClick={() => !focusedCamId && focusCamera(cam.id)}
                >
                  {/* MJPEG stream */}
                  <div className="relative flex-1 min-h-0 aspect-video">
                    {live ? (
                      <img
                        src={`${STREAM_BASE}/api/stream/${cam.id}`}
                        alt={cam.name}
                        className="w-full h-full object-contain bg-black"
                      />
                    ) : (
                      <div className="flex h-full w-full items-center justify-center bg-black text-xs text-neutral-400">
                        {statusLabel(cam)}
                      </div>
                    )}

                    {/* Camera name overlay */}
                    <div className="absolute top-0 left-0 right-0 p-2.5 space-y-1">
                      <div className="inline-flex items-center gap-2 bg-black/70 backdrop-blur-sm rounded-md px-2.5 py-1">
                        <span className="relative flex h-2 w-2">
                          {live && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />}
                          <span className={cn(
                            "relative inline-flex h-2 w-2 rounded-full",
                            live ? "bg-emerald-500" : "bg-neutral-500"
                          )} />
                        </span>
                        <span className="text-white text-xs font-medium">{cam.name}</span>
                      </div>
                      <div className="flex gap-1.5">
                        <Badge variant="default" className="bg-black/50 text-white/90 text-[10px] border-0">
                          {cam.zone}
                        </Badge>
                      </div>
                    </div>

                    {/* Maximize hint on hover */}
                    {!focusedCamId && (
                      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <div className="bg-black/60 backdrop-blur-sm rounded-md p-1.5">
                          <Maximize2 size={14} className="text-white" />
                        </div>
                      </div>
                    )}

                    {/* Active alert count */}
                    {activeAlerts.filter((a) => a.cameraId === cam.id).length > 0 && (
                      <div className="absolute bottom-2 right-2">
                        <span className="inline-flex items-center gap-1 bg-red-600 text-white text-[10px] font-bold rounded-full px-2 py-0.5 animate-pulse">
                          <AlertTriangle size={10} />
                          {activeAlerts.filter((a) => a.cameraId === cam.id).length} alert(s)
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Rules bar */}
                  <div className="bg-neutral-900 border-t border-neutral-800 px-3 py-2">
                    <div className="flex flex-wrap gap-1">
                      {cam.rules.map((rule) => (
                        <span key={rule} className="text-[10px] text-neutral-400 bg-neutral-800 rounded px-1.5 py-0.5">
                          {rule}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
                  )
                })()
              ))}
            </div>
          )}
        </div>

        {/* VLM result bar */}
        {vlmResult && vlmResult.text && (
          <div className="border-t bg-purple-50 px-4 py-3">
            <div className="flex items-start gap-2">
              <Brain size={16} className="text-purple-600 mt-0.5 shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-semibold text-purple-700">VLM Scene Analysis</span>
                  <span className="text-[10px] text-purple-500">{vlmResult.elapsed}s inference</span>
                </div>
                <p className="text-xs text-purple-900 leading-relaxed">{vlmResult.text}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {showOccupancyPanel && occupancyReport && (
        <aside className="hidden xl:flex w-[380px] shrink-0 flex-col border-l border-[var(--color-border-default)] bg-[var(--color-bg-primary)]">
          <div className="border-b border-[var(--color-border-default)] px-4 py-3">
            <div className="flex items-center gap-2">
              <FileText size={17} className="text-[var(--color-accent)]" />
              <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">Chair Occupancy Snapshot</h2>
            </div>
            <p className="mt-1 text-xs text-[var(--color-text-secondary)]">
              Work hours {occupancyReport.workHours.start}-{occupancyReport.workHours.end}; lunch excluded.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 border-b border-[var(--color-border-default)] p-4">
            <div className="rounded-[var(--radius-md)] border border-[var(--color-border-default)] p-3">
              <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
                <Armchair size={14} />
                Occupied chairs
              </div>
              <div className="mt-2 text-2xl font-semibold text-[var(--color-text-primary)]">
                {occupiedChairCount}/{occupancyChairs.length}
              </div>
            </div>
            <div className="rounded-[var(--radius-md)] border border-[var(--color-border-default)] p-3">
              <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
                <Clock size={14} />
                Empty chairs
              </div>
              <div className="mt-2 text-2xl font-semibold text-[var(--color-text-primary)]">
                {emptyChairCount}
              </div>
            </div>
          </div>

          <div className="border-b border-[var(--color-border-default)] px-4 py-3">
            <div className="rounded-[var(--radius-md)] bg-amber-50 px-3 py-2 text-xs text-amber-800">
              Current chair status is live. Day-end utilization should be generated from persisted work-hour samples, not from this short live snapshot.
            </div>
          </div>

          <div className="flex-1 overflow-auto p-4">
            <div className="space-y-2">
              {occupancyReport.chairs.map((chair) => (
                <div
                  key={chair.id}
                  className="rounded-[var(--radius-md)] border border-[var(--color-border-default)] p-3"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-sm font-medium text-[var(--color-text-primary)]">{chair.name}</div>
                      <div className="mt-1 text-xs text-[var(--color-text-secondary)]">
                        {chair.occupied ? "Person detected in chair zone" : `Empty for ${formatDuration(chair.emptyForSeconds)}`}
                      </div>
                    </div>
                    <span
                      className={cn(
                        "rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
                        chair.occupied
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-amber-100 text-amber-700"
                      )}
                    >
                      {chair.status}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center justify-between text-xs text-[var(--color-text-secondary)]">
                    <span>sample {formatDuration(chair.sampleSeconds)}</span>
                    <span>current snapshot</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="border-t border-[var(--color-border-default)] p-4">
            <div className="rounded-[var(--radius-md)] bg-[var(--color-bg-secondary)] p-3 text-xs text-[var(--color-text-secondary)]">
              End-of-day export would summarize occupied time, empty time, and long absence events for each marked chair after work-hour samples are persisted.
            </div>
          </div>
        </aside>
      )}

    </div>
  )
}
