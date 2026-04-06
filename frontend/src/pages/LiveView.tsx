import { useState, useEffect } from "react"
import { useSearchParams } from "react-router-dom"
import {
  Maximize2,
  AlertTriangle,
  Brain,
  Cpu,
  WifiOff,
  Grid2x2,
  Grid3x3,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { API_BASE, getCameras as fetchCameras, getToken } from "@/lib/api"
import { useAlertStore } from "@/stores/alertStore"
import { useAlertConnection } from "@/components/AlertProvider"

interface CameraInfo {
  id: string
  name: string
  zone: string
  demo: string
  rules: string[]
  status: string
  detectionsCount: number
  yoloe_classes?: string[]
}

export function LiveView() {
  const [searchParams, setSearchParams] = useSearchParams()
  const focusedCamId = searchParams.get("cam")
  const [gridCols, setGridCols] = useState(2)
  const [cameras, setCameras] = useState<CameraInfo[]>([])
  const [vlmResult, setVlmResult] = useState<{ text: string; timestamp: string; elapsed: number } | null>(null)
  const connected = useAlertConnection((s) => s.connected)
  const alerts = useAlertStore((s) => s.alerts)

  const focusCamera = (camId: string) => setSearchParams({ cam: camId })
  const unfocusCamera = () => setSearchParams({})
  const displayedCameras = focusedCamId ? cameras.filter((c) => c.id === focusedCamId) : cameras

  // Fetch cameras on mount
  useEffect(() => {
    fetchCameras()
      .then((data) => setCameras(data))
      .catch(() => setCameras([]))
  }, [])

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
        </div>

        {/* Camera grid */}
        <div className="flex-1 overflow-auto p-4 bg-[var(--color-bg-secondary)]">
          {cameras.length === 0 ? (
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
          ) : (
            <div
              className="grid gap-3 h-full"
              style={{ gridTemplateColumns: `repeat(${focusedCamId ? 1 : gridCols}, minmax(0, 1fr))` }}
            >
              {displayedCameras.map((cam) => (
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
                    <img
                      src={`${API_BASE}/api/stream/${cam.id}`}
                      alt={cam.name}
                      className="w-full h-full object-contain bg-black"
                    />

                    {/* Camera name overlay */}
                    <div className="absolute top-0 left-0 right-0 p-2.5 space-y-1">
                      <div className="inline-flex items-center gap-2 bg-black/70 backdrop-blur-sm rounded-md px-2.5 py-1">
                        <span className="relative flex h-2 w-2">
                          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                          <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
                        </span>
                        <span className="text-white text-xs font-medium">{cam.name}</span>
                      </div>
                      <div className="flex gap-1.5">
                        <Badge variant="default" className="bg-black/50 text-white/90 text-[10px] border-0">
                          {cam.zone}
                        </Badge>
                        <Badge
                          variant={cam.demo === "yolo+vlm" ? "warning" : cam.demo === "yoloe" ? "info" : "success"}
                          className="text-[10px]"
                        >
                          {cam.demo === "yolo+vlm" ? (
                            <span className="flex items-center gap-1"><Brain size={10} /> YOLO + VLM</span>
                          ) : cam.demo === "yoloe" ? (
                            <span className="flex items-center gap-1"><Cpu size={10} /> YOLOe</span>
                          ) : (
                            <span className="flex items-center gap-1"><Cpu size={10} /> YOLO</span>
                          )}
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
                    {focusedCamId && cam.demo === "yoloe" && cam.yoloe_classes && cam.yoloe_classes.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1.5 pt-1.5 border-t border-neutral-800">
                        <span className="text-[10px] text-neutral-500 mr-1">YOLOe classes:</span>
                        {cam.yoloe_classes.map((cls) => (
                          <span key={cls} className="text-[10px] text-emerald-400 bg-emerald-950 rounded px-1.5 py-0.5">
                            {cls}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
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

    </div>
  )
}
