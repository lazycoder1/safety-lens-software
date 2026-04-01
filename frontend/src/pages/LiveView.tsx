import { useState, useEffect, useRef, useCallback } from "react"
import { useSearchParams } from "react-router-dom"
import {
  Grid2X2,
  Grid3X3,
  Maximize2,
  Minimize2,
  PanelRightOpen,
  PanelRightClose,
  CheckCircle2,
  AlertTriangle,
  Activity,
  Brain,
  Cpu,
  WifiOff,
  Clock,
} from "lucide-react"
import { severityConfig } from "@/data/mock"
import type { Severity } from "@/data/mock"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"
import { API_BASE, WS_BASE } from "@/lib/api"
import { playP1AlertSound } from "@/lib/alertSound"

interface LiveAlert {
  id: string
  severity: Severity
  status: string
  rule: string
  cameraId: string
  cameraName: string
  zone: string
  confidence: number
  timestamp: string
  source: string
  description: string
}

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

const severityVariantMap: Record<Severity, "critical" | "high" | "warning" | "info"> = {
  P1: "critical",
  P2: "high",
  P3: "warning",
  P4: "info",
}

function timeAgo(isoString: string): string {
  const seconds = Math.floor((Date.now() - new Date(isoString).getTime()) / 1000)
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  return `${Math.floor(minutes / 60)}h ago`
}

export function LiveView() {
  const [searchParams, setSearchParams] = useSearchParams()
  const focusedCamId = searchParams.get("cam")
  const [gridCols, setGridCols] = useState(2)
  const [panelOpen, setPanelOpen] = useState(true)
  const [cameras, setCameras] = useState<CameraInfo[]>([])
  const [liveAlerts, setLiveAlerts] = useState<LiveAlert[]>([])
  const [vlmResult, setVlmResult] = useState<{ text: string; timestamp: string; elapsed: number } | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(true)
  const [clock, setClock] = useState(() => new Date().toLocaleTimeString("en-IN", { hour12: false }))

  const focusCamera = (camId: string) => setSearchParams({ cam: camId })
  const unfocusCamera = () => setSearchParams({})
  const displayedCameras = focusedCamId ? cameras.filter((c) => c.id === focusedCamId) : cameras

  // Clock tick
  useEffect(() => {
    const id = setInterval(() => setClock(new Date().toLocaleTimeString("en-IN", { hour12: false })), 1000)
    return () => clearInterval(id)
  }, [])

  // Fetch cameras on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/cameras`)
      .then((r) => r.json())
      .then((data) => { setCameras(data); setConnected(true) })
      .catch(() => { setCameras([]); setConnected(false) })
  }, [])

  // Poll VLM result (backend returns { cam_id: { text, timestamp, elapsed } })
  useEffect(() => {
    const interval = setInterval(() => {
      fetch(`${API_BASE}/api/vlm/latest`)
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

  // WebSocket for alerts
  useEffect(() => {
    let retryDelay = 2000
    let cancelled = false

    function connect() {
      if (cancelled) return
      const ws = new WebSocket(`${WS_BASE}/ws/alerts`)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        retryDelay = 2000
      }

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)
        if (msg.type === "alert") {
          const alert = msg.data as LiveAlert
          setLiveAlerts((prev) => [alert, ...prev].slice(0, 100))

          // Fire toast
          const sev = severityConfig[alert.severity as Severity]
          toast.custom(
            () => (
              <div className="flex gap-0 bg-white rounded-lg shadow-lg border overflow-hidden max-w-sm">
                <div className="w-1.5 shrink-0" style={{ backgroundColor: sev?.color || "#666" }} />
                <div className="px-3 py-2.5 flex-1">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <span className="text-xs font-semibold text-[var(--color-text-primary)]">{alert.rule}</span>
                    <Badge variant={severityVariantMap[alert.severity as Severity] || "default"}>
                      {alert.severity}
                    </Badge>
                  </div>
                  <p className="text-[11px] text-[var(--color-text-secondary)]">{alert.cameraName}</p>
                  {alert.description && (
                    <p className="text-[10px] text-[var(--color-text-tertiary)] mt-1 line-clamp-2">{alert.description}</p>
                  )}
                  <div className="flex items-center gap-2 mt-1.5">
                    <span className="text-[10px] text-[var(--color-text-tertiary)]">
                      {alert.source} | {Math.round(alert.confidence * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            ),
            { duration: alert.severity === "P1" ? 30000 : 8000 }
          )

          if (alert.severity === "P1") {
            playP1AlertSound()
          }
        }
      }

      ws.onclose = () => {
        setConnected(false)
        if (!cancelled) {
          setTimeout(connect, retryDelay)
          retryDelay = Math.min(retryDelay * 1.5, 15000)
        }
      }

      ws.onerror = () => {
        ws.close()
      }
    }

    connect()
    return () => {
      cancelled = true
      wsRef.current?.close()
    }
  }, [])

  const acknowledge = useCallback((alertId: string) => {
    setLiveAlerts((prev) => prev.map((a) => (a.id === alertId ? { ...a, status: "acknowledged" } : a)))
    wsRef.current?.send(JSON.stringify({ type: "acknowledge", alertId }))
    toast.success("Alert acknowledged")
  }, [])

  const activeAlerts = liveAlerts.filter((a) => a.status === "active")

  return (
    <div className="flex h-full overflow-hidden">
      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Toolbar */}
        <div className="flex items-center justify-between border-b px-4 py-2.5 bg-white">
          <div className="flex items-center gap-1.5">
            {focusedCamId ? (
              <button
                onClick={unfocusCamera}
                className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-[var(--radius-md)] text-sm font-medium text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] transition-colors cursor-pointer"
              >
                <Minimize2 size={16} />
                Back to Grid
              </button>
            ) : (
              <>
                <span className="text-sm font-medium text-[var(--color-text-secondary)] mr-2">Grid</span>
                {[
                  { cols: 1, label: "1x1", icon: Maximize2 },
                  { cols: 2, label: "2x2", icon: Grid2X2 },
                  { cols: 3, label: "3x3", icon: Grid3X3 },
                ].map((opt) => {
                  const Icon = opt.icon
                  return (
                    <button
                      key={opt.cols}
                      onClick={() => setGridCols(opt.cols)}
                      className={cn(
                        "inline-flex items-center justify-center w-8 h-8 rounded-[var(--radius-md)] transition-colors cursor-pointer",
                        gridCols === opt.cols
                          ? "bg-[var(--color-text-primary)] text-white"
                          : "text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)]"
                      )}
                      title={opt.label}
                    >
                      <Icon size={16} />
                    </button>
                  )
                })}
              </>
            )}
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-xs font-mono text-[var(--color-text-secondary)]">
              <Clock size={14} />
              <span>{clock}</span>
            </div>
            <div className="w-px h-4 bg-[var(--color-border-default)]" />
            <div className="flex items-center gap-1.5 text-xs text-[var(--color-text-secondary)]">
              <Activity size={14} className={connected ? "text-[var(--color-success)]" : "text-[var(--color-critical)]"} />
              <span>{cameras.length} cameras</span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setPanelOpen(!panelOpen)}
              title={panelOpen ? "Hide alerts panel" : "Show alerts panel"}
            >
              {panelOpen ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
            </Button>
          </div>
        </div>

        {/* Connection lost banner */}
        {!connected && (
          <div className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white text-xs font-medium">
            <WifiOff size={14} />
            Connection to backend lost. Retrying...
          </div>
        )}

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

      {/* Right panel - Live Alerts */}
      {panelOpen && (
        <div className="w-80 border-l bg-white flex flex-col shrink-0">
          <div className="flex items-center justify-between px-4 py-3 border-b">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">Live Alerts</h2>
              {activeAlerts.length > 0 && (
                <span className="inline-flex items-center justify-center min-w-[20px] h-5 rounded-full bg-red-600 text-white text-[10px] font-bold px-1.5">
                  {activeAlerts.length}
                </span>
              )}
            </div>
            <span className="text-[10px] text-[var(--color-text-tertiary)]">
              {liveAlerts.length} total
            </span>
          </div>

          <div className="flex-1 overflow-y-auto">
            {activeAlerts.length === 0 && liveAlerts.length === 0 && (
              <div className="flex flex-col items-center justify-center py-12 text-[var(--color-text-secondary)]">
                <CheckCircle2 size={32} className="mb-2 opacity-40" />
                <span className="text-sm">No alerts yet</span>
                <span className="text-xs text-[var(--color-text-tertiary)] mt-1">Waiting for detections...</span>
              </div>
            )}

            {liveAlerts.map((alert) => {
              const sev = severityConfig[alert.severity as Severity]
              const isActive = alert.status === "active"
              return (
                <div
                  key={alert.id}
                  className={cn(
                    "border-b last:border-b-0 transition-colors",
                    isActive ? "bg-white hover:bg-[var(--color-bg-secondary)]" : "bg-[var(--color-bg-secondary)] opacity-60"
                  )}
                >
                  <div className="flex gap-0">
                    <div className="w-1 shrink-0" style={{ backgroundColor: sev?.color || "#999" }} />
                    <div className="flex-1 px-3 py-2.5">
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <div className="min-w-0">
                          <p className="text-xs font-semibold text-[var(--color-text-primary)] truncate">
                            {alert.rule}
                          </p>
                          <p className="text-[11px] text-[var(--color-text-secondary)] truncate">
                            {alert.cameraName}
                          </p>
                        </div>
                        <div className="flex items-center gap-1 shrink-0">
                          {alert.source.includes("VLM") && (
                            <Brain size={12} className="text-purple-500" />
                          )}
                          <Badge variant={severityVariantMap[alert.severity as Severity] || "default"}>
                            {alert.severity}
                          </Badge>
                        </div>
                      </div>

                      {alert.description && (
                        <p className="text-[10px] text-[var(--color-text-tertiary)] line-clamp-2 mb-1.5">
                          {alert.description}
                        </p>
                      )}

                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-[10px] text-[var(--color-text-tertiary)]">
                          <span>{timeAgo(alert.timestamp)}</span>
                          <span>|</span>
                          <span>{alert.source}</span>
                          <span>|</span>
                          <span>{Math.round(alert.confidence * 100)}%</span>
                        </div>
                        {isActive && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-[10px] h-5 px-1.5"
                            onClick={() => acknowledge(alert.id)}
                          >
                            Ack
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
