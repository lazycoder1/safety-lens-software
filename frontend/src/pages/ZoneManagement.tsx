import { useState, useEffect, useRef, useCallback } from "react"
import { Plus, Trash2, X, Shield, AlertTriangle, HardHat } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getCameras, getZones, addZone, deleteZone, API_BASE } from "@/lib/api"

interface Camera {
  id: string
  name: string
  zone: string
  status: string
}

interface Zone {
  id: string
  name: string
  type: string
  color: string
  points: number[][]
}

const ZONE_TYPES = [
  { key: "restricted", label: "Restricted Area", color: "#dc2626", icon: Shield, desc: "No entry — triggers P1 alert" },
  { key: "caution", label: "Caution Zone", color: "#f59e0b", icon: AlertTriangle, desc: "Extra care needed — triggers P2 alert" },
  { key: "ppe_required", label: "PPE Required", color: "#2563eb", icon: HardHat, desc: "Must wear PPE in this zone" },
]

export function ZoneManagement() {
  const [cameras, setCameras] = useState<Camera[]>([])
  const [selectedCamId, setSelectedCamId] = useState<string>("")
  const [zones, setZones] = useState<Zone[]>([])
  const [drawingMode, setDrawingMode] = useState(false)
  const [currentPoints, setCurrentPoints] = useState<number[][]>([])
  const [newZoneName, setNewZoneName] = useState("")
  const [newZoneType, setNewZoneType] = useState("restricted")
  const [saving, setSaving] = useState(false)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)

  // Load cameras
  useEffect(() => {
    getCameras().then((cams) => {
      const online = cams.filter((c: any) => c.status === "online")
      setCameras(online)
      if (online.length > 0 && !selectedCamId) {
        setSelectedCamId(online[0].id)
      }
    }).catch(() => {})
  }, [])

  // Load zones when camera changes
  useEffect(() => {
    if (selectedCamId) {
      getZones(selectedCamId).then(setZones).catch(() => setZones([]))
    }
  }, [selectedCamId])

  // Draw zones on canvas
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return

    const rect = img.getBoundingClientRect()
    canvas.width = rect.width
    canvas.height = rect.height

    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw saved zones
    for (const zone of zones) {
      if (zone.points.length < 3) continue
      ctx.beginPath()
      ctx.moveTo(zone.points[0][0] * canvas.width, zone.points[0][1] * canvas.height)
      for (let i = 1; i < zone.points.length; i++) {
        ctx.lineTo(zone.points[i][0] * canvas.width, zone.points[i][1] * canvas.height)
      }
      ctx.closePath()
      ctx.fillStyle = zone.color + "20"
      ctx.fill()
      ctx.strokeStyle = zone.color
      ctx.lineWidth = 2
      ctx.stroke()

      // Label
      const cx = zone.points.reduce((s, p) => s + p[0], 0) / zone.points.length * canvas.width
      const cy = zone.points.reduce((s, p) => s + p[1], 0) / zone.points.length * canvas.height
      ctx.fillStyle = zone.color
      ctx.font = "bold 12px Inter, sans-serif"
      ctx.textAlign = "center"
      ctx.fillText(zone.name, cx, cy)
    }

    // Draw current drawing
    if (currentPoints.length > 0) {
      const typeConfig = ZONE_TYPES.find((t) => t.key === newZoneType)
      const color = typeConfig?.color || "#dc2626"

      ctx.beginPath()
      ctx.moveTo(currentPoints[0][0] * canvas.width, currentPoints[0][1] * canvas.height)
      for (let i = 1; i < currentPoints.length; i++) {
        ctx.lineTo(currentPoints[i][0] * canvas.width, currentPoints[i][1] * canvas.height)
      }
      if (currentPoints.length >= 3) {
        ctx.closePath()
        ctx.fillStyle = color + "15"
        ctx.fill()
      }
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.setLineDash([6, 3])
      ctx.stroke()
      ctx.setLineDash([])

      // Draw vertices
      for (const pt of currentPoints) {
        ctx.beginPath()
        ctx.arc(pt[0] * canvas.width, pt[1] * canvas.height, 5, 0, Math.PI * 2)
        ctx.fillStyle = color
        ctx.fill()
        ctx.strokeStyle = "white"
        ctx.lineWidth = 2
        ctx.stroke()
      }
    }
  }, [zones, currentPoints, newZoneType])

  useEffect(() => {
    drawCanvas()
    const interval = setInterval(drawCanvas, 200)
    return () => clearInterval(interval)
  }, [drawCanvas])

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!drawingMode) return
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height

    setCurrentPoints((prev) => [...prev, [x, y]])
  }

  async function handleSaveZone() {
    if (!selectedCamId || currentPoints.length < 3 || !newZoneName.trim()) return
    setSaving(true)
    try {
      const typeConfig = ZONE_TYPES.find((t) => t.key === newZoneType)
      await addZone(selectedCamId, {
        name: newZoneName.trim(),
        type: newZoneType,
        color: typeConfig?.color || "#dc2626",
        points: currentPoints,
      })
      const updated = await getZones(selectedCamId)
      setZones(updated)
      setCurrentPoints([])
      setNewZoneName("")
      setDrawingMode(false)
    } catch {
      // ignore
    } finally {
      setSaving(false)
    }
  }

  async function handleDeleteZone(zoneId: string) {
    if (!selectedCamId) return
    try {
      await deleteZone(selectedCamId, zoneId)
      setZones((prev) => prev.filter((z) => z.id !== zoneId))
    } catch {
      // ignore
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Zone Management</h1>
          <p className="text-sm text-[var(--color-text-tertiary)] mt-1">
            Draw restricted areas on camera feeds to detect intrusions
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Camera selector + zone list */}
        <div className="space-y-4">
          <Card>
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">Select Camera</h3>
            <div className="space-y-1.5">
              {cameras.map((cam) => (
                <button
                  key={cam.id}
                  onClick={() => {
                    setSelectedCamId(cam.id)
                    setDrawingMode(false)
                    setCurrentPoints([])
                  }}
                  className={`w-full text-left px-3 py-2 rounded-[var(--radius-md)] text-sm transition-colors cursor-pointer ${
                    selectedCamId === cam.id
                      ? "bg-[var(--color-text-primary)] text-white"
                      : "hover:bg-[var(--color-bg-tertiary)] text-[var(--color-text-primary)]"
                  }`}
                >
                  <span className="font-medium">{cam.name}</span>
                  <span className={`text-xs ml-2 ${selectedCamId === cam.id ? "text-white/70" : "text-[var(--color-text-tertiary)]"}`}>
                    {cam.zone}
                  </span>
                </button>
              ))}
              {cameras.length === 0 && (
                <p className="text-xs text-[var(--color-text-tertiary)] py-4 text-center">No cameras online</p>
              )}
            </div>
          </Card>

          <Card>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">Zones</h3>
              <span className="text-xs text-[var(--color-text-tertiary)]">{zones.length} zone{zones.length !== 1 ? "s" : ""}</span>
            </div>
            <div className="space-y-2">
              {zones.map((zone) => (
                <div
                  key={zone.id}
                  className="flex items-center justify-between p-2.5 rounded-[var(--radius-md)] border"
                >
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full" style={{ backgroundColor: zone.color }} />
                    <div>
                      <span className="text-sm font-medium text-[var(--color-text-primary)]">{zone.name}</span>
                      <span className="text-xs text-[var(--color-text-tertiary)] block">{zone.type} ({zone.points.length} pts)</span>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteZone(zone.id)}
                    className="p-1 rounded hover:bg-[var(--color-critical-bg)] text-[var(--color-text-tertiary)] hover:text-[var(--color-critical)] cursor-pointer transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
              {zones.length === 0 && (
                <p className="text-xs text-[var(--color-text-tertiary)] py-3 text-center">
                  No zones drawn. Click "Draw Zone" to start.
                </p>
              )}
            </div>
          </Card>
        </div>

        {/* Right: Camera feed + canvas overlay */}
        <div className="lg:col-span-2 space-y-4">
          {selectedCamId ? (
            <>
              <Card className="p-0 overflow-hidden">
                <div className="relative bg-black">
                  <img
                    ref={imgRef}
                    src={`${API_BASE}/api/stream/${selectedCamId}`}
                    alt="Camera feed"
                    className="w-full aspect-video object-contain"
                    onLoad={drawCanvas}
                  />
                  <canvas
                    ref={canvasRef}
                    onClick={handleCanvasClick}
                    className={`absolute inset-0 w-full h-full ${
                      drawingMode ? "cursor-crosshair" : "pointer-events-none"
                    }`}
                  />
                  {drawingMode && (
                    <div className="absolute top-3 left-3 bg-black/70 text-white text-xs px-3 py-1.5 rounded-full">
                      Click to add points ({currentPoints.length} placed) — need at least 3
                    </div>
                  )}
                </div>
              </Card>

              {/* Drawing controls */}
              {drawingMode ? (
                <Card>
                  <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">New Zone</h3>
                  <div className="space-y-3">
                    <div className="space-y-1.5">
                      <label className="text-sm font-medium text-[var(--color-text-primary)]">Zone Name</label>
                      <input
                        type="text"
                        value={newZoneName}
                        onChange={(e) => setNewZoneName(e.target.value)}
                        placeholder="e.g. Motor Bay - Restricted"
                        className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
                      />
                    </div>

                    <div className="space-y-1.5">
                      <label className="text-sm font-medium text-[var(--color-text-primary)]">Zone Type</label>
                      <div className="grid grid-cols-3 gap-2">
                        {ZONE_TYPES.map((zt) => {
                          const Icon = zt.icon
                          return (
                            <button
                              key={zt.key}
                              onClick={() => setNewZoneType(zt.key)}
                              className={`flex flex-col items-center gap-1.5 p-3 rounded-[var(--radius-md)] border text-xs cursor-pointer transition-colors ${
                                newZoneType === zt.key
                                  ? "border-2"
                                  : "border-[var(--color-border)] hover:border-[var(--color-border-strong)]"
                              }`}
                              style={newZoneType === zt.key ? { borderColor: zt.color, backgroundColor: zt.color + "10" } : {}}
                            >
                              <Icon className="w-5 h-5" style={{ color: zt.color }} />
                              <span className="font-medium text-[var(--color-text-primary)]">{zt.label}</span>
                            </button>
                          )
                        })}
                      </div>
                    </div>

                    <div className="flex items-center gap-2 pt-2">
                      <Button
                        onClick={handleSaveZone}
                        disabled={currentPoints.length < 3 || !newZoneName.trim() || saving}
                      >
                        {saving ? "Saving..." : `Save Zone (${currentPoints.length} points)`}
                      </Button>
                      <Button
                        variant="secondary"
                        onClick={() => setCurrentPoints((p) => p.slice(0, -1))}
                        disabled={currentPoints.length === 0}
                      >
                        Undo Point
                      </Button>
                      <Button
                        variant="secondary"
                        onClick={() => {
                          setDrawingMode(false)
                          setCurrentPoints([])
                          setNewZoneName("")
                        }}
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                </Card>
              ) : (
                <Button onClick={() => setDrawingMode(true)}>
                  <Plus className="w-4 h-4" />
                  Draw Zone
                </Button>
              )}
            </>
          ) : (
            <Card className="flex items-center justify-center h-64">
              <p className="text-sm text-[var(--color-text-tertiary)]">Select a camera to manage zones</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
