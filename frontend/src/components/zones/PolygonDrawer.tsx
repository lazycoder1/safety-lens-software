import { useState, useEffect, useRef, useCallback } from "react"
import { Plus, Shield, AlertTriangle, HardHat } from "lucide-react"
import type { LucideIcon } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import type { Zone } from "@/types"

export interface ZoneType {
  key: string
  label: string
  color: string
  icon: LucideIcon
  desc: string
}

export const ZONE_TYPES: ZoneType[] = [
  { key: "restricted", label: "Restricted Area", color: "#dc2626", icon: Shield, desc: "No entry — triggers P1 alert" },
  { key: "caution", label: "Caution Zone", color: "#f59e0b", icon: AlertTriangle, desc: "Extra care needed — triggers P2 alert" },
  { key: "ppe_required", label: "PPE Required", color: "#2563eb", icon: HardHat, desc: "Must wear PPE in this zone" },
]

export interface PolygonDrawerProps {
  imageUrl: string
  existingZones?: Zone[]
  onSave: (zone: { name: string; type: string; color: string; points: number[][] }) => Promise<void>
}

export function PolygonDrawer({ imageUrl, existingZones = [], onSave }: PolygonDrawerProps) {
  const [drawingMode, setDrawingMode] = useState(false)
  const [currentPoints, setCurrentPoints] = useState<number[][]>([])
  const [newZoneName, setNewZoneName] = useState("")
  const [newZoneType, setNewZoneType] = useState("restricted")
  const [saving, setSaving] = useState(false)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)

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
    for (const zone of existingZones) {
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
  }, [existingZones, currentPoints, newZoneType])

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

  async function handleSave() {
    if (currentPoints.length < 3 || !newZoneName.trim()) return
    setSaving(true)
    try {
      const typeConfig = ZONE_TYPES.find((t) => t.key === newZoneType)
      await onSave({
        name: newZoneName.trim(),
        type: newZoneType,
        color: typeConfig?.color || "#dc2626",
        points: currentPoints,
      })
      setCurrentPoints([])
      setNewZoneName("")
      setDrawingMode(false)
    } catch {
      // ignore — parent surfaces errors
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card className="p-0 overflow-hidden">
        <div className="relative bg-black">
          <img
            ref={imgRef}
            src={imageUrl}
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
                      type="button"
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
                onClick={handleSave}
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
    </div>
  )
}
