import { useState, useEffect, useRef, useCallback } from "react"
import { Plus, Shield, AlertTriangle, HardHat, Monitor, Pencil, Trash2, X } from "lucide-react"
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
  { key: "workstation", label: "Chair", color: "#059669", icon: Monitor, desc: "Chair/seat zone for occupancy tracking" },
  { key: "gangway", label: "Gangway", color: "#7c3aed", icon: Monitor, desc: "Pedestrian or material movement path" },
  { key: "motor_risk_zone", label: "Motor Risk", color: "#be123c", icon: AlertTriangle, desc: "Machine or motor danger area" },
  { key: "queue", label: "Queue", color: "#0891b2", icon: Monitor, desc: "Queue density and dwell monitoring" },
  { key: "object_watch", label: "Object Watch", color: "#0f766e", icon: Monitor, desc: "Watched-object dwell and removal monitoring" },
  { key: "loading_bay", label: "Loading Bay", color: "#9333ea", icon: Monitor, desc: "Truck or loading operations area" },
  { key: "forklift_lane", label: "Forklift Lane", color: "#ea580c", icon: AlertTriangle, desc: "Forklift movement lane" },
  { key: "exclusion", label: "Exclusion Zone", color: "#4b5563", icon: Shield, desc: "Ignore or suppress detections in this area" },
]

export interface PolygonDrawerProps {
  imageUrl: string
  existingZones?: Zone[]
  defaultZoneType?: string
  onSave: (zone: Omit<Zone, "id">) => Promise<void>
  onDelete?: (zoneId: string) => Promise<void>
  onUpdate?: (zoneId: string, updates: Partial<Zone>) => Promise<void>
  onDraftStateChange?: (hasDraft: boolean) => void
}

function getDefaultZoneName(zoneType: string, zoneCount: number): string {
  const zoneTypeLabel =
    ZONE_TYPES.find((item) => item.key === zoneType)?.label.replace(" Area", "") ||
    "Zone"
  return `${zoneTypeLabel} ${zoneCount}`
}

function defaultAnalyticsForZone(zoneType: string): string {
  const mapping: Record<string, string> = {
    restricted: "intrusion",
    caution: "intrusion",
    ppe_required: "ppe",
    workstation: "occupancy",
    queue: "queue",
    object_watch: "object_lifecycle",
    gangway: "obstruction",
    loading_bay: "vehicle_presence",
    forklift_lane: "vehicle_presence",
    motor_risk_zone: "intrusion",
    exclusion: "exclusion",
  }
  return mapping[zoneType] || "intrusion"
}

/* ── geometry helpers ── */

function isPointInPolygon(px: number, py: number, polygon: number[][]): boolean {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1]
    const xj = polygon[j][0], yj = polygon[j][1]
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside
    }
  }
  return inside
}

function distPx(
  ax: number, ay: number, bx: number, by: number,
  canvasW: number, canvasH: number,
): number {
  const dx = (ax - bx) * canvasW
  const dy = (ay - by) * canvasH
  return Math.sqrt(dx * dx + dy * dy)
}

/* ── component ── */

export function PolygonDrawer({
  imageUrl,
  existingZones = [],
  defaultZoneType = "restricted",
  onSave,
  onDelete,
  onUpdate,
  onDraftStateChange,
}: PolygonDrawerProps) {
  // drawing state
  const [drawingMode, setDrawingMode] = useState(false)
  const [currentPoints, setCurrentPoints] = useState<number[][]>([])
  const [newZoneName, setNewZoneName] = useState("")
  const [newZoneType, setNewZoneType] = useState(defaultZoneType)
  const [newZoneAnalytics, setNewZoneAnalytics] = useState(defaultAnalyticsForZone(defaultZoneType))
  const [newZoneClasses, setNewZoneClasses] = useState("")
  const [newZoneThreshold, setNewZoneThreshold] = useState("")
  const [newZoneMinDuration, setNewZoneMinDuration] = useState("")
  const [newZoneArea, setNewZoneArea] = useState("")
  const [newZoneP1Threshold, setNewZoneP1Threshold] = useState("")
  const [newZoneP2Threshold, setNewZoneP2Threshold] = useState("")
  const [saving, setSaving] = useState(false)

  // selection & editing
  const [selectedZoneId, setSelectedZoneId] = useState<string | null>(null)
  const [hoveredZoneId, setHoveredZoneId] = useState<string | null>(null)
  const [editingZoneId, setEditingZoneId] = useState<string | null>(null)

  // vertex dragging
  const [draggingVertex, setDraggingVertex] = useState<{ zoneId: string; pointIndex: number } | null>(null)
  const [localZoneOverrides, setLocalZoneOverrides] = useState<Record<string, number[][]>>({})

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const rafRef = useRef<number>(0)

  /* ── resolve zone points (local override during drag, else prop) ── */
  const getZonePoints = useCallback(
    (zone: Zone) => localZoneOverrides[zone.id] ?? zone.points,
    [localZoneOverrides],
  )

  /* ── canvas draw ── */
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

    const W = canvas.width
    const H = canvas.height

    // draw existing zones
    for (const zone of existingZones) {
      const pts = getZonePoints(zone)
      if (pts.length < 3) continue

      const isSelected = zone.id === selectedZoneId
      const isHovered = zone.id === hoveredZoneId
      const isBeingRedrawn = zone.id === editingZoneId

      ctx.beginPath()
      ctx.moveTo(pts[0][0] * W, pts[0][1] * H)
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0] * W, pts[i][1] * H)
      ctx.closePath()

      if (isBeingRedrawn) {
        // dimmed dashed reference
        ctx.fillStyle = zone.color + "10"
        ctx.fill()
        ctx.setLineDash([6, 4])
        ctx.strokeStyle = zone.color + "60"
        ctx.lineWidth = 1.5
        ctx.stroke()
        ctx.setLineDash([])
      } else {
        const fillAlpha = isSelected ? "40" : isHovered ? "35" : "20"
        ctx.fillStyle = zone.color + fillAlpha
        ctx.fill()
        ctx.strokeStyle = zone.color
        ctx.lineWidth = isSelected ? 3 : 2
        ctx.stroke()
      }

      // zone label
      if (!isBeingRedrawn) {
        const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length * W
        const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length * H
        ctx.fillStyle = zone.color
        ctx.font = "bold 12px Inter, sans-serif"
        ctx.textAlign = "center"
        ctx.fillText(zone.name, cx, cy)
      }

      // vertex handles on selected zone
      if (isSelected && !isBeingRedrawn) {
        for (const pt of pts) {
          ctx.beginPath()
          ctx.arc(pt[0] * W, pt[1] * H, 7, 0, Math.PI * 2)
          ctx.fillStyle = "white"
          ctx.fill()
          ctx.strokeStyle = zone.color
          ctx.lineWidth = 2.5
          ctx.stroke()
        }
      }
    }

    // draw current polygon in progress
    if (currentPoints.length > 0) {
      const typeConfig = ZONE_TYPES.find((t) => t.key === newZoneType)
      const color = typeConfig?.color || "#dc2626"

      ctx.beginPath()
      ctx.moveTo(currentPoints[0][0] * W, currentPoints[0][1] * H)
      for (let i = 1; i < currentPoints.length; i++) {
        ctx.lineTo(currentPoints[i][0] * W, currentPoints[i][1] * H)
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

      for (let i = 0; i < currentPoints.length; i++) {
        const pt = currentPoints[i]
        const isFirst = i === 0 && currentPoints.length >= 3
        ctx.beginPath()
        ctx.arc(pt[0] * W, pt[1] * H, isFirst ? 8 : 5, 0, Math.PI * 2)
        ctx.fillStyle = isFirst ? "white" : color
        ctx.fill()
        ctx.strokeStyle = isFirst ? color : "white"
        ctx.lineWidth = isFirst ? 3 : 2
        ctx.stroke()
      }
    }
  }, [existingZones, currentPoints, newZoneType, selectedZoneId, hoveredZoneId, editingZoneId, getZonePoints])

  /* ── redraw on state changes + ResizeObserver ── */
  useEffect(() => {
    drawCanvas()
  }, [drawCanvas])

  useEffect(() => {
    onDraftStateChange?.(currentPoints.length > 0)
  }, [currentPoints.length, onDraftStateChange])

  useEffect(() => {
    const img = imgRef.current
    if (!img) return
    const ro = new ResizeObserver(() => drawCanvas())
    ro.observe(img)
    return () => ro.disconnect()
  }, [drawCanvas])

  // periodic redraw for MJPEG live stream (image updates but no React state change)
  useEffect(() => {
    const interval = setInterval(drawCanvas, 300)
    return () => clearInterval(interval)
  }, [drawCanvas])

  /* ── keyboard shortcuts ── */
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (!drawingMode) return
      if (e.key === "Escape") {
        e.preventDefault()
        cancelDrawing()
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault()
        setCurrentPoints((p) => p.slice(0, -1))
      }
    }
    window.addEventListener("keydown", handleKey)
    return () => window.removeEventListener("keydown", handleKey)
  }, [drawingMode])

  /* ── canvas click ── */
  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (draggingVertex) return // ignore clicks during drag release

    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height

    if (drawingMode) {
      // check if clicking near first point to close polygon
      if (currentPoints.length >= 3) {
        const dist = distPx(x, y, currentPoints[0][0], currentPoints[0][1], rect.width, rect.height)
        if (dist < 15) {
          handleSave()
          return
        }
      }
      setCurrentPoints((prev) => [...prev, [x, y]])
      return
    }

    // not drawing — check zone hit for selection
    // iterate in reverse so topmost zone wins
    for (let i = existingZones.length - 1; i >= 0; i--) {
      const zone = existingZones[i]
      const pts = getZonePoints(zone)
      if (pts.length >= 3 && isPointInPolygon(x, y, pts)) {
        setSelectedZoneId(zone.id === selectedZoneId ? null : zone.id)
        return
      }
    }
    // clicked outside all zones
    setSelectedZoneId(null)
  }

  /* ── mouse move: hover + vertex drag ── */
  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height

    // vertex dragging
    if (draggingVertex) {
      setLocalZoneOverrides((prev) => {
        const zone = existingZones.find((z) => z.id === draggingVertex.zoneId)
        if (!zone) return prev
        const base = prev[zone.id] ?? [...zone.points.map((p) => [...p])]
        const updated = base.map((p, i) => (i === draggingVertex.pointIndex ? [x, y] : p))
        return { ...prev, [zone.id]: updated }
      })
      rafRef.current = requestAnimationFrame(drawCanvas)
      return
    }

    if (drawingMode) return

    // hover detection
    let found: string | null = null
    for (let i = existingZones.length - 1; i >= 0; i--) {
      const zone = existingZones[i]
      const pts = getZonePoints(zone)
      if (pts.length >= 3 && isPointInPolygon(x, y, pts)) {
        found = zone.id
        break
      }
    }
    if (found !== hoveredZoneId) setHoveredZoneId(found)

    // check for vertex hover on selected zone
    if (selectedZoneId && onUpdate) {
      const selZone = existingZones.find((z) => z.id === selectedZoneId)
      if (selZone) {
        const pts = getZonePoints(selZone)
        for (const pt of pts) {
          if (distPx(x, y, pt[0], pt[1], rect.width, rect.height) < 10) {
            canvas.style.cursor = "grab"
            return
          }
        }
      }
    }

    canvas.style.cursor = found ? "pointer" : drawingMode ? "crosshair" : "default"
  }

  /* ── mouse down: start vertex drag ── */
  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    if (drawingMode || !selectedZoneId || !onUpdate) return
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height

    const selZone = existingZones.find((z) => z.id === selectedZoneId)
    if (!selZone) return
    const pts = getZonePoints(selZone)

    for (let i = 0; i < pts.length; i++) {
      if (distPx(x, y, pts[i][0], pts[i][1], rect.width, rect.height) < 10) {
        e.preventDefault()
        setDraggingVertex({ zoneId: selZone.id, pointIndex: i })
        canvas.style.cursor = "grabbing"
        return
      }
    }
  }

  /* ── mouse up: finish vertex drag ── */
  async function handleMouseUp() {
    if (!draggingVertex || !onUpdate) {
      setDraggingVertex(null)
      return
    }
    const override = localZoneOverrides[draggingVertex.zoneId]
    if (override) {
      try {
        await onUpdate(draggingVertex.zoneId, { points: override })
      } catch {
        // revert
      }
    }
    setDraggingVertex(null)
    setLocalZoneOverrides((prev) => {
      const next = { ...prev }
      delete next[draggingVertex.zoneId]
      return next
    })
    const canvas = canvasRef.current
    if (canvas) canvas.style.cursor = "default"
  }

  /* ── save / cancel ── */
  async function handleSave() {
    if (currentPoints.length < 3 || !newZoneName.trim()) return
    setSaving(true)
    try {
      const typeConfig = ZONE_TYPES.find((t) => t.key === newZoneType)
      if (editingZoneId && onUpdate) {
        await onUpdate(editingZoneId, { points: currentPoints })
      } else {
        const severityThresholds: Record<string, number> = {}
        if (newZoneP1Threshold.trim()) severityThresholds.P1 = Number(newZoneP1Threshold)
        if (newZoneP2Threshold.trim()) severityThresholds.P2 = Number(newZoneP2Threshold)
        await onSave({
          name: newZoneName.trim(),
          type: newZoneType,
          color: typeConfig?.color || "#dc2626",
          points: currentPoints,
          analytics: newZoneAnalytics || null,
          classes: newZoneClasses.split(",").map((item) => item.trim()).filter(Boolean),
          threshold: numberOrNull(newZoneThreshold),
          min_duration_seconds: numberOrNull(newZoneMinDuration),
          area_square_meters: numberOrNull(newZoneArea),
          severity_thresholds: Object.keys(severityThresholds).length ? severityThresholds : null,
        })
      }
      resetDrawing()
    } catch {
      // parent surfaces errors
    } finally {
      setSaving(false)
    }
  }

  function cancelDrawing() {
    setDrawingMode(false)
    setCurrentPoints([])
    setNewZoneName("")
    setNewZoneType(defaultZoneType)
    setNewZoneAnalytics(defaultAnalyticsForZone(defaultZoneType))
    setNewZoneClasses("")
    setNewZoneThreshold("")
    setNewZoneMinDuration("")
    setNewZoneArea("")
    setNewZoneP1Threshold("")
    setNewZoneP2Threshold("")
    setEditingZoneId(null)
  }

  function resetDrawing() {
    setCurrentPoints([])
    setNewZoneName("")
    setNewZoneType(defaultZoneType)
    setNewZoneAnalytics(defaultAnalyticsForZone(defaultZoneType))
    setNewZoneClasses("")
    setNewZoneThreshold("")
    setNewZoneMinDuration("")
    setNewZoneArea("")
    setNewZoneP1Threshold("")
    setNewZoneP2Threshold("")
    setDrawingMode(false)
    setEditingZoneId(null)
    setSelectedZoneId(null)
  }

  function startRedraw(zone: Zone) {
    setEditingZoneId(zone.id)
    setNewZoneName(zone.name)
    const zt = ZONE_TYPES.find((t) => t.key === zone.type)
    setNewZoneType(zt ? zt.key : "restricted")
    setNewZoneAnalytics(zone.analytics || defaultAnalyticsForZone(zt ? zt.key : "restricted"))
    setNewZoneClasses((zone.classes || []).join(", "))
    setNewZoneThreshold(numberToInput(zone.threshold))
    setNewZoneMinDuration(numberToInput(zone.min_duration_seconds))
    setNewZoneArea(numberToInput(zone.area_square_meters))
    setNewZoneP1Threshold(numberToInput(zone.severity_thresholds?.P1))
    setNewZoneP2Threshold(numberToInput(zone.severity_thresholds?.P2))
    setCurrentPoints([])
    setDrawingMode(true)
    setSelectedZoneId(null)
  }

  function startNewZone() {
    setCurrentPoints([])
    setNewZoneType(defaultZoneType)
    setNewZoneAnalytics(defaultAnalyticsForZone(defaultZoneType))
    setNewZoneClasses("")
    setNewZoneThreshold("")
    setNewZoneMinDuration("")
    setNewZoneArea("")
    setNewZoneP1Threshold("")
    setNewZoneP2Threshold("")
    setNewZoneName(getDefaultZoneName(defaultZoneType, existingZones.length + 1))
    setDrawingMode(true)
    setEditingZoneId(null)
    setSelectedZoneId(null)
  }

  async function handleInlineDelete(zoneId: string) {
    if (!onDelete) return
    try {
      await onDelete(zoneId)
      setSelectedZoneId(null)
    } catch {
      // parent surfaces errors
    }
  }

  const selectedZone = existingZones.find((z) => z.id === selectedZoneId)
  const cursorClass = drawingMode
    ? "cursor-crosshair"
    : ""

  // instruction text
  let instructionText = ""
  if (drawingMode) {
    if (currentPoints.length < 3) {
      instructionText = `Click to place points (${currentPoints.length} placed) — need at least 3`
    } else {
      instructionText = `${currentPoints.length} points — click first point to close, or keep adding`
    }
    if (editingZoneId) instructionText = `Redrawing zone — ${instructionText}`
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
            onMouseMove={handleMouseMove}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => { setHoveredZoneId(null); handleMouseUp() }}
            className={`absolute inset-0 w-full h-full ${cursorClass}`}
          />
          {drawingMode && instructionText && (
            <div className="absolute top-3 left-3 bg-black/70 text-white text-xs px-3 py-1.5 rounded-full">
              {instructionText}
            </div>
          )}
          {!drawingMode && !selectedZoneId && existingZones.length > 0 && (
            <div className="absolute bottom-3 left-3 bg-black/50 text-white text-xs px-3 py-1.5 rounded-full">
              Click a zone to select it
            </div>
          )}
        </div>
      </Card>

      {/* Selected zone toolbar */}
      {selectedZone && !drawingMode && (
        <Card className="flex items-center gap-3 py-2 px-4">
          <div
            className="w-3 h-3 rounded-full flex-shrink-0"
            style={{ backgroundColor: selectedZone.color }}
          />
          <span className="text-sm font-medium text-[var(--color-text-primary)] flex-1">
            {selectedZone.name}
          </span>
          <span className="text-xs text-[var(--color-text-tertiary)]">
            {ZONE_TYPES.find((t) => t.key === selectedZone.type)?.label ?? selectedZone.type}
            {selectedZone.analytics ? ` · ${selectedZone.analytics}` : ""}
          </span>
          {onUpdate && (
            <Button variant="secondary" size="sm" onClick={() => startRedraw(selectedZone)}>
              <Pencil className="w-3.5 h-3.5 mr-1" />
              Redraw
            </Button>
          )}
          {onDelete && (
            <Button variant="secondary" size="sm" onClick={() => handleInlineDelete(selectedZone.id)}
              className="text-red-600 hover:text-red-700"
            >
              <Trash2 className="w-3.5 h-3.5 mr-1" />
              Delete
            </Button>
          )}
          <button
            type="button"
            onClick={() => setSelectedZoneId(null)}
            className="p-1 rounded hover:bg-[var(--color-bg-subtle)] text-[var(--color-text-tertiary)]"
          >
            <X className="w-4 h-4" />
          </button>
        </Card>
      )}

      {/* Drawing form */}
      {drawingMode ? (
        <Card>
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">
            {editingZoneId ? "Redraw Zone" : "New Zone"}
          </h3>
          <div className="space-y-3">
            {!editingZoneId && (
              <>
                <p className="text-sm text-[var(--color-text-secondary)]">
                  Zones are saved separately from the camera. Click <strong>Save Zone</strong> here before saving the camera.
                </p>
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
                  <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                    {ZONE_TYPES.map((zt) => {
                      const Icon = zt.icon
                      return (
                        <button
                          key={zt.key}
                          type="button"
                          onClick={() => {
                            setNewZoneType(zt.key)
                            setNewZoneAnalytics(defaultAnalyticsForZone(zt.key))
                          }}
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

                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">Analytics Type</label>
                    <select
                      value={newZoneAnalytics}
                      onChange={(event) => setNewZoneAnalytics(event.target.value)}
                      className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
                    >
                      <option value="intrusion">Restricted entry</option>
                      <option value="ppe">PPE required</option>
                      <option value="occupancy">Occupancy</option>
                      <option value="queue">Queue</option>
                      <option value="obstruction">Route obstruction</option>
                      <option value="vehicle_presence">Vehicle presence</option>
                      <option value="object_lifecycle">Object lifecycle</option>
                      <option value="exclusion">Exclusion / ignore</option>
                    </select>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">Target Classes</label>
                    <input
                      type="text"
                      value={newZoneClasses}
                      onChange={(event) => setNewZoneClasses(event.target.value)}
                      placeholder="person, forklift, truck"
                      className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
                    />
                  </div>
                </div>

                <div className="grid gap-3 md:grid-cols-5">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">Threshold</label>
                    <input
                      type="number"
                      min="0"
                      step="0.1"
                      value={newZoneThreshold}
                      onChange={(event) => setNewZoneThreshold(event.target.value)}
                      placeholder="default"
                      className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">Min Dwell</label>
                    <input
                      type="number"
                      min="0"
                      step="0.5"
                      value={newZoneMinDuration}
                      onChange={(event) => setNewZoneMinDuration(event.target.value)}
                      placeholder="seconds"
                      className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">Area m2</label>
                    <input
                      type="number"
                      min="0"
                      step="0.1"
                      value={newZoneArea}
                      onChange={(event) => setNewZoneArea(event.target.value)}
                      placeholder="optional"
                      className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">P1 At</label>
                    <input
                      type="number"
                      min="0"
                      step="0.1"
                      value={newZoneP1Threshold}
                      onChange={(event) => setNewZoneP1Threshold(event.target.value)}
                      placeholder="optional"
                      className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">P2 At</label>
                    <input
                      type="number"
                      min="0"
                      step="0.1"
                      value={newZoneP2Threshold}
                      onChange={(event) => setNewZoneP2Threshold(event.target.value)}
                      placeholder="optional"
                      className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
                    />
                  </div>
                </div>
              </>
            )}

            {editingZoneId && (
              <p className="text-sm text-[var(--color-text-secondary)]">
                Draw new boundary for <strong>{newZoneName}</strong>. The old boundary is shown as a dashed outline for reference.
              </p>
            )}

            <div className="flex items-center gap-2 pt-2">
              <Button
                onClick={handleSave}
                disabled={currentPoints.length < 3 || (!editingZoneId && !newZoneName.trim()) || saving}
              >
                {saving ? "Saving..." : editingZoneId
                  ? `Update Zone (${currentPoints.length} points)`
                  : `Save Zone (${currentPoints.length} points)`}
              </Button>
              <Button
                variant="secondary"
                onClick={() => setCurrentPoints((p) => p.slice(0, -1))}
                disabled={currentPoints.length === 0}
              >
                Undo Point
              </Button>
              <Button variant="secondary" onClick={cancelDrawing}>
                Cancel
              </Button>
              <span className="text-xs text-[var(--color-text-tertiary)] ml-auto">
                Esc to cancel &middot; Ctrl+Z to undo
              </span>
            </div>
          </div>
        </Card>
      ) : (
        <Button onClick={startNewZone}>
          <Plus className="w-4 h-4" />
          Draw Zone
        </Button>
      )}
    </div>
  )
}

function numberOrNull(value: string): number | null {
  if (!value.trim()) return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function numberToInput(value: unknown): string {
  if (value === null || value === undefined || value === "") return ""
  const parsed = Number(value)
  return Number.isFinite(parsed) ? String(parsed) : ""
}
