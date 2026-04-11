import { useState, useEffect } from "react"
import { Trash2 } from "lucide-react"
import { Card } from "@/components/ui/card"
import { getCameras, getZones, addZone, deleteZone, API_BASE } from "@/lib/api"
import { PolygonDrawer } from "@/components/zones/PolygonDrawer"
import type { Zone } from "@/types"

interface Camera {
  id: string
  name: string
  zone: string
  status: string
}

export function ZoneManagement() {
  const [cameras, setCameras] = useState<Camera[]>([])
  const [selectedCamId, setSelectedCamId] = useState<string>("")
  const [zones, setZones] = useState<Zone[]>([])

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

  async function handleSaveZone(zone: { name: string; type: string; color: string; points: number[][] }) {
    if (!selectedCamId) return
    await addZone(selectedCamId, zone)
    const updated = await getZones(selectedCamId)
    setZones(updated)
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
                  onClick={() => setSelectedCamId(cam.id)}
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
        <div className="lg:col-span-2">
          {selectedCamId ? (
            <PolygonDrawer
              key={selectedCamId}
              imageUrl={`${API_BASE}/api/stream/${selectedCamId}`}
              existingZones={zones}
              onSave={handleSaveZone}
            />
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
