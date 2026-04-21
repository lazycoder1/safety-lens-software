import { useCallback, useEffect, useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import { Plus, SearchCheck } from "lucide-react"
import { Button } from "@/components/ui/button"
import { getCameras, deleteCamera, getSafetyRules, getZones } from "@/lib/api"
import { useAuthStore } from "@/stores/authStore"
import type { Camera, SafetyRule } from "@/types"
import { CameraCard } from "@/components/cameras/CameraCard"
import { SearchInput } from "@/components/ui/SearchInput"
import { getConfiguredDetectionKeys, usesZoneIntrusion } from "@/components/cameras/detectionCatalog"

export function CameraConfig() {
  const navigate = useNavigate()
  const userRole = useAuthStore((state) => state.user?.role)
  const isAdmin = userRole === "admin"

  const [cameras, setCameras] = useState<Camera[]>([])
  const [safetyRules, setSafetyRules] = useState<SafetyRule[]>([])
  const [search, setSearch] = useState("")
  const [zoneCounts, setZoneCounts] = useState<Record<string, number>>({})

  const fetchData = useCallback(async () => {
    try {
      const [cameraData, ruleData]: [Camera[], SafetyRule[]] = await Promise.all([getCameras(), getSafetyRules()])
      setCameras(cameraData)
      setSafetyRules(ruleData)

      const zoneCameraIds = cameraData
        .filter((camera: Camera) => usesZoneIntrusion(getConfiguredDetectionKeys(camera, ruleData)))
        .map((camera: Camera) => camera.id)

      const zoneEntries = await Promise.all(
        zoneCameraIds.map(async (cameraId: string) => {
          const zones = await getZones(cameraId).catch(() => [])
          return [cameraId, zones.length] as const
        })
      )

      setZoneCounts(Object.fromEntries(zoneEntries))
    } catch {
      // silently fail for now
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  const filtered = useMemo(() => {
    if (!search.trim()) return cameras
    const query = search.toLowerCase()
    return cameras.filter(
      (camera) =>
        camera.name.toLowerCase().includes(query) ||
        camera.zone.toLowerCase().includes(query)
    )
  }, [search, cameras])

  const onlineCount = cameras.filter((camera) => camera.status === "online").length

  async function handleDelete(id: string) {
    const camera = cameras.find((item) => item.id === id)
    if (!window.confirm(`Delete camera "${camera?.name || id}"? This cannot be undone.`)) return
    try {
      await deleteCamera(id)
      await fetchData()
    } catch {
      // ignore
    }
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Cameras</h1>
          <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
            Browse camera health, enabled detections, and zone readiness.
          </p>
        </div>
        {isAdmin && (
          <div className="flex items-center gap-2">
            <Button onClick={() => navigate("/configure/cameras/discover")}>
              <SearchCheck className="w-4 h-4" />
              Discover Cameras
            </Button>
            <Button variant="secondary" onClick={() => navigate("/configure/cameras/new")}>
              <Plus className="w-4 h-4" />
              Add Camera
            </Button>
          </div>
        )}
      </div>

      <div className="flex items-center gap-4 text-sm text-[var(--color-text-secondary)]">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-full bg-[var(--color-success)]" />
          <span className="font-medium">{onlineCount}</span> Online
        </span>
        <span className="text-[var(--color-text-tertiary)]">
          &mdash; <span className="font-medium text-[var(--color-text-primary)]">{cameras.length}</span> Total
        </span>
      </div>

      <SearchInput
        value={search}
        onChange={setSearch}
        placeholder="Filter cameras by name or zone..."
        className="max-w-sm"
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {filtered.map((camera) => (
          <CameraCard
            key={camera.id}
            camera={camera}
            safetyRules={safetyRules}
            zoneCount={zoneCounts[camera.id] ?? null}
            canEdit={isAdmin}
            onView={() => navigate(`/configure/cameras/${camera.id}`)}
            onEdit={() => navigate(`/configure/cameras/${camera.id}/edit`)}
            onDelete={() => handleDelete(camera.id)}
          />
        ))}
        {filtered.length === 0 && (
          <p className="col-span-full py-12 text-center text-sm text-[var(--color-text-tertiary)]">
            {cameras.length === 0
              ? "No cameras configured yet."
              : "No cameras match your search."}
          </p>
        )}
      </div>
    </div>
  )
}
