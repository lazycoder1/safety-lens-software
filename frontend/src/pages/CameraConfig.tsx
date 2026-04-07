import { useState, useMemo, useEffect, useCallback } from "react"
import { Plus, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { getCameras, deleteCamera } from "@/lib/api"
import type { Camera } from "@/types"
import { CameraCard } from "@/components/cameras/CameraCard"
import { CameraDetailPanel } from "@/components/cameras/CameraDetailPanel"
import { AddCameraModal } from "@/components/cameras/AddCameraModal"
import type { CameraRole } from "@/components/cameras/constants"
import { SearchInput } from "@/components/ui/SearchInput"

export function CameraConfig() {
  const [cameras, setCameras] = useState<Camera[]>([])
  const [search, setSearch] = useState("")
  const [modalOpen, setModalOpen] = useState(false)
  const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null)
  const [cameraRoles, setCameraRoles] = useState<Record<string, CameraRole>>({})

  const fetchCameras = useCallback(async () => {
    try {
      const data = await getCameras()
      setCameras(data)
    } catch {
      // silently fail — cameras stay empty
    }
  }, [])

  useEffect(() => {
    fetchCameras()
  }, [fetchCameras])

  const filtered = useMemo(() => {
    if (!search.trim()) return cameras
    const q = search.toLowerCase()
    return cameras.filter(
      (c) => c.name.toLowerCase().includes(q) || c.zone.toLowerCase().includes(q)
    )
  }, [search, cameras])

  const onlineCount = cameras.filter((c) => c.status === "online").length

  async function handleDelete(id: string) {
    const cam = cameras.find((c) => c.id === id)
    if (!window.confirm(`Delete camera "${cam?.name || id}"? This cannot be undone.`)) return
    try {
      await deleteCamera(id)
      await fetchCameras()
      if (selectedCamera?.id === id) setSelectedCamera(null)
    } catch {
      // ignore
    }
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Cameras</h1>
        <Button onClick={() => setModalOpen(true)}>
          <Plus className="w-4 h-4" />
          Add Camera
        </Button>
      </div>

      {/* Stats bar */}
      <div className="flex items-center gap-4 text-sm text-[var(--color-text-secondary)]">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2 h-2 rounded-full bg-[var(--color-success)]" />
          <span className="font-medium">{onlineCount}</span> Online
        </span>
        <span className="text-[var(--color-text-tertiary)]">
          &mdash; <span className="font-medium text-[var(--color-text-primary)]">{cameras.length}</span> Total
        </span>
      </div>

      {/* Search */}
      <SearchInput
        value={search}
        onChange={setSearch}
        placeholder="Filter cameras by name or zone..."
        className="max-w-sm"
      />

      {/* Camera grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {filtered.map((cam) => (
          <CameraCard
            key={cam.id}
            camera={cam}
            role={cameraRoles[cam.id] || "general"}
            onClick={() => setSelectedCamera(cam)}
            onDelete={() => handleDelete(cam.id)}
          />
        ))}
        {filtered.length === 0 && (
          <p className="col-span-full text-center text-sm text-[var(--color-text-tertiary)] py-12">
            {cameras.length === 0 ? "No cameras configured. Click Add Camera to get started." : "No cameras match your search."}
          </p>
        )}
      </div>

      {/* Modal */}
      {modalOpen && (
        <AddCameraModal
          onClose={() => setModalOpen(false)}
          onAdded={fetchCameras}
        />
      )}

      {/* Camera Detail Panel */}
      {selectedCamera && (
        <CameraDetailPanel
          camera={selectedCamera}
          role={cameraRoles[selectedCamera.id] || "general"}
          onRoleChange={(role) => setCameraRoles((prev) => ({ ...prev, [selectedCamera.id]: role }))}
          onClose={() => setSelectedCamera(null)}
          onUpdated={fetchCameras}
          onDelete={() => handleDelete(selectedCamera.id)}
        />
      )}
    </div>
  )
}
