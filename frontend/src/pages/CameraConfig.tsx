import { useState, useMemo, useEffect, useCallback } from "react"
import { Plus, Search, Trash2, X } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getCameras, getVideos, addCamera, deleteCamera, updateCamera, API_BASE } from "@/lib/api"
import { cn } from "@/lib/utils"

interface Camera {
  id: string
  name: string
  zone: string
  demo: string
  rules: string[]
  status: string
  detectionsCount: number
  video: string
  fps: number
  enabled: boolean
  yoloe_classes: string[]
  stream_type: string
  rtsp_url: string
  alert_classes: string[]
}

type CameraRole = "general" | "gate_anpr_face" | "work_zone_ppe" | "manual"

const CAMERA_ROLES: { value: CameraRole; label: string; models: string[]; gpuImpact: string }[] = [
  { value: "general", label: "General", models: ["YOLO26n", "YOLOE-11s"], gpuImpact: "~5ms/frame" },
  { value: "gate_anpr_face", label: "Gate (ANPR + Face)", models: ["YOLO26n", "YOLOE-11s", "YOLO26s", "PaddleOCR", "SCRFD", "ArcFace"], gpuImpact: "~18ms/frame" },
  { value: "work_zone_ppe", label: "Work Zone (PPE)", models: ["YOLO26n", "YOLOE-11s"], gpuImpact: "~5ms/frame" },
  { value: "manual", label: "Manual — configure via Rules", models: [], gpuImpact: "varies" },
]

const ALERT_CLASS_OPTIONS = [
  { key: "mobile_phone", label: "Mobile Phone", desc: "Detect cell phone usage", severity: "P3" },
  { key: "animal_intrusion", label: "Animal Intrusion", desc: "Detect dogs, cats", severity: "P3" },
  { key: "person_detected", label: "Person Detected", desc: "Headcount / zone presence", severity: "P4" },
  { key: "vehicle_detected", label: "Vehicle Detected", desc: "Trucks, cars, motorcycles", severity: "P4" },
]

const statusVariant: Record<string, "success" | "critical" | "default"> = {
  online: "success",
  error: "critical",
  offline: "default",
}

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
      <div className="relative max-w-sm">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--color-text-tertiary)]" />
        <input
          type="text"
          placeholder="Filter cameras by name or zone..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full pl-9 pr-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
        />
      </div>

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

function CameraCard({
  camera,
  role,
  onClick,
  onDelete,
}: {
  camera: Camera
  role: CameraRole
  onClick: () => void
  onDelete: () => void
}) {
  const roleInfo = CAMERA_ROLES.find((r) => r.value === role) || CAMERA_ROLES[0]
  const variant = statusVariant[camera.status] || "default"

  return (
    <Card
      className="flex flex-col gap-3 cursor-pointer hover:border-[var(--color-border-active)] hover:shadow-sm transition-all"
      onClick={onClick}
    >
      {/* Info */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <p className="text-sm font-semibold text-[var(--color-text-primary)] truncate">
            {camera.name}
          </p>
          <Badge variant={variant}>{camera.status}</Badge>
        </div>
        <div className="flex items-center gap-2">
          <Badge>{camera.zone}</Badge>
          <Badge variant={camera.demo === "yolo+vlm" ? "warning" : camera.demo === "yoloe" ? "success" : "info"}>{camera.demo}</Badge>
        </div>
        <p className="text-xs text-[var(--color-text-tertiary)] truncate">
          {camera.video}
        </p>
        {camera.rules.length > 0 && (
          <p className="text-xs text-[var(--color-text-tertiary)]">
            {camera.rules.length} rule{camera.rules.length !== 1 ? "s" : ""}: {camera.rules.join(", ")}
          </p>
        )}
        <div className="space-y-0.5">
          <Badge variant="info">{roleInfo.label}</Badge>
          {roleInfo.models.length > 0 && (
            <p className="text-[10px] text-[var(--color-text-tertiary)] truncate">
              {roleInfo.models.join(", ")}
            </p>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-1 pt-1 border-t mt-auto">
        <Button
          variant="ghost"
          size="sm"
          className="text-[var(--color-critical)] hover:text-[var(--color-critical)]"
          onClick={(e) => {
            e.stopPropagation()
            onDelete()
          }}
        >
          <Trash2 className="w-3.5 h-3.5" />
          Delete
        </Button>
      </div>
    </Card>
  )
}

function CameraDetailPanel({
  camera,
  role,
  onRoleChange,
  onClose,
  onUpdated,
  onDelete,
}: {
  camera: Camera
  role: CameraRole
  onRoleChange: (role: CameraRole) => void
  onClose: () => void
  onUpdated: () => Promise<void>
  onDelete: () => void
}) {
  const [editing, setEditing] = useState(false)
  const [name, setName] = useState(camera.name)
  const [zone, setZone] = useState(camera.zone)
  const [demo, setDemo] = useState(camera.demo)
  const [rules, setRules] = useState(camera.rules.join(", "))
  const [fps, setFps] = useState(camera.fps)
  const [yoloeClasses, setYoloeClasses] = useState((camera.yoloe_classes || ["person"]).join(", "))
  const [streamType, setStreamType] = useState(camera.stream_type || "file")
  const [rtspUrl, setRtspUrl] = useState(camera.rtsp_url || "")
  const [alertClasses, setAlertClasses] = useState<string[]>(camera.alert_classes || ["mobile_phone", "animal_intrusion"])
  const [saving, setSaving] = useState(false)

  const variant = statusVariant[camera.status] || "default"

  function toggleAlertClass(key: string) {
    setAlertClasses((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    )
  }

  async function handleSave() {
    setSaving(true)
    try {
      await updateCamera(camera.id, {
        name,
        zone,
        demo,
        rules: rules.split(",").map((r) => r.trim()).filter(Boolean),
        fps,
        yoloe_classes: yoloeClasses.split(",").map((c) => c.trim()).filter(Boolean),
        stream_type: streamType,
        rtsp_url: streamType === "rtsp" ? rtspUrl : "",
        alert_classes: alertClasses,
      })
      await onUpdated()
      setEditing(false)
    } catch {
      // ignore
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/30" onClick={onClose} />
      <div className="relative w-full max-w-lg bg-white shadow-xl overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b px-6 py-4 flex items-center justify-between z-10">
          <div>
            <h2 className="text-base font-semibold text-[var(--color-text-primary)]">{camera.name}</h2>
            <div className="flex items-center gap-2 mt-1">
              <Badge variant={variant}>{camera.status}</Badge>
              <Badge>{camera.zone}</Badge>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-[var(--radius-md)] hover:bg-[var(--color-bg-tertiary)] text-[var(--color-text-tertiary)] cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Stream preview */}
          <div className="aspect-video rounded-[var(--radius-lg)] overflow-hidden bg-[var(--color-bg-tertiary)] border">
            <img
              src={`${API_BASE}/api/stream/${camera.id}`}
              alt={camera.name}
              className="w-full h-full object-cover"
            />
          </div>

          {editing ? (
            <div className="space-y-4">
              <Field label="Name">
                <input type="text" value={name} onChange={(e) => setName(e.target.value)}
                  className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
              </Field>
              <Field label="Zone">
                <input type="text" value={zone} onChange={(e) => setZone(e.target.value)}
                  className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
              </Field>
              <Field label="FPS">
                <input type="number" value={fps} onChange={(e) => setFps(Number(e.target.value))} min={1} max={30}
                  className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
              </Field>

              <div className="border-t pt-4">
                <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">Source</p>
                <Field label="Stream Type">
                  <div className="flex gap-2">
                    {(["file", "rtsp"] as const).map((t) => (
                      <button key={t} onClick={() => setStreamType(t)}
                        className={`px-4 py-2 text-sm font-medium rounded-[var(--radius-md)] border cursor-pointer transition-colors ${
                          streamType === t ? "bg-[var(--color-text-primary)] text-white border-transparent" : "bg-white text-[var(--color-text-secondary)]"
                        }`}>{t === "file" ? "Video File" : "RTSP Stream"}</button>
                    ))}
                  </div>
                </Field>
                {streamType === "rtsp" && (
                  <Field label="RTSP URL">
                    <input type="text" value={rtspUrl} onChange={(e) => setRtspUrl(e.target.value)}
                      placeholder="rtsp://192.168.1.100:554/stream1"
                      className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 font-mono text-xs" />
                  </Field>
                )}
              </div>

              <div className="border-t pt-4">
                <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">Detection</p>
                <Field label="Detection Mode">
                  <select value={demo} onChange={(e) => setDemo(e.target.value)}
                    className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer">
                    <option value="yolo">COCO (80 classes)</option>
                    <option value="yoloe">YOLOe Open-Vocab</option>
                    <option value="yolo+vlm">COCO + VLM</option>
                  </select>
                </Field>
                {demo === "yoloe" && (
                  <Field label="YOLOe Classes (comma-separated)">
                    <input type="text" value={yoloeClasses} onChange={(e) => setYoloeClasses(e.target.value)}
                      placeholder="person, hard hat, safety vest"
                      className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
                  </Field>
                )}
              </div>

              <div className="border-t pt-4">
                <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">Role</p>
                <Field label="Camera Role">
                  <select
                    value={role}
                    onChange={(e) => onRoleChange(e.target.value as CameraRole)}
                    className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer"
                  >
                    {CAMERA_ROLES.map((r) => (
                      <option key={r.value} value={r.value}>{r.label}</option>
                    ))}
                  </select>
                </Field>
                {(() => {
                  const roleInfo = CAMERA_ROLES.find((r) => r.value === role) || CAMERA_ROLES[0]
                  return (
                    <div className="mt-2 p-3 rounded-[var(--radius-md)] bg-[var(--color-bg-secondary)] border text-xs space-y-1">
                      <p className="text-[var(--color-text-secondary)]">
                        <span className="font-medium">Models:</span>{" "}
                        {roleInfo.models.length > 0 ? roleInfo.models.join(", ") : "None (configure via Rules)"}
                      </p>
                      <p className="text-[var(--color-text-secondary)]">
                        <span className="font-medium">Estimated GPU impact:</span> {roleInfo.gpuImpact}
                      </p>
                    </div>
                  )
                })()}
              </div>

              {demo !== "yoloe" && (
                <div className="border-t pt-4">
                  <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">Alert Rules</p>
                  <div className="space-y-2">
                    {ALERT_CLASS_OPTIONS.map((opt) => (
                      <label key={opt.key}
                        className={`flex items-center gap-3 p-3 rounded-[var(--radius-md)] border cursor-pointer transition-colors ${
                          alertClasses.includes(opt.key) ? "border-[var(--color-info)] bg-blue-50" : "border-[var(--color-border)]"
                        }`}>
                        <input type="checkbox" checked={alertClasses.includes(opt.key)} onChange={() => toggleAlertClass(opt.key)}
                          className="accent-[var(--color-info)]" />
                        <div className="flex-1">
                          <span className="text-sm font-medium text-[var(--color-text-primary)]">{opt.label}</span>
                          <span className="text-xs text-[var(--color-text-tertiary)] ml-2">{opt.desc}</span>
                        </div>
                        <Badge variant={opt.severity === "P3" ? "warning" : "info"}>{opt.severity}</Badge>
                      </label>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex gap-2 pt-2">
                <Button onClick={handleSave} disabled={saving}>{saving ? "Saving..." : "Save"}</Button>
                <Button variant="secondary" onClick={() => setEditing(false)}>Cancel</Button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <InfoRow label="Source" value={camera.stream_type === "rtsp" ? camera.rtsp_url : camera.video} />
              <InfoRow label="Stream Type" value={camera.stream_type === "rtsp" ? "RTSP" : "Video File"} />
              <InfoRow label="Zone" value={camera.zone} />
              <InfoRow label="Detection Mode" value={camera.demo === "yoloe" ? "YOLOe Open-Vocab" : camera.demo === "yolo+vlm" ? "COCO + VLM" : "COCO (80 classes)"} />
              {camera.demo === "yoloe" && (
                <div>
                  <span className="text-xs font-medium text-[var(--color-text-secondary)] block mb-1">YOLOe Classes</span>
                  <div className="flex flex-wrap gap-1">
                    {(camera.yoloe_classes || []).map((c) => (
                      <Badge key={c} variant="success">{c}</Badge>
                    ))}
                  </div>
                </div>
              )}
              {(() => {
                const roleInfo = CAMERA_ROLES.find((r) => r.value === role) || CAMERA_ROLES[0]
                return (
                  <>
                    <InfoRow label="Role" value={roleInfo.label} />
                    {roleInfo.models.length > 0 && (
                      <div>
                        <span className="text-xs font-medium text-[var(--color-text-secondary)] block mb-1">Models</span>
                        <div className="flex flex-wrap gap-1">
                          {roleInfo.models.map((m) => (
                            <Badge key={m}>{m}</Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )
              })()}
              <InfoRow label="FPS" value={String(camera.fps)} />
              <InfoRow label="Live Detections" value={String(camera.detectionsCount)} />
              {camera.demo !== "yoloe" && (
                <div>
                  <span className="text-xs font-medium text-[var(--color-text-secondary)] block mb-1">Alert Rules</span>
                  <div className="flex flex-wrap gap-1">
                    {(camera.alert_classes || []).map((key) => {
                      const opt = ALERT_CLASS_OPTIONS.find((o) => o.key === key)
                      return <Badge key={key} variant="info">{opt?.label || key}</Badge>
                    })}
                    {(!camera.alert_classes || camera.alert_classes.length === 0) && (
                      <span className="text-xs text-[var(--color-text-tertiary)]">No alert rules configured</span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Quick actions */}
          {!editing && (
            <div className="flex gap-2 pt-2 border-t">
              <Button variant="secondary" size="sm" className="flex-1" onClick={() => setEditing(true)}>
                Edit
              </Button>
              <Button
                variant="danger"
                size="sm"
                className="flex-1"
                onClick={() => {
                  onDelete()
                  onClose()
                }}
              >
                <Trash2 className="w-3.5 h-3.5" />
                Delete
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function AddCameraModal({
  onClose,
  onAdded,
}: {
  onClose: () => void
  onAdded: () => Promise<void>
}) {
  const [videos, setVideos] = useState<string[]>([])
  const [name, setName] = useState("")
  const [video, setVideo] = useState("")
  const [zone, setZone] = useState("")
  const [demo, setDemo] = useState("yolo")
  const [rules, setRules] = useState("")
  const [yoloeClasses, setYoloeClasses] = useState("person")
  const [streamType, setStreamType] = useState("file")
  const [rtspUrl, setRtspUrl] = useState("")
  const [alertClasses, setAlertClasses] = useState<string[]>(["mobile_phone", "animal_intrusion"])
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    getVideos().then((v) => {
      setVideos(v)
      if (v.length > 0) setVideo(v[0])
    }).catch(() => {})
  }, [])

  function toggleAlertClass(key: string) {
    setAlertClasses((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    )
  }

  async function handleSave() {
    setSaving(true)
    try {
      await addCamera({
        name,
        video: streamType === "file" ? video : "",
        zone,
        demo,
        rules: rules.split(",").map((r) => r.trim()).filter(Boolean),
        yoloe_classes: demo === "yoloe" ? yoloeClasses.split(",").map((c) => c.trim()).filter(Boolean) : undefined,
        stream_type: streamType,
        rtsp_url: streamType === "rtsp" ? rtspUrl : "",
        alert_classes: alertClasses,
      } as any)
      await onAdded()
      onClose()
    } catch {
      // ignore
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative bg-white rounded-[var(--radius-xl)] shadow-xl w-full max-w-lg mx-4 max-h-[85vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b">
          <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Add Camera</h2>
          <button
            onClick={onClose}
            className="p-1 rounded-[var(--radius-sm)] hover:bg-[var(--color-bg-tertiary)] text-[var(--color-text-tertiary)] transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-4">
          <Field label="Camera Name">
            <input
              type="text"
              placeholder="e.g. Welding Bay - Entry"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
            />
          </Field>

          <Field label="Zone">
            <input
              type="text"
              placeholder="e.g. Welding Bay"
              value={zone}
              onChange={(e) => setZone(e.target.value)}
              className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
            />
          </Field>

          {/* ── Source ── */}
          <div className="border-t pt-4 mt-2">
            <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">Video Source</p>
            <Field label="Stream Type">
              <div className="flex gap-2">
                {(["file", "rtsp"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setStreamType(t)}
                    className={`px-4 py-2 text-sm font-medium rounded-[var(--radius-md)] border cursor-pointer transition-colors ${
                      streamType === t
                        ? "bg-[var(--color-text-primary)] text-white border-transparent"
                        : "bg-white text-[var(--color-text-secondary)] border-[var(--color-border)]"
                    }`}
                  >
                    {t === "file" ? "Video File" : "RTSP Stream"}
                  </button>
                ))}
              </div>
            </Field>

            {streamType === "file" ? (
              <Field label="Video File">
                <select
                  value={video}
                  onChange={(e) => setVideo(e.target.value)}
                  className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer"
                >
                  {videos.length === 0 && <option value="">Loading...</option>}
                  {videos.map((v) => (
                    <option key={v} value={v}>{v}</option>
                  ))}
                </select>
              </Field>
            ) : (
              <Field label="RTSP URL">
                <input
                  type="text"
                  placeholder="rtsp://192.168.1.100:554/stream1"
                  value={rtspUrl}
                  onChange={(e) => setRtspUrl(e.target.value)}
                  className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 font-mono text-xs"
                />
              </Field>
            )}
          </div>

          {/* ── Detection Model ── */}
          <div className="border-t pt-4 mt-2">
            <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">Detection</p>
            <Field label="Detection Mode">
              <select
                value={demo}
                onChange={(e) => setDemo(e.target.value)}
                className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer"
              >
                <option value="yolo">COCO (80 classes — person, phone, animal, vehicle)</option>
                <option value="yoloe">YOLOe Open-Vocab (custom text prompts — PPE, etc.)</option>
                <option value="yolo+vlm">COCO + VLM (periodic scene analysis)</option>
              </select>
            </Field>

            {demo === "yoloe" && (
              <Field label="YOLOe Classes (comma-separated)">
                <input
                  type="text"
                  placeholder="person, hard hat, safety vest, safety goggles"
                  value={yoloeClasses}
                  onChange={(e) => setYoloeClasses(e.target.value)}
                  className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
                />
                <p className="text-xs text-[var(--color-text-tertiary)] mt-1">
                  Open-vocabulary — type any object class to detect (e.g. hard hat, safety vest, goggles)
                </p>
              </Field>
            )}
          </div>

          {/* ── Alert Rules ── */}
          {demo !== "yoloe" && (
            <div className="border-t pt-4 mt-2">
              <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">Alert Rules</p>
              <p className="text-xs text-[var(--color-text-tertiary)] mb-3">Select which detections should trigger alerts on this camera</p>
              <div className="space-y-2">
                {ALERT_CLASS_OPTIONS.map((opt) => (
                  <label
                    key={opt.key}
                    className={`flex items-center gap-3 p-3 rounded-[var(--radius-md)] border cursor-pointer transition-colors ${
                      alertClasses.includes(opt.key)
                        ? "border-[var(--color-info)] bg-blue-50"
                        : "border-[var(--color-border)] hover:border-[var(--color-border-strong)]"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={alertClasses.includes(opt.key)}
                      onChange={() => toggleAlertClass(opt.key)}
                      className="accent-[var(--color-info)]"
                    />
                    <div className="flex-1">
                      <span className="text-sm font-medium text-[var(--color-text-primary)]">{opt.label}</span>
                      <span className="text-xs text-[var(--color-text-tertiary)] ml-2">{opt.desc}</span>
                    </div>
                    <Badge variant={opt.severity === "P3" ? "warning" : "info"}>{opt.severity}</Badge>
                  </label>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-center gap-2 pt-4 border-t">
            <Button onClick={handleSave} disabled={!name || (streamType === "file" && !video) || (streamType === "rtsp" && !rtspUrl) || saving}>
              {saving ? "Saving..." : "Add Camera"}
            </Button>
            <Button variant="secondary" onClick={onClose}>
              Cancel
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium text-[var(--color-text-primary)]">{label}</label>
      {children}
    </div>
  )
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs font-medium text-[var(--color-text-secondary)]">{label}</span>
      <span className="text-sm text-[var(--color-text-primary)]">{value}</span>
    </div>
  )
}
