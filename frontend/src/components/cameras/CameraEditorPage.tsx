import { useEffect, useMemo, useRef, useState } from "react"
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom"
import { AlertTriangle, ArrowLeft, Trash2 } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import {
  addCamera,
  addZone,
  API_BASE,
  ApiError,
  deleteCamera,
  deleteZone,
  getCameraById,
  getSafetyRules,
  getVideos,
  getZones,
  installModels,
  previewCameraPlan,
  updateCamera,
} from "@/lib/api"
import type { Camera, CameraProfile, ExecutionPlan, SafetyRule, Zone } from "@/types"
import {
  cameraNeedsLegacyNormalization,
  deriveCameraPurpose,
  getConfiguredDetectionKeys,
  getDetectionLabelsFromKeys,
  getSelectedRuleIds,
  getUnmappedRuleIds,
  usesZoneIntrusion,
  type CameraDetectionKey,
} from "./detectionCatalog"
import { PROFILE_DEFAULTS, PROFILE_OPTIONS } from "./profileOptions"
import { DetectionChecklist } from "./DetectionChecklist"
import { Field } from "./helpers"
import { PolygonDrawer } from "@/components/zones/PolygonDrawer"
import { useModelInstallModal } from "@/components/ModelInstallModal"

interface CameraEditorPageProps {
  mode: "create" | "edit"
}

export function CameraEditorPage({ mode }: CameraEditorPageProps) {
  const navigate = useNavigate()
  const params = useParams()
  const [searchParams] = useSearchParams()
  const openInstallModal = useModelInstallModal((state) => state.open)
  const cameraId = params.cameraId || null
  const isEdit = mode === "edit"
  const showZoneFocus = searchParams.get("focus") === "zones"
  const zoneSectionRef = useRef<HTMLDivElement>(null)

  const [loading, setLoading] = useState(isEdit)
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [videos, setVideos] = useState<string[]>([])
  const [safetyRules, setSafetyRules] = useState<SafetyRule[]>([])
  const [camera, setCamera] = useState<Camera | null>(null)
  const [zones, setZones] = useState<Zone[]>([])
  const [planPreview, setPlanPreview] = useState<{ execution_plan: ExecutionPlan; missing_model_keys: string[] } | null>(null)

  const [name, setName] = useState("")
  const [zone, setZone] = useState("")
  const [profile, setProfile] = useState<CameraProfile>("general_safety")
  const [streamType, setStreamType] = useState<"file" | "rtsp">("file")
  const [video, setVideo] = useState("")
  const [rtspUrl, setRtspUrl] = useState("")
  const [rtspUsername, setRtspUsername] = useState("")
  const [rtspPassword, setRtspPassword] = useState("")
  const [replaceStoredCredentials, setReplaceStoredCredentials] = useState(false)
  const [selectedDetections, setSelectedDetections] = useState<CameraDetectionKey[]>([])

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const [ruleData, videoData] = await Promise.all([getSafetyRules(), getVideos()])
        if (cancelled) return
        setSafetyRules(ruleData)
        setVideos(videoData)

        if (!isEdit || !cameraId) {
          if (videoData.length > 0) {
            setVideo((current) => current || videoData[0])
          }
          setLoading(false)
          return
        }

        const existingCamera = await getCameraById(cameraId)
        if (cancelled) return
        const zoneData = await getZones(existingCamera.id).catch(() => [])
        if (cancelled) return

        setCamera(existingCamera)
        setZones(zoneData)
        setName(existingCamera.name)
        setZone(existingCamera.zone)
        setProfile(existingCamera.profile || "general_safety")
        setStreamType((existingCamera.stream_type || "file") as "file" | "rtsp")
        setVideo(existingCamera.video || videoData[0] || "")
        setRtspUrl(existingCamera.connection_summary || existingCamera.rtsp_url || "")
        setRtspUsername("")
        setRtspPassword("")
        setReplaceStoredCredentials(false)
        setSelectedDetections(getConfiguredDetectionKeys(existingCamera, ruleData))
      } catch (error: any) {
        toast.error(error.message || "Failed to load camera")
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [cameraId, isEdit])

  const requiresZone = usesZoneIntrusion(selectedDetections)
  const selectedRuleIds = useMemo(() => getSelectedRuleIds(selectedDetections, safetyRules), [selectedDetections, safetyRules])
  const selectedLabels = useMemo(() => getDetectionLabelsFromKeys(selectedDetections), [selectedDetections])
  const preservedRuleIds = useMemo(
    () => (camera ? getUnmappedRuleIds(camera.safety_rule_ids || [], safetyRules) : []),
    [camera, safetyRules]
  )
  const purpose = preservedRuleIds.length > 0 ? "Custom monitoring" : deriveCameraPurpose(selectedDetections)
  const showLegacyNote = camera ? cameraNeedsLegacyNormalization(camera) : false

  useEffect(() => {
    if (!showZoneFocus || !requiresZone || !zoneSectionRef.current) return
    zoneSectionRef.current.scrollIntoView({ behavior: "smooth", block: "start" })
  }, [showZoneFocus, requiresZone, zones.length])

  useEffect(() => {
    if (loading) return
    const shouldSendCredentialUpdate =
      streamType === "rtsp" &&
      (!isEdit || replaceStoredCredentials || Boolean(rtspUsername.trim()) || Boolean(rtspPassword))
    let cancelled = false
    const timer = window.setTimeout(async () => {
      setPreviewLoading(true)
      try {
        const preview = await previewCameraPlan({
          name,
          zone,
          profile,
          capabilities: selectedDetections,
          stream_type: streamType,
          video: streamType === "file" ? video : "",
          rtsp_url: streamType === "rtsp" ? rtspUrl : "",
          username: shouldSendCredentialUpdate ? rtspUsername.trim() : undefined,
          password: shouldSendCredentialUpdate ? rtspPassword : undefined,
          safety_rule_ids: [...selectedRuleIds, ...preservedRuleIds],
        })
        if (!cancelled) {
          setPlanPreview(preview)
        }
      } catch {
        if (!cancelled) setPlanPreview(null)
      } finally {
        if (!cancelled) setPreviewLoading(false)
      }
    }, 250)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [
    loading,
    name,
    zone,
    profile,
    streamType,
    video,
    rtspUrl,
    rtspUsername,
    rtspPassword,
    replaceStoredCredentials,
    selectedDetections,
    selectedRuleIds,
    preservedRuleIds,
    isEdit,
  ])

  function handleProfileChange(nextProfile: CameraProfile) {
    setProfile(nextProfile)
    if (selectedDetections.length === 0) {
      setSelectedDetections(PROFILE_DEFAULTS[nextProfile])
    }
  }

  async function refreshZones(targetCameraId: string) {
    const updatedZones = await getZones(targetCameraId).catch(() => [])
    setZones(updatedZones)
  }

  async function handleSaveZone(newZone: { name: string; type: string; color: string; points: number[][] }) {
    if (!cameraId) return
    try {
      await addZone(cameraId, newZone)
      await refreshZones(cameraId)
      toast.success("Zone saved")
    } catch (error: any) {
      toast.error(error.message || "Failed to save zone")
    }
  }

  async function handleDeleteZone(zoneId: string) {
    if (!cameraId) return
    try {
      await deleteZone(cameraId, zoneId)
      setZones((current) => current.filter((item) => item.id !== zoneId))
      toast.success("Zone removed")
    } catch (error: any) {
      toast.error(error.message || "Failed to remove zone")
    }
  }

  async function handleDeleteCamera() {
    if (!cameraId || !camera) return
    if (!window.confirm(`Delete camera "${camera.name}"? This cannot be undone.`)) return
    setDeleting(true)
    try {
      await deleteCamera(cameraId)
      toast.success("Camera deleted")
      navigate("/configure/cameras")
    } catch (error: any) {
      toast.error(error.message || "Failed to delete camera")
    } finally {
      setDeleting(false)
    }
  }

  async function submitCamera(isRetry = false) {
    if (!name.trim()) {
      toast.error("Camera name is required")
      return
    }
    if (!zone.trim()) {
      toast.error("Zone / location is required")
      return
    }
    if (streamType === "file" && !video) {
      toast.error("Choose a video file")
      return
    }
    if (streamType === "rtsp" && !rtspUrl.trim()) {
      toast.error("RTSP URL is required")
      return
    }

    const shouldSendCredentialUpdate =
      streamType === "rtsp" &&
      (!isEdit || replaceStoredCredentials || Boolean(rtspUsername.trim()) || Boolean(rtspPassword))

    const payload = {
      name: name.trim(),
      zone: zone.trim(),
      profile,
      capabilities: selectedDetections,
      demo: planPreview?.execution_plan?.derived_demo || "yolo",
      video: streamType === "file" ? video : "",
      rules: [],
      stream_type: streamType,
      rtsp_url: streamType === "rtsp" ? rtspUrl.trim() : "",
      username: shouldSendCredentialUpdate ? rtspUsername.trim() : undefined,
      password: shouldSendCredentialUpdate ? rtspPassword : undefined,
      safety_rule_ids: [...selectedRuleIds, ...preservedRuleIds],
    }

    setSaving(true)
    try {
      const saved = isEdit && cameraId ? await updateCamera(cameraId, payload) : await addCamera(payload)

      if (requiresZone && !isEdit) {
        toast.success("Camera created. Draw at least one zone to finish setup.")
        navigate(`/configure/cameras/${saved.id}/edit?focus=zones`)
      } else {
        toast.success(isEdit ? "Camera updated" : "Camera created")
        navigate(`/configure/cameras/${saved.id}`)
      }
    } catch (error) {
      if (error instanceof ApiError && error.status === 409 && error.payload?.code === "missing_models" && !isRetry) {
        try {
          const job = await installModels(error.payload.missing_model_keys || [])
          openInstallModal(job.id, {
            title: "Setting up required models",
            subtitle: "The selected camera needs missing model files. This modal will stay open until download and setup finish.",
            onReady: () => {
              void submitCamera(true)
            },
          })
          return
        } catch (installError: any) {
          toast.error(installError.message || "Failed to start model setup")
          return
        }
      }
      toast.error(error instanceof Error ? error.message : "Failed to save camera")
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="p-6">
        <Card className="p-6">
          <p className="text-sm text-[var(--color-text-secondary)]">Loading camera setup…</p>
        </Card>
      </div>
    )
  }

  if (isEdit && !camera) {
    return (
      <div className="p-6">
        <Card className="space-y-4 p-6">
          <div>
            <h1 className="text-xl font-semibold text-[var(--color-text-primary)]">Camera not found</h1>
            <p className="mt-1 text-sm text-[var(--color-text-secondary)]">The requested camera could not be loaded.</p>
          </div>
          <Button variant="secondary" onClick={() => navigate("/configure/cameras")}>
            <ArrowLeft className="w-4 h-4" />
            Back to Cameras
          </Button>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
            <Link to="/configure/cameras" className="hover:text-[var(--color-text-primary)]">Cameras</Link>
            <span>/</span>
            <span>{isEdit ? "Edit Camera" : "New Camera"}</span>
          </div>
          <h1 className="mt-2 text-2xl font-semibold text-[var(--color-text-primary)]">
            {isEdit ? `Edit ${camera?.name || "Camera"}` : "Set Up Camera"}
          </h1>
          <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
            Configure the camera profile, detections, and execution plan in one place.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={() => navigate(isEdit && cameraId ? `/configure/cameras/${cameraId}` : "/configure/cameras")}>
            Cancel
          </Button>
          {isEdit && (
            <Button variant="danger" onClick={handleDeleteCamera} disabled={deleting}>
              <Trash2 className="w-4 h-4" />
              {deleting ? "Deleting…" : "Delete Camera"}
            </Button>
          )}
        </div>
      </div>

      {showZoneFocus && requiresZone && zones.length === 0 && (
        <Card className="border-[var(--color-warning)] bg-[var(--color-warning-bg)] p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-4 w-4 text-[var(--color-warning)]" />
            <div>
              <p className="text-sm font-semibold text-[var(--color-text-primary)]">One more step: draw a restricted zone</p>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                This camera has Zone Intrusion enabled. Add at least one zone before treating setup as complete.
              </p>
            </div>
          </div>
        </Card>
      )}

      {isEdit && camera && showLegacyNote && (
        <Card className="border-[var(--color-warning)] bg-[var(--color-warning-bg)] p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-4 w-4 text-[var(--color-warning)]" />
            <div>
              <p className="text-sm font-semibold text-[var(--color-text-primary)]">This camera is still on a legacy setup</p>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                Saving this screen will normalize the camera into the capability-planned setup.
              </p>
            </div>
          </div>
        </Card>
      )}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-6">
          <Card className="space-y-4">
            <div>
              <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Basics</h2>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                Start with the camera name, location, and source.
              </p>
            </div>

            <Field label="Camera Name">
              <input
                type="text"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="e.g. Main Gate Camera 1"
                className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
              />
            </Field>

            <Field label="Zone / Location">
              <input
                type="text"
                value={zone}
                onChange={(event) => setZone(event.target.value)}
                placeholder="e.g. Main Gate"
                className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
              />
            </Field>

            <Field label="Camera Profile">
              <div className="grid gap-2 md:grid-cols-3">
                {PROFILE_OPTIONS.map((option) => (
                  <button
                    key={option.key}
                    type="button"
                    onClick={() => handleProfileChange(option.key)}
                    className={`rounded-[var(--radius-lg)] border p-3 text-left transition-colors ${
                      profile === option.key
                        ? "border-[var(--color-info)] bg-[var(--color-info-bg)]"
                        : "border-[var(--color-border-default)] bg-white hover:border-[var(--color-border-active)]"
                    }`}
                  >
                    <p className="text-sm font-semibold text-[var(--color-text-primary)]">{option.label}</p>
                    <p className="mt-1 text-xs leading-5 text-[var(--color-text-secondary)]">{option.description}</p>
                  </button>
                ))}
              </div>
            </Field>

            <Field label="Source Type">
              <div className="flex gap-2">
                {(["file", "rtsp"] as const).map((value) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setStreamType(value)}
                    className={`rounded-[var(--radius-md)] border px-4 py-2 text-sm font-medium transition-colors ${
                      streamType === value
                        ? "border-transparent bg-[var(--color-text-primary)] text-white"
                        : "border-[var(--color-border-default)] bg-white text-[var(--color-text-secondary)] hover:border-[var(--color-border-active)]"
                    }`}
                  >
                    {value === "file" ? "Video File" : "RTSP Stream"}
                  </button>
                ))}
              </div>
            </Field>

            {streamType === "file" ? (
              <Field label="Video File">
                <select
                  value={video}
                  onChange={(event) => setVideo(event.target.value)}
                  className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)]"
                >
                  {videos.length === 0 && <option value="">No video files available</option>}
                  {videos.map((item) => (
                    <option key={item} value={item}>{item}</option>
                  ))}
                </select>
              </Field>
            ) : (
              <div className="space-y-4">
                <Field label="RTSP URL">
                  <input
                    type="text"
                    value={rtspUrl}
                    onChange={(event) => setRtspUrl(event.target.value)}
                    placeholder="rtsp://192.168.1.100:554/stream1"
                    className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 font-mono text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
                  />
                </Field>

                {isEdit && camera?.credentials_configured && !replaceStoredCredentials ? (
                  <div className="rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-[var(--color-bg-tertiary)] px-3 py-3">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-sm font-medium text-[var(--color-text-primary)]">Stored credentials are configured</p>
                        <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                          Existing camera credentials will be preserved unless you choose to replace them.
                        </p>
                      </div>
                      <Button variant="secondary" size="sm" type="button" onClick={() => setReplaceStoredCredentials(true)}>
                        Replace Credentials
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="grid gap-4 md:grid-cols-2">
                    <Field label={isEdit ? "Username Override" : "Username"}>
                      <input
                        type="text"
                        value={rtspUsername}
                        onChange={(event) => setRtspUsername(event.target.value)}
                        placeholder="admin"
                        className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
                      />
                    </Field>
                    <Field label={isEdit ? "Password Override" : "Password"}>
                      <input
                        type="password"
                        value={rtspPassword}
                        onChange={(event) => setRtspPassword(event.target.value)}
                        placeholder={isEdit ? "Leave blank to keep current unless replacing" : "Camera password"}
                        className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
                      />
                    </Field>
                  </div>
                )}
              </div>
            )}
          </Card>

          <Card className="space-y-4">
            <div>
              <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Enabled Detections</h2>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                Choose the capabilities this camera should run.
              </p>
            </div>
            <DetectionChecklist safetyRules={safetyRules} selectedKeys={selectedDetections} onChange={setSelectedDetections} />
          </Card>

          {isEdit && requiresZone && cameraId && (
            <div ref={zoneSectionRef}>
              <Card className="space-y-4">
                <div>
                  <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Zones</h2>
                  <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                    Zone Intrusion is enabled. Draw at least one restricted area for this camera.
                  </p>
                </div>

                <PolygonDrawer imageUrl={`${API_BASE}/api/stream/${cameraId}`} existingZones={zones} onSave={handleSaveZone} />

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-[var(--color-text-primary)]">Configured Zones</p>
                    <span className="text-xs text-[var(--color-text-secondary)]">
                      {zones.length} zone{zones.length !== 1 ? "s" : ""}
                    </span>
                  </div>
                  {zones.length === 0 ? (
                    <p className="text-sm text-[var(--color-text-tertiary)]">No zones configured yet.</p>
                  ) : (
                    <div className="space-y-2">
                      {zones.map((item) => (
                        <div key={item.id} className="flex items-center justify-between rounded-[var(--radius-md)] border px-3 py-2">
                          <div className="flex items-center gap-2">
                            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                            <div>
                              <p className="text-sm font-medium text-[var(--color-text-primary)]">{item.name}</p>
                              <p className="text-xs text-[var(--color-text-secondary)]">{item.type} · {item.points.length} points</p>
                            </div>
                          </div>
                          <Button variant="ghost" size="sm" onClick={() => handleDeleteZone(item.id)}>
                            <Trash2 className="w-3.5 h-3.5" />
                            Remove
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </Card>
            </div>
          )}
        </div>

        <div className="space-y-6">
          <Card className="space-y-4 xl:sticky xl:top-20">
            <div>
              <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Execution Plan</h2>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                This is the grouped runtime plan the backend will compile for this camera.
              </p>
            </div>

            <div className="space-y-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">Purpose</p>
                <p className="mt-1 text-sm font-medium text-[var(--color-text-primary)]">{purpose}</p>
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">Selected Detections</p>
                {selectedLabels.length > 0 ? (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {selectedLabels.map((label) => (
                      <Badge key={label} variant="info">{label}</Badge>
                    ))}
                  </div>
                ) : (
                  <p className="mt-1 text-sm text-[var(--color-text-secondary)]">Monitoring only</p>
                )}
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">Profile</p>
                <p className="mt-1 text-sm font-medium text-[var(--color-text-primary)]">
                  {PROFILE_OPTIONS.find((option) => option.key === profile)?.label || profile}
                </p>
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">Model Stack</p>
                {planPreview?.execution_plan?.model_stack?.length ? (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {planPreview.execution_plan.model_stack.map((item) => (
                      <Badge key={item} variant="info">{item}</Badge>
                    ))}
                  </div>
                ) : (
                  <p className="mt-1 text-sm text-[var(--color-text-secondary)]">No model execution planned yet.</p>
                )}
              </div>

              <div className="grid gap-3 md:grid-cols-2">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">Expected Load</p>
                  <p className="mt-1 text-sm font-medium text-[var(--color-text-primary)]">
                    {previewLoading ? "Refreshing…" : planPreview?.execution_plan?.runtime_load || "Low"}
                  </p>
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">Zones</p>
                  <p className="mt-1 text-sm font-medium text-[var(--color-text-primary)]">
                    {requiresZone
                      ? isEdit
                        ? zones.length > 0
                          ? `${zones.length} configured`
                          : "Setup required"
                        : "Required after save"
                      : "Not required"}
                  </p>
                </div>
              </div>

              {planPreview?.missing_model_keys?.length ? (
                <div className="rounded-[var(--radius-lg)] border border-[var(--color-warning)] bg-[var(--color-warning-bg)] p-4">
                  <p className="text-sm font-semibold text-[var(--color-text-primary)]">Models need setup</p>
                  <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                    Saving will pause and set up: {planPreview.missing_model_keys.join(", ")}.
                  </p>
                </div>
              ) : (
                <div className="rounded-[var(--radius-lg)] border border-[var(--color-success)] bg-[var(--color-success-bg)] p-4">
                  <p className="text-sm font-semibold text-[var(--color-text-primary)]">Required models are ready</p>
                  <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                    The current execution plan can run immediately after save.
                  </p>
                </div>
              )}

              {preservedRuleIds.length > 0 && (
                <p className="text-xs text-[var(--color-text-secondary)]">
                  Additional custom rules already attached to this camera will be preserved.
                </p>
              )}
            </div>

            <div className="flex flex-col gap-2 pt-2">
              <Button onClick={() => void submitCamera()} disabled={saving}>
                {saving
                  ? isEdit ? "Saving…" : "Creating…"
                  : isEdit ? "Save Camera Changes" : requiresZone ? "Save and Continue to Zones" : "Create Camera"}
              </Button>
              <Button variant="secondary" onClick={() => navigate(isEdit && cameraId ? `/configure/cameras/${cameraId}` : "/configure/cameras")}>
                Cancel
              </Button>
            </div>
          </Card>

          {import.meta.env.DEV && (
            <details className="rounded-[var(--radius-lg)] border bg-white p-4">
              <summary className="cursor-pointer text-sm font-semibold text-[var(--color-text-primary)]">Diagnostics</summary>
              <div className="mt-3 space-y-2 text-sm text-[var(--color-text-secondary)]">
                <p><span className="font-medium text-[var(--color-text-primary)]">Camera ID:</span> {camera?.id || "Not created yet"}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Source Type:</span> {streamType === "rtsp" ? "RTSP" : "Video File"}</p>
                <p className="break-all"><span className="font-medium text-[var(--color-text-primary)]">Source Value:</span> {streamType === "rtsp" ? rtspUrl || "Not set" : video || "Not set"}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Credentials Update:</span> {streamType === "rtsp" ? (replaceStoredCredentials || rtspUsername || rtspPassword ? "Provided in request" : camera?.credentials_configured ? "Preserve stored" : "None") : "N/A"}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Profile:</span> {profile}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Capabilities:</span> {selectedDetections.join(", ") || "None"}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Rule IDs:</span> {selectedRuleIds.join(", ") || "None"}</p>
              </div>
            </details>
          )}
        </div>
      </div>
    </div>
  )
}
