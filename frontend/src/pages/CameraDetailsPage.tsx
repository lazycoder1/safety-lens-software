import { useEffect, useMemo, useState } from "react"
import { Link, useNavigate, useParams } from "react-router-dom"
import { AlertTriangle, ArrowLeft, Trash2 } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { deleteCamera, getAlertOutputs, getAutomationRules, getCameraById, getSafetyRules, getZones, STREAM_BASE } from "@/lib/api"
import { useAuthStore } from "@/stores/authStore"
import type { AlertOutput, Camera, CapabilityKey, CapabilityWindow, EngineRule, SafetyRule, Zone } from "@/types"
import { statusVariant } from "@/components/cameras/constants"
import {
  cameraHasDetectionModeMismatch,
  cameraNeedsLegacyNormalization,
  deriveConfiguredCameraPurpose,
  getConfiguredDetectionKeys,
  getConfiguredDetectionLabels,
  getDetectionLabelsFromKeys,
  usesConfiguredZones,
} from "@/components/cameras/detectionCatalog"
import { InfoRow } from "@/components/cameras/helpers"
import { CameraEventPolicySummary, normalizeCapabilityWindows } from "@/components/cameras/CameraEventPolicyPanel"

export function CameraDetailsPage() {
  const navigate = useNavigate()
  const params = useParams()
  const userRole = useAuthStore((state) => state.user?.role)
  const cameraId = params.cameraId || ""
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState(false)
  const [camera, setCamera] = useState<Camera | null>(null)
  const [safetyRules, setSafetyRules] = useState<SafetyRule[]>([])
  const [alertOutputs, setAlertOutputs] = useState<AlertOutput[]>([])
  const [automationRules, setAutomationRules] = useState<EngineRule[]>([])
  const [zones, setZones] = useState<Zone[]>([])

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const [cameraData, ruleData, outputData, automationRuleData] = await Promise.all([
          getCameraById(cameraId),
          getSafetyRules(),
          getAlertOutputs(),
          getAutomationRules(),
        ])
        if (cancelled) return
        const zoneData = await getZones(cameraData.id).catch(() => cameraData.zones || [])
        if (cancelled) return
        setCamera(cameraData)
        setSafetyRules(ruleData)
        setAlertOutputs(outputData)
        setAutomationRules(automationRuleData)
        setZones(zoneData)
      } catch (error: any) {
        toast.error(error.message || "Failed to load camera")
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [cameraId])

  const detectionKeys = useMemo(
    () => (camera ? getConfiguredDetectionKeys(camera, safetyRules) : []),
    [camera, safetyRules]
  )
  const detectionLabels = useMemo(
    () => (camera ? getConfiguredDetectionLabels(camera, safetyRules) : []),
    [camera, safetyRules]
  )
  const purpose = camera ? deriveConfiguredCameraPurpose(camera, safetyRules) : "Monitoring only"
  const requiresZone = usesConfiguredZones(detectionKeys)
  const statusVariantValue = camera ? statusVariant[camera.status] || "default" : "default"
  const isAdmin = userRole === "admin"
  const showLegacyNote = camera ? cameraNeedsLegacyNormalization(camera) : false
  const showModeMismatch = camera ? cameraHasDetectionModeMismatch(camera, safetyRules) : false
  const detectionSchedule = camera ? summarizeDetectionSchedule(camera) : null
  const trainedPpeEnabled = camera ? cameraUsesTrainedPpeModel(camera) : false

  async function handleDelete() {
    if (!camera) return
    if (!window.confirm(`Delete camera "${camera.name}"? This cannot be undone.`)) return
    setDeleting(true)
    try {
      await deleteCamera(camera.id)
      toast.success("Camera deleted")
      navigate("/configure/cameras")
    } catch (error: any) {
      toast.error(error.message || "Failed to delete camera")
    } finally {
      setDeleting(false)
    }
  }

  if (loading) {
    return (
      <div className="space-y-6 p-6">
        <div className="flex items-center justify-between gap-4">
          <div>
            <Skeleton className="h-4 w-40" />
            <Skeleton className="h-8 w-56 mt-2" />
            <Skeleton className="h-4 w-64 mt-1" />
          </div>
          <div className="flex items-center gap-2">
            <Skeleton className="h-9 w-20 rounded-[var(--radius-md)]" />
            <Skeleton className="h-9 w-28 rounded-[var(--radius-md)]" />
          </div>
        </div>
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
          <div className="space-y-6">
            <Skeleton className="aspect-video w-full rounded-[var(--radius-lg)]" />
            <Card className="space-y-4">
              <Skeleton className="h-5 w-24" />
              <div className="space-y-3">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <Skeleton className="h-4 w-20" />
                    <Skeleton className="h-4 w-32" />
                  </div>
                ))}
              </div>
            </Card>
            <Card className="space-y-4">
              <Skeleton className="h-5 w-40" />
              <div className="flex flex-wrap gap-2">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-6 w-24 rounded-full" />
                ))}
              </div>
            </Card>
          </div>
          <Card className="space-y-4">
            <Skeleton className="h-5 w-32" />
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-5 w-16 rounded-full" />
              </div>
              <div className="flex items-center justify-between">
                <Skeleton className="h-4 w-28" />
                <Skeleton className="h-4 w-8" />
              </div>
            </div>
          </Card>
        </div>
      </div>
    )
  }

  if (!camera) {
    return (
      <div className="p-6">
        <Card className="space-y-4 p-6">
          <div>
            <h1 className="text-xl font-semibold text-[var(--color-text-primary)]">Camera not found</h1>
            <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
              The requested camera could not be loaded.
            </p>
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
            <Link to="/configure/cameras" className="hover:text-[var(--color-text-primary)]">
              Cameras
            </Link>
            <span>/</span>
            <span>{camera.name}</span>
          </div>
          <h1 className="mt-2 text-2xl font-semibold text-[var(--color-text-primary)]">{camera.name}</h1>
          <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
            View the operational setup for this camera.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={() => navigate("/configure/cameras")}>
            <ArrowLeft className="w-4 h-4" />
            Back
          </Button>
          {isAdmin && (
            <>
              <Button onClick={() => navigate(`/configure/cameras/${camera.id}/edit`)}>
                Edit Camera
              </Button>
              <Button variant="danger" onClick={handleDelete} disabled={deleting}>
                <Trash2 className="w-4 h-4" />
                {deleting ? "Deleting…" : "Delete"}
              </Button>
            </>
          )}
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-6">
          {isAdmin && (showLegacyNote || showModeMismatch) && (
            <Card className="border-[var(--color-warning)] bg-[var(--color-warning-bg)] p-4">
              <div className="flex items-start gap-3">
                <AlertTriangle className="mt-0.5 h-4 w-4 text-[var(--color-warning)]" />
                <div>
                  <p className="text-sm font-semibold text-[var(--color-text-primary)]">
                    Camera setup needs one cleanup pass
                  </p>
                  <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                    {showLegacyNote
                      ? "This camera is still using legacy detection settings. Open Edit and save once so the visible settings and alert behavior stay aligned."
                      : "This camera has PPE detections selected under a non-PPE detector mode. Open Edit and save once to realign the camera setup."}
                  </p>
                </div>
              </div>
            </Card>
          )}

          <Card className="overflow-hidden p-0">
            <img
              src={`${STREAM_BASE}/api/stream/${camera.id}`}
              alt={camera.name}
              className="aspect-video w-full object-cover"
            />
          </Card>

          <Card className="space-y-4">
            <div>
              <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Overview</h2>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                High-level setup and current state.
              </p>
            </div>
            <div className="space-y-3">
              <InfoRow label="Zone" value={camera.zone} />
              <InfoRow
                label="Health"
                value={<Badge variant={statusVariantValue}>{camera.status}</Badge>}
              />
              <InfoRow
                label="Runtime"
                value={
                  <Badge variant={camera.runtime_status === "awaiting_model_install" ? "warning" : "default"}>
                    {camera.runtime_status}
                  </Badge>
                }
              />
              <InfoRow label="Purpose" value={purpose} />
              <InfoRow label="Profile" value={camera.profile} />
              {camera.stream_type === "rtsp" && (
                <InfoRow label="Connection" value={camera.connection_summary || camera.rtsp_url || "Configured"} />
              )}
            </div>
          </Card>

          <Card className="space-y-4">
            <div>
              <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Enabled Detections</h2>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                These are the detections configured for this camera.
              </p>
            </div>
            {detectionLabels.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {detectionLabels.map((label) => (
                  <Badge key={label} variant="info">
                    {label}
                  </Badge>
                ))}
              </div>
            ) : (
              <p className="text-sm text-[var(--color-text-secondary)]">
                No rule-based detections are enabled. This camera is used for monitoring only.
              </p>
            )}
          </Card>

          <CameraEventPolicySummary
            cameraId={camera.id}
            rules={automationRules}
            alertOutputs={alertOutputs}
          />

          <Card className="space-y-4">
            <div>
              <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Detection Runtime</h2>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                Runtime behavior for this camera's selected detections.
              </p>
            </div>
            <div className="space-y-3">
              {trainedPpeEnabled && (
                <InfoRow
                  label="PPE detection"
                  value={<Badge variant="info">Trained apron / harness detector</Badge>}
                />
              )}
              <InfoRow
                label="Detection schedule"
                value={detectionSchedule?.windowText || "Always active"}
              />
              <InfoRow
                label="Detector status"
                value={
                  detectionSchedule?.suppressedLabels.length ? (
                    <Badge variant="warning">
                      Inactive for {detectionSchedule.suppressedLabels.join(", ")}
                    </Badge>
                  ) : (
                    <Badge variant="success">Active</Badge>
                  )
                }
              />
              <InfoRow label="Expected load" value={camera.execution_plan?.runtime_load || "unknown"} />
              <InfoRow
                label="Tracking"
                value={camera.execution_plan?.tracking_enabled ? "Enabled" : "Not required"}
              />
              <InfoRow
                label="Association"
                value={camera.execution_plan?.association_enabled ? "Enabled" : "Not required"}
              />
            </div>
          </Card>

          <Card className="space-y-4">
            <div>
              <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Zones</h2>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                Zone requirements for this camera.
              </p>
            </div>
            <div className="space-y-3">
              <InfoRow
                label="Required"
                value={requiresZone ? <Badge variant="warning">Yes</Badge> : <Badge>Not required</Badge>}
              />
              <InfoRow
                label="Configured"
                value={
                  requiresZone
                    ? zones.length > 0
                      ? `${zones.length} zone${zones.length !== 1 ? "s" : ""}`
                      : "Needs setup"
                    : "Not applicable"
                }
              />
              {zones.length > 0 && (
                <div className="pt-2">
                  <p className="text-sm text-[var(--color-text-secondary)]">Saved zones</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {zones.map((item) => (
                      <Badge key={item.id} variant="info">
                        {item.name}{item.analytics ? ` · ${item.analytics}` : ""}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          <Card className="space-y-4 xl:sticky xl:top-20">
            <div>
              <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Activity & Health</h2>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                Lightweight operational signals available in the current app.
              </p>
            </div>
            <div className="space-y-3">
              <InfoRow
                label="Status"
                value={<Badge variant={statusVariantValue}>{camera.status}</Badge>}
              />
              <InfoRow label="Live detections" value={String(camera.detectionsCount)} />
              <InfoRow label="Stream processing target" value={`${camera.fps} FPS`} />
              <InfoRow label="Primary inference target" value={`${camera.inference_fps} FPS`} />
            </div>
          </Card>

          {import.meta.env.DEV && (
            <details className="rounded-[var(--radius-lg)] border bg-white p-4">
              <summary className="cursor-pointer text-sm font-semibold text-[var(--color-text-primary)]">
                Diagnostics
              </summary>
              <div className="mt-3 space-y-2 text-sm text-[var(--color-text-secondary)]">
                <p><span className="font-medium text-[var(--color-text-primary)]">Camera ID:</span> {camera.id}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Source Type:</span> {camera.stream_type === "rtsp" ? "RTSP" : "Video File"}</p>
                <p className="break-all"><span className="font-medium text-[var(--color-text-primary)]">Source Value:</span> {camera.stream_type === "rtsp" ? camera.connection_summary || camera.rtsp_url : camera.video}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">FPS:</span> {camera.fps}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Inference FPS:</span> {camera.inference_fps} ({camera.inference_fps_source || "camera"})</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Detection Engine:</span> {camera.demo}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Runtime Status:</span> {camera.runtime_status}</p>
                <p className="break-all"><span className="font-medium text-[var(--color-text-primary)]">Rule IDs:</span> {(camera.safety_rule_ids || []).join(", ") || "None"}</p>
                <p className="break-all"><span className="font-medium text-[var(--color-text-primary)]">Model Stack:</span> {camera.execution_plan?.model_stack?.join(", ") || "None"}</p>
                <p className="break-all"><span className="font-medium text-[var(--color-text-primary)]">Model Overrides:</span> {JSON.stringify(camera.capability_model_overrides || {})}</p>
                <p className="break-all"><span className="font-medium text-[var(--color-text-primary)]">Schedule Telemetry:</span> {JSON.stringify(camera.scheduleTelemetry || {})}</p>
              </div>
            </details>
          )}
        </div>
      </div>
    </div>
  )
}

function summarizeDetectionSchedule(camera: Camera): { windowText: string; suppressedLabels: string[] } {
  const windows = normalizeCapabilityWindows(camera.capability_windows || camera.execution_plan?.capability_windows)
  const suppressed = getSuppressedCapabilities(camera)
  return {
    windowText: windows.length ? windows.map(formatCapabilityWindow).join("; ") : "Always active",
    suppressedLabels: getDetectionLabelsFromKeys(suppressed),
  }
}

function getSuppressedCapabilities(camera: Camera): CapabilityKey[] {
  const raw =
    camera.scheduleTelemetry?.scheduleState?.suppressedCapabilities ||
    camera.scheduleTelemetry?.suppressedCapabilities ||
    camera.execution_plan?.suppressed_capabilities ||
    []
  return raw.filter((item): item is CapabilityKey => Boolean(item))
}

function formatCapabilityWindow(window: CapabilityWindow): string {
  const labels = getDetectionLabelsFromKeys(window.capabilities || [])
  const detectionText = labels.length ? labels.join(", ") : "Selected detections"
  const rangeText = (window.windows || []).map((item) => {
    const days = item.days?.length ? item.days.map(formatDay).join("/") : "daily"
    return `${days} ${item.from}-${item.to}`
  })
  return `${detectionText}: ${rangeText.join(", ") || "scheduled"}`
}

function formatDay(value: string): string {
  const labels: Record<string, string> = {
    mon: "Mon",
    tue: "Tue",
    wed: "Wed",
    thu: "Thu",
    fri: "Fri",
    sat: "Sat",
    sun: "Sun",
  }
  return labels[value] || value
}

function cameraUsesTrainedPpeModel(camera: Camera): boolean {
  const overrides = camera.capability_model_overrides || camera.execution_plan?.capability_model_overrides || {}
  return (
    overrides.apron_required === "ppe_closed_set_candidate" ||
    overrides.harness_required === "ppe_closed_set_candidate"
  )
}
