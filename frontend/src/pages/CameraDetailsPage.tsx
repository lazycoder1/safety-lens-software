import { useEffect, useMemo, useState } from "react"
import { Link, useNavigate, useParams } from "react-router-dom"
import { AlertTriangle, ArrowLeft, Trash2 } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { deleteCamera, getCameraById, getSafetyRules, getZones, API_BASE } from "@/lib/api"
import { useAuthStore } from "@/stores/authStore"
import type { Camera, SafetyRule, Zone } from "@/types"
import { statusVariant } from "@/components/cameras/constants"
import {
  cameraHasDetectionModeMismatch,
  cameraNeedsLegacyNormalization,
  deriveConfiguredCameraPurpose,
  getConfiguredDetectionKeys,
  getConfiguredDetectionLabels,
  usesZoneIntrusion,
} from "@/components/cameras/detectionCatalog"
import { InfoRow } from "@/components/cameras/helpers"

export function CameraDetailsPage() {
  const navigate = useNavigate()
  const params = useParams()
  const userRole = useAuthStore((state) => state.user?.role)
  const cameraId = params.cameraId || ""
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState(false)
  const [camera, setCamera] = useState<Camera | null>(null)
  const [safetyRules, setSafetyRules] = useState<SafetyRule[]>([])
  const [zones, setZones] = useState<Zone[]>([])

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const [cameraData, ruleData] = await Promise.all([getCameraById(cameraId), getSafetyRules()])
        if (cancelled) return
        const zoneData = await getZones(cameraData.id).catch(() => [])
        if (cancelled) return
        setCamera(cameraData)
        setSafetyRules(ruleData)
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
  const requiresZone = usesZoneIntrusion(detectionKeys)
  const statusVariantValue = camera ? statusVariant[camera.status] || "default" : "default"
  const isAdmin = userRole === "admin"
  const showLegacyNote = camera ? cameraNeedsLegacyNormalization(camera) : false
  const showModeMismatch = camera ? cameraHasDetectionModeMismatch(camera, safetyRules) : false

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
              src={`${API_BASE}/api/stream/${camera.id}`}
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

          <Card className="space-y-4">
            <div>
              <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Execution Plan</h2>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                Grouped model stack and runtime requirements for this camera.
              </p>
            </div>
            <div className="space-y-3">
              <InfoRow label="Expected load" value={camera.execution_plan?.runtime_load || "unknown"} />
              <InfoRow
                label="Tracking"
                value={camera.execution_plan?.tracking_enabled ? "Enabled" : "Not required"}
              />
              <InfoRow
                label="Association"
                value={camera.execution_plan?.association_enabled ? "Enabled" : "Not required"}
              />
              <InfoRow
                label="Models"
                value={camera.execution_plan?.model_stack?.join(", ") || "No model stack"}
              />
            </div>
          </Card>

          <Card className="space-y-4">
            <div>
              <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Zones</h2>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                Restricted area requirements for this camera.
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
                <p><span className="font-medium text-[var(--color-text-primary)]">Detection Engine:</span> {camera.demo}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Runtime Status:</span> {camera.runtime_status}</p>
                <p className="break-all"><span className="font-medium text-[var(--color-text-primary)]">Rule IDs:</span> {(camera.safety_rule_ids || []).join(", ") || "None"}</p>
              </div>
            </details>
          )}
        </div>
      </div>
    </div>
  )
}
