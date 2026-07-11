import { Eye, Pencil, Trash2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import type { Camera, SafetyRule } from "@/types"
import { statusVariant } from "./constants"
import {
  cameraHasDetectionModeMismatch,
  cameraNeedsLegacyNormalization,
  deriveConfiguredCameraPurpose,
  getConfiguredDetectionLabels,
  getConfiguredDetectionKeys,
  usesConfiguredZones,
} from "./detectionCatalog"

interface CameraCardProps {
  camera: Camera
  safetyRules: SafetyRule[]
  zoneCount: number | null
  canEdit: boolean
  onView: () => void
  onEdit: () => void
  onDelete: () => void
}

export function CameraCard({
  camera,
  safetyRules,
  zoneCount,
  canEdit,
  onView,
  onEdit,
  onDelete,
}: CameraCardProps) {
  const detectionKeys = getConfiguredDetectionKeys(camera, safetyRules)
  const purpose = deriveConfiguredCameraPurpose(camera, safetyRules)
  const summary = getConfiguredDetectionLabels(camera, safetyRules)
  const requiresZone = usesConfiguredZones(detectionKeys)
  const hasDetectorWindow = Boolean((camera.capability_windows || camera.execution_plan?.capability_windows || []).length)
  const suppressedCount = getSuppressedDetectionCount(camera)
  const trainedPpeEnabled = cameraUsesTrainedPpeModel(camera)
  const variant = statusVariant[camera.status] || "default"
  const needsReview = cameraNeedsLegacyNormalization(camera) || cameraHasDetectionModeMismatch(camera, safetyRules)
  const runtimeLabel =
    camera.runtime_status === "awaiting_model_install"
      ? "Waiting on models"
      : camera.runtime_status === "starting"
        ? "Starting"
        : camera.runtime_status

  return (
    <Card className="flex h-full flex-col gap-4">
      <div className="space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate text-base font-semibold text-[var(--color-text-primary)]">
              {camera.name}
            </p>
            <p className="mt-1 text-sm text-[var(--color-text-secondary)]">{camera.zone}</p>
          </div>
          <div className="flex flex-col items-end gap-2">
            <Badge variant={variant}>{camera.status}</Badge>
            <Badge variant={camera.runtime_status === "awaiting_model_install" ? "warning" : "default"}>
              {runtimeLabel}
            </Badge>
          </div>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">
            Purpose
          </p>
          <p className="mt-1 text-sm font-medium text-[var(--color-text-primary)]">
            {purpose}
          </p>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">
            Detection Runtime
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            <Badge variant={summary.length > 0 ? "success" : "default"}>
              {summary.length > 0 ? "Runs selected detections" : "Monitoring only"}
            </Badge>
            <Badge variant={suppressedCount > 0 ? "warning" : hasDetectorWindow ? "info" : "default"}>
              {suppressedCount > 0
                ? `${suppressedCount} inactive now`
                : hasDetectorWindow
                  ? "Scheduled"
                  : "Always active"}
            </Badge>
            {trainedPpeEnabled && <Badge variant="info">Trained PPE</Badge>}
          </div>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">
            Enabled Detections
          </p>
          {summary.length > 0 ? (
            <div className="mt-2 flex flex-wrap gap-2">
              {summary.slice(0, 3).map((label) => (
                <Badge key={label} variant="info">
                  {label}
                </Badge>
              ))}
              {summary.length > 3 && (
                <Badge variant="info">+{summary.length - 3} more</Badge>
              )}
            </div>
          ) : (
            <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
              Monitoring only
            </p>
          )}
        </div>

        {requiresZone && (
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">
              Zones
            </p>
            <div className="mt-2">
              {zoneCount && zoneCount > 0 ? (
                <Badge variant="success">
                  {zoneCount} zone{zoneCount !== 1 ? "s" : ""} configured
                </Badge>
              ) : (
                <Badge variant="warning">Zone setup needed</Badge>
              )}
            </div>
          </div>
        )}

        {canEdit && needsReview && (
          <div className="rounded-[var(--radius-md)] border border-[var(--color-warning)] bg-[var(--color-warning-bg)] px-3 py-2">
            <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-warning)]">
              Review Needed
            </p>
            <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
              Legacy camera settings need one save pass to align what is configured with what the camera runs.
            </p>
          </div>
        )}
      </div>

      <div className="mt-auto flex flex-wrap items-center gap-2 border-t pt-3">
        <Button variant="secondary" size="sm" onClick={onView}>
          <Eye className="w-3.5 h-3.5" />
          View
        </Button>
        {canEdit && (
          <>
            <Button variant="ghost" size="sm" onClick={onEdit}>
              <Pencil className="w-3.5 h-3.5" />
              Edit
            </Button>
            <Button variant="ghost" size="sm" className="text-[var(--color-critical)] hover:text-[var(--color-critical)]" onClick={onDelete}>
              <Trash2 className="w-3.5 h-3.5" />
              Delete
            </Button>
          </>
        )}
      </div>
    </Card>
  )
}

function getSuppressedDetectionCount(camera: Camera): number {
  return (
    camera.scheduleTelemetry?.scheduleState?.suppressedCapabilities ||
    camera.scheduleTelemetry?.suppressedCapabilities ||
    camera.execution_plan?.suppressed_capabilities ||
    []
  ).length
}

function cameraUsesTrainedPpeModel(camera: Camera): boolean {
  const overrides = camera.capability_model_overrides || camera.execution_plan?.capability_model_overrides || {}
  return (
    overrides.apron_required === "ppe_closed_set_candidate" ||
    overrides.harness_required === "ppe_closed_set_candidate"
  )
}
