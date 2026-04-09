import { Trash2 } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import type { Camera, SafetyRule } from "@/types"
import { CAMERA_ROLES, statusVariant } from "./constants"
import type { CameraRole } from "./constants"

interface CameraCardProps {
  camera: Camera
  role: CameraRole
  safetyRules: SafetyRule[]
  onClick: () => void
  onDelete: () => void
}

export function CameraCard({ camera, role, safetyRules, onClick, onDelete }: CameraCardProps) {
  const roleInfo = CAMERA_ROLES.find((r) => r.value === role) || CAMERA_ROLES[0]
  const variant = statusVariant[camera.status] || "default"
  const assignedRules = safetyRules.filter((r) => (camera.safety_rule_ids || []).includes(r.id))

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
        {assignedRules.length > 0 ? (
          <div className="flex flex-wrap gap-1">
            {assignedRules.slice(0, 3).map((rule) => (
              <Badge key={rule.id} variant={rule.type === "ppe" ? "info" : "warning"}>
                {rule.name}
              </Badge>
            ))}
            {assignedRules.length > 3 && (
              <Badge>+{assignedRules.length - 3}</Badge>
            )}
          </div>
        ) : (
          <p className="text-xs text-[var(--color-text-tertiary)] italic">No safety rules</p>
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
