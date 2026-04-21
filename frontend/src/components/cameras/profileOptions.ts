import type { CameraProfile } from "@/types"
import type { CameraDetectionKey } from "./detectionCatalog"

export const PROFILE_OPTIONS: Array<{ key: CameraProfile; label: string; description: string }> = [
  {
    key: "general_safety",
    label: "General Safety",
    description: "Default safety monitoring for people, phones, vehicles, animals, and zones.",
  },
  {
    key: "work_zone_ppe",
    label: "Work Zone / PPE",
    description: "PPE-focused cameras that still need person and zone context.",
  },
  {
    key: "demo_advanced",
    label: "Demo / Advanced",
    description: "Advanced cameras that use long-tail detections such as fire and smoke.",
  },
]

export const PROFILE_DEFAULTS: Record<CameraProfile, CameraDetectionKey[]> = {
  general_safety: ["person_presence", "mobile_phone"],
  work_zone_ppe: ["helmet_required", "vest_required", "zone_intrusion"],
  demo_advanced: ["fire_smoke"],
}
