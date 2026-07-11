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
    key: "office_occupancy",
    label: "Office Occupancy",
    description: "Chair and desk occupancy monitoring with optional phone usage checks.",
  },
  {
    key: "demo_advanced",
    label: "Demo / Advanced",
    description: "Advanced cameras for fire, smoke, Fall / Man Down, and custom detections.",
  },
]

export const PROFILE_DEFAULTS: Record<CameraProfile, CameraDetectionKey[]> = {
  general_safety: ["person_presence", "mobile_phone"],
  work_zone_ppe: ["helmet_required", "vest_required", "zone_intrusion"],
  office_occupancy: ["office_occupancy", "mobile_phone"],
  demo_advanced: ["fire_smoke", "fall_detection"],
}
