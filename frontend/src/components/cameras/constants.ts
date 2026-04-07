export type CameraRole = "general" | "gate_anpr_face" | "work_zone_ppe" | "manual"

export const CAMERA_ROLES: { value: CameraRole; label: string; models: string[]; gpuImpact: string }[] = [
  { value: "general", label: "General", models: ["YOLO26n", "YOLOE-11s"], gpuImpact: "~5ms/frame" },
  { value: "gate_anpr_face", label: "Gate (ANPR + Face)", models: ["YOLO26n", "YOLOE-11s", "YOLO26s", "PaddleOCR", "SCRFD", "ArcFace"], gpuImpact: "~18ms/frame" },
  { value: "work_zone_ppe", label: "Work Zone (PPE)", models: ["YOLO26n", "YOLOE-11s"], gpuImpact: "~5ms/frame" },
  { value: "manual", label: "Manual — configure via Rules", models: [], gpuImpact: "varies" },
]


export const statusVariant: Record<string, "success" | "critical" | "default"> = {
  online: "success",
  error: "critical",
  offline: "default",
}
