export type Severity = "P1" | "P2" | "P3" | "P4"
export type AlertStatus = "active" | "acknowledged" | "resolved" | "snoozed"
export type CameraStatus = "online" | "offline" | "error"

export interface Camera {
  id: string
  name: string
  zone: string
  status: CameraStatus
  rtspUrl: string
  rulesCount: number
  thumbnail: string
  assignedRuleIds: string[]
}

export interface PipelineStage {
  id: string
  name: string
  model: string
  description: string
  fps: string
  runsOn: string
  color: string
  rules: string[]
}

export interface Alert {
  id: string
  severity: Severity
  status: AlertStatus
  rule: string
  cameraId: string
  cameraName: string
  zone: string
  confidence: number
  timestamp: Date
  acknowledgedBy?: string
  acknowledgedAt?: Date
  thumbnail: string
}

export interface DetectionRule {
  id: string
  name: string
  model: "YOLOE" | "YOLO26" | "YOLO-pose" | "VLM"
  promptType: "text" | "visual" | "internal"
  prompts: string[]
  confidenceThreshold: number
  severity: Severity
  enabled: boolean
  camerasCount: number
  category: string
}

export interface Zone {
  id: string
  name: string
  riskLevel: "hazardous" | "caution" | "general"
  camerasCount: number
  activeRules: number
  color: string
}

export interface KPI {
  label: string
  value: string | number
  change: number
  changeLabel: string
}

const PLACEHOLDER_IMG = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='320' height='240' fill='%23f5f5f5'%3E%3Crect width='320' height='240'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' dy='.3em' fill='%23a3a3a3' font-family='system-ui' font-size='14'%3ECamera Feed%3C/text%3E%3C/svg%3E"

export const zones: Zone[] = [
  { id: "z1", name: "Welding Bay", riskLevel: "hazardous", camerasCount: 4, activeRules: 6, color: "#dc2626" },
  { id: "z2", name: "Battery Room", riskLevel: "hazardous", camerasCount: 3, activeRules: 7, color: "#dc2626" },
  { id: "z3", name: "Assembly Line", riskLevel: "caution", camerasCount: 5, activeRules: 5, color: "#f59e0b" },
  { id: "z4", name: "Loading Dock", riskLevel: "caution", camerasCount: 3, activeRules: 4, color: "#f59e0b" },
  { id: "z5", name: "Testing Area", riskLevel: "caution", camerasCount: 3, activeRules: 5, color: "#f59e0b" },
  { id: "z6", name: "Main Entrance", riskLevel: "general", camerasCount: 2, activeRules: 3, color: "#059669" },
]

export const cameras: Camera[] = [
  { id: "c1", name: "Welding Bay - Entry", zone: "Welding Bay", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.101:554/stream1", rulesCount: 5, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r3", "r6", "r10"] },
  { id: "c2", name: "Welding Bay - Overview", zone: "Welding Bay", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.102:554/stream1", rulesCount: 6, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r3", "r5", "r6", "r10"] },
  { id: "c3", name: "Welding Bay - Workstation A", zone: "Welding Bay", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.103:554/stream1", rulesCount: 4, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r6", "r10"] },
  { id: "c4", name: "Welding Bay - Workstation B", zone: "Welding Bay", status: "offline", rtspUrl: "rtsp://admin:pass@192.168.1.104:554/stream1", rulesCount: 4, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r6", "r10"] },
  { id: "c5", name: "Battery Room - Charging", zone: "Battery Room", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.105:554/stream1", rulesCount: 7, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r3", "r4", "r6", "r7", "r10"] },
  { id: "c6", name: "Battery Room - Assembly", zone: "Battery Room", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.106:554/stream1", rulesCount: 6, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r4", "r6", "r7", "r10"] },
  { id: "c7", name: "Battery Room - Storage", zone: "Battery Room", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.107:554/stream1", rulesCount: 5, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r6", "r8", "r10"] },
  { id: "c8", name: "Assembly Line - Section 1", zone: "Assembly Line", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.108:554/stream1", rulesCount: 5, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r6", "r7", "r10"] },
  { id: "c9", name: "Assembly Line - Section 2", zone: "Assembly Line", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.109:554/stream1", rulesCount: 5, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r6", "r7", "r10"] },
  { id: "c10", name: "Assembly Line - Section 3", zone: "Assembly Line", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.110:554/stream1", rulesCount: 4, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r6", "r10"] },
  { id: "c11", name: "Assembly Line - QC Station", zone: "Assembly Line", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.111:554/stream1", rulesCount: 3, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r6"] },
  { id: "c12", name: "Assembly Line - Packing", zone: "Assembly Line", status: "error", rtspUrl: "rtsp://admin:pass@192.168.1.112:554/stream1", rulesCount: 4, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r6", "r9"] },
  { id: "c13", name: "Loading Dock - Bay 1", zone: "Loading Dock", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.113:554/stream1", rulesCount: 4, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r4", "r12"] },
  { id: "c14", name: "Loading Dock - Bay 2", zone: "Loading Dock", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.114:554/stream1", rulesCount: 4, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r4", "r12"] },
  { id: "c15", name: "Loading Dock - Overview", zone: "Loading Dock", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.115:554/stream1", rulesCount: 3, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r8"] },
  { id: "c16", name: "Testing Area - Motor Bay", zone: "Testing Area", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.116:554/stream1", rulesCount: 5, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r5", "r6", "r10"] },
  { id: "c17", name: "Testing Area - Drive Bay", zone: "Testing Area", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.117:554/stream1", rulesCount: 5, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r5", "r6", "r10"] },
  { id: "c18", name: "Testing Area - High Voltage", zone: "Testing Area", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.118:554/stream1", rulesCount: 6, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r2", "r3", "r5", "r6", "r10"] },
  { id: "c19", name: "Main Entrance - Gate", zone: "Main Entrance", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.119:554/stream1", rulesCount: 3, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r8", "r10"] },
  { id: "c20", name: "Main Entrance - Lobby", zone: "Main Entrance", status: "online", rtspUrl: "rtsp://admin:pass@192.168.1.120:554/stream1", rulesCount: 2, thumbnail: PLACEHOLDER_IMG, assignedRuleIds: ["r1", "r10"] },
]

const rules = ["No Helmet", "No Safety Vest", "Zone Intrusion", "Person Fall", "Mobile Phone Usage", "No Safety Goggles", "Animal Detected", "Gangway Blocked", "No Harness", "Fire/Smoke"]
const severities: Severity[] = ["P1", "P2", "P3"]

function randomDate(hoursBack: number): Date {
  return new Date(Date.now() - Math.random() * hoursBack * 60 * 60 * 1000)
}

export const alerts: Alert[] = Array.from({ length: 50 }, (_, i) => {
  const cam = cameras[Math.floor(Math.random() * cameras.length)]
  const rule = rules[Math.floor(Math.random() * rules.length)]
  const severity = rule === "Person Fall" || rule === "Fire/Smoke" ? "P1" : rule === "Animal Detected" || rule === "Gangway Blocked" ? "P3" : severities[Math.floor(Math.random() * 2)]
  const statuses: AlertStatus[] = ["active", "active", "acknowledged", "resolved", "resolved", "snoozed"]
  const status = i < 5 ? "active" : statuses[Math.floor(Math.random() * statuses.length)]
  return {
    id: `alert-${i + 1}`,
    severity,
    status,
    rule,
    cameraId: cam.id,
    cameraName: cam.name,
    zone: cam.zone,
    confidence: 0.6 + Math.random() * 0.35,
    timestamp: randomDate(24),
    acknowledgedBy: status === "acknowledged" || status === "resolved" ? "Ravi K." : undefined,
    thumbnail: PLACEHOLDER_IMG,
  }
}).sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())

export const detectionRules: DetectionRule[] = [
  { id: "r1", name: "Hard Hat Detection", model: "YOLOE", promptType: "text", prompts: ["hard hat", "safety helmet"], confidenceThreshold: 0.5, severity: "P2", enabled: true, camerasCount: 16, category: "PPE" },
  { id: "r2", name: "Safety Vest Detection", model: "YOLOE", promptType: "text", prompts: ["safety vest", "high visibility vest"], confidenceThreshold: 0.5, severity: "P2", enabled: true, camerasCount: 18, category: "PPE" },
  { id: "r3", name: "Safety Goggles", model: "YOLOE", promptType: "text", prompts: ["safety goggles", "protective eyewear"], confidenceThreshold: 0.45, severity: "P2", enabled: true, camerasCount: 6, category: "PPE" },
  { id: "r4", name: "Safety Harness", model: "YOLOE", promptType: "visual", prompts: ["harness-ref.jpg"], confidenceThreshold: 0.5, severity: "P2", enabled: true, camerasCount: 4, category: "PPE" },
  { id: "r5", name: "Zone Intrusion", model: "YOLO26", promptType: "internal", prompts: ["person"], confidenceThreshold: 0.6, severity: "P1", enabled: true, camerasCount: 8, category: "Zone Safety" },
  { id: "r6", name: "Person Fall Detection", model: "YOLO-pose", promptType: "internal", prompts: ["pose-keypoints"], confidenceThreshold: 0.55, severity: "P1", enabled: true, camerasCount: 20, category: "Emergency" },
  { id: "r7", name: "Mobile Phone Usage", model: "YOLO26", promptType: "internal", prompts: ["cell phone"], confidenceThreshold: 0.6, severity: "P2", enabled: true, camerasCount: 14, category: "Behavior" },
  { id: "r8", name: "Animal Detection", model: "YOLOE", promptType: "text", prompts: ["snake", "dog", "cat"], confidenceThreshold: 0.4, severity: "P3", enabled: true, camerasCount: 6, category: "Environment" },
  { id: "r9", name: "Gangway Blockage", model: "VLM", promptType: "text", prompts: ["Is the gangway clear and unobstructed?"], confidenceThreshold: 0.5, severity: "P3", enabled: true, camerasCount: 4, category: "Environment" },
  { id: "r10", name: "Fire / Smoke Detection", model: "YOLOE", promptType: "text", prompts: ["fire", "smoke", "flames"], confidenceThreshold: 0.35, severity: "P1", enabled: true, camerasCount: 20, category: "Emergency" },
  { id: "r11", name: "Head Cap vs Helmet", model: "YOLOE", promptType: "text", prompts: ["hair net", "head cap", "hard hat"], confidenceThreshold: 0.45, severity: "P2", enabled: false, camerasCount: 0, category: "PPE" },
  { id: "r12", name: "Forklift Operator Helmet", model: "YOLOE", promptType: "text", prompts: ["forklift", "hard hat"], confidenceThreshold: 0.5, severity: "P2", enabled: true, camerasCount: 5, category: "PPE" },
]

export const kpis: KPI[] = [
  { label: "Incidents Today", value: 14, change: -42, changeLabel: "vs yesterday" },
  { label: "Compliance Rate", value: "94.2%", change: 2.1, changeLabel: "vs last shift" },
  { label: "Active Cameras", value: "18 / 20", change: 0, changeLabel: "2 offline" },
  { label: "Avg Response Time", value: "2m 14s", change: -18, changeLabel: "vs last shift" },
]

export const complianceByZoneShift = [
  { zone: "Welding Bay", day: 92, evening: 88, night: 78 },
  { zone: "Battery Room", day: 96, evening: 94, night: 91 },
  { zone: "Assembly Line", day: 97, evening: 95, night: 93 },
  { zone: "Loading Dock", day: 94, evening: 90, night: 85 },
  { zone: "Testing Area", day: 98, evening: 96, night: 94 },
  { zone: "Main Entrance", day: 99, evening: 98, night: 97 },
]

export const violationTrend = Array.from({ length: 24 }, (_, i) => ({
  hour: `${String(i).padStart(2, "0")}:00`,
  "No Helmet": Math.floor(Math.random() * 8),
  "No Vest": Math.floor(Math.random() * 5),
  "Zone Intrusion": Math.floor(Math.random() * 3),
  "Other": Math.floor(Math.random() * 4),
}))

export const severityConfig: Record<Severity, { label: string; color: string; bg: string; textColor: string }> = {
  P1: { label: "Critical", color: "#dc2626", bg: "#fef2f2", textColor: "#991b1b" },
  P2: { label: "High", color: "#f97316", bg: "#fff7ed", textColor: "#9a3412" },
  P3: { label: "Medium", color: "#f59e0b", bg: "#fffbeb", textColor: "#92400e" },
  P4: { label: "Low", color: "#2563eb", bg: "#eff6ff", textColor: "#1e40af" },
}

export const pipelineStages: PipelineStage[] = [
  {
    id: "yolo26",
    name: "YOLO26",
    model: "YOLO26n (fine-tuned)",
    description: "Always-on real-time detector for trained COCO classes. Runs every frame.",
    fps: "29 FPS",
    runsOn: "Jetson GPU (TensorRT)",
    color: "#059669",
    rules: ["Mobile Phone Usage", "Person Fall Detection", "Zone Intrusion", "Forklift Operator Helmet"],
  },
  {
    id: "yoloe",
    name: "YOLOE",
    model: "YOLOE (open-vocab)",
    description: "Zero-shot detection via text/visual prompts. No training needed. Handles novel objects.",
    fps: "161 FPS (T4)",
    runsOn: "Jetson GPU",
    color: "#2563eb",
    rules: ["Hard Hat Detection", "Safety Vest Detection", "Safety Goggles", "Safety Harness", "Animal Detection", "Fire / Smoke Detection", "Head Cap vs Helmet"],
  },
  {
    id: "yolo-pose",
    name: "YOLO-pose",
    model: "YOLO26n-pose",
    description: "Keypoint-based pose estimation. Detects falls via shoulder/knee geometry.",
    fps: "25 FPS",
    runsOn: "Jetson GPU",
    color: "#f97316",
    rules: ["Person Fall Detection"],
  },
  {
    id: "vlm",
    name: "VLM (qwen3-vl)",
    model: "qwen3-vl-flash / plus",
    description: "Scene reasoning for spatial understanding, ambiguous cases, and incident investigation. Triggered by YOLO or periodic scan.",
    fps: "~1-5s/frame",
    runsOn: "Cloud API (Alibaba DashScope)",
    color: "#8b5cf6",
    rules: ["Gangway Blockage", "Drugs/Syringes (escalation)", "Incident Investigation (why unsafe?)"],
  },
]

export const pipelineFlow = [
  { from: "Camera Frame", to: "yolo26", label: "Every frame" },
  { from: "Camera Frame", to: "yoloe", label: "Every frame (text/visual prompts)" },
  { from: "Camera Frame", to: "yolo-pose", label: "Every frame (pose)" },
  { from: "yoloe", to: "vlm", label: "Low confidence → escalate" },
  { from: "Camera Frame", to: "vlm", label: "Every 60s (periodic scan)" },
  { from: "yolo26", to: "Alert", label: "Detection → alert logic" },
  { from: "yoloe", to: "Alert", label: "Detection → alert logic" },
  { from: "yolo-pose", to: "Alert", label: "Fall detected → P1 alert" },
  { from: "vlm", to: "Alert", label: "Reasoning → alert" },
]
