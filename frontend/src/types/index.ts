// Canonical type definitions for the SafetyLens frontend

export type Severity = "P1" | "P2" | "P3" | "P4"
export type AlertStatus = "active" | "acknowledged" | "resolved" | "snoozed"
export type CameraStatus = "online" | "offline" | "error"

export interface Alert {
  id: string
  severity: Severity
  status: AlertStatus
  rule: string
  cameraId: string
  cameraName: string
  zone: string
  confidence: number
  timestamp: string
  source: string
  description: string
  snapshotUrl: string | null
  cleanSnapshotUrl: string | null
  bboxes: Array<{ label: string; bbox: [number, number, number, number]; confidence: number }>
  acknowledgedBy?: string | null
  acknowledgedAt?: string | null
  resolvedAt?: string | null
  snoozedUntil?: string | null
  falsePositive?: boolean
}

export interface Camera {
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
  ppe_rule_ids?: string[]
  safety_rule_ids?: string[]
}

export interface SafetyRule {
  id: string
  name: string
  type: "ppe" | "alert"
  classes: string[]
  model: "yolo" | "yoloe"
  severity: Severity
  enabled: boolean
}

export interface PPERule {
  id: string
  name: string
  yoloe_classes: string[]
  severity: Severity
  enabled: boolean
}

export interface DetectionRule {
  id: string
  name: string
  model: string
  promptType: string
  prompts: string[]
  confidenceThreshold: number
  severity: Severity
  enabled: boolean
  camerasCount: number
  category: string
}

export interface User {
  id: string
  username: string
  role: "admin" | "operator" | "viewer"
  status: string
  mustChangePassword: boolean
  createdAt: string
  lastLogin: string | null
}
