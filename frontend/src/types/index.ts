// Canonical type definitions for the SafetyLens frontend

export type Severity = "P1" | "P2" | "P3" | "P4"
export type AlertStatus = "active" | "acknowledged" | "resolved" | "snoozed"
export type CameraStatus = "online" | "offline" | "error"
export type CameraProfile = "general_safety" | "work_zone_ppe" | "demo_advanced"
export type CapabilityKey =
  | "person_presence"
  | "vehicle_presence"
  | "animal_presence"
  | "mobile_phone"
  | "zone_intrusion"
  | "helmet_required"
  | "vest_required"
  | "gloves_required"
  | "hairnet_required"
  | "face_mask_required"
  | "apron_required"
  | "boots_required"
  | "goggles_required"
  | "fire_smoke"
  | "custom_long_tail"
export type CameraRuntimeStatus = "running" | "starting" | "awaiting_model_install" | "offline" | "error"
export type ModelKey = "coco_primary" | "ppe_specialist" | "yoloe_long_tail"

export interface ExecutionPlan {
  profile: CameraProfile
  capabilities: CapabilityKey[]
  required_model_keys: ModelKey[]
  run_coco_primary: boolean
  run_ppe_specialist: boolean
  run_yoloe_long_tail: boolean
  tracking_enabled: boolean
  zones_required: boolean
  association_enabled: boolean
  runtime_load: "low" | "medium" | "high"
  ppe_prompt_terms: string[]
  yoloe_prompt_terms: string[]
  derived_demo: string
  model_stack: string[]
}

export interface ModelStatus {
  model_key: ModelKey
  display_name: string
  filename: string
  local_path: string
  active_path: string | null
  download_url: string
  warmup_behavior: string
  status: "ready" | "not_downloaded" | "installing" | "failed"
  error: string | null
  job_id: string | null
  shared_asset_key: string
  is_ready: boolean
  is_downloaded: boolean
}

export interface ModelInstallJob {
  id: string
  model_keys: ModelKey[]
  asset_keys: string[]
  status: "queued" | "running" | "ready" | "failed"
  stage: "queued" | "checking_disk" | "downloading" | "verifying" | "preparing_assets" | "loading" | "warming_up" | "ready" | "failed"
  current_model_key: ModelKey
  progress_percent: number
  bytes_downloaded: number
  total_bytes: number | null
  error: string | null
  created_at: number
  updated_at: number
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
  profile: CameraProfile
  capabilities: CapabilityKey[]
  execution_plan: ExecutionPlan
  runtime_status: CameraRuntimeStatus
  demo: string
  rules: string[]
  status: string
  detectionsCount: number
  video: string
  fps: number
  enabled: boolean
  yoloe_classes?: string[] | null
  stream_type: string
  rtsp_url: string
  host?: string | null
  rtsp_port?: number | null
  onvif_port?: number | null
  stream_path?: string | null
  preferred_stream?: string | null
  onvif_uuid?: string | null
  discovery_fingerprint?: string | null
  credentials_configured?: boolean
  connection_summary?: string | null
  alert_classes?: string[] | null
  ppe_rule_ids?: string[] | null
  safety_rule_ids?: string[] | null
  custom_long_tail_terms?: string[] | null
}

export interface DiscoveredCamera {
  fingerprint: string
  source?: string
  ip: string
  host: string
  name: string
  vendor: string
  model: string
  location?: string
  onvif_uuid?: string | null
  onvif_xaddr?: string | null
  onvif_port?: number | null
  rtsp_port?: number | null
  stream_path?: string
  stream_candidates?: Array<{ label: string; name?: string; path: string }>
  recommended_stream?: string
  auth_state?: string
  duplicate_state?: "none" | "exact" | "potential"
  existing_camera_id?: string | null
}

export type DiscoveryCredentialMode = "inherit" | "override"

export interface DiscoveryRowOverride {
  displayName: string
  zone: string
  profile: CameraProfile
  capabilities: CapabilityKey[]
  preferredStream: "main" | "sub" | "mjpeg" | "custom"
  streamPath: string
  credentialMode: DiscoveryCredentialMode
  username: string
  password: string
}

export interface DiscoveryTestResult {
  ok: boolean
  auth_state: string
  error?: string | null
  error_code?: string | null
  host: string
  rtsp_port?: number | null
  onvif_port?: number | null
  stream_path?: string
  preferred_stream?: string
  stream_candidates?: Array<{ label: string; name?: string; path: string }>
  recommended_stream?: string
  preview_data_url?: string
  latency_ms?: number
  width?: number
  height?: number
  onvif_uuid?: string | null
  discovery_fingerprint?: string | null
  vendor?: string
  model?: string
  name?: string
  duplicate_state?: "none" | "exact" | "potential"
  existing_camera_id?: string | null
}

export interface DiscoveryImportResult {
  created: Array<{
    camera_id: string
    name: string
    zone: string
    needs_zone_setup: boolean
  }>
  failed: Array<{
    fingerprint: string
    error_code: string
    error: string
    existing_camera_id?: string | null
  }>
  needs_zone_setup: Array<{
    camera_id: string
    name: string
    zone: string
    needs_zone_setup: boolean
  }>
}

export interface Zone {
  id: string
  name: string
  type: string
  color: string
  points: number[][]
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

export interface RuleCondition {
  type: string
  params: Record<string, string>
}

export interface RuleAction {
  type: string
  params: Record<string, string>
}

export interface EngineRule {
  id: string
  name: string
  description: string
  enabled: boolean
  trigger: string
  cameras: string[]
  conditions: RuleCondition[]
  thenActions: RuleAction[]
  elseActions: RuleAction[]
  cooldownSeconds: number
  priority: number
  lastTriggered: string | null
  preset: string | null
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
