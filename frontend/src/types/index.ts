// Canonical type definitions for the Rakshak Lens frontend

export type Severity = "P1" | "P2" | "P3" | "P4"
export type AlertStatus = "active" | "acknowledged" | "resolved" | "snoozed"
export type CameraStatus = "online" | "offline" | "error"
export type CameraProfile = "general_safety" | "work_zone_ppe" | "office_occupancy" | "demo_advanced"
export type CapabilityKey =
  | "person_presence"
  | "office_occupancy"
  | "crowd_count_threshold"
  | "queue_monitoring"
  | "route_obstruction"
  | "object_lifecycle"
  | "vehicle_presence"
  | "animal_presence"
  | "mobile_phone"
  | "zone_intrusion"
  | "helmet_required"
  | "rider_helmet_required"
  | "vest_required"
  | "gloves_required"
  | "hairnet_required"
  | "face_mask_required"
  | "face_shield_required"
  | "apron_required"
  | "boots_required"
  | "harness_required"
  | "goggles_required"
  | "fire_smoke"
  | "face_recognition"
  | "plate_recognition"
  | "fall_detection"
  | "custom_long_tail"
export type CameraRuntimeStatus = "running" | "starting" | "awaiting_model_install" | "offline" | "error"
export type ModelKey = "coco_primary" | "ppe_specialist" | "ppe_closed_set_candidate" | "yoloe_long_tail" | "fire_smoke_specialist" | "face_recognition" | "pose_specialist" | "plate_recognition"

export interface RuleScheduleWindow {
  days?: string[]
  from: string
  to: string
}

export interface CapabilityWindow {
  id?: string
  capabilities: CapabilityKey[]
  mode: "detection" | "detector" | "detector_off" | "alert_policy"
  windows: RuleScheduleWindow[]
  active?: boolean
}

export interface ExecutionPlan {
  profile: CameraProfile
  capabilities: CapabilityKey[]
  required_model_keys: ModelKey[]
  capability_model_overrides?: Partial<Record<CapabilityKey, ModelKey>>
  run_coco_primary: boolean
  run_ppe_specialist: boolean
  run_ppe_closed_set_candidate?: boolean
  run_yoloe_long_tail: boolean
  run_fire_smoke_specialist?: boolean
  run_face_recognition?: boolean
  run_pose_specialist?: boolean
  run_plate_recognition?: boolean
  tracking_enabled: boolean
  zones_required: boolean
  association_enabled: boolean
  runtime_load: "low" | "medium" | "high"
  ppe_prompt_terms: string[]
  yoloe_prompt_terms: string[]
  capability_windows?: CapabilityWindow[]
  active_capabilities?: CapabilityKey[]
  suppressed_capabilities?: CapabilityKey[]
  derived_demo: string
  model_stack: string[]
}

export interface CameraScheduleTelemetry {
  scheduleState?: {
    suppressedCapabilities?: CapabilityKey[]
    activeCapabilities?: CapabilityKey[]
    evaluatedAt?: string | null
  }
  activeCapabilities?: CapabilityKey[]
  suppressedCapabilities?: CapabilityKey[]
  modelInvocations?: Record<string, number>
  [key: string]: any
}

export interface ModelStatus {
  model_key: ModelKey
  display_name: string
  filename: string
  local_path: string
  active_path: string | null
  download_url: string
  warmup_behavior: string
  status: "ready" | "not_downloaded" | "installing" | "failed" | "remote_unavailable"
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
  deliveryResults?: AlertDeliveryResult[]
  policyId?: string | null
  priority?: number | null
  message?: string | null
  metadata?: Record<string, any>
  acknowledgedBy?: string | null
  acknowledgedAt?: string | null
  resolvedAt?: string | null
  snoozedUntil?: string | null
  falsePositive?: boolean
}

export type AlertOutputType =
  | "in_app"
  | "browser_sound"
  | "telegram"
  | "email"
  | "webhook"
  | "pushover"
  | "ip_speaker"
  | "relay"
  | "plc"

export type AlertOutputStatus =
  | "ready"
  | "needs_setup"
  | "simulated"
  | "failed"
  | "disabled"
  | "not_implemented"

export type AlertDeliveryStatus = "delivered" | "simulated" | "skipped" | "failed"

export interface AlertDeliveryResult {
  id: string
  outputId: string
  outputName: string
  type: AlertOutputType
  status: AlertDeliveryStatus
  message: string
  timestamp: string
  details?: Record<string, any>
}

export interface AlertOutput {
  id: string
  name: string
  type: AlertOutputType
  enabled: boolean
  severities: Severity[]
  zones: string[]
  mode: string
  status: AlertOutputStatus
  lastTestAt?: string | null
  lastFiredAt?: string | null
  lastError?: string
  settings: Record<string, any>
}

export interface Camera {
  id: string
  name: string
  zone: string
  profile: CameraProfile
  capabilities: CapabilityKey[]
  capability_model_overrides?: Partial<Record<CapabilityKey, ModelKey>>
  execution_plan: ExecutionPlan
  runtime_status: CameraRuntimeStatus
  demo: string
  rules: string[]
  status: string
  detectionsCount: number
  video: string
  fps: number
  inference_fps: number
  inference_fps_source?: "camera" | "global"
  enabled: boolean
  yoloe_classes?: string[] | null
  stream_type: string
  zones?: Zone[] | null
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
  safety_rule_overrides?: Record<string, { confidence?: number | null; threshold?: number | null }> | null
  custom_long_tail_terms?: string[] | null
  capability_windows?: CapabilityWindow[] | null
  scheduleTelemetry?: CameraScheduleTelemetry
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
  analytics?: string | null
  classes?: string[] | null
  threshold?: number | null
  min_duration_seconds?: number | null
  area_square_meters?: number | null
  severity_thresholds?: Record<string, number> | null
}

export interface SafetyRule {
  id: string
  name: string
  type: "ppe" | "alert"
  classes: string[]
  model: "yolo" | "yoloe" | "fire_smoke_specialist" | "pose"
  severity: Severity
  enabled: boolean
  threshold?: number | null
  confidence?: number | null
}

export interface RuleCondition {
  type: string
  params: Record<string, string>
}

export interface RuleAction {
  type: string
  params: Record<string, string>
}

export interface RuleSchedule {
  timezone?: string
  windows: RuleScheduleWindow[]
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
  severity?: Severity | null
  outputIds?: string[]
  messageTemplate?: string
  schedule?: RuleSchedule | null
  lastTriggered: string | null
  preset: string | null
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
