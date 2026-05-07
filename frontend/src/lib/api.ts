export const API_BASE =
  import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8000`

export const WS_BASE =
  import.meta.env.VITE_WS_URL ||
  `ws://${window.location.hostname}:8000`

import { reportError } from "@/lib/errorReporter"

const TOKEN_KEY = "safetylens_token"

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY)
}

export class ApiError extends Error {
  status?: number
  isNetwork: boolean
  isTimeout: boolean
  payload?: any
  constructor(message: string, opts: { status?: number; isNetwork?: boolean; isTimeout?: boolean; payload?: any } = {}) {
    super(message)
    this.name = "ApiError"
    this.status = opts.status
    this.isNetwork = opts.isNetwork ?? false
    this.isTimeout = opts.isTimeout ?? false
    this.payload = opts.payload
  }
}

// Backoff delays for network-error retries. Only genuine TypeError (request
// never reached server) is retried — not timeouts or HTTP errors.
const NETWORK_RETRY_DELAYS_MS = [250, 750, 1500]
const DEFAULT_TIMEOUT_MS = 30000

async function request(path: string, options?: RequestInit & { retryOnTimeout?: boolean }) {
  const { retryOnTimeout, ...fetchOptions } = options ?? {}
  const token = getToken()
  const headers: Record<string, string> = {
    ...(fetchOptions?.headers as Record<string, string>),
  }
  if (token) {
    headers["Authorization"] = `Bearer ${token}`
  }

  const totalAttempts = NETWORK_RETRY_DELAYS_MS.length + 1
  for (let attempt = 0; attempt < totalAttempts; attempt++) {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS)
    let res: Response
    try {
      res = await fetch(`${API_BASE}${path}`, { ...fetchOptions, headers, signal: controller.signal })
    } catch (err: unknown) {
      clearTimeout(timer)
      const isAbort = (err as { name?: string })?.name === "AbortError"
      const isNetwork = err instanceof TypeError
      const isRetryable = isNetwork || (isAbort && retryOnTimeout)
      if (isAbort && !retryOnTimeout) {
        reportError(err, { url: path, extra: { type: "timeout" } })
        throw new ApiError("Server didn't respond in time. Please try again.", { isTimeout: true })
      }
      if (isRetryable && attempt < totalAttempts - 1) {
        await new Promise((r) => setTimeout(r, NETWORK_RETRY_DELAYS_MS[attempt]))
        continue
      }
      if (isAbort) {
        reportError(err, { url: path, extra: { type: "timeout_exhausted" } })
        throw new ApiError("Could not reach server. Check that the backend is running.", { isTimeout: true })
      }
      if (isNetwork) {
        reportError(err, { url: path, extra: { type: "network_exhausted" } })
        throw new ApiError("Can't reach the server. Check your connection and try again.", { isNetwork: true })
      }
      throw err
    }
    clearTimeout(timer)

    if (res.status === 401 && !path.startsWith("/api/auth/")) {
      clearToken()
      window.location.href = "/login"
      throw new ApiError("Session expired", { status: 401 })
    }
    if (!res.ok) {
      const text = await res.text().catch(() => "")
      let parsedPayload: any = null
      let message: string
      try {
        parsedPayload = JSON.parse(text)
        const candidate = parsedPayload.detail || parsedPayload.message || parsedPayload.error || ""
        message = typeof candidate === "string" ? candidate : ""
      } catch {
        message = ""
      }
      if (!message) {
        if (res.status === 401) message = "Invalid credentials or account not active"
        else if (res.status === 403) message = "You don't have permission to do that"
        else if (res.status === 404) message = "Resource not found"
        else if (res.status === 409) message = "This action conflicts with existing data"
        else if (res.status >= 500) message = "Server error — please try again later"
        else message = "Something went wrong"
      }
      const apiErr = new ApiError(message, { status: res.status, payload: parsedPayload })
      if (res.status >= 500) {
        reportError(apiErr, { url: path, requestId: res.headers.get("X-Request-ID") || undefined })
      }
      throw apiErr
    }
    return res.json()
  }
  // Unreachable — loop always returns or throws.
  throw new ApiError("Request failed", { isNetwork: true })
}

// Auth
export async function apiLogin(username: string, password: string) {
  return request("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
    retryOnTimeout: true,
  })
}

export async function apiRegister(username: string, password: string) {
  return request("/api/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  })
}

export async function apiChangePassword(currentPassword: string, newPassword: string) {
  return request("/api/auth/change-password", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ currentPassword, newPassword }),
  })
}

export async function getMe() {
  return request("/api/auth/me")
}

// Admin user management
export async function getUsers() {
  return request("/api/admin/users")
}

export async function approveUser(id: string) {
  return request(`/api/admin/users/${id}/approve`, { method: "PUT" })
}

export async function rejectUser(id: string) {
  return request(`/api/admin/users/${id}/reject`, { method: "PUT" })
}

export async function updateUserRole(id: string, role: string) {
  return request(`/api/admin/users/${id}/role`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role }),
  })
}

export async function deleteUser(id: string) {
  return request(`/api/admin/users/${id}`, { method: "DELETE" })
}

export async function resetUserPassword(id: string, newPassword?: string) {
  return request(`/api/admin/users/${id}/reset-password`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ newPassword: newPassword || null }),
  })
}

// Config
export async function getConfig() {
  return request("/api/config")
}

export async function updateGlobalConfig(settings: Record<string, any>) {
  return request("/api/config/global", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  })
}

export async function updateVlmConfig(settings: Record<string, any>) {
  return request("/api/config/vlm", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  })
}

export async function updateTelegramConfig(settings: Record<string, any>) {
  return request("/api/config/telegram", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  })
}

export async function testTelegramConfig() {
  return request("/api/config/telegram/test", { method: "POST" })
}

export async function fetchTelegramGroups(bot_token: string) {
  return request("/api/config/telegram/groups", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ bot_token }),
  })
}

export async function getAlertTimeSeries(hours: number = 24, cameraId?: string) {
  const params = new URLSearchParams({ hours: String(hours) })
  if (cameraId) params.set("cameraId", cameraId)
  return request(`/api/alerts/time-series?${params}`)
}

export async function getComplianceMetrics(hours: number = 24, cameraId?: string) {
  const params = new URLSearchParams({ hours: String(hours) })
  if (cameraId) params.set("cameraId", cameraId)
  return request(`/api/alerts/compliance?${params}`)
}

// Zones
export async function getZones(cameraId: string) {
  return request(`/api/cameras/${cameraId}/zones`)
}

export async function addZone(cameraId: string, zone: { name: string; type: string; color: string; points: number[][] }) {
  return request(`/api/cameras/${cameraId}/zones`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(zone),
  })
}

export async function updateZone(cameraId: string, zoneId: string, updates: Record<string, any>) {
  return request(`/api/cameras/${cameraId}/zones/${zoneId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  })
}

export async function deleteZone(cameraId: string, zoneId: string) {
  return request(`/api/cameras/${cameraId}/zones/${zoneId}`, { method: "DELETE" })
}

// RTSP
export async function testRtspConnection(rtspUrl: string): Promise<{
  success: boolean
  resolution: [number, number] | null
  snapshot_b64: string | null
  error: string | null
}> {
  return request("/api/rtsp/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rtsp_url: rtspUrl }),
  })
}

// Cameras
export async function getCameras() {
  return request("/api/cameras")
}

export async function getCameraById(id: string) {
  const cameras = await getCameras()
  const camera = cameras.find((item: any) => item.id === id)
  if (!camera) {
    throw new ApiError("Camera not found", { status: 404 })
  }
  return camera
}

export async function addCamera(camera: {
  name: string
  video: string
  zone: string
  profile?: string
  capabilities?: string[]
  demo: string
  rules: string[]
  fps?: number
  yoloe_classes?: string[]
  stream_type?: string
  rtsp_url?: string
  host?: string
  rtsp_port?: number | null
  onvif_port?: number | null
  stream_path?: string
  preferred_stream?: string
  username?: string
  password?: string
  onvif_uuid?: string
  discovery_fingerprint?: string
  safety_rule_ids?: string[]
  custom_long_tail_terms?: string[]
}) {
  return request("/api/cameras", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(camera),
  })
}

export async function updateCamera(id: string, updates: Record<string, any>) {
  return request(`/api/cameras/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  })
}

export async function deleteCamera(id: string) {
  return request(`/api/cameras/${id}`, { method: "DELETE" })
}

export interface EnrolledFaceApi {
  id: string
  name: string
  group: "employees" | "visitors" | "contractors" | "watchlist"
  validUntil: string | null
  enrolledAt: string
  consentMethod: string
  consentConfirmed: boolean
  photoUrl: string | null
  active: boolean
}

export interface FaceLogApi {
  id: string
  cameraId: string
  cameraName: string
  timestamp: string
  eventType: "face_match" | "face_unknown" | "face_low_quality"
  matchedFaceId: string | null
  personId: string | null
  personName: string | null
  personGroup?: string | null
  confidence: number | null
  snapshotUrl: string | null
  qualityReason: string | null
  isUnknown: boolean
}

export async function getFaces(): Promise<EnrolledFaceApi[]> {
  return request("/api/faces")
}

export async function getFaceLogs(params?: {
  cameraId?: string
  eventType?: string
  limit?: number
  offset?: number
}): Promise<FaceLogApi[]> {
  const searchParams = new URLSearchParams()
  if (params?.cameraId) searchParams.set("cameraId", params.cameraId)
  if (params?.eventType) searchParams.set("eventType", params.eventType)
  if (params?.limit) searchParams.set("limit", String(params.limit))
  if (params?.offset) searchParams.set("offset", String(params.offset))
  const qs = searchParams.toString()
  return request(`/api/faces/logs${qs ? `?${qs}` : ""}`)
}

export async function enrollFace(payload: {
  name: string
  group: string
  validUntil?: string | null
  consentMethod: string
  consentConfirmed: boolean
  photo: File
}): Promise<EnrolledFaceApi> {
  const form = new FormData()
  form.set("name", payload.name)
  form.set("group", payload.group)
  if (payload.validUntil) form.set("validUntil", payload.validUntil)
  form.set("consentMethod", payload.consentMethod)
  form.set("consentConfirmed", String(payload.consentConfirmed))
  form.set("photo", payload.photo)
  return request("/api/faces/enroll", { method: "POST", body: form })
}

export async function enrollFaceLive(payload: {
  cameraId: string
  name: string
  group: string
  validUntil?: string | null
  consentMethod: string
  consentConfirmed: boolean
}): Promise<EnrolledFaceApi> {
  return request("/api/faces/enroll/live", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
}

export async function deleteFace(id: string): Promise<EnrolledFaceApi> {
  return request(`/api/faces/${id}`, { method: "DELETE" })
}

export async function previewCameraPlan(payload: {
  name?: string
  zone?: string
  profile?: string
  capabilities?: string[]
  stream_type?: string
  video?: string
  rtsp_url?: string
  host?: string
  rtsp_port?: number | null
  onvif_port?: number | null
  stream_path?: string
  preferred_stream?: string
  username?: string
  password?: string
  onvif_uuid?: string
  discovery_fingerprint?: string
  safety_rule_ids?: string[]
  yoloe_classes?: string[]
  custom_long_tail_terms?: string[]
}) {
  return request("/api/camera-plans/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
}

export async function discoverCameras(payload?: {
  cidrs?: string[]
  timeout_seconds?: number
}) {
  return request("/api/cameras/discover", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  })
}

export async function testDiscoveredCamera(payload: {
  fingerprint?: string
  host?: string
  ip?: string
  name?: string
  vendor?: string
  model?: string
  onvif_uuid?: string
  onvif_xaddr?: string
  onvif_port?: number | null
  rtsp_port?: number | null
  preferred_stream?: string
  stream_path?: string
  username?: string
  password?: string
}) {
  return request("/api/cameras/discover/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
}

export async function importDiscoveredCameras(payload: {
  devices: Array<{
    fingerprint?: string
    host?: string
    ip?: string
    name: string
    zone: string
    profile?: string
    capabilities?: string[]
    preferred_stream?: string
    stream_path?: string
    username?: string
    password?: string
    onvif_uuid?: string
    onvif_xaddr?: string
    onvif_port?: number | null
    rtsp_port?: number | null
  }>
}) {
  return request("/api/cameras/discover/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
}

// Alerts
export async function getAlerts(params?: {
  severity?: string
  status?: string
  cameraId?: string
  limit?: number
  offset?: number
}) {
  const searchParams = new URLSearchParams()
  if (params?.severity) searchParams.set("severity", params.severity)
  if (params?.status) searchParams.set("status", params.status)
  if (params?.cameraId) searchParams.set("cameraId", params.cameraId)
  if (params?.limit) searchParams.set("limit", String(params.limit))
  if (params?.offset) searchParams.set("offset", String(params.offset))
  const qs = searchParams.toString()
  return request(`/api/alerts${qs ? `?${qs}` : ""}`)
}

export async function getAlertStats(cameraId?: string) {
  const params = cameraId ? `?cameraId=${cameraId}` : ""
  return request(`/api/alerts/stats${params}`)
}

export async function acknowledgeAlert(id: string) {
  return request(`/api/alerts/${id}/acknowledge`, { method: "PUT" })
}

export async function resolveAlert(id: string) {
  return request(`/api/alerts/${id}/resolve`, { method: "PUT" })
}

export async function snoozeAlert(id: string, minutes: number = 15) {
  return request(`/api/alerts/${id}/snooze?minutes=${minutes}`, { method: "PUT" })
}

export async function markFalsePositive(id: string) {
  return request(`/api/alerts/${id}/false-positive`, { method: "PUT" })
}

// Safety Rules
import type { EngineRule, ModelInstallJob, ModelStatus, SafetyRule } from "@/types"

export async function getSafetyRules(): Promise<SafetyRule[]> {
  return request("/api/safety-rules")
}

export async function createSafetyRule(rule: { name: string; type: string; classes: string[]; model: string; severity: string; threshold?: number | null }): Promise<SafetyRule> {
  return request("/api/safety-rules", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(rule),
  })
}

export async function updateSafetyRule(id: string, updates: Partial<Omit<SafetyRule, "threshold">> & { threshold?: number | null }): Promise<SafetyRule> {
  return request(`/api/safety-rules/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  })
}

export async function toggleSafetyRule(id: string): Promise<SafetyRule> {
  return request(`/api/safety-rules/${id}/toggle`, { method: "PUT" })
}

export async function deleteSafetyRule(id: string): Promise<void> {
  return request(`/api/safety-rules/${id}`, { method: "DELETE" })
}

export async function assignRuleCameras(ruleId: string, cameraIds: string[]) {
  return request(`/api/safety-rules/${ruleId}/cameras`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ camera_ids: cameraIds }),
  })
}

// Automation Rules

export async function getAutomationRules(): Promise<EngineRule[]> {
  return request("/api/automation-rules")
}

export async function createAutomationRule(rule: Omit<EngineRule, "id" | "lastTriggered">): Promise<EngineRule> {
  return request("/api/automation-rules", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(rule),
  })
}

export async function updateAutomationRule(id: string, updates: Partial<EngineRule>): Promise<EngineRule> {
  return request(`/api/automation-rules/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  })
}

export async function toggleAutomationRule(id: string): Promise<EngineRule> {
  return request(`/api/automation-rules/${id}/toggle`, { method: "PUT" })
}

export async function deleteAutomationRule(id: string): Promise<void> {
  return request(`/api/automation-rules/${id}`, { method: "DELETE" })
}

// Models
export async function getModels(): Promise<{ models: ModelStatus[] }> {
  return request("/api/models")
}

export async function installModels(modelKeys: string[]): Promise<ModelInstallJob> {
  return request("/api/models/install", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_keys: modelKeys }),
  })
}

export async function getModelInstallJob(jobId: string): Promise<ModelInstallJob> {
  return request(`/api/models/install/${jobId}`)
}

export async function retryModelInstallJob(jobId: string): Promise<ModelInstallJob> {
  return request(`/api/models/install/${jobId}/retry`, { method: "POST" })
}

// Videos
export async function getVideos(): Promise<string[]> {
  return request("/api/videos")
}

// License
export type LicenseState = "valid" | "warning" | "grace" | "suspended"

export interface LicenseInfo {
  schema_version: number
  license_id: string
  customer_name: string
  issued_by_partner: string
  max_cameras: number
  features: string[]
  issued_at: string
  expires_at: string
  heartbeat_url: string
}

export interface HeartbeatInfo {
  schema_version: number
  license_id: string
  issued_at: string
  valid_until: string
}

export interface LicenseStatusResponse {
  state: LicenseState
  license: LicenseInfo | null
  heartbeat: HeartbeatInfo | null
  reason: string
  days_until_suspension: number | null
  has_license: boolean
  inference_allowed: boolean
}

export async function getLicenseStatus(): Promise<LicenseStatusResponse> {
  return request("/api/license")
}

async function uploadLicenseFile(path: string, file: File): Promise<LicenseStatusResponse> {
  const form = new FormData()
  form.append("file", file)
  // FormData boundary is set by the browser, so we deliberately do NOT set
  // Content-Type — letting fetch handle it.
  return request(path, { method: "POST", body: form })
}

export async function uploadLicense(file: File): Promise<LicenseStatusResponse> {
  return uploadLicenseFile("/api/license/upload", file)
}

export async function uploadHeartbeat(file: File): Promise<LicenseStatusResponse> {
  return uploadLicenseFile("/api/license/heartbeat", file)
}
