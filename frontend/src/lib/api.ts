export const API_BASE =
  import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8000`

export const WS_BASE =
  import.meta.env.VITE_WS_URL ||
  `ws://${window.location.hostname}:8000`

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

async function request(path: string, options?: RequestInit) {
  const token = getToken()
  const headers: Record<string, string> = {
    ...(options?.headers as Record<string, string>),
  }
  if (token) {
    headers["Authorization"] = `Bearer ${token}`
  }
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers })
  if (res.status === 401 && !path.startsWith("/api/auth/")) {
    clearToken()
    window.location.href = "/login"
    throw new Error("Session expired")
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "")
    let message: string
    try {
      const json = JSON.parse(text)
      message = json.detail || json.message || json.error || ""
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
    throw new Error(message)
  }
  return res.json()
}

// Auth
export async function apiLogin(username: string, password: string) {
  return request("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
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

export async function getAlertTimeSeries(hours: number = 24) {
  return request(`/api/alerts/time-series?hours=${hours}`)
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

// Cameras
export async function getCameras() {
  return request("/api/cameras")
}

export async function addCamera(camera: {
  name: string
  video: string
  zone: string
  demo: string
  rules: string[]
  fps?: number
  yoloe_classes?: string[]
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

export async function getAlertStats() {
  return request("/api/alerts/stats")
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

// Detection Rules
export async function getDetectionRules() {
  return request("/api/detection-rules")
}

export async function toggleDetectionRule(ruleId: string) {
  return request(`/api/detection-rules/${ruleId}/toggle`, { method: "PUT" })
}

export async function createDetectionRule(rule: {
  name: string
  model: string
  promptType: string
  prompts: string[]
  confidenceThreshold: number
  severity: string
  category?: string
}) {
  return request("/api/detection-rules", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(rule),
  })
}

// Safety Rules
import type { PPERule, SafetyRule } from "@/types"

export async function getSafetyRules(): Promise<SafetyRule[]> {
  return request("/api/safety-rules")
}

export async function createSafetyRule(rule: { name: string; type: string; classes: string[]; model: string; severity: string }): Promise<SafetyRule> {
  return request("/api/safety-rules", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(rule),
  })
}

export async function updateSafetyRule(id: string, updates: Partial<SafetyRule>): Promise<SafetyRule> {
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

// PPE Rules (deprecated — use Safety Rules)

export async function getPPERules(): Promise<PPERule[]> {
  return request("/api/ppe-rules")
}

export async function createPPERule(rule: { name: string; yoloe_classes: string[]; severity: string }): Promise<PPERule> {
  return request("/api/ppe-rules", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(rule),
  })
}

export async function updatePPERule(id: string, updates: Partial<PPERule>): Promise<PPERule> {
  return request(`/api/ppe-rules/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  })
}

export async function togglePPERule(id: string): Promise<PPERule> {
  return request(`/api/ppe-rules/${id}/toggle`, { method: "PUT" })
}

export async function deletePPERule(id: string): Promise<void> {
  return request(`/api/ppe-rules/${id}`, { method: "DELETE" })
}

// Videos
export async function getVideos(): Promise<string[]> {
  return request("/api/videos")
}
