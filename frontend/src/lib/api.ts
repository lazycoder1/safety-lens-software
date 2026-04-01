export const API_BASE =
  import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8000`

export const WS_BASE =
  import.meta.env.VITE_WS_URL ||
  `ws://${window.location.hostname}:8000`

async function request(path: string, options?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`API ${res.status}: ${text}`)
  }
  return res.json()
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

// Videos
export async function getVideos(): Promise<string[]> {
  return request("/api/videos")
}
