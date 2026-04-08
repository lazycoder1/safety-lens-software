import { useState, useEffect } from "react"
import { Trash2, X, ChevronDown, ChevronRight } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { SeverityBadge } from "@/components/ui/SeverityBadge"
import { updateCamera, API_BASE, getSafetyRules } from "@/lib/api"
import type { Camera, SafetyRule } from "@/types"
import { CAMERA_ROLES, statusVariant } from "./constants"
import type { CameraRole } from "./constants"
import { Field, InfoRow } from "./helpers"

interface CameraDetailPanelProps {
  camera: Camera
  role: CameraRole
  onRoleChange: (role: CameraRole) => void
  onClose: () => void
  onUpdated: () => Promise<void>
  onDelete: () => void
}

export function CameraDetailPanel({
  camera,
  role,
  onRoleChange,
  onClose,
  onUpdated,
  onDelete,
}: CameraDetailPanelProps) {
  const [editing, setEditing] = useState(false)
  const [name, setName] = useState(camera.name)
  const [zone, setZone] = useState(camera.zone)
  const [demo, setDemo] = useState(camera.demo)
  const [rules, setRules] = useState(camera.rules.join(", "))
  const [fps, setFps] = useState(camera.fps)
  const [yoloeClasses, setYoloeClasses] = useState((camera.yoloe_classes || ["person"]).join(", "))
  const [streamType, setStreamType] = useState(camera.stream_type || "file")
  const [rtspUrl, setRtspUrl] = useState(camera.rtsp_url || "")
  const [saving, setSaving] = useState(false)
  const [safetyRules, setSafetyRules] = useState<SafetyRule[]>([])
  const [safetyRuleIds, setSafetyRuleIds] = useState<string[]>(camera.safety_rule_ids || [])
  const [showAdvancedClasses, setShowAdvancedClasses] = useState(false)

  useEffect(() => {
    getSafetyRules().then(setSafetyRules).catch(() => {})
  }, [])

  const variant = statusVariant[camera.status] || "default"

  function toggleSafetyRule(ruleId: string) {
    setSafetyRuleIds((prev) =>
      prev.includes(ruleId) ? prev.filter((id) => id !== ruleId) : [...prev, ruleId]
    )
  }

  const ppeRulesEnabled = safetyRules.filter((r) => r.type === "ppe" && r.enabled)
  const alertRulesEnabled = safetyRules.filter((r) => r.type === "alert" && r.enabled)

  const derivedClasses = Array.from(
    new Set(
      safetyRules
        .filter((r) => safetyRuleIds.includes(r.id) && r.enabled)
        .flatMap((r) => r.classes)
    )
  )

  async function handleSave() {
    setSaving(true)
    try {
      await updateCamera(camera.id, {
        name,
        zone,
        demo,
        rules: rules.split(",").map((r) => r.trim()).filter(Boolean),
        fps,
        yoloe_classes: yoloeClasses.split(",").map((c) => c.trim()).filter(Boolean),
        stream_type: streamType,
        rtsp_url: streamType === "rtsp" ? rtspUrl : "",
        safety_rule_ids: safetyRuleIds,
      })
      await onUpdated()
      setEditing(false)
    } catch {
      // ignore
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/30" onClick={onClose} />
      <div className="relative w-full max-w-lg bg-white shadow-xl overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b px-6 py-4 flex items-center justify-between z-10">
          <div>
            <h2 className="text-base font-semibold text-[var(--color-text-primary)]">{camera.name}</h2>
            <div className="flex items-center gap-2 mt-1">
              <Badge variant={variant}>{camera.status}</Badge>
              <Badge>{camera.zone}</Badge>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-[var(--radius-md)] hover:bg-[var(--color-bg-tertiary)] text-[var(--color-text-tertiary)] cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Stream preview */}
          <div className="aspect-video rounded-[var(--radius-lg)] overflow-hidden bg-[var(--color-bg-tertiary)] border">
            <img
              src={`${API_BASE}/api/stream/${camera.id}`}
              alt={camera.name}
              className="w-full h-full object-cover"
            />
          </div>

          {editing ? (
            <div className="space-y-4">
              <Field label="Name">
                <input type="text" value={name} onChange={(e) => setName(e.target.value)}
                  className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
              </Field>
              <Field label="Zone">
                <input type="text" value={zone} onChange={(e) => setZone(e.target.value)}
                  className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
              </Field>
              <Field label="FPS">
                <input type="number" value={fps} onChange={(e) => setFps(Number(e.target.value))} min={1} max={30}
                  className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
              </Field>

              <div className="border-t pt-4">
                <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">Source</p>
                <Field label="Stream Type">
                  <div className="flex gap-2">
                    {(["file", "rtsp"] as const).map((t) => (
                      <button key={t} onClick={() => setStreamType(t)}
                        className={`px-4 py-2 text-sm font-medium rounded-[var(--radius-md)] border cursor-pointer transition-colors ${
                          streamType === t ? "bg-[var(--color-text-primary)] text-white border-transparent" : "bg-white text-[var(--color-text-secondary)]"
                        }`}>{t === "file" ? "Video File" : "RTSP Stream"}</button>
                    ))}
                  </div>
                </Field>
                {streamType === "rtsp" && (
                  <Field label="RTSP URL">
                    <input type="text" value={rtspUrl} onChange={(e) => setRtspUrl(e.target.value)}
                      placeholder="rtsp://192.168.1.100:554/stream1"
                      className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 font-mono text-xs" />
                  </Field>
                )}
              </div>

              <div className="border-t pt-4">
                <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">Detection</p>
                <Field label="Detection Mode">
                  <select value={demo} onChange={(e) => setDemo(e.target.value)}
                    className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer">
                    <option value="yolo">COCO (80 classes)</option>
                    <option value="yoloe">YOLOe Open-Vocab</option>
                    <option value="yolo+vlm">COCO + VLM</option>
                  </select>
                </Field>
                {(ppeRulesEnabled.length > 0 || alertRulesEnabled.length > 0) && (
                  <div className="border-t pt-4">
                    <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">Safety Rules</p>
                    {ppeRulesEnabled.length > 0 && (
                      <div className="mb-3">
                        <p className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">PPE Rules</p>
                        <div className="space-y-2">
                          {ppeRulesEnabled.map((rule) => (
                            <label
                              key={rule.id}
                              className={`flex items-center gap-3 p-3 rounded-[var(--radius-md)] border cursor-pointer transition-colors ${
                                safetyRuleIds.includes(rule.id) ? "border-[var(--color-info)] bg-blue-50" : "border-[var(--color-border)]"
                              }`}
                            >
                              <input
                                type="checkbox"
                                checked={safetyRuleIds.includes(rule.id)}
                                onChange={() => toggleSafetyRule(rule.id)}
                                className="accent-[var(--color-info)]"
                              />
                              <div className="flex-1">
                                <span className="text-sm font-medium text-[var(--color-text-primary)]">{rule.name}</span>
                                <span className="text-xs text-[var(--color-text-tertiary)] ml-2">
                                  {rule.classes.join(", ")}
                                </span>
                              </div>
                              <SeverityBadge severity={rule.severity} />
                            </label>
                          ))}
                        </div>
                      </div>
                    )}
                    {alertRulesEnabled.length > 0 && (
                      <div className="mb-3">
                        <p className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">Alert Rules</p>
                        <div className="space-y-2">
                          {alertRulesEnabled.map((rule) => (
                            <label
                              key={rule.id}
                              className={`flex items-center gap-3 p-3 rounded-[var(--radius-md)] border cursor-pointer transition-colors ${
                                safetyRuleIds.includes(rule.id) ? "border-[var(--color-info)] bg-blue-50" : "border-[var(--color-border)]"
                              }`}
                            >
                              <input
                                type="checkbox"
                                checked={safetyRuleIds.includes(rule.id)}
                                onChange={() => toggleSafetyRule(rule.id)}
                                className="accent-[var(--color-info)]"
                              />
                              <div className="flex-1">
                                <span className="text-sm font-medium text-[var(--color-text-primary)]">{rule.name}</span>
                                <span className="text-xs text-[var(--color-text-tertiary)] ml-2">
                                  {rule.classes.join(", ")}
                                </span>
                              </div>
                              <SeverityBadge severity={rule.severity} />
                            </label>
                          ))}
                        </div>
                      </div>
                    )}
                    {derivedClasses.length > 0 && demo === "yoloe" && (
                      <p className="text-xs text-[var(--color-text-tertiary)]">
                        Derived YOLOe classes: {derivedClasses.join(", ")}
                      </p>
                    )}
                  </div>
                )}
                {demo === "yoloe" && (
                  <div>
                    <button
                      type="button"
                      onClick={() => setShowAdvancedClasses(!showAdvancedClasses)}
                      className="flex items-center gap-1 text-xs font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] cursor-pointer mb-1"
                    >
                      {showAdvancedClasses ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                      Advanced: Custom YOLOe Classes
                    </button>
                    {showAdvancedClasses && (
                      <Field label="YOLOe Classes (comma-separated)">
                        <input type="text" value={yoloeClasses} onChange={(e) => setYoloeClasses(e.target.value)}
                          placeholder="person, hard hat, safety vest"
                          className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
                      </Field>
                    )}
                  </div>
                )}
              </div>

              <div className="border-t pt-4">
                <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">Role</p>
                <Field label="Camera Role">
                  <select
                    value={role}
                    onChange={(e) => onRoleChange(e.target.value as CameraRole)}
                    className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer"
                  >
                    {CAMERA_ROLES.map((r) => (
                      <option key={r.value} value={r.value}>{r.label}</option>
                    ))}
                  </select>
                </Field>
                {(() => {
                  const roleInfo = CAMERA_ROLES.find((r) => r.value === role) || CAMERA_ROLES[0]
                  return (
                    <div className="mt-2 p-3 rounded-[var(--radius-md)] bg-[var(--color-bg-secondary)] border text-xs space-y-1">
                      <p className="text-[var(--color-text-secondary)]">
                        <span className="font-medium">Models:</span>{" "}
                        {roleInfo.models.length > 0 ? roleInfo.models.join(", ") : "None (configure via Rules)"}
                      </p>
                      <p className="text-[var(--color-text-secondary)]">
                        <span className="font-medium">Estimated GPU impact:</span> {roleInfo.gpuImpact}
                      </p>
                    </div>
                  )
                })()}
              </div>

              <div className="flex gap-2 pt-2">
                <Button onClick={handleSave} disabled={saving}>{saving ? "Saving..." : "Save"}</Button>
                <Button variant="secondary" onClick={() => setEditing(false)}>Cancel</Button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <InfoRow label="Source" value={camera.stream_type === "rtsp" ? camera.rtsp_url : camera.video} />
              <InfoRow label="Stream Type" value={camera.stream_type === "rtsp" ? "RTSP" : "Video File"} />
              <InfoRow label="Zone" value={camera.zone} />
              <InfoRow label="Detection Mode" value={camera.demo === "yoloe" ? "YOLOe Open-Vocab" : camera.demo === "yolo+vlm" ? "COCO + VLM" : "COCO (80 classes)"} />
              {(camera.safety_rule_ids || []).length > 0 && (
                <div>
                  <span className="text-xs font-medium text-[var(--color-text-secondary)] block mb-1">Safety Rules</span>
                  <div className="flex flex-wrap gap-1">
                    {safetyRules
                      .filter((r) => (camera.safety_rule_ids || []).includes(r.id))
                      .map((r) => (
                        <Badge key={r.id} variant={r.type === "ppe" ? "info" : "warning"}>{r.name}</Badge>
                      ))}
                  </div>
                </div>
              )}
              {camera.demo === "yoloe" && (
                <div>
                  <span className="text-xs font-medium text-[var(--color-text-secondary)] block mb-1">YOLOe Classes</span>
                  <div className="flex flex-wrap gap-1">
                    {(camera.yoloe_classes || []).map((c) => (
                      <Badge key={c} variant="success">{c}</Badge>
                    ))}
                  </div>
                </div>
              )}
              {(() => {
                const roleInfo = CAMERA_ROLES.find((r) => r.value === role) || CAMERA_ROLES[0]
                return (
                  <>
                    <InfoRow label="Role" value={roleInfo.label} />
                    {roleInfo.models.length > 0 && (
                      <div>
                        <span className="text-xs font-medium text-[var(--color-text-secondary)] block mb-1">Models</span>
                        <div className="flex flex-wrap gap-1">
                          {roleInfo.models.map((m) => (
                            <Badge key={m}>{m}</Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )
              })()}
              <InfoRow label="FPS" value={String(camera.fps)} />
              <InfoRow label="Live Detections" value={String(camera.detectionsCount)} />
            </div>
          )}

          {/* Quick actions */}
          {!editing && (
            <div className="flex gap-2 pt-2 border-t">
              <Button variant="secondary" size="sm" className="flex-1" onClick={() => setEditing(true)}>
                Edit
              </Button>
              <Button
                variant="danger"
                size="sm"
                className="flex-1"
                onClick={() => {
                  onDelete()
                  onClose()
                }}
              >
                <Trash2 className="w-3.5 h-3.5" />
                Delete
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
