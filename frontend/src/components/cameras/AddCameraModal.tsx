import { useState, useEffect } from "react"
import { X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { SeverityBadge } from "@/components/ui/SeverityBadge"
import { getVideos, addCamera, getSafetyRules } from "@/lib/api"
import type { SafetyRule } from "@/types"
import { Field } from "./helpers"

interface AddCameraModalProps {
  onClose: () => void
  onAdded: () => Promise<void>
}

export function AddCameraModal({ onClose, onAdded }: AddCameraModalProps) {
  const [videos, setVideos] = useState<string[]>([])
  const [name, setName] = useState("")
  const [video, setVideo] = useState("")
  const [zone, setZone] = useState("")
  const [demo, setDemo] = useState("yolo")
  const [rules, setRules] = useState("")
  const [yoloeClasses, setYoloeClasses] = useState("person")
  const [streamType, setStreamType] = useState("file")
  const [rtspUrl, setRtspUrl] = useState("")
  const [safetyRules, setSafetyRules] = useState<SafetyRule[]>([])
  const [safetyRuleIds, setSafetyRuleIds] = useState<string[]>([])
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    getVideos().then((v) => {
      setVideos(v)
      if (v.length > 0) setVideo(v[0])
    }).catch(() => {})
    getSafetyRules().then(setSafetyRules).catch(() => {})
  }, [])

  function toggleSafetyRule(ruleId: string) {
    setSafetyRuleIds((prev) =>
      prev.includes(ruleId) ? prev.filter((id) => id !== ruleId) : [...prev, ruleId]
    )
  }

  const ppeRulesEnabled = safetyRules.filter((r) => r.type === "ppe" && r.enabled)
  const alertRulesEnabled = safetyRules.filter((r) => r.type === "alert" && r.enabled)

  async function handleSave() {
    setSaving(true)
    try {
      await addCamera({
        name,
        video: streamType === "file" ? video : "",
        zone,
        demo,
        rules: rules.split(",").map((r) => r.trim()).filter(Boolean),
        yoloe_classes: demo === "yoloe" ? yoloeClasses.split(",").map((c) => c.trim()).filter(Boolean) : undefined,
        stream_type: streamType,
        rtsp_url: streamType === "rtsp" ? rtspUrl : "",
        safety_rule_ids: safetyRuleIds,
      } as any)
      await onAdded()
      onClose()
    } catch {
      // ignore
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative bg-white rounded-[var(--radius-xl)] shadow-xl w-full max-w-lg mx-4 max-h-[85vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b">
          <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Add Camera</h2>
          <button
            onClick={onClose}
            className="p-1 rounded-[var(--radius-sm)] hover:bg-[var(--color-bg-tertiary)] text-[var(--color-text-tertiary)] transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-4">
          <Field label="Camera Name">
            <input
              type="text"
              placeholder="e.g. Welding Bay - Entry"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
            />
          </Field>

          <Field label="Zone">
            <input
              type="text"
              placeholder="e.g. Welding Bay"
              value={zone}
              onChange={(e) => setZone(e.target.value)}
              className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
            />
          </Field>

          {/* ── Source ── */}
          <div className="border-t pt-4 mt-2">
            <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">Video Source</p>
            <Field label="Stream Type">
              <div className="flex gap-2">
                {(["file", "rtsp"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setStreamType(t)}
                    className={`px-4 py-2 text-sm font-medium rounded-[var(--radius-md)] border cursor-pointer transition-colors ${
                      streamType === t
                        ? "bg-[var(--color-text-primary)] text-white border-transparent"
                        : "bg-white text-[var(--color-text-secondary)] border-[var(--color-border)]"
                    }`}
                  >
                    {t === "file" ? "Video File" : "RTSP Stream"}
                  </button>
                ))}
              </div>
            </Field>

            {streamType === "file" ? (
              <Field label="Video File">
                <select
                  value={video}
                  onChange={(e) => setVideo(e.target.value)}
                  className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer"
                >
                  {videos.length === 0 && <option value="">Loading...</option>}
                  {videos.map((v) => (
                    <option key={v} value={v}>{v}</option>
                  ))}
                </select>
              </Field>
            ) : (
              <Field label="RTSP URL">
                <input
                  type="text"
                  placeholder="rtsp://192.168.1.100:554/stream1"
                  value={rtspUrl}
                  onChange={(e) => setRtspUrl(e.target.value)}
                  className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 font-mono text-xs"
                />
              </Field>
            )}
          </div>

          {/* ── Detection Model ── */}
          <div className="border-t pt-4 mt-2">
            <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">Detection</p>
            <Field label="Detection Mode">
              <select
                value={demo}
                onChange={(e) => setDemo(e.target.value)}
                className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer"
              >
                <option value="yolo">COCO (80 classes — person, phone, animal, vehicle)</option>
                <option value="yoloe">YOLOe Open-Vocab (custom text prompts — PPE, etc.)</option>
                <option value="yolo+vlm">COCO + VLM (periodic scene analysis)</option>
              </select>
            </Field>

            {demo === "yoloe" && (
              <Field label="YOLOe Classes (comma-separated)">
                <input
                  type="text"
                  placeholder="person, hard hat, safety vest, safety goggles"
                  value={yoloeClasses}
                  onChange={(e) => setYoloeClasses(e.target.value)}
                  className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
                />
                <p className="text-xs text-[var(--color-text-tertiary)] mt-1">
                  Open-vocabulary — type any object class to detect (e.g. hard hat, safety vest, goggles)
                </p>
              </Field>
            )}
          </div>

          {/* ── Safety Rules ── */}
          {(ppeRulesEnabled.length > 0 || alertRulesEnabled.length > 0) && (
            <div className="border-t pt-4 mt-2">
              <p className="text-xs font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-3">Safety Rules</p>
              <p className="text-xs text-[var(--color-text-tertiary)] mb-3">Select which safety rules should apply to this camera</p>
              {ppeRulesEnabled.length > 0 && (
                <div className="mb-3">
                  <p className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">PPE Rules</p>
                  <div className="space-y-2">
                    {ppeRulesEnabled.map((rule) => (
                      <label
                        key={rule.id}
                        className={`flex items-center gap-3 p-3 rounded-[var(--radius-md)] border cursor-pointer transition-colors ${
                          safetyRuleIds.includes(rule.id)
                            ? "border-[var(--color-info)] bg-blue-50"
                            : "border-[var(--color-border)] hover:border-[var(--color-border-strong)]"
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
                          <span className="text-xs text-[var(--color-text-tertiary)] ml-2">{rule.classes.join(", ")}</span>
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
                          safetyRuleIds.includes(rule.id)
                            ? "border-[var(--color-info)] bg-blue-50"
                            : "border-[var(--color-border)] hover:border-[var(--color-border-strong)]"
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
                          <span className="text-xs text-[var(--color-text-tertiary)] ml-2">{rule.classes.join(", ")}</span>
                        </div>
                        <SeverityBadge severity={rule.severity} />
                      </label>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="flex items-center gap-2 pt-4 border-t">
            <Button onClick={handleSave} disabled={!name || (streamType === "file" && !video) || (streamType === "rtsp" && !rtspUrl) || saving}>
              {saving ? "Saving..." : "Add Camera"}
            </Button>
            <Button variant="secondary" onClick={onClose}>
              Cancel
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
