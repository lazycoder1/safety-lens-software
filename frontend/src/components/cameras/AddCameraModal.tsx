import { useState, useEffect } from "react"
import { X, Eye, Zap, ChevronDown, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { getVideos, addCamera, getSafetyRules } from "@/lib/api"
import type { SafetyRule } from "@/types"
import { Field, inferDetectionMode } from "./helpers"
import { SafetyRuleSelector } from "./SafetyRuleSelector"

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
  const [demoManuallySet, setDemoManuallySet] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  useEffect(() => {
    getVideos().then((v) => {
      setVideos(v)
      if (v.length > 0) setVideo(v[0])
    }).catch(() => {})
    getSafetyRules().then(setSafetyRules).catch(() => {})
  }, [])

  const inferred = inferDetectionMode(safetyRuleIds, safetyRules)

  useEffect(() => {
    if (!demoManuallySet && safetyRuleIds.length > 0) {
      setDemo(inferred.mode)
    }
  }, [safetyRuleIds, demoManuallySet, inferred.mode])

  function toggleSafetyRule(ruleId: string) {
    setSafetyRuleIds((prev) =>
      prev.includes(ruleId) ? prev.filter((id) => id !== ruleId) : [...prev, ruleId]
    )
  }

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
      })
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

          {/* ── Safety Rules (Hero Section) ── */}
          <div className="border-t pt-4 mt-2">
            <div className="flex items-center gap-2 mb-1">
              <Eye className="w-4 h-4 text-[var(--color-info)]" />
              <p className="text-sm font-semibold text-[var(--color-text-primary)]">
                What Should This Camera Watch For?
              </p>
            </div>
            <p className="text-xs text-[var(--color-text-tertiary)] mb-3">
              Select the safety rules this camera will enforce
            </p>
            <SafetyRuleSelector
              safetyRules={safetyRules}
              selectedIds={safetyRuleIds}
              onToggle={toggleSafetyRule}
              prominent
            />
            {safetyRuleIds.length > 0 && (
              <div className="mt-3 flex items-center gap-2 p-2 rounded-[var(--radius-md)] bg-[var(--color-info-bg)]">
                <Zap className="w-3.5 h-3.5 text-[var(--color-info)] shrink-0" />
                <span className="text-xs text-[var(--color-text-secondary)]">
                  Detection: {inferred.mode === "yoloe" ? "YOLOe Open-Vocab" : "COCO (80 classes)"} — {inferred.reason}
                </span>
              </div>
            )}
          </div>

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

          {/* ── Advanced Detection (collapsible) ── */}
          <div className="border-t pt-4 mt-2">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1 text-xs font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] cursor-pointer"
            >
              {showAdvanced ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              Advanced: Detection Mode Override
            </button>
            {showAdvanced && (
              <div className="mt-3 space-y-3">
                <Field label="Detection Mode">
                  <select
                    value={demo}
                    onChange={(e) => {
                      setDemo(e.target.value)
                      setDemoManuallySet(true)
                    }}
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

                {demoManuallySet && (
                  <button
                    type="button"
                    onClick={() => {
                      setDemoManuallySet(false)
                      if (safetyRuleIds.length > 0) setDemo(inferred.mode)
                    }}
                    className="text-xs text-[var(--color-info)] hover:underline cursor-pointer"
                  >
                    Reset to auto-detected mode
                  </button>
                )}
              </div>
            )}
          </div>

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
