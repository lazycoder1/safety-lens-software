import { useState, useEffect } from "react"
import {
  Shield,
  ShieldAlert,
  PersonStanding,
  Bug,
  Flame,
  Construction,
  ChevronDown,
  ChevronUp,
  Plus,
  X,
  Settings2,
  ToggleLeft,
  ToggleRight,
} from "lucide-react"
import { pipelineStages } from "@/data/mock"
import type { Severity, DetectionRule } from "@/types"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { SeverityBadge } from "@/components/ui/SeverityBadge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { getDetectionRules, toggleDetectionRule, createDetectionRule, getCameras } from "@/lib/api"

/* ------------------------------------------------------------------ */
/*  Types & constants                                                  */
/* ------------------------------------------------------------------ */

type Mode = "simple" | "advanced" | "pipeline"
type SensitivityLevel = "Low" | "Medium" | "High"

interface PresetScenario {
  id: string
  title: string
  description: string
  icon: React.ElementType
  enabled: boolean
  camerasCount: number
  subRules: { name: string; enabled: boolean }[]
  sensitivity: SensitivityLevel
  category: string
  ruleIds: string[]
}

function buildPresets(rules: DetectionRule[]): PresetScenario[] {
  const ruleMap = Object.fromEntries(rules.map((r) => [r.id, r]))
  return [
    {
      id: "ppe",
      title: "PPE Detection",
      description: "Detect missing personal protective equipment on workers",
      icon: Shield,
      enabled: ["r1", "r2", "r3", "r4"].some((id) => ruleMap[id]?.enabled),
      camerasCount: 3,
      subRules: [
        { name: "Hard Hat", enabled: ruleMap["r1"]?.enabled ?? true },
        { name: "Safety Vest", enabled: ruleMap["r2"]?.enabled ?? true },
        { name: "Goggles", enabled: ruleMap["r3"]?.enabled ?? true },
        { name: "Harness", enabled: ruleMap["r4"]?.enabled ?? true },
      ],
      sensitivity: "Medium",
      category: "PPE",
      ruleIds: ["r1", "r2", "r3", "r4"],
    },
    {
      id: "zone",
      title: "Zone Intrusion",
      description: "Alert when personnel enter restricted or hazardous zones",
      icon: ShieldAlert,
      enabled: ruleMap["r5"]?.enabled ?? true,
      camerasCount: 2,
      subRules: [
        { name: "Restricted Area", enabled: true },
        { name: "Hazardous Zone", enabled: true },
        { name: "After-Hours Access", enabled: false },
      ],
      sensitivity: "High",
      category: "Zone Safety",
      ruleIds: ["r5"],
    },
    {
      id: "fall",
      title: "Fall Detection",
      description: "Detect person falls using pose estimation models",
      icon: PersonStanding,
      enabled: ruleMap["r6"]?.enabled ?? true,
      camerasCount: 3,
      subRules: [
        { name: "Standing to Ground", enabled: true },
        { name: "Collapse Detection", enabled: true },
        { name: "Slip Detection", enabled: false },
      ],
      sensitivity: "High",
      category: "Emergency",
      ruleIds: ["r6"],
    },
    {
      id: "animal",
      title: "Animal Detection",
      description: "Detect stray animals or snakes in the factory premises",
      icon: Bug,
      enabled: ruleMap["r8"]?.enabled ?? true,
      camerasCount: 1,
      subRules: [
        { name: "Snake", enabled: true },
        { name: "Dog", enabled: true },
        { name: "Cat", enabled: true },
      ],
      sensitivity: "Medium",
      category: "Environment",
      ruleIds: ["r8"],
    },
    {
      id: "fire",
      title: "Fire / Smoke",
      description: "Early detection of fire, smoke, or thermal anomalies",
      icon: Flame,
      enabled: ruleMap["r10"]?.enabled ?? true,
      camerasCount: 3,
      subRules: [
        { name: "Flames", enabled: true },
        { name: "Smoke", enabled: true },
        { name: "Sparks", enabled: false },
      ],
      sensitivity: "High",
      category: "Emergency",
      ruleIds: ["r10"],
    },
    {
      id: "gangway",
      title: "Gangway Blockage",
      description: "Monitor gangways and exits for obstructions",
      icon: Construction,
      enabled: ruleMap["r9"]?.enabled ?? true,
      camerasCount: 1,
      subRules: [
        { name: "Exit Blocked", enabled: true },
        { name: "Aisle Obstruction", enabled: true },
      ],
      sensitivity: "Medium",
      category: "Environment",
      ruleIds: ["r9"],
    },
  ]
}

const modelBadgeVariant: Record<string, "critical" | "high" | "warning" | "info" | "success" | "default"> = {
  YOLOE: "info",
  YOLO26: "success",
  "YOLO-pose": "high",
  VLM: "warning",
}

/* ------------------------------------------------------------------ */
/*  New Rule Form                                                      */
/* ------------------------------------------------------------------ */

interface NewRuleForm {
  name: string
  model: string
  promptType: string
  prompts: string[]
  promptInput: string
  confidenceThreshold: number
  severity: Severity
}

const emptyForm: NewRuleForm = {
  name: "",
  model: "YOLOE",
  promptType: "text",
  prompts: [],
  promptInput: "",
  confidenceThreshold: 50,
  severity: "P2",
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function DetectionRules() {
  const [mode, setMode] = useState<Mode>("simple")
  const [rules, setRules] = useState<DetectionRule[]>([])
  const [expandedPreset, setExpandedPreset] = useState<string | null>(null)
  const [showNewRule, setShowNewRule] = useState(false)
  const [form, setForm] = useState<NewRuleForm>(emptyForm)
  const [cameraCount, setCameraCount] = useState(0)

  useEffect(() => {
    getDetectionRules().then(setRules).catch(() => {})
    getCameras().then((cams: any) => {
      const list = Array.isArray(cams) ? cams : []
      setCameraCount(list.length)
    }).catch(() => {})
  }, [])

  const presets = buildPresets(rules)

  /* Simple mode handlers */
  async function togglePreset(presetId: string) {
    const preset = presets.find((p) => p.id === presetId)
    if (!preset) return
    for (const ruleId of preset.ruleIds) {
      try {
        const updated = await toggleDetectionRule(ruleId)
        setRules((prev) => prev.map((r) => (r.id === ruleId ? updated : r)))
      } catch {}
    }
  }

  /* Advanced mode handlers */
  async function toggleRule(id: string) {
    try {
      const updated = await toggleDetectionRule(id)
      setRules((prev) => prev.map((r) => (r.id === id ? updated : r)))
    } catch {}
  }

  function addPromptChip(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && form.promptInput.trim()) {
      e.preventDefault()
      setForm((prev) => ({
        ...prev,
        prompts: [...prev.prompts, prev.promptInput.trim()],
        promptInput: "",
      }))
    }
  }

  function removePromptChip(index: number) {
    setForm((prev) => ({
      ...prev,
      prompts: prev.prompts.filter((_, i) => i !== index),
    }))
  }

  async function handleSaveRule() {
    if (!form.name.trim()) return
    try {
      const newRule = await createDetectionRule({
        name: form.name,
        model: form.model,
        promptType: form.promptType,
        prompts: form.prompts,
        confidenceThreshold: form.confidenceThreshold / 100,
        severity: form.severity,
        category: "Custom",
      })
      setRules((prev) => [...prev, newRule])
      setForm(emptyForm)
      setShowNewRule(false)
    } catch {}
  }

  return (
    <div className="h-full overflow-auto bg-[var(--color-bg-secondary)]">
      <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-xl font-bold text-[var(--color-text-primary)]">
              Detection Rules
            </h1>
            <p className="text-sm text-[var(--color-text-secondary)] mt-0.5">
              Configure what SafetyLens detects across your cameras
            </p>
          </div>

          {/* Mode toggle */}
          <div className="flex items-center bg-white border rounded-[var(--radius-lg)] p-0.5">
            {(["simple", "pipeline", "advanced"] as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={cn(
                  "px-3 py-1.5 text-xs font-medium rounded-[var(--radius-md)] transition-colors cursor-pointer capitalize",
                  mode === m
                    ? "bg-[var(--color-text-primary)] text-white"
                    : "text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
                )}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        {/* ============================================================ */}
        {/*  SIMPLE MODE                                                  */}
        {/* ============================================================ */}
        {mode === "simple" && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {presets.map((preset) => {
              const Icon = preset.icon
              const isExpanded = expandedPreset === preset.id
              return (
                <Card key={preset.id} className="flex flex-col">
                  {/* Card top */}
                  <div className="flex items-start justify-between">
                    <div
                      className="flex items-start gap-3 flex-1 cursor-pointer"
                      onClick={() =>
                        setExpandedPreset(isExpanded ? null : preset.id)
                      }
                    >
                      <div
                        className={cn(
                          "flex items-center justify-center w-10 h-10 rounded-[var(--radius-md)] shrink-0",
                          preset.enabled
                            ? "bg-[var(--color-info-bg)] text-[var(--color-info)]"
                            : "bg-[var(--color-bg-tertiary)] text-[var(--color-text-tertiary)]"
                        )}
                      >
                        <Icon size={20} />
                      </div>
                      <div className="min-w-0">
                        <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
                          {preset.title}
                        </h3>
                        <p className="text-xs text-[var(--color-text-secondary)] mt-0.5 line-clamp-2">
                          {preset.description}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => togglePreset(preset.id)}
                      className="shrink-0 ml-2 cursor-pointer"
                      title={preset.enabled ? "Disable" : "Enable"}
                    >
                      {preset.enabled ? (
                        <ToggleRight size={28} className="text-[var(--color-info)]" />
                      ) : (
                        <ToggleLeft size={28} className="text-[var(--color-text-tertiary)]" />
                      )}
                    </button>
                  </div>

                  {/* Camera count badge */}
                  <div className="flex items-center gap-2 mt-3">
                    <Badge variant="default">
                      {preset.camerasCount} camera{preset.camerasCount !== 1 ? "s" : ""}
                    </Badge>
                    <button
                      onClick={() =>
                        setExpandedPreset(isExpanded ? null : preset.id)
                      }
                      className="ml-auto text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)] cursor-pointer"
                    >
                      {isExpanded ? (
                        <ChevronUp size={16} />
                      ) : (
                        <ChevronDown size={16} />
                      )}
                    </button>
                  </div>

                  {/* Expanded panel */}
                  {isExpanded && (
                    <div className="mt-4 pt-4 border-t space-y-4">
                      {/* Sub-rules */}
                      <div>
                        <p className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">
                          Sub-rules
                        </p>
                        <div className="space-y-1.5">
                          {preset.subRules.map((sr) => (
                            <label
                              key={sr.name}
                              className="flex items-center gap-2 cursor-pointer"
                            >
                              <input
                                type="checkbox"
                                checked={sr.enabled}
                                readOnly
                                className="rounded border-[var(--color-border-default)] text-[var(--color-info)] focus:ring-[var(--color-info)] cursor-pointer"
                              />
                              <span className="text-xs text-[var(--color-text-primary)]">
                                {sr.name}
                              </span>
                            </label>
                          ))}
                        </div>
                      </div>

                      {/* Sensitivity */}
                      <div>
                        <p className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">
                          Sensitivity
                        </p>
                        <div className="flex gap-1">
                          {(["Low", "Medium", "High"] as SensitivityLevel[]).map(
                            (level) => (
                              <button
                                key={level}
                                className={cn(
                                  "px-3 py-1 text-xs rounded-[var(--radius-md)] font-medium transition-colors cursor-pointer",
                                  preset.sensitivity === level
                                    ? level === "High"
                                      ? "bg-red-100 text-red-700"
                                      : level === "Medium"
                                      ? "bg-amber-100 text-amber-700"
                                      : "bg-emerald-100 text-emerald-700"
                                    : "bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] hover:bg-[var(--color-border-default)]"
                                )}
                              >
                                {level}
                              </button>
                            )
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </Card>
              )
            })}
          </div>
        )}

        {/* ============================================================ */}
        {/*  PIPELINE MODE                                                */}
        {/* ============================================================ */}
        {mode === "pipeline" && (
          <div className="space-y-6">
            <p className="text-sm text-[var(--color-text-secondary)]">
              How detection flows from camera frame through models to alerts.
            </p>

            {/* Flow diagram */}
            <div className="relative">
              {/* Source: Camera Frame */}
              <div className="flex justify-center mb-4">
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-[var(--radius-lg)] bg-[var(--color-bg-tertiary)] border-2 border-[var(--color-border-active)]">
                  <div className="w-3 h-3 rounded-full bg-[var(--color-success)] animate-pulse" />
                  <span className="text-sm font-semibold text-[var(--color-text-primary)]">Camera Frame</span>
                  <span className="text-xs text-[var(--color-text-tertiary)]">(every frame, tiered FPS)</span>
                </div>
              </div>

              {/* Arrows down */}
              <div className="flex justify-center mb-4">
                <div className="flex gap-16 text-[var(--color-text-tertiary)]">
                  <div className="text-center">
                    <div className="w-px h-6 bg-[var(--color-border-active)] mx-auto" />
                    <ChevronDown className="w-4 h-4 mx-auto -mt-1" />
                  </div>
                  <div className="text-center">
                    <div className="w-px h-6 bg-[var(--color-border-active)] mx-auto" />
                    <ChevronDown className="w-4 h-4 mx-auto -mt-1" />
                  </div>
                  <div className="text-center">
                    <div className="w-px h-6 bg-[var(--color-border-active)] mx-auto" />
                    <ChevronDown className="w-4 h-4 mx-auto -mt-1" />
                  </div>
                </div>
              </div>

              {/* Model stages */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                {pipelineStages.map((stage) => {
                  const bgColor = stage.id === "yolo26" ? "bg-emerald-50 border-emerald-200" :
                    stage.id === "yoloe" ? "bg-blue-50 border-blue-200" :
                    stage.id === "yolo-pose" ? "bg-orange-50 border-orange-200" :
                    "bg-purple-50 border-purple-200"
                  const textColor = stage.id === "yolo26" ? "text-emerald-700" :
                    stage.id === "yoloe" ? "text-blue-700" :
                    stage.id === "yolo-pose" ? "text-orange-700" :
                    "text-purple-700"
                  const dotColor = stage.color

                  return (
                    <Card key={stage.id} className={cn("border-2", bgColor)}>
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: dotColor }} />
                          <span className={cn("text-sm font-bold", textColor)}>{stage.name}</span>
                        </div>
                        <span className="text-[10px] font-mono text-[var(--color-text-tertiary)]">{stage.fps}</span>
                      </div>
                      <p className="text-xs text-[var(--color-text-secondary)] mb-1">{stage.model}</p>
                      <p className="text-[11px] text-[var(--color-text-tertiary)] mb-3">{stage.description}</p>
                      <div className="text-[10px] font-medium text-[var(--color-text-tertiary)] uppercase tracking-wider mb-1.5">
                        Runs on: {stage.runsOn}
                      </div>
                      <div className="border-t pt-2 mt-1">
                        <p className="text-[10px] font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-1.5">Rules handled</p>
                        <div className="space-y-1">
                          {stage.rules.map((r) => (
                            <div key={r} className="flex items-center gap-1.5 text-xs text-[var(--color-text-primary)]">
                              <div className="w-1 h-1 rounded-full bg-[var(--color-text-tertiary)]" />
                              {r}
                            </div>
                          ))}
                        </div>
                      </div>
                      {stage.id === "yoloe" && (
                        <div className="mt-3 pt-2 border-t border-dashed flex items-center gap-1.5 text-[10px] text-purple-600 font-medium">
                          Low confidence detections escalate to VLM for reasoning
                        </div>
                      )}
                    </Card>
                  )
                })}
              </div>

              {/* Arrows down to alert */}
              <div className="flex justify-center mb-4">
                <div className="text-center text-[var(--color-text-tertiary)]">
                  <div className="w-px h-6 bg-[var(--color-border-active)] mx-auto" />
                  <ChevronDown className="w-4 h-4 mx-auto -mt-1" />
                </div>
              </div>

              {/* Alert output */}
              <div className="flex justify-center">
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-[var(--radius-lg)] bg-[var(--color-critical-bg)] border-2 border-red-200">
                  <div className="w-3 h-3 rounded-full bg-[var(--color-critical)]" />
                  <span className="text-sm font-semibold text-[#991b1b]">Alert System</span>
                  <span className="text-xs text-[#991b1b]/60">(Toast, Telegram, PLC, WhatsApp)</span>
                </div>
              </div>
            </div>

            {/* GPU Budget Estimate */}
            <Card className="border-2 border-[var(--color-border-active)]">
              <h4 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">GPU Budget Estimate</h4>
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-[var(--color-text-secondary)]">Jetson Device</span>
                  <span className="font-semibold">Orin Nano 8GB (40 TOPS)</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-[var(--color-text-secondary)]">Active Cameras</span>
                  <span className="font-semibold">{cameraCount}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-[var(--color-text-secondary)]">Active Rules</span>
                  <span className="font-semibold">{rules.filter(r => r.enabled).length} / {rules.length}</span>
                </div>
                <div>
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-[var(--color-text-secondary)]">Estimated GPU Utilization</span>
                    <span className="font-bold text-[var(--color-high)]">~72%</span>
                  </div>
                  <div className="w-full h-3 bg-[var(--color-bg-tertiary)] rounded-full overflow-hidden">
                    <div className="h-full rounded-full bg-[var(--color-high)] transition-all" style={{ width: "72%" }} />
                  </div>
                  <div className="flex justify-between text-[10px] text-[var(--color-text-tertiary)] mt-1">
                    <span>0%</span>
                    <span className="text-[var(--color-warning)]">85% (auto-degrade threshold)</span>
                    <span>100%</span>
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-2 pt-2 border-t">
                  {[
                    { model: "YOLO26", cost: "18%", color: "#059669" },
                    { model: "YOLOE", cost: "28%", color: "#2563eb" },
                    { model: "YOLO-pose", cost: "22%", color: "#f97316" },
                    { model: "VLM", cost: "0%", color: "#8b5cf6" },
                  ].map(m => (
                    <div key={m.model} className="text-center">
                      <div className="text-xs font-bold" style={{ color: m.color }}>{m.cost}</div>
                      <div className="text-[10px] text-[var(--color-text-tertiary)]">{m.model}</div>
                    </div>
                  ))}
                </div>
                <p className="text-[10px] text-[var(--color-text-tertiary)]">
                  VLM runs on cloud (Alibaba DashScope) — no local GPU cost. At 85% utilization, the system auto-degrades low-priority rules (animal detection, PPE sampling) to free resources. Safety-critical rules (fall, fire, intrusion) are never degraded.
                </p>
              </div>
            </Card>

            {/* Key insight */}
            <Card className="bg-[var(--color-info-bg)] border-blue-200">
              <h4 className="text-sm font-semibold text-[#1e40af] mb-2">How it works</h4>
              <div className="space-y-2 text-xs text-[#1e40af]/80">
                <p><strong>YOLO26</strong> and <strong>YOLOE</strong> run on every frame in parallel on the Jetson GPU. YOLO26 handles trained classes (phone, person), YOLOE handles open-vocabulary detection via text prompts (no training needed).</p>
                <p><strong>YOLO-pose</strong> runs keypoint detection for fall detection (checks if shoulder Y &gt; knee Y).</p>
                <p><strong>VLM (qwen3-vl)</strong> is called only when needed: periodic scene scans (every 60s for gangway blockage) or when YOLOE has a low-confidence detection that needs reasoning (e.g., "is this a syringe?").</p>
                <p>This architecture keeps cloud API costs at ~$1/day for 10 cameras while maintaining real-time detection.</p>
              </div>
            </Card>
          </div>
        )}

        {/* ============================================================ */}
        {/*  ADVANCED MODE                                                */}
        {/* ============================================================ */}
        {mode === "advanced" && (
          <div className="space-y-4">
            {/* Toolbar */}
            <div className="flex items-center justify-between">
              <p className="text-sm text-[var(--color-text-secondary)]">
                {rules.length} rule{rules.length !== 1 ? "s" : ""} configured
              </p>
              <Button
                size="sm"
                onClick={() => setShowNewRule(true)}
              >
                <Plus size={14} />
                New Rule
              </Button>
            </div>

            {/* New Rule Form */}
            {showNewRule && (
              <Card className="border-[var(--color-info)] border-2">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Settings2 size={16} className="text-[var(--color-info)]" />
                    <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
                      Create New Rule
                    </h3>
                  </div>
                  <button
                    onClick={() => {
                      setShowNewRule(false)
                      setForm(emptyForm)
                    }}
                    className="text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] cursor-pointer"
                  >
                    <X size={16} />
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Rule name */}
                  <div>
                    <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">
                      Rule Name
                    </label>
                    <input
                      type="text"
                      value={form.name}
                      onChange={(e) =>
                        setForm((prev) => ({ ...prev, name: e.target.value }))
                      }
                      placeholder="e.g. Welding Mask Detection"
                      className="w-full px-3 py-1.5 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent"
                    />
                  </div>

                  {/* Model selector */}
                  <div>
                    <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">
                      Model
                    </label>
                    <select
                      value={form.model}
                      onChange={(e) =>
                        setForm((prev) => ({
                          ...prev,
                          model: e.target.value,
                        }))
                      }
                      className="w-full px-3 py-1.5 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent cursor-pointer"
                    >
                      <option value="YOLOE">YOLOE</option>
                      <option value="YOLO26">YOLO26</option>
                      <option value="YOLO-pose">YOLO-pose</option>
                      <option value="VLM">VLM</option>
                    </select>
                  </div>

                  {/* Prompt type (for YOLOE) */}
                  {form.model === "YOLOE" && (
                    <div>
                      <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">
                        Prompt Type
                      </label>
                      <select
                        value={form.promptType}
                        onChange={(e) =>
                          setForm((prev) => ({
                            ...prev,
                            promptType: e.target.value,
                          }))
                        }
                        className="w-full px-3 py-1.5 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent cursor-pointer"
                      >
                        <option value="text">Text</option>
                        <option value="visual">Visual</option>
                        <option value="internal">Internal</option>
                      </select>
                    </div>
                  )}

                  {/* Severity */}
                  <div>
                    <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">
                      Severity
                    </label>
                    <select
                      value={form.severity}
                      onChange={(e) =>
                        setForm((prev) => ({
                          ...prev,
                          severity: e.target.value as Severity,
                        }))
                      }
                      className="w-full px-3 py-1.5 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent cursor-pointer"
                    >
                      <option value="P1">P1 - Critical</option>
                      <option value="P2">P2 - High</option>
                      <option value="P3">P3 - Medium</option>
                      <option value="P4">P4 - Low</option>
                    </select>
                  </div>

                  {/* Prompts (tag input) */}
                  <div className="md:col-span-2">
                    <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">
                      Prompts
                    </label>
                    <div className="flex flex-wrap items-center gap-1.5 min-h-[38px] px-3 py-1.5 border rounded-[var(--radius-md)] bg-white focus-within:ring-2 focus-within:ring-[var(--color-info)] focus-within:border-transparent">
                      {form.prompts.map((p, i) => (
                        <span
                          key={i}
                          className="inline-flex items-center gap-1 bg-[var(--color-info-bg)] text-[var(--color-info)] text-xs font-medium rounded-md px-2 py-0.5"
                        >
                          {p}
                          <button
                            onClick={() => removePromptChip(i)}
                            className="hover:text-red-500 cursor-pointer"
                          >
                            <X size={12} />
                          </button>
                        </span>
                      ))}
                      <input
                        type="text"
                        value={form.promptInput}
                        onChange={(e) =>
                          setForm((prev) => ({
                            ...prev,
                            promptInput: e.target.value,
                          }))
                        }
                        onKeyDown={addPromptChip}
                        placeholder={
                          form.prompts.length === 0
                            ? "Type a prompt and press Enter"
                            : ""
                        }
                        className="flex-1 min-w-[120px] text-sm bg-transparent text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] focus:outline-none"
                      />
                    </div>
                  </div>

                  {/* Confidence threshold slider */}
                  <div className="md:col-span-2">
                    <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">
                      Confidence Threshold: {form.confidenceThreshold}%
                    </label>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={form.confidenceThreshold}
                      onChange={(e) =>
                        setForm((prev) => ({
                          ...prev,
                          confidenceThreshold: Number(e.target.value),
                        }))
                      }
                      className="w-full accent-[var(--color-info)] cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-[var(--color-text-tertiary)] mt-0.5">
                      <span>0%</span>
                      <span>50%</span>
                      <span>100%</span>
                    </div>
                  </div>
                </div>

                {/* Form actions */}
                <div className="flex items-center justify-end gap-2 mt-4 pt-4 border-t">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => {
                      setShowNewRule(false)
                      setForm(emptyForm)
                    }}
                  >
                    Cancel
                  </Button>
                  <Button
                    size="sm"
                    onClick={handleSaveRule}
                    disabled={!form.name.trim() || form.prompts.length === 0}
                  >
                    Save Rule
                  </Button>
                </div>
              </Card>
            )}

            {/* Rules table */}
            <Card className="p-0 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-[var(--color-bg-secondary)]">
                      <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">
                        Name
                      </th>
                      <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">
                        Model
                      </th>
                      <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">
                        Prompts
                      </th>
                      <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">
                        Confidence
                      </th>
                      <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">
                        Severity
                      </th>
                      <th className="text-center px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">
                        Cameras
                      </th>
                      <th className="text-center px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">
                        Status
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {rules.map((rule) => (
                      <tr
                        key={rule.id}
                        className="border-b last:border-b-0 hover:bg-[var(--color-bg-secondary)] transition-colors"
                      >
                        <td className="px-4 py-3">
                          <span className="font-medium text-[var(--color-text-primary)]">
                            {rule.name}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <Badge variant={modelBadgeVariant[rule.model] || "default"}>
                            {rule.model}
                          </Badge>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex flex-wrap gap-1 max-w-[200px]">
                            {rule.prompts.map((p, i) => (
                              <span
                                key={i}
                                className="inline-block bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] text-[11px] rounded px-1.5 py-0.5"
                              >
                                {p}
                              </span>
                            ))}
                          </div>
                        </td>
                        <td className="px-4 py-3 text-[var(--color-text-secondary)]">
                          {Math.round(rule.confidenceThreshold * 100)}%
                        </td>
                        <td className="px-4 py-3">
                          <SeverityBadge severity={rule.severity} />
                        </td>
                        <td className="px-4 py-3 text-center text-[var(--color-text-secondary)]">
                          {rule.camerasCount}
                        </td>
                        <td className="px-4 py-3 text-center">
                          <button
                            onClick={() => toggleRule(rule.id)}
                            className="cursor-pointer"
                            title={rule.enabled ? "Disable" : "Enable"}
                          >
                            {rule.enabled ? (
                              <ToggleRight
                                size={24}
                                className="text-[var(--color-info)]"
                              />
                            ) : (
                              <ToggleLeft
                                size={24}
                                className="text-[var(--color-text-tertiary)]"
                              />
                            )}
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}
