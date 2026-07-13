import { Cpu, SlidersHorizontal } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import type { Camera, CapabilityKey, CapabilityWindow, ModelKey, RuleScheduleWindow, SafetyRule } from "@/types"
import { cn } from "@/lib/utils"
import { defaultScheduleWindow, normalizeCapabilityWindows, ScheduleWindowsEditor } from "./CameraEventPolicyPanel"

export interface CameraCapabilityEditorOption {
  key: CapabilityKey
  label: string
  ruleId: string | null
  classes: string[]
  supportsClosedSetCandidate: boolean
}

export interface CameraCapabilityConfigDraft {
  limitRuntime: boolean
  windows: RuleScheduleWindow[]
  confidence: string
  threshold: string
  modelKey: "default" | "ppe_closed_set_candidate"
  allowedHelmetColours: string[]
}

export type CameraCapabilityConfigDrafts = Partial<Record<CapabilityKey, CameraCapabilityConfigDraft>>

const inputClasses =
  "w-full rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 disabled:bg-[var(--color-bg-tertiary)] disabled:text-[var(--color-text-tertiary)]"

const HELMET_COLOURS = [
  { key: "white", label: "White", swatch: "#f8fafc" },
  { key: "yellow", label: "Yellow", swatch: "#facc15" },
  { key: "orange", label: "Orange", swatch: "#f97316" },
  { key: "red", label: "Red", swatch: "#dc2626" },
  { key: "blue", label: "Blue", swatch: "#2563eb" },
  { key: "green", label: "Green", swatch: "#16a34a" },
  { key: "black", label: "Black", swatch: "#111827" },
]

export function buildCapabilityEditorOptions(
  selected: CapabilityKey[],
  safetyRules: SafetyRule[],
  definitions: Array<{ key: CapabilityKey; label: string; ruleId: string | null }>
): CameraCapabilityEditorOption[] {
  const rulesById = new Map(safetyRules.map((rule) => [rule.id, rule]))
  const selectedSet = new Set(selected)
  return definitions
    .filter((definition) => selectedSet.has(definition.key))
    .map((definition) => {
      const rule = definition.ruleId ? rulesById.get(definition.ruleId) : null
      return {
        key: definition.key,
        label: definition.label,
        ruleId: definition.ruleId,
        classes: rule?.classes || [],
        supportsClosedSetCandidate: definition.key === "apron_required" || definition.key === "harness_required",
      }
    })
}

export function defaultCapabilityConfigDraft(capability?: CapabilityKey): CameraCapabilityConfigDraft {
  const helmetColour = capability === "helmet_color_compliance"
  return {
    limitRuntime: false,
    windows: [defaultScheduleWindow()],
    confidence: helmetColour ? "0.45" : "",
    threshold: helmetColour ? "3" : "",
    modelKey: "default",
    allowedHelmetColours: [],
  }
}

export function capabilityConfigDraftsFromCamera(
  camera: Camera | null,
  options: CameraCapabilityEditorOption[]
): CameraCapabilityConfigDrafts {
  const windows = normalizeCapabilityWindows(camera?.capability_windows || camera?.execution_plan?.capability_windows)
  const windowsByCapability = new Map<CapabilityKey, CapabilityWindow>()
  for (const window of windows) {
    for (const capability of window.capabilities || []) {
      if (!windowsByCapability.has(capability)) {
        windowsByCapability.set(capability, window)
      }
    }
  }

  const overrides = camera?.safety_rule_overrides || {}
  const modelOverrides = camera?.capability_model_overrides || camera?.execution_plan?.capability_model_overrides || {}
  const helmetColourPolicy = camera?.helmet_colour_policy
  const drafts: CameraCapabilityConfigDrafts = {}
  for (const option of options) {
    const window = windowsByCapability.get(option.key)
    const ruleOverride = option.ruleId ? overrides[option.ruleId] : null
    const isHelmetColour = option.key === "helmet_color_compliance"
    drafts[option.key] = {
      limitRuntime: Boolean(window?.windows?.length),
      windows: window?.windows?.length ? window.windows : [defaultScheduleWindow()],
      confidence: isHelmetColour ? numberString(helmetColourPolicy?.min_confidence ?? 0.45) : numberString(ruleOverride?.confidence),
      threshold: isHelmetColour ? numberString(helmetColourPolicy?.confirmation_threshold ?? 3) : numberString(ruleOverride?.threshold),
      modelKey: modelOverrides[option.key] === "ppe_closed_set_candidate" ? "ppe_closed_set_candidate" : "default",
      allowedHelmetColours: isHelmetColour ? [...(helmetColourPolicy?.allowed_colours || [])] : [],
    }
  }
  return drafts
}

export function reconcileCapabilityConfigDrafts(
  current: CameraCapabilityConfigDrafts,
  options: CameraCapabilityEditorOption[]
): CameraCapabilityConfigDrafts {
  const next: CameraCapabilityConfigDrafts = {}
  for (const option of options) {
    next[option.key] = current[option.key] || defaultCapabilityConfigDraft(option.key)
  }
  return next
}

export function validateCapabilityConfigDrafts(
  options: CameraCapabilityEditorOption[],
  drafts: CameraCapabilityConfigDrafts
): string | null {
  for (const option of options) {
    const draft = drafts[option.key]
    if (!draft) continue
    if (draft.limitRuntime) {
      if (!draft.windows.length) return `${option.label} needs at least one detector active window.`
      if (draft.windows.some((window) => !window.from || !window.to)) return `${option.label} detector windows need start and end times.`
      if (draft.windows.some((window) => window.days?.length === 0)) return `${option.label} detector windows need at least one day.`
    }
    const confidence = optionalNumber(draft.confidence)
    if (confidence !== null && (confidence <= 0 || confidence > 1)) return `${option.label} confidence must be between 0 and 1.`
    const threshold = optionalNumber(draft.threshold)
    if (threshold !== null && threshold < 1) return `${option.label} confirmation threshold must be 1 or higher.`
  }
  return null
}

export function buildCapabilityWindowsPayload(
  options: CameraCapabilityEditorOption[],
  drafts: CameraCapabilityConfigDrafts
): CapabilityWindow[] {
  return options.flatMap((option) => {
    const draft = drafts[option.key]
    if (!draft?.limitRuntime) return []
    return [{
      id: `capability_window_${option.key}`,
      capabilities: [option.key],
      mode: "detection" as const,
      windows: draft.windows.map((window) => ({
        days: window.days || [],
        from: window.from,
        to: window.to,
      })),
    }]
  })
}

export function buildSafetyRuleOverridesPayload(
  options: CameraCapabilityEditorOption[],
  drafts: CameraCapabilityConfigDrafts
): Record<string, { confidence?: number; threshold?: number }> {
  const payload: Record<string, { confidence?: number; threshold?: number }> = {}
  for (const option of options) {
    if (!option.ruleId) continue
    const draft = drafts[option.key]
    if (!draft) continue
    const confidence = optionalNumber(draft.confidence)
    const threshold = optionalNumber(draft.threshold)
    const override: { confidence?: number; threshold?: number } = {}
    if (confidence !== null) override.confidence = confidence
    if (threshold !== null) override.threshold = Math.round(threshold)
    if (Object.keys(override).length > 0) payload[option.ruleId] = override
  }
  return payload
}

export function buildCapabilityModelOverridesPayload(
  options: CameraCapabilityEditorOption[],
  drafts: CameraCapabilityConfigDrafts
): Partial<Record<CapabilityKey, ModelKey>> {
  const payload: Partial<Record<CapabilityKey, ModelKey>> = {}
  for (const option of options) {
    const draft = drafts[option.key]
    if (option.supportsClosedSetCandidate && draft?.modelKey === "ppe_closed_set_candidate") {
      payload[option.key] = "ppe_closed_set_candidate"
    }
  }
  return payload
}

export function buildHelmetColourPolicyPayload(
  options: CameraCapabilityEditorOption[],
  drafts: CameraCapabilityConfigDrafts
): {
  enabled: boolean
  allowed_colours: string[]
  min_confidence: number
  confirmation_threshold: number
  severity: "P2"
} {
  const option = options.find((item) => item.key === "helmet_color_compliance")
  const draft = option ? drafts[option.key] : null
  return {
    enabled: Boolean(option && draft),
    allowed_colours: draft?.allowedHelmetColours || [],
    min_confidence: optionalNumber(draft?.confidence || "") ?? 0.45,
    confirmation_threshold: Math.round(optionalNumber(draft?.threshold || "") ?? 3),
    severity: "P2",
  }
}

export function CameraCapabilityConfigPanel({
  options,
  drafts,
  onChange,
}: {
  options: CameraCapabilityEditorOption[]
  drafts: CameraCapabilityConfigDrafts
  onChange: (drafts: CameraCapabilityConfigDrafts) => void
}) {
  function patch(capability: CapabilityKey, updates: Partial<CameraCapabilityConfigDraft>) {
    onChange({
      ...drafts,
      [capability]: {
        ...defaultCapabilityConfigDraft(capability),
        ...drafts[capability],
        ...updates,
      },
    })
  }

  return (
    <Card className="space-y-4">
      <div className="flex items-start gap-3">
        <div className="rounded-[var(--radius-md)] bg-[var(--color-info-bg)] p-2 text-[#1e40af]">
          <SlidersHorizontal className="h-4 w-4" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Per-Detection Runtime</h2>
          <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
            Tune detector windows, confidence, confirmation thresholds, and supported model routing per detection.
          </p>
        </div>
      </div>

      {options.length === 0 ? (
        <p className="text-sm text-[var(--color-text-secondary)]">
          Select detections above to configure runtime behavior.
        </p>
      ) : (
        <div className="space-y-3">
          {options.map((option) => {
            const draft = drafts[option.key] || defaultCapabilityConfigDraft(option.key)
            const isHelmetColour = option.key === "helmet_color_compliance"
            return (
              <details
                key={option.key}
                className="rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white p-3"
              >
                <summary className="flex cursor-pointer list-none flex-wrap items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="text-sm font-semibold text-[var(--color-text-primary)]">{option.label}</span>
                    {draft.limitRuntime ? (
                      <Badge variant="info">{draft.windows.length} active window{draft.windows.length !== 1 ? "s" : ""}</Badge>
                    ) : (
                      <Badge variant="success">Always runs</Badge>
                    )}
                    {draft.confidence && <Badge variant="default">conf {draft.confidence}</Badge>}
                    {draft.threshold && <Badge variant="default">{draft.threshold} hits</Badge>}
                    {isHelmetColour && draft.allowedHelmetColours.length > 0 && (
                      <Badge variant="info">Allow {draft.allowedHelmetColours.join(", ")}</Badge>
                    )}
                    {draft.modelKey === "ppe_closed_set_candidate" && <Badge variant="info">Trained detector</Badge>}
                  </div>
                  <span className="text-xs text-[var(--color-text-secondary)]">Configure</span>
                </summary>

                <div className="mt-4 space-y-4 border-t border-[var(--color-border-default)] pt-4">
                  <label className="inline-flex items-center gap-2 text-sm font-medium text-[var(--color-text-primary)]">
                    <input
                      type="checkbox"
                      checked={draft.limitRuntime}
                      onChange={(event) => patch(option.key, { limitRuntime: event.target.checked })}
                      className="rounded"
                    />
                    Run this detector only during selected windows
                  </label>

                  {draft.limitRuntime && (
                    <ScheduleWindowsEditor
                      windows={draft.windows}
                      onChange={(windows) => patch(option.key, { windows })}
                    />
                  )}

                  {isHelmetColour && (
                    <div className="space-y-2 rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-[var(--color-bg-tertiary)] p-3">
                      <div>
                        <p className="text-sm font-medium text-[var(--color-text-primary)]">Allowed helmet colours</p>
                        <p className="mt-1 text-xs text-[var(--color-text-secondary)]">
                          Select accepted colours to alert on a mismatch. Leave all unchecked to label colours without creating colour-compliance alerts.
                        </p>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {HELMET_COLOURS.map((colour) => {
                          const selected = draft.allowedHelmetColours.includes(colour.key)
                          return (
                            <label
                              key={colour.key}
                              className={cn(
                                "inline-flex cursor-pointer items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium",
                                selected
                                  ? "border-[var(--color-info)] bg-[var(--color-info-bg)] text-[var(--color-text-primary)]"
                                  : "border-[var(--color-border-default)] bg-white text-[var(--color-text-secondary)]"
                              )}
                            >
                              <input
                                type="checkbox"
                                checked={selected}
                                onChange={() => patch(option.key, {
                                  allowedHelmetColours: selected
                                    ? draft.allowedHelmetColours.filter((item) => item !== colour.key)
                                    : [...draft.allowedHelmetColours, colour.key],
                                })}
                                className="sr-only"
                              />
                              <span
                                className="h-3 w-3 rounded-full border border-black/15"
                                style={{ backgroundColor: colour.swatch }}
                              />
                              {colour.label}
                            </label>
                          )
                        })}
                      </div>
                    </div>
                  )}

                  <div className="grid gap-3 md:grid-cols-3">
                    <label className="space-y-1.5">
                      <span className="text-xs font-medium text-[var(--color-text-secondary)]">
                        {isHelmetColour ? "Minimum colour confidence" : "Minimum confidence"}
                      </span>
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.05"
                        value={draft.confidence}
                        onChange={(event) => patch(option.key, { confidence: event.target.value })}
                        placeholder={isHelmetColour ? "0.45" : "Rule default"}
                        className={inputClasses}
                      />
                    </label>
                    <label className="space-y-1.5">
                      <span className="text-xs font-medium text-[var(--color-text-secondary)]">
                        {isHelmetColour ? "Mismatch confirmation hits" : "Confirmation hits"}
                      </span>
                      <input
                        type="number"
                        min="1"
                        step="1"
                        value={draft.threshold}
                        onChange={(event) => patch(option.key, { threshold: event.target.value })}
                        placeholder={isHelmetColour ? "3" : "Rule default"}
                        className={inputClasses}
                      />
                    </label>
                    <label className="space-y-1.5">
                      <span className="text-xs font-medium text-[var(--color-text-secondary)]">Runtime route</span>
                      <select
                        value={draft.modelKey}
                        disabled={!option.supportsClosedSetCandidate}
                        onChange={(event) => patch(option.key, { modelKey: event.target.value as CameraCapabilityConfigDraft["modelKey"] })}
                        className={cn(inputClasses, !option.supportsClosedSetCandidate && "text-[var(--color-text-tertiary)]")}
                      >
                        <option value="default">Default detector</option>
                        <option value="ppe_closed_set_candidate">Trained apron / harness detector</option>
                      </select>
                    </label>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 text-xs text-[var(--color-text-secondary)]">
                    <Cpu className="h-3.5 w-3.5" />
                    <span>
                      {isHelmetColour
                        ? "Reuses the gated helmet detector; colour classification runs only inside fresh helmet boxes."
                        : option.ruleId
                          ? `YAML rule override: ${option.ruleId}`
                          : "No safety rule is mapped for threshold overrides."}
                    </span>
                    {option.classes.length > 0 && <span>Classes: {option.classes.join(", ")}</span>}
                  </div>
                </div>
              </details>
            )
          })}
        </div>
      )}
    </Card>
  )
}

function optionalNumber(value: string): number | null {
  if (!value.trim()) return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function numberString(value: unknown): string {
  if (value === null || value === undefined || value === "") return ""
  const parsed = Number(value)
  return Number.isFinite(parsed) ? String(parsed) : ""
}
