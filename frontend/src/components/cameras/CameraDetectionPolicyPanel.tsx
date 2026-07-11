import { BellRing, Mail, MessageSquare, Radio, Volume2, Webhook } from "lucide-react"
import type { ReactNode } from "react"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import type { AlertOutput, CapabilityKey, EngineRule, RuleAction, RuleCondition, RuleScheduleWindow, Severity } from "@/types"
import {
  defaultOutputIds,
  defaultScheduleWindow,
  DEFAULT_CAMERA_EVENT_MESSAGE,
  ScheduleWindowsEditor,
} from "./CameraEventPolicyPanel"
import type { CameraCapabilityEditorOption } from "./CameraCapabilityConfigPanel"

export const CAMERA_DETECTION_POLICY_PRESET_PREFIX = "camera_detection_policy:"

type DetectionSeverityMode = Severity | "inherit"

export interface CameraDetectionPolicyDraft {
  enabled: boolean
  severity: DetectionSeverityMode
  outputIds: string[]
  scheduleEnabled: boolean
  scheduleWindows: RuleScheduleWindow[]
  priority: number
  cooldownSeconds: number
  minConfidence: number
  messageTemplate: string
}

export type CameraDetectionPolicyDrafts = Partial<Record<CapabilityKey, CameraDetectionPolicyDraft>>

const ACTION_BY_OUTPUT_ID: Record<string, string> = {
  browser_sound: "play_sound",
  telegram: "send_telegram",
  email: "send_email",
  webhook: "webhook",
  pushover: "pushover",
  relay: "trigger_relay",
  relay_buzzer: "trigger_relay",
  plc: "trigger_plc",
}

const outputIcons: Record<string, ReactNode> = {
  browser_sound: <Volume2 className="h-4 w-4" />,
  telegram: <MessageSquare className="h-4 w-4" />,
  email: <Mail className="h-4 w-4" />,
  webhook: <Webhook className="h-4 w-4" />,
  relay: <Radio className="h-4 w-4" />,
  plc: <Radio className="h-4 w-4" />,
}

const inputClasses =
  "w-full rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 disabled:bg-[var(--color-bg-tertiary)] disabled:text-[var(--color-text-tertiary)]"

export function defaultDetectionPolicyDraft(outputs: AlertOutput[] = []): CameraDetectionPolicyDraft {
  return {
    enabled: false,
    severity: "inherit",
    outputIds: defaultOutputIds(outputs),
    scheduleEnabled: false,
    scheduleWindows: [defaultScheduleWindow()],
    priority: 5,
    cooldownSeconds: 60,
    minConfidence: 0.6,
    messageTemplate: DEFAULT_CAMERA_EVENT_MESSAGE,
  }
}

export function detectionPolicyPreset(capability: CapabilityKey): string {
  return `${CAMERA_DETECTION_POLICY_PRESET_PREFIX}${capability}`
}

export function findDetectionPolicyRule(
  rules: EngineRule[],
  cameraId: string,
  capability: CapabilityKey
): EngineRule | null {
  return rules.find(
    (rule) => (rule.cameras || []).includes(cameraId) && rule.preset === detectionPolicyPreset(capability)
  ) ?? null
}

export function detectionPolicyDraftsFromRules(
  rules: EngineRule[],
  cameraId: string | null,
  options: CameraCapabilityEditorOption[],
  outputs: AlertOutput[]
): CameraDetectionPolicyDrafts {
  const drafts: CameraDetectionPolicyDrafts = {}
  for (const option of options) {
    const rule = cameraId ? findDetectionPolicyRule(rules, cameraId, option.key) : null
    drafts[option.key] = detectionPolicyDraftFromRule(rule, outputs)
  }
  return drafts
}

export function reconcileDetectionPolicyDrafts(
  current: CameraDetectionPolicyDrafts,
  options: CameraCapabilityEditorOption[],
  outputs: AlertOutput[]
): CameraDetectionPolicyDrafts {
  const next: CameraDetectionPolicyDrafts = {}
  for (const option of options) {
    next[option.key] = current[option.key] || defaultDetectionPolicyDraft(outputs)
  }
  return next
}

export function validateDetectionPolicyDrafts(
  options: CameraCapabilityEditorOption[],
  drafts: CameraDetectionPolicyDrafts
): string | null {
  for (const option of options) {
    const draft = drafts[option.key]
    if (!draft?.enabled) continue
    if (draft.outputIds.length === 0) return `Select at least one alert channel for ${option.label}.`
    if (!Number.isFinite(draft.priority) || draft.priority < 1) return `${option.label} alert priority must be 1 or higher.`
    if (!Number.isFinite(draft.cooldownSeconds) || draft.cooldownSeconds < 0) return `${option.label} cooldown must be 0 or higher.`
    if (!Number.isFinite(draft.minConfidence) || draft.minConfidence < 0 || draft.minConfidence > 1) {
      return `${option.label} alert confidence must be between 0 and 1.`
    }
    if (draft.scheduleEnabled) {
      if (!draft.scheduleWindows.length) return `${option.label} alert policy needs at least one active window.`
      if (draft.scheduleWindows.some((window) => !window.from || !window.to)) return `${option.label} alert windows need start and end times.`
      if (draft.scheduleWindows.some((window) => window.days?.length === 0)) return `${option.label} alert windows need at least one day.`
    }
  }
  return null
}

export function buildDetectionPolicyPayload({
  cameraId,
  cameraName,
  cameraZone,
  option,
  draft,
  existingRule,
}: {
  cameraId: string
  cameraName: string
  cameraZone: string
  option: CameraCapabilityEditorOption
  draft: CameraDetectionPolicyDraft
  existingRule?: EngineRule | null
}): Omit<EngineRule, "id" | "lastTriggered"> {
  const conditions: RuleCondition[] = [
    { type: "class_is", params: { classes: detectionClassesForPolicy(option).join(",") } },
  ]
  if (draft.minConfidence > 0) {
    conditions.push({ type: "confidence_above", params: { value: String(roundConfidence(draft.minConfidence)) } })
  }

  const thenActions: RuleAction[] = [{ type: "create_alert", params: draft.severity === "inherit" ? {} : { severity: draft.severity } }]
  for (const outputId of draft.outputIds) {
    const actionType = ACTION_BY_OUTPUT_ID[outputId]
    if (actionType && !thenActions.some((action) => action.type === actionType)) {
      thenActions.push({ type: actionType, params: {} })
    }
  }

  return {
    name: existingRule?.name || `${option.label} Alerts - ${cameraName}`,
    description: existingRule?.description || `Per-detection alert policy for ${option.label} on ${cameraName} in ${cameraZone}.`,
    enabled: draft.enabled,
    trigger: "detection",
    cameras: [cameraId],
    conditions,
    thenActions,
    elseActions: existingRule?.elseActions || [],
    cooldownSeconds: Math.max(0, Math.round(draft.cooldownSeconds)),
    priority: Math.max(1, Math.round(draft.priority)),
    severity: draft.severity === "inherit" ? null : draft.severity,
    outputIds: draft.outputIds,
    messageTemplate: draft.messageTemplate.trim() || DEFAULT_CAMERA_EVENT_MESSAGE,
    schedule: draft.scheduleEnabled
      ? { windows: draft.scheduleWindows.map((window) => ({ days: window.days || [], from: window.from, to: window.to })) }
      : null,
    preset: detectionPolicyPreset(option.key),
  }
}

export function CameraDetectionPolicyEditor({
  options,
  drafts,
  onChange,
  alertOutputs,
}: {
  options: CameraCapabilityEditorOption[]
  drafts: CameraDetectionPolicyDrafts
  onChange: (drafts: CameraDetectionPolicyDrafts) => void
  alertOutputs: AlertOutput[]
}) {
  function patch(capability: CapabilityKey, updates: Partial<CameraDetectionPolicyDraft>) {
    onChange({
      ...drafts,
      [capability]: {
        ...defaultDetectionPolicyDraft(alertOutputs),
        ...drafts[capability],
        ...updates,
      },
    })
  }

  return (
    <Card className="space-y-4">
      <div className="flex items-start gap-3">
        <div className="rounded-[var(--radius-md)] bg-[var(--color-info-bg)] p-2 text-[#1e40af]">
          <BellRing className="h-4 w-4" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Per-Detection Alert Policies</h2>
          <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
            Override channels, severity, cooldown, message, and alert windows for individual detections on this camera.
          </p>
        </div>
      </div>

      {options.length === 0 ? (
        <p className="text-sm text-[var(--color-text-secondary)]">
          Select detections above to configure individual alert policies.
        </p>
      ) : (
        <div className="space-y-3">
          {options.map((option) => {
            const draft = drafts[option.key] || defaultDetectionPolicyDraft(alertOutputs)
            const disabled = !draft.enabled
            return (
              <details
                key={option.key}
                className="rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white p-3"
              >
                <summary className="flex cursor-pointer list-none flex-wrap items-center justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="text-sm font-semibold text-[var(--color-text-primary)]">{option.label}</span>
                    <Badge variant={draft.enabled ? "info" : "default"}>{draft.enabled ? "Custom policy" : "Global policy"}</Badge>
                    {draft.enabled && <Badge variant="default">{draft.outputIds.length} channel{draft.outputIds.length !== 1 ? "s" : ""}</Badge>}
                    {draft.scheduleEnabled && <Badge variant="info">{draft.scheduleWindows.length} alert window{draft.scheduleWindows.length !== 1 ? "s" : ""}</Badge>}
                  </div>
                  <span className="text-xs text-[var(--color-text-secondary)]">Configure</span>
                </summary>

                <div className="mt-4 space-y-4 border-t border-[var(--color-border-default)] pt-4">
                  <label className="inline-flex items-center gap-2 text-sm font-medium text-[var(--color-text-primary)]">
                    <input
                      type="checkbox"
                      checked={draft.enabled}
                      onChange={(event) => patch(option.key, { enabled: event.target.checked })}
                      className="rounded"
                    />
                    Use a custom alert policy for {option.label}
                  </label>

                  <div className={cn("space-y-4", disabled && "opacity-60")}>
                    <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                      {alertOutputs.map((output) => {
                        const checked = draft.outputIds.includes(output.id)
                        return (
                          <label
                            key={output.id}
                            className={cn(
                              "flex min-h-14 cursor-pointer items-start gap-2 rounded-[var(--radius-md)] border p-2 text-sm",
                              checked ? "border-[var(--color-info)] bg-[var(--color-info-bg)]" : "border-[var(--color-border-default)] bg-white",
                              disabled && "cursor-default"
                            )}
                          >
                            <input
                              type="checkbox"
                              disabled={disabled}
                              checked={checked}
                              onChange={(event) => {
                                patch(option.key, {
                                  outputIds: event.target.checked
                                    ? [...draft.outputIds, output.id]
                                    : draft.outputIds.filter((id) => id !== output.id),
                                })
                              }}
                              className="mt-1 rounded"
                            />
                            <span className="min-w-0">
                              <span className="flex items-center gap-1.5 font-medium text-[var(--color-text-primary)]">
                                {outputIcons[output.type] || <Radio className="h-4 w-4" />}
                                <span className="truncate">{output.name}</span>
                              </span>
                              <span className="text-xs text-[var(--color-text-secondary)]">{output.status.replace("_", " ")}</span>
                            </span>
                          </label>
                        )
                      })}
                    </div>

                    <div className="grid gap-3 md:grid-cols-4">
                      <label className="space-y-1.5">
                        <span className="text-xs font-medium text-[var(--color-text-secondary)]">Severity</span>
                        <select
                          disabled={disabled}
                          value={draft.severity}
                          onChange={(event) => patch(option.key, { severity: event.target.value as DetectionSeverityMode })}
                          className={inputClasses}
                        >
                          <option value="inherit">Use detection severity</option>
                          <option value="P1">P1 - Critical</option>
                          <option value="P2">P2 - High</option>
                          <option value="P3">P3 - Medium</option>
                          <option value="P4">P4 - Low</option>
                        </select>
                      </label>
                      <label className="space-y-1.5">
                        <span className="text-xs font-medium text-[var(--color-text-secondary)]">Priority</span>
                        <input
                          type="number"
                          min="1"
                          disabled={disabled}
                          value={draft.priority}
                          onChange={(event) => patch(option.key, { priority: numericValue(event.target.value, 1) })}
                          className={inputClasses}
                        />
                      </label>
                      <label className="space-y-1.5">
                        <span className="text-xs font-medium text-[var(--color-text-secondary)]">Cooldown seconds</span>
                        <input
                          type="number"
                          min="0"
                          disabled={disabled}
                          value={draft.cooldownSeconds}
                          onChange={(event) => patch(option.key, { cooldownSeconds: numericValue(event.target.value, 0) })}
                          className={inputClasses}
                        />
                      </label>
                      <label className="space-y-1.5">
                        <span className="text-xs font-medium text-[var(--color-text-secondary)]">Alert confidence</span>
                        <input
                          type="number"
                          min="0"
                          max="1"
                          step="0.05"
                          disabled={disabled}
                          value={draft.minConfidence}
                          onChange={(event) => patch(option.key, { minConfidence: numericValue(event.target.value, 0) })}
                          className={inputClasses}
                        />
                      </label>
                    </div>

                    <label className="space-y-1.5">
                      <span className="text-xs font-medium text-[var(--color-text-secondary)]">Message template</span>
                      <textarea
                        rows={2}
                        disabled={disabled}
                        value={draft.messageTemplate}
                        onChange={(event) => patch(option.key, { messageTemplate: event.target.value })}
                        className={cn(inputClasses, "resize-y")}
                      />
                    </label>

                    <div className="space-y-3 rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-[var(--color-bg-tertiary)] p-3">
                      <label className="inline-flex items-center gap-2 text-sm font-medium text-[var(--color-text-primary)]">
                        <input
                          type="checkbox"
                          disabled={disabled}
                          checked={draft.scheduleEnabled}
                          onChange={(event) => patch(option.key, { scheduleEnabled: event.target.checked })}
                          className="rounded"
                        />
                        Send alerts only during selected windows
                      </label>
                      {draft.scheduleEnabled && (
                        <ScheduleWindowsEditor
                          windows={draft.scheduleWindows}
                          disabled={disabled}
                          onChange={(scheduleWindows) => patch(option.key, { scheduleWindows })}
                        />
                      )}
                    </div>
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

function detectionPolicyDraftFromRule(rule: EngineRule | null, outputs: AlertOutput[]): CameraDetectionPolicyDraft {
  if (!rule) return defaultDetectionPolicyDraft(outputs)
  const window = rule.schedule?.windows?.length ? rule.schedule.windows : [defaultScheduleWindow()]
  return {
    enabled: rule.enabled,
    severity: rule.severity ?? severityFromActions(rule.thenActions) ?? "inherit",
    outputIds: rule.outputIds?.length ? rule.outputIds : outputIdsFromActions(rule.thenActions, outputs),
    scheduleEnabled: Boolean(rule.schedule?.windows?.length),
    scheduleWindows: window,
    priority: rule.priority ?? 5,
    cooldownSeconds: rule.cooldownSeconds ?? 60,
    minConfidence: confidenceFromConditions(rule.conditions),
    messageTemplate: rule.messageTemplate || DEFAULT_CAMERA_EVENT_MESSAGE,
  }
}

function detectionClassesForPolicy(option: CameraCapabilityEditorOption): string[] {
  const classes = new Set<string>()
  for (const value of [option.key, option.label, ...option.classes]) {
    if (value) classes.add(String(value).toLowerCase().replace(/[_-]+/g, " ").trim())
  }
  for (const value of Array.from(classes)) {
    classes.add(`missing ${value}`)
    classes.add(`no ${value}`)
  }
  return Array.from(classes)
}

function severityFromActions(actions: RuleAction[]): Severity | null {
  const value = actions.find((action) => action.type === "create_alert")?.params?.severity
  return value === "P1" || value === "P2" || value === "P3" || value === "P4" ? value : null
}

function outputIdsFromActions(actions: RuleAction[], outputs: AlertOutput[]): string[] {
  const byAction = new Map(Object.entries(ACTION_BY_OUTPUT_ID).map(([outputId, action]) => [action, outputId]))
  const ids = actions
    .map((action) => byAction.get(action.type))
    .filter((id): id is string => Boolean(id))
  return ids.length ? Array.from(new Set(ids)) : defaultOutputIds(outputs)
}

function confidenceFromConditions(conditions: RuleCondition[]): number {
  const raw = conditions.find((condition) => condition.type === "confidence_above")?.params?.value
  const parsed = Number(raw)
  return Number.isFinite(parsed) ? parsed : 0.6
}

function numericValue(value: string, fallback: number): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function roundConfidence(value: number): number {
  return Math.round(value * 100) / 100
}
