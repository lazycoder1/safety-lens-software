import type { ReactNode } from "react"
import { BellRing, CalendarClock, Mail, MessageSquare, Radio, SlidersHorizontal, Volume2, Webhook } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import type {
  AlertOutput,
  CapabilityKey,
  CapabilityWindow,
  EngineRule,
  RuleAction,
  RuleCondition,
  RuleSchedule,
  RuleScheduleWindow,
  Severity,
} from "@/types"

export const CAMERA_EVENT_POLICY_PRESET = "camera_event_default"
export const DEFAULT_CAMERA_EVENT_MESSAGE = "{severity} {violation_type} on {camera} in {zone}"

const DAYS = [
  { value: "mon", label: "Mon" },
  { value: "tue", label: "Tue" },
  { value: "wed", label: "Wed" },
  { value: "thu", label: "Thu" },
  { value: "fri", label: "Fri" },
  { value: "sat", label: "Sat" },
  { value: "sun", label: "Sun" },
]

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

const inputClasses =
  "w-full rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 disabled:bg-[var(--color-bg-tertiary)] disabled:text-[var(--color-text-tertiary)]"

const severityVariants: Record<Severity, "critical" | "high" | "warning" | "default"> = {
  P1: "critical",
  P2: "high",
  P3: "warning",
  P4: "default",
}

const outputIcons: Record<string, ReactNode> = {
  in_app: <BellRing className="h-4 w-4" />,
  browser_sound: <Volume2 className="h-4 w-4" />,
  telegram: <MessageSquare className="h-4 w-4" />,
  email: <Mail className="h-4 w-4" />,
  webhook: <Webhook className="h-4 w-4" />,
  relay: <Radio className="h-4 w-4" />,
  plc: <SlidersHorizontal className="h-4 w-4" />,
}

export type EventSeverityMode = Severity | "inherit"

export interface CameraEventPolicyDraft {
  enabled: boolean
  severity: EventSeverityMode
  outputIds: string[]
  scheduleEnabled: boolean
  scheduleWindows: RuleScheduleWindow[]
  priority: number
  cooldownSeconds: number
  minConfidence: number
  messageTemplate: string
}

export interface CameraDetectionWindowDraft {
  enabled: boolean
  days: string[]
  from: string
  to: string
}

export function defaultCameraEventPolicyDraft(outputs: AlertOutput[] = []): CameraEventPolicyDraft {
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

export function defaultCameraDetectionWindowDraft(): CameraDetectionWindowDraft {
  return {
    enabled: false,
    days: DAYS.map((day) => day.value),
    from: "00:00",
    to: "23:59",
  }
}

export function defaultScheduleWindow(): RuleScheduleWindow {
  return {
    days: DAYS.map((day) => day.value),
    from: "00:00",
    to: "23:59",
  }
}

export function normalizeCapabilityWindows(raw: CapabilityWindow[] | Record<string, any> | null | undefined): CapabilityWindow[] {
  if (!raw) return []
  if (Array.isArray(raw)) return raw.filter((item): item is CapabilityWindow => Boolean(item && typeof item === "object"))
  if (typeof raw !== "object") return []
  const normalized: CapabilityWindow[] = []
  Object.entries(raw).forEach(([capability, value], index) => {
    if (Array.isArray(value)) {
      normalized.push({
        id: `capability_window_${index + 1}`,
        capabilities: [capability as CapabilityKey],
        mode: "detection",
        windows: value,
      })
      return
    }
    if (value && typeof value === "object") {
      const window: CapabilityWindow = {
        id: String(value.id || `capability_window_${index + 1}`),
        capabilities: Array.isArray(value.capabilities)
          ? value.capabilities
          : [String(value.capability || capability) as CapabilityKey],
        mode: (value.mode || "detection") as CapabilityWindow["mode"],
        windows: Array.isArray(value.windows) ? value.windows : [],
      }
      if (typeof value.active === "boolean") {
        window.active = value.active
      }
      normalized.push(window)
    }
  })
  return normalized
}

export function defaultOutputIds(outputs: AlertOutput[]): string[] {
  const inApp = outputs.find((output) => output.id === "in_app")
  if (inApp) return [inApp.id]
  const firstEnabled = outputs.find((output) => output.enabled)
  return firstEnabled ? [firstEnabled.id] : []
}

export function cameraDetectionWindowDraftFromWindows(
  windows: CapabilityWindow[] | Record<string, any> | null | undefined,
  selectedCapabilities: CapabilityKey[] = []
): CameraDetectionWindowDraft {
  const selected = new Set(selectedCapabilities)
  const first = normalizeCapabilityWindows(windows).find((window) => {
    if (!["detection", "detector", "detector_off"].includes(window.mode || "detection")) return false
    if (!window.windows?.[0]) return false
    if (selected.size === 0) return true
    return (window.capabilities || []).some((capability) => selected.has(capability))
  })
  const activeWindow = first?.windows?.[0]
  if (!activeWindow) return defaultCameraDetectionWindowDraft()
  return {
    enabled: true,
    days: activeWindow.days?.length ? activeWindow.days : DAYS.map((day) => day.value),
    from: activeWindow.from || "00:00",
    to: activeWindow.to || "23:59",
  }
}

export function cameraScopedAutomationRules(rules: EngineRule[], cameraId: string): EngineRule[] {
  return rules.filter((rule) => (rule.cameras || []).includes(cameraId))
}

export function findDefaultCameraEventRule(rules: EngineRule[], cameraId: string): EngineRule | null {
  return cameraScopedAutomationRules(rules, cameraId).find((rule) => rule.preset === CAMERA_EVENT_POLICY_PRESET) ?? null
}

export function cameraEventPolicyDraftFromRule(rule: EngineRule | null, outputs: AlertOutput[]): CameraEventPolicyDraft {
  if (!rule) return defaultCameraEventPolicyDraft(outputs)
  const windows = rule.schedule?.windows?.length ? rule.schedule.windows : [defaultScheduleWindow()]
  return {
    enabled: rule.enabled,
    severity: rule.severity ?? severityFromActions(rule.thenActions) ?? "inherit",
    outputIds: rule.outputIds?.length ? rule.outputIds : outputIdsFromActions(rule.thenActions, outputs),
    scheduleEnabled: Boolean(rule.schedule?.windows?.length),
    scheduleWindows: windows,
    priority: rule.priority ?? 5,
    cooldownSeconds: rule.cooldownSeconds ?? 60,
    minConfidence: confidenceFromConditions(rule.conditions),
    messageTemplate: rule.messageTemplate || DEFAULT_CAMERA_EVENT_MESSAGE,
  }
}

export function validateCameraEventPolicyDraft(draft: CameraEventPolicyDraft): string | null {
  if (!draft.enabled) return null
  if (draft.outputIds.length === 0) return "Select at least one event channel for this camera."
  if (draft.scheduleEnabled) {
    if (draft.scheduleWindows.length === 0) return "Add at least one event active window for this camera."
    if (draft.scheduleWindows.some((window) => !window.from || !window.to)) return "Each event active window needs a start and end time."
    if (draft.scheduleWindows.some((window) => window.days?.length === 0)) return "Each event active window needs at least one day."
  }
  if (!Number.isFinite(draft.priority) || draft.priority < 1) return "Priority must be 1 or higher."
  if (!Number.isFinite(draft.cooldownSeconds) || draft.cooldownSeconds < 0) return "Cooldown must be 0 seconds or higher."
  if (!Number.isFinite(draft.minConfidence) || draft.minConfidence < 0 || draft.minConfidence > 1) {
    return "Minimum confidence must be between 0 and 1."
  }
  return null
}

export function validateCameraDetectionWindowDraft(
  draft: CameraDetectionWindowDraft,
  selectedCapabilities: CapabilityKey[]
): string | null {
  if (!draft.enabled) return null
  if (selectedCapabilities.length === 0) return "Select at least one detection before enabling a detector window."
  if (draft.days.length === 0) return "Select at least one detector-active day."
  if (!draft.from || !draft.to) return "Detector window needs a start and end time."
  return null
}

export function buildCapabilityWindowsPayload(
  selectedCapabilities: CapabilityKey[],
  draft: CameraDetectionWindowDraft
): CapabilityWindow[] {
  if (!draft.enabled || selectedCapabilities.length === 0) return []
  return [
    {
      id: "camera_detection_active_window",
      capabilities: selectedCapabilities,
      mode: "detection",
      windows: [{ days: draft.days, from: draft.from, to: draft.to }],
    },
  ]
}

export function buildCameraEventPolicyPayload({
  cameraId,
  cameraName,
  cameraZone,
  draft,
  existingRule,
}: {
  cameraId: string
  cameraName: string
  cameraZone: string
  draft: CameraEventPolicyDraft
  existingRule?: EngineRule | null
}): Omit<EngineRule, "id" | "lastTriggered"> {
  const preservedConditions = (existingRule?.conditions || []).filter(
    (condition) => condition.type !== "confidence_above"
  )
  const conditions: RuleCondition[] = [...preservedConditions]
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
    name: existingRule?.name || `Default Event Policy - ${cameraName}`,
    description: existingRule?.description || `Camera-level event routing for ${cameraName} in ${cameraZone}.`,
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
    preset: CAMERA_EVENT_POLICY_PRESET,
  }
}

export function CameraDetectionWindowEditor({
  draft,
  onChange,
  selectedLabels,
}: {
  draft: CameraDetectionWindowDraft
  onChange: (draft: CameraDetectionWindowDraft) => void
  selectedLabels: string[]
}) {
  const disabled = !draft.enabled

  function patch(updates: Partial<CameraDetectionWindowDraft>) {
    onChange({ ...draft, ...updates })
  }

  return (
    <Card className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className="rounded-[var(--radius-md)] bg-[var(--color-info-bg)] p-2 text-[#1e40af]">
            <CalendarClock className="h-4 w-4" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Detection Schedule</h2>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {selectedLabels.length ? (
                selectedLabels.map((label) => (
                  <Badge key={label} variant="default">
                    {label}
                  </Badge>
                ))
              ) : (
                <Badge variant="warning">No detections selected</Badge>
              )}
            </div>
          </div>
        </div>
        <label className="inline-flex items-center gap-2 rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white px-3 py-2 text-sm font-medium text-[var(--color-text-primary)]">
          <input
            type="checkbox"
            checked={draft.enabled}
            onChange={(event) => patch({ enabled: event.target.checked })}
            className="rounded"
          />
          Limit detector runtime
        </label>
      </div>

      {draft.enabled && (
        <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_150px_150px]">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 text-xs font-medium text-[var(--color-text-secondary)]">
              <CalendarClock className="h-3.5 w-3.5" />
              Active days
            </div>
            <div className="flex flex-wrap gap-2">
              {DAYS.map((day) => (
                <label
                  key={day.value}
                  className={cn(
                    "inline-flex cursor-pointer items-center gap-1.5 rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-[var(--color-bg-tertiary)] px-2 py-1 text-xs text-[var(--color-text-primary)]",
                    draft.days.includes(day.value) && "border-[var(--color-info)] bg-[var(--color-info-bg)]",
                    disabled && "cursor-default opacity-60"
                  )}
                >
                  <input
                    type="checkbox"
                    disabled={disabled}
                    checked={draft.days.includes(day.value)}
                    onChange={(event) => {
                      patch({
                        days: event.target.checked
                          ? [...draft.days, day.value]
                          : draft.days.filter((value) => value !== day.value),
                      })
                    }}
                    className="rounded"
                  />
                  {day.label}
                </label>
              ))}
            </div>
          </div>
          <label className="space-y-1.5">
            <span className="text-xs font-medium text-[var(--color-text-secondary)]">From</span>
            <input
              type="time"
              disabled={disabled}
              value={draft.from}
              onChange={(event) => patch({ from: event.target.value })}
              className={inputClasses}
            />
          </label>
          <label className="space-y-1.5">
            <span className="text-xs font-medium text-[var(--color-text-secondary)]">To</span>
            <input
              type="time"
              disabled={disabled}
              value={draft.to}
              onChange={(event) => patch({ to: event.target.value })}
              className={inputClasses}
            />
          </label>
        </div>
      )}
    </Card>
  )
}

export function CameraEventPolicyEditor({
  draft,
  onChange,
  alertOutputs,
  cameraScopedRules,
}: {
  draft: CameraEventPolicyDraft
  onChange: (draft: CameraEventPolicyDraft) => void
  alertOutputs: AlertOutput[]
  cameraScopedRules: EngineRule[]
}) {
  const advancedRules = cameraScopedRules.filter((rule) => rule.preset !== CAMERA_EVENT_POLICY_PRESET)
  const disabled = !draft.enabled
  const messagePreview = renderPreview(draft.messageTemplate)

  function patch(updates: Partial<CameraEventPolicyDraft>) {
    onChange({ ...draft, ...updates })
  }

  return (
    <Card className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className="rounded-[var(--radius-md)] bg-[var(--color-info-bg)] p-2 text-[#1e40af]">
            <BellRing className="h-4 w-4" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Event Handling</h2>
            <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
              Camera-specific channels, active windows, severity, priority, and message text.
            </p>
          </div>
        </div>
        <label className="inline-flex items-center gap-2 rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white px-3 py-2 text-sm font-medium text-[var(--color-text-primary)]">
          <input
            type="checkbox"
            checked={draft.enabled}
            onChange={(event) => patch({ enabled: event.target.checked })}
            className="rounded"
          />
          Use camera policy
        </label>
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_220px]">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium text-[var(--color-text-primary)]">
            <BellRing className="h-4 w-4" />
            Event channels
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {alertOutputs.map((output) => {
              const checked = draft.outputIds.includes(output.id)
              return (
                <label
                  key={output.id}
                  className={cn(
                    "flex min-h-16 cursor-pointer items-start gap-3 rounded-[var(--radius-md)] border p-3 text-sm transition-colors",
                    checked
                      ? "border-[var(--color-info)] bg-[var(--color-info-bg)]"
                      : "border-[var(--color-border-default)] bg-white hover:border-[var(--color-border-active)]",
                    disabled && "cursor-default opacity-60"
                  )}
                >
                  <input
                    type="checkbox"
                    disabled={disabled}
                    checked={checked}
                    onChange={(event) => {
                      patch({
                        outputIds: event.target.checked
                          ? [...draft.outputIds, output.id]
                          : draft.outputIds.filter((id) => id !== output.id),
                      })
                    }}
                    className="mt-1 rounded"
                  />
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-2 font-medium text-[var(--color-text-primary)]">
                      {outputIcons[output.type] || <Radio className="h-4 w-4" />}
                      <span className="truncate">{output.name}</span>
                    </span>
                    <span className="mt-1 flex flex-wrap items-center gap-1.5">
                      <Badge variant={output.enabled ? "success" : "warning"}>{output.enabled ? "enabled" : "off"}</Badge>
                      <Badge variant={output.status === "ready" ? "success" : output.status === "failed" ? "critical" : "default"}>
                        {output.status.replace("_", " ")}
                      </Badge>
                    </span>
                  </span>
                </label>
              )
            })}
            {alertOutputs.length === 0 && (
              <div className="rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white p-3 text-sm text-[var(--color-text-secondary)]">
                No event channels are configured yet.
              </div>
            )}
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
          <label className="space-y-1.5">
            <span className="text-xs font-medium text-[var(--color-text-secondary)]">Severity</span>
            <select
              disabled={disabled}
              value={draft.severity}
              onChange={(event) => patch({ severity: event.target.value as EventSeverityMode })}
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
              max="100"
              disabled={disabled}
              value={draft.priority}
              onChange={(event) => patch({ priority: numericValue(event.target.value, 1) })}
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
              onChange={(event) => patch({ cooldownSeconds: numericValue(event.target.value, 0) })}
              className={inputClasses}
            />
          </label>

          <label className="space-y-1.5">
            <span className="text-xs font-medium text-[var(--color-text-secondary)]">Minimum confidence</span>
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              disabled={disabled}
              value={draft.minConfidence}
              onChange={(event) => patch({ minConfidence: numericValue(event.target.value, 0) })}
              className={inputClasses}
            />
          </label>
        </div>
      </div>

      <div className="space-y-3 rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white p-3">
        <label className="flex items-center gap-2 text-sm font-medium text-[var(--color-text-primary)]">
          <input
            type="checkbox"
            disabled={disabled}
            checked={draft.scheduleEnabled}
            onChange={(event) => patch({ scheduleEnabled: event.target.checked })}
            className="rounded"
          />
          Active only during selected times
        </label>
        {draft.scheduleEnabled && (
          <ScheduleWindowsEditor
            windows={draft.scheduleWindows}
            disabled={disabled}
            onChange={(scheduleWindows) => patch({ scheduleWindows })}
          />
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_280px]">
        <label className="space-y-1.5">
          <span className="text-xs font-medium text-[var(--color-text-secondary)]">Message template</span>
          <textarea
            disabled={disabled}
            rows={3}
            value={draft.messageTemplate}
            onChange={(event) => patch({ messageTemplate: event.target.value })}
            className={cn(inputClasses, "resize-y")}
          />
        </label>
        <div className="rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-[var(--color-bg-tertiary)] p-3">
          <p className="text-xs font-medium text-[var(--color-text-secondary)]">Preview</p>
          <p className="mt-2 text-sm font-medium text-[var(--color-text-primary)]">{messagePreview}</p>
        </div>
      </div>

      {advancedRules.length > 0 && (
        <div className="rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-[var(--color-bg-tertiary)] px-3 py-2">
          <p className="text-sm font-medium text-[var(--color-text-primary)]">
            {advancedRules.length} advanced camera rule{advancedRules.length !== 1 ? "s" : ""} will stay attached.
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            {advancedRules.map((rule) => (
              <Badge key={rule.id} variant={rule.enabled ? "info" : "default"}>
                {rule.name}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </Card>
  )
}

export function ScheduleWindowsEditor({
  windows,
  disabled,
  onChange,
}: {
  windows: RuleScheduleWindow[]
  disabled?: boolean
  onChange: (windows: RuleScheduleWindow[]) => void
}) {
  const normalizedWindows = windows.length ? windows : [defaultScheduleWindow()]

  function updateWindow(index: number, updates: Partial<RuleScheduleWindow>) {
    onChange(normalizedWindows.map((window, itemIndex) => (itemIndex === index ? { ...window, ...updates } : window)))
  }

  function removeWindow(index: number) {
    const next = normalizedWindows.filter((_, itemIndex) => itemIndex !== index)
    onChange(next.length ? next : [defaultScheduleWindow()])
  }

  return (
    <div className="space-y-3">
      {normalizedWindows.map((window, index) => {
        const days = window.days || []
        return (
          <div
            key={`${index}-${window.from}-${window.to}`}
            className="rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-[var(--color-bg-tertiary)] p-3"
          >
            <div className="mb-3 flex items-center justify-between gap-3">
              <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">
                Window {index + 1}
              </p>
              {normalizedWindows.length > 1 && (
                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => removeWindow(index)}
                  className="text-xs font-medium text-[var(--color-danger)] hover:underline disabled:opacity-50"
                >
                  Remove
                </button>
              )}
            </div>
            <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_140px_140px]">
              <div className="space-y-1.5">
                <div className="flex items-center gap-2 text-xs font-medium text-[var(--color-text-secondary)]">
                  <CalendarClock className="h-3.5 w-3.5" />
                  Days
                </div>
                <div className="flex flex-wrap gap-2">
                  {DAYS.map((day) => (
                    <label
                      key={day.value}
                      className={cn(
                        "inline-flex cursor-pointer items-center gap-1.5 rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white px-2 py-1 text-xs text-[var(--color-text-primary)]",
                        days.includes(day.value) && "border-[var(--color-info)] bg-[var(--color-info-bg)]",
                        disabled && "cursor-default opacity-60"
                      )}
                    >
                      <input
                        type="checkbox"
                        disabled={disabled}
                        checked={days.includes(day.value)}
                        onChange={(event) => {
                          updateWindow(index, {
                            days: event.target.checked
                              ? [...days, day.value]
                              : days.filter((value) => value !== day.value),
                          })
                        }}
                        className="rounded"
                      />
                      {day.label}
                    </label>
                  ))}
                </div>
              </div>
              <label className="space-y-1.5">
                <span className="text-xs font-medium text-[var(--color-text-secondary)]">From</span>
                <input
                  type="time"
                  disabled={disabled}
                  value={window.from}
                  onChange={(event) => updateWindow(index, { from: event.target.value })}
                  className={inputClasses}
                />
              </label>
              <label className="space-y-1.5">
                <span className="text-xs font-medium text-[var(--color-text-secondary)]">To</span>
                <input
                  type="time"
                  disabled={disabled}
                  value={window.to}
                  onChange={(event) => updateWindow(index, { to: event.target.value })}
                  className={inputClasses}
                />
              </label>
            </div>
          </div>
        )
      })}
      <button
        type="button"
        disabled={disabled}
        onClick={() => onChange([...normalizedWindows, defaultScheduleWindow()])}
        className="text-sm font-medium text-[var(--color-info)] hover:underline disabled:opacity-50"
      >
        Add another window
      </button>
    </div>
  )
}

export function CameraEventPolicySummary({
  cameraId,
  rules,
  alertOutputs,
}: {
  cameraId: string
  rules: EngineRule[]
  alertOutputs: AlertOutput[]
}) {
  const cameraRules = cameraScopedAutomationRules(rules, cameraId)
  const defaultRule = findDefaultCameraEventRule(rules, cameraId)
  const advancedRules = cameraRules.filter((rule) => rule.preset !== CAMERA_EVENT_POLICY_PRESET)
  const channelText = defaultRule ? outputNames(defaultRule, alertOutputs) : "Global alert routing"
  const severity = defaultRule?.severity ?? severityFromActions(defaultRule?.thenActions || [])

  return (
    <Card className="space-y-4">
      <div>
        <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Event Handling</h2>
        <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
          Camera-level delivery policy and active alert windows.
        </p>
      </div>
      {defaultRule ? (
        <div className="space-y-3">
          <InfoLine label="Policy" value={<Badge variant={defaultRule.enabled ? "success" : "default"}>{defaultRule.enabled ? "Active" : "Off"}</Badge>} />
          <InfoLine label="Channels" value={channelText} />
          <InfoLine label="Active window" value={scheduleSummary(defaultRule.schedule)} />
          <InfoLine
            label="Severity"
            value={
              severity ? (
                <Badge variant={severityVariants[severity]}>{severity}</Badge>
              ) : (
                "Detection severity"
              )
            }
          />
          <InfoLine label="Priority" value={String(defaultRule.priority ?? 5)} />
          <InfoLine label="Cooldown" value={`${defaultRule.cooldownSeconds ?? 60}s`} />
          <InfoLine label="Message" value={defaultRule.messageTemplate || DEFAULT_CAMERA_EVENT_MESSAGE} />
        </div>
      ) : (
        <p className="text-sm text-[var(--color-text-secondary)]">
          No default camera policy is configured. This camera currently uses global routing and rule behavior.
        </p>
      )}
      {advancedRules.length > 0 && (
        <div className="border-t border-[var(--color-border-default)] pt-3">
          <p className="text-sm text-[var(--color-text-secondary)]">Advanced camera rules</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {advancedRules.map((rule) => (
              <Badge key={rule.id} variant={rule.enabled ? "info" : "default"}>
                {rule.name}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </Card>
  )
}

function InfoLine({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <span className="text-sm text-[var(--color-text-secondary)]">{label}</span>
      <span className="min-w-0 max-w-[65%] text-right text-sm font-medium text-[var(--color-text-primary)]">{value}</span>
    </div>
  )
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

function outputNames(rule: EngineRule, outputs: AlertOutput[]): string {
  const ids = rule.outputIds || []
  if (ids.length === 0) return "Global alert routing"
  const names = ids.map((id) => outputs.find((output) => output.id === id)?.name || id)
  return names.join(", ")
}

function scheduleSummary(schedule?: RuleSchedule | null): string {
  const windows = schedule?.windows || []
  if (!windows.length) return "Always active"
  return windows
    .map((window) => {
      const dayLabels = window.days?.length
        ? window.days.map((value) => DAYS.find((day) => day.value === value)?.label || value).join(", ")
        : "Every day"
      return `${dayLabels}, ${window.from} to ${window.to}`
    })
    .join("; ")
}

function renderPreview(template: string): string {
  return (template || DEFAULT_CAMERA_EVENT_MESSAGE)
    .replaceAll("{severity}", "P2")
    .replaceAll("{violation_type}", "Missing Helmet")
    .replaceAll("{camera}", "Main Gate Camera")
    .replaceAll("{camera_id}", "camera")
    .replaceAll("{zone}", "Main Gate")
    .replaceAll("{confidence}", "0.86")
    .replaceAll("{timestamp}", "10:30")
    .replaceAll("{description}", "Person detected without required PPE")
}

function numericValue(value: string, fallback: number): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function roundConfidence(value: number): number {
  return Math.round(value * 100) / 100
}
