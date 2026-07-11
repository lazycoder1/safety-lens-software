import { useState, useEffect } from "react"
import type { Dispatch, ReactNode, SetStateAction } from "react"
import { useParams, useNavigate, useLocation } from "react-router-dom"
import {
  ArrowLeft,
  Plus,
  X,
  ChevronDown,
  ChevronRight,
  HardHat,
  Flame,
  DoorOpen,
  Moon,
  Users,
  Settings,
} from "lucide-react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import {
  triggerOptions,
  conditionTypes,
  actionTypes,
  presetTemplates,
} from "@/data/mockRules"
import { createAutomationRule, getAlertOutputs, getAutomationRules, getCameras, updateAutomationRule } from "@/lib/api"
import type { AlertOutput, Camera, EngineRule, RuleAction, RuleCondition, Severity } from "@/types"

/* ── constants ───────────────────────────────────────────────────── */

const DEFAULT_MESSAGE_TEMPLATE = "{severity} {violation_type} on {camera} in {zone}"

const DAYS = [
  { value: "mon", label: "Mon" },
  { value: "tue", label: "Tue" },
  { value: "wed", label: "Wed" },
  { value: "thu", label: "Thu" },
  { value: "fri", label: "Fri" },
  { value: "sat", label: "Sat" },
  { value: "sun", label: "Sun" },
]

const ACTION_OUTPUT_IDS: Record<string, string> = {
  send_telegram: "telegram",
  send_email: "email",
  webhook: "webhook",
  play_sound: "browser_sound",
  trigger_plc: "plc",
  trigger_relay: "relay_buzzer",
  relay: "relay_buzzer",
  pushover: "pushover",
}

const PRESET_ICONS: Record<string, ReactNode> = {
  HardHat: <HardHat className="h-5 w-5" />,
  Flame: <Flame className="h-5 w-5" />,
  DoorOpen: <DoorOpen className="h-5 w-5" />,
  Moon: <Moon className="h-5 w-5" />,
  Users: <Users className="h-5 w-5" />,
  Settings: <Settings className="h-5 w-5" />,
}

const inputClasses =
  "w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"

const selectClasses = cn(inputClasses, "cursor-pointer")

/* ── helpers ─────────────────────────────────────────────────────── */

function emptyRule(): Omit<EngineRule, "id" | "lastTriggered"> {
  return {
    name: "",
    description: "",
    enabled: true,
    trigger: "detection",
    cameras: [],
    conditions: [],
    thenActions: [],
    elseActions: [],
    cooldownSeconds: 60,
    priority: 5,
    severity: "P2",
    outputIds: [],
    messageTemplate: DEFAULT_MESSAGE_TEMPLATE,
    schedule: null,
    preset: null,
  }
}

function severityFromActions(actions: RuleAction[], fallback: Severity = "P2"): Severity {
  const createAlert = actions.find((action) => action.type === "create_alert")
  const value = createAlert?.params?.severity
  return value === "P1" || value === "P2" || value === "P3" || value === "P4" ? value : fallback
}

function outputIdsFromActions(actions: RuleAction[]): string[] {
  return Array.from(
    new Set(
      actions
        .map((action) => ACTION_OUTPUT_IDS[action.type])
        .filter((outputId): outputId is string => Boolean(outputId))
    )
  )
}

function firstTimeWindow(conditions: RuleCondition[]) {
  return conditions.find((condition) => condition.type === "time_between")?.params
}

/* ── component ───────────────────────────────────────────────────── */

export function RuleEditor() {
  const { ruleId } = useParams<{ ruleId: string }>()
  const navigate = useNavigate()
  const location = useLocation()

  const isNew = !ruleId || ruleId === "new"

  // form state
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [trigger, setTrigger] = useState("detection")
  const [cameras, setCameras] = useState<string[]>([])
  const [allCameras, setAllCameras] = useState(true)
  const [conditions, setConditions] = useState<RuleCondition[]>([])
  const [thenActions, setThenActions] = useState<RuleAction[]>([])
  const [elseActions, setElseActions] = useState<RuleAction[]>([])
  const [showElse, setShowElse] = useState(false)
  const [cooldown, setCooldown] = useState(60)
  const [priority, setPriority] = useState(5)
  const [severity, setSeverity] = useState<Severity>("P2")
  const [outputIds, setOutputIds] = useState<string[]>([])
  const [messageTemplate, setMessageTemplate] = useState(DEFAULT_MESSAGE_TEMPLATE)
  const [scheduleEnabled, setScheduleEnabled] = useState(false)
  const [scheduleDays, setScheduleDays] = useState<string[]>(DAYS.map((day) => day.value))
  const [scheduleFrom, setScheduleFrom] = useState("00:00")
  const [scheduleTo, setScheduleTo] = useState("23:59")
  const [preset, setPreset] = useState<string | null>(null)
  const [showPresets, setShowPresets] = useState(isNew)
  const [existingRule, setExistingRule] = useState<EngineRule | null>(null)
  const [availableCameras, setAvailableCameras] = useState<Camera[]>([])
  const [alertOutputs, setAlertOutputs] = useState<AlertOutput[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [loadError, setLoadError] = useState("")
  const [saveError, setSaveError] = useState("")

  // load cameras, outputs, and existing rule
  useEffect(() => {
    let mounted = true
    async function load() {
      setLoading(true)
      setLoadError("")
      try {
        const [rules, cameraList, outputs] = await Promise.all([
          getAutomationRules(),
          getCameras(),
          getAlertOutputs(),
        ])
        if (!mounted) return
        setAvailableCameras(cameraList)
        setAlertOutputs(outputs)
        if (!isNew) {
          const rule = rules.find((item) => item.id === ruleId)
          if (!rule) {
            setLoadError("Rule not found")
            return
          }
          setExistingRule(rule)
          applyRule(rule)
          setShowPresets(false)
        }
      } catch (err) {
        if (!mounted) return
        setLoadError(err instanceof Error ? err.message : "Could not load rule editor")
      } finally {
        if (mounted) setLoading(false)
      }
    }
    load()
    return () => {
      mounted = false
    }
  }, [isNew, ruleId])

  // check for preset passed via location state
  useEffect(() => {
    const state = location.state as { preset?: string } | null
    if (state?.preset) {
      applyPreset(state.preset)
    }
  }, [location.state])

  function applyRule(rule: EngineRule) {
    setName(rule.name)
    setDescription(rule.description)
    setTrigger(rule.trigger)
    setCameras(rule.cameras)
    setAllCameras(rule.cameras.length === 0)
    setConditions(rule.conditions)
    setThenActions(rule.thenActions)
    setElseActions(rule.elseActions)
    setShowElse(rule.elseActions.length > 0)
    setCooldown(rule.cooldownSeconds)
    setPriority(rule.priority)
    setSeverity(rule.severity ?? severityFromActions(rule.thenActions))
    setOutputIds(rule.outputIds ?? outputIdsFromActions(rule.thenActions))
    setMessageTemplate(rule.messageTemplate || DEFAULT_MESSAGE_TEMPLATE)
    const window = rule.schedule?.windows?.[0]
    setScheduleEnabled(Boolean(window))
    setScheduleDays(window?.days?.length ? window.days : DAYS.map((day) => day.value))
    setScheduleFrom(window?.from ?? "00:00")
    setScheduleTo(window?.to ?? "23:59")
    setPreset(rule.preset)
  }

  function applyPreset(key: string) {
    const tpl = presetTemplates.find((p) => p.key === key)
    if (!tpl) return
    const t = tpl.template
    const window = firstTimeWindow(t.conditions)
    setName(t.name)
    setDescription(t.description)
    setTrigger(t.trigger)
    setCameras(t.cameras)
    setAllCameras(t.cameras.length === 0)
    setConditions(t.conditions.map((c) => ({ ...c, params: { ...c.params } })))
    setThenActions(t.thenActions.map((a) => ({ ...a, params: { ...a.params } })))
    setElseActions(t.elseActions.map((a) => ({ ...a, params: { ...a.params } })))
    setShowElse(t.elseActions.length > 0)
    setCooldown(t.cooldownSeconds)
    setPriority(t.priority)
    setSeverity(severityFromActions(t.thenActions))
    setOutputIds(outputIdsFromActions(t.thenActions))
    setMessageTemplate(DEFAULT_MESSAGE_TEMPLATE)
    setScheduleEnabled(Boolean(window))
    setScheduleDays(DAYS.map((day) => day.value))
    setScheduleFrom(window?.from ?? "00:00")
    setScheduleTo(window?.to ?? "23:59")
    setPreset(t.preset)
    setShowPresets(false)
  }

  async function handleSave() {
    setSaving(true)
    setSaveError("")
    const payload: Omit<EngineRule, "id" | "lastTriggered"> = {
      name,
      description,
      enabled: existingRule?.enabled ?? true,
      trigger,
      cameras: allCameras ? [] : cameras,
      conditions,
      thenActions,
      elseActions: showElse ? elseActions : [],
      cooldownSeconds: cooldown,
      priority,
      severity,
      outputIds,
      messageTemplate,
      schedule: scheduleEnabled
        ? { windows: [{ days: scheduleDays, from: scheduleFrom, to: scheduleTo }] }
        : null,
      preset,
    }
    try {
      if (isNew) {
        await createAutomationRule(payload)
      } else {
        await updateAutomationRule(ruleId!, payload)
      }
      navigate("/configure/rules")
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Could not save rule")
    } finally {
      setSaving(false)
    }
  }

  /* ── condition rows ───────────────────────────────────────────── */

  function updateCondition(index: number, updates: Partial<RuleCondition>) {
    setConditions((prev) =>
      prev.map((c, i) => {
        if (i !== index) return c
        if (updates.type && updates.type !== c.type) {
          return { type: updates.type, params: {} }
        }
        return { ...c, ...updates, params: { ...c.params, ...updates.params } }
      })
    )
  }

  function renderConditionParams(c: RuleCondition, index: number) {
    switch (c.type) {
      case "plate_in_list":
        return (
          <select
            value={c.params.list ?? ""}
            onChange={(e) => updateCondition(index, { params: { list: e.target.value } })}
            className={selectClasses}
          >
            <option value="">Select list...</option>
            <option value="Whitelist">Whitelist</option>
            <option value="Blocked">Blocked</option>
            <option value="Visitors">Visitors</option>
          </select>
        )
      case "face_in_group":
        return (
          <select
            value={c.params.group ?? ""}
            onChange={(e) => updateCondition(index, { params: { group: e.target.value } })}
            className={selectClasses}
          >
            <option value="">Select group...</option>
            <option value="Employees">Employees</option>
            <option value="Visitors">Visitors</option>
            <option value="Contractors">Contractors</option>
          </select>
        )
      case "confidence_above":
        return (
          <input
            type="number"
            min="0.1"
            max="1.0"
            step="0.05"
            placeholder="0.7"
            value={c.params.value ?? ""}
            onChange={(e) => updateCondition(index, { params: { value: e.target.value } })}
            className={inputClasses}
          />
        )
      case "zone_is":
        return (
          <input
            type="text"
            placeholder="Zone name"
            value={c.params.zone ?? ""}
            onChange={(e) => updateCondition(index, { params: { zone: e.target.value } })}
            className={inputClasses}
          />
        )
      case "time_between":
        return (
          <div className="flex items-center gap-2">
            <input
              type="time"
              value={c.params.from ?? ""}
              onChange={(e) =>
                updateCondition(index, { params: { ...c.params, from: e.target.value } })
              }
              className={inputClasses}
            />
            <span className="text-xs text-[var(--color-text-tertiary)]">to</span>
            <input
              type="time"
              value={c.params.to ?? ""}
              onChange={(e) =>
                updateCondition(index, { params: { ...c.params, to: e.target.value } })
              }
              className={inputClasses}
            />
          </div>
        )
      case "class_is":
        return (
          <input
            type="text"
            placeholder="fire,smoke (comma-separated)"
            value={c.params.classes ?? ""}
            onChange={(e) => updateCondition(index, { params: { classes: e.target.value } })}
            className={inputClasses}
          />
        )
      case "count_exceeds":
        return (
          <input
            type="number"
            min="1"
            placeholder="15"
            value={c.params.count ?? ""}
            onChange={(e) => updateCondition(index, { params: { count: e.target.value } })}
            className={inputClasses}
          />
        )
      default:
        return null
    }
  }

  /* ── action rows ──────────────────────────────────────────────── */

  function updateAction(
    list: RuleAction[],
    setList: Dispatch<SetStateAction<RuleAction[]>>,
    index: number,
    updates: Partial<RuleAction>
  ) {
    setList(
      list.map((a, i) => {
        if (i !== index) return a
        if (updates.type && updates.type !== a.type) {
          return { type: updates.type, params: {} }
        }
        return { ...a, ...updates, params: { ...a.params, ...updates.params } }
      })
    )
  }

  function renderActionParams(
    a: RuleAction,
    index: number,
    list: RuleAction[],
    setList: Dispatch<SetStateAction<RuleAction[]>>
  ) {
    switch (a.type) {
      case "create_alert":
        return (
          <select
            value={a.params.severity ?? ""}
            onChange={(e) =>
              updateAction(list, setList, index, { params: { severity: e.target.value } })
            }
            className={selectClasses}
          >
            <option value="">Severity...</option>
            <option value="P1">P1 — Critical</option>
            <option value="P2">P2 — High</option>
            <option value="P3">P3 — Medium</option>
            <option value="P4">P4 — Low</option>
          </select>
        )
      case "open_gate":
      case "close_gate":
      case "trigger_plc":
        return (
          <input
            type="text"
            placeholder="Device name"
            value={a.params.device ?? ""}
            onChange={(e) =>
              updateAction(list, setList, index, { params: { device: e.target.value } })
            }
            className={inputClasses}
          />
        )
      case "webhook":
        return (
          <input
            type="url"
            placeholder="https://hooks.example.com/..."
            value={a.params.url ?? ""}
            onChange={(e) =>
              updateAction(list, setList, index, { params: { url: e.target.value } })
            }
            className={inputClasses}
          />
        )
      default:
        return null
    }
  }

  function renderActionRows(
    list: RuleAction[],
    setList: Dispatch<SetStateAction<RuleAction[]>>
  ) {
    return (
      <div className="space-y-2">
        {list.map((a, i) => (
          <div
            key={i}
            className="flex items-start gap-3 rounded-[var(--radius-md)] bg-[var(--color-bg-tertiary)] p-3"
          >
            <div className="flex-1 grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-[var(--color-text-secondary)]">
                  Action
                </label>
                <select
                  value={a.type}
                  onChange={(e) => updateAction(list, setList, i, { type: e.target.value })}
                  className={selectClasses}
                >
                  <option value="">Select action...</option>
                  {actionTypes.map((at) => (
                    <option key={at.value} value={at.value}>
                      {at.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-[var(--color-text-secondary)]">
                  Parameters
                </label>
                {renderActionParams(a, i, list, setList) ?? (
                  <p className="px-3 py-2 text-xs text-[var(--color-text-tertiary)]">
                    No parameters needed
                  </p>
                )}
              </div>
            </div>
            <button
              onClick={() => setList(list.filter((_, j) => j !== i))}
              className="mt-6 text-[var(--color-text-tertiary)] hover:text-[var(--color-critical)] cursor-pointer"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        ))}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setList([...list, { type: "", params: {} }])}
        >
          <Plus className="h-3.5 w-3.5" />
          Add action
        </Button>
      </div>
    )
  }

  /* ── preset picker ─────────────────────────────────────────────── */

  if (!isNew && loading) {
    return (
      <div className="p-6 text-sm text-[var(--color-text-secondary)]">
        Loading rule...
      </div>
    )
  }

  if (!isNew && loadError) {
    return (
      <div className="space-y-4 p-6">
        <Button variant="ghost" size="sm" onClick={() => navigate("/configure/rules")}>
          <ArrowLeft className="h-4 w-4" />
          Back
        </Button>
        <Card className="text-sm text-[var(--color-critical)]">{loadError}</Card>
      </div>
    )
  }

  if (showPresets && isNew) {
    return (
      <div className="space-y-6 p-6">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={() => navigate("/configure/rules")}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <h1 className="text-lg font-semibold text-[var(--color-text-primary)]">
              Create Rule
            </h1>
            <p className="text-sm text-[var(--color-text-secondary)]">
              Choose a template to get started
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {presetTemplates.map((p) => (
            <Card key={p.key} hover className="flex flex-col gap-3">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]">
                  {PRESET_ICONS[p.icon] ?? <Settings className="h-5 w-5" />}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-[var(--color-text-primary)]">
                    {p.name}
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)]">{p.description}</p>
                </div>
              </div>
              <Button
                variant="secondary"
                size="sm"
                className="w-full"
                onClick={() => applyPreset(p.key)}
              >
                Use template
              </Button>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  /* ── main form ─────────────────────────────────────────────────── */

  return (
    <div className="space-y-6 p-6">
      {/* header */}
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="sm" onClick={() => navigate("/configure/rules")}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <div>
          <h1 className="text-lg font-semibold text-[var(--color-text-primary)]">
            {isNew ? "Create Rule" : `Edit Rule: ${existingRule?.name ?? name}`}
          </h1>
          {preset && (
            <Badge variant="info" className="mt-1">
              Based on {presetTemplates.find((p) => p.key === preset)?.name ?? preset} template
            </Badge>
          )}
        </div>
      </div>

      {/* Section 1: When this happens */}
      <Card className="space-y-4">
        <h2 className="text-base font-semibold text-[var(--color-text-primary)]">
          When this happens...
        </h2>

        {/* trigger */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-[var(--color-text-secondary)]">
            Trigger event
          </label>
          <select
            value={trigger}
            onChange={(e) => setTrigger(e.target.value)}
            className={selectClasses}
          >
            {triggerOptions.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </div>

        {/* cameras */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-[var(--color-text-secondary)]">Cameras</label>
          <label className="flex items-center gap-2 text-sm text-[var(--color-text-primary)] cursor-pointer">
            <input
              type="checkbox"
              checked={allCameras}
              onChange={(e) => {
                setAllCameras(e.target.checked)
                if (e.target.checked) setCameras([])
              }}
              className="rounded"
            />
            All cameras
          </label>
          {!allCameras && (
            <div className="flex flex-wrap gap-3 mt-2">
              {availableCameras.map((cam) => (
                <label
                  key={cam.id}
                  className="flex items-center gap-2 text-sm text-[var(--color-text-primary)] cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={cameras.includes(cam.id)}
                    onChange={(e) =>
                      setCameras(
                        e.target.checked
                          ? [...cameras, cam.id]
                          : cameras.filter((c) => c !== cam.id)
                      )
                    }
                    className="rounded"
                  />
                  {cam.name}
                </label>
              ))}
              {!availableCameras.length && (
                <p className="text-xs text-[var(--color-text-tertiary)]">
                  No cameras configured yet
                </p>
              )}
            </div>
          )}
        </div>

        {/* conditions */}
        <div className="space-y-2">
          <label className="text-xs font-medium text-[var(--color-text-secondary)]">
            Conditions
          </label>
          {conditions.map((c, i) => (
            <div
              key={i}
              className="flex items-start gap-3 rounded-[var(--radius-md)] bg-[var(--color-bg-tertiary)] p-3"
            >
              <div className="flex-1 grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[var(--color-text-secondary)]">
                    Type
                  </label>
                  <select
                    value={c.type}
                    onChange={(e) => updateCondition(i, { type: e.target.value })}
                    className={selectClasses}
                  >
                    <option value="">Select condition...</option>
                    {conditionTypes.map((ct) => (
                      <option key={ct.value} value={ct.value}>
                        {ct.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-[var(--color-text-secondary)]">
                    Parameters
                  </label>
                  {renderConditionParams(c, i) ?? (
                    <p className="px-3 py-2 text-xs text-[var(--color-text-tertiary)]">
                      Select a condition type
                    </p>
                  )}
                </div>
              </div>
              <button
                onClick={() => setConditions(conditions.filter((_, j) => j !== i))}
                className="mt-6 text-[var(--color-text-tertiary)] hover:text-[var(--color-critical)] cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          ))}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setConditions([...conditions, { type: "", params: {} }])}
          >
            <Plus className="h-3.5 w-3.5" />
            Add condition
          </Button>
        </div>
      </Card>

      {/* Section 2: Do this */}
      <Card className="space-y-4">
        <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Do this...</h2>
        {renderActionRows(thenActions, setThenActions)}
      </Card>

      {/* Alert behavior */}
      <Card className="space-y-4">
        <h2 className="text-base font-semibold text-[var(--color-text-primary)]">
          Alert behavior
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[var(--color-text-secondary)]">
              Severity
            </label>
            <select
              value={severity}
              onChange={(e) => setSeverity(e.target.value as Severity)}
              className={selectClasses}
            >
              <option value="P1">P1 - Critical</option>
              <option value="P2">P2 - High</option>
              <option value="P3">P3 - Medium</option>
              <option value="P4">P4 - Low</option>
            </select>
          </div>

          <div className="lg:col-span-2 space-y-1.5">
            <label className="text-xs font-medium text-[var(--color-text-secondary)]">
              Notification channels
            </label>
            <div className="flex flex-wrap gap-3 rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white p-3">
              {alertOutputs.map((output) => (
                <label
                  key={output.id}
                  className="flex items-center gap-2 text-sm text-[var(--color-text-primary)] cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={outputIds.includes(output.id)}
                    onChange={(e) =>
                      setOutputIds(
                        e.target.checked
                          ? [...outputIds, output.id]
                          : outputIds.filter((id) => id !== output.id)
                      )
                    }
                    className="rounded"
                  />
                  {output.name}
                </label>
              ))}
              {!alertOutputs.length && (
                <span className="text-xs text-[var(--color-text-tertiary)]">
                  No channels configured
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-[var(--color-text-secondary)]">
            Message
          </label>
          <textarea
            value={messageTemplate}
            onChange={(e) => setMessageTemplate(e.target.value)}
            rows={3}
            className={cn(inputClasses, "resize-y")}
          />
        </div>

        <div className="space-y-3 rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white p-3">
          <label className="flex items-center gap-2 text-sm font-medium text-[var(--color-text-primary)] cursor-pointer">
            <input
              type="checkbox"
              checked={scheduleEnabled}
              onChange={(e) => setScheduleEnabled(e.target.checked)}
              className="rounded"
            />
            Active only during selected times
          </label>
          {scheduleEnabled && (
            <div className="grid grid-cols-1 lg:grid-cols-[1fr_180px_180px] gap-3">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-[var(--color-text-secondary)]">
                  Days
                </label>
                <div className="flex flex-wrap gap-2">
                  {DAYS.map((day) => (
                    <label
                      key={day.value}
                      className="flex items-center gap-1.5 text-sm text-[var(--color-text-primary)] cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={scheduleDays.includes(day.value)}
                        onChange={(e) =>
                          setScheduleDays(
                            e.target.checked
                              ? [...scheduleDays, day.value]
                              : scheduleDays.filter((value) => value !== day.value)
                          )
                        }
                        className="rounded"
                      />
                      {day.label}
                    </label>
                  ))}
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-[var(--color-text-secondary)]">
                  From
                </label>
                <input
                  type="time"
                  value={scheduleFrom}
                  onChange={(e) => setScheduleFrom(e.target.value)}
                  className={inputClasses}
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-[var(--color-text-secondary)]">
                  To
                </label>
                <input
                  type="time"
                  value={scheduleTo}
                  onChange={(e) => setScheduleTo(e.target.value)}
                  className={inputClasses}
                />
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Section 3: Otherwise */}
      <Card className="space-y-4">
        <button
          onClick={() => setShowElse(!showElse)}
          className="flex items-center gap-2 text-base font-semibold text-[var(--color-text-primary)] cursor-pointer w-full text-left"
        >
          {showElse ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
          Otherwise...
          <span className="text-xs font-normal text-[var(--color-text-tertiary)]">
            (optional)
          </span>
        </button>
        {showElse && renderActionRows(elseActions, setElseActions)}
      </Card>

      {/* Settings */}
      <Card className="space-y-4">
        <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Settings</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[var(--color-text-secondary)]">Name</label>
            <input
              type="text"
              placeholder="Rule name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className={inputClasses}
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[var(--color-text-secondary)]">
              Description
            </label>
            <input
              type="text"
              placeholder="Short description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className={inputClasses}
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[var(--color-text-secondary)]">
              Cooldown (seconds)
            </label>
            <input
              type="number"
              min="0"
              value={cooldown}
              onChange={(e) => setCooldown(Number(e.target.value))}
              className={inputClasses}
            />
            <p className="text-xs text-[var(--color-text-tertiary)]">
              Recommended: 60s for PPE, 10s for fire
            </p>
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[var(--color-text-secondary)]">
              Priority
            </label>
            <input
              type="number"
              min="1"
              max="100"
              value={priority}
              onChange={(e) => setPriority(Number(e.target.value))}
              className={inputClasses}
            />
            <p className="text-xs text-[var(--color-text-tertiary)]">
              Higher = evaluated first
            </p>
          </div>
        </div>
      </Card>

      {/* Bottom bar */}
      <div className="flex items-center justify-end gap-3 border-t border-[var(--color-border-default)] pt-4">
        {saveError && (
          <p className="mr-auto text-sm text-[var(--color-critical)]">{saveError}</p>
        )}
        <Button variant="secondary" onClick={() => navigate("/configure/rules")}>
          Cancel
        </Button>
        <Button onClick={handleSave} disabled={saving || !name.trim()}>
          {saving ? "Saving..." : "Save Rule"}
        </Button>
      </div>
    </div>
  )
}
