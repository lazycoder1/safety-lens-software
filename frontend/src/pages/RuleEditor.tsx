import { useState, useEffect } from "react"
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
  mockRules,
  triggerOptions,
  conditionTypes,
  actionTypes,
  presetTemplates,
  type EngineRule,
  type RuleCondition,
  type RuleAction,
} from "@/data/mockRules"

/* ── constants ───────────────────────────────────────────────────── */

const CAMERA_LIST = [
  { id: "cam-01", name: "Gate Camera" },
  { id: "cam-02", name: "Assembly Line A" },
  { id: "cam-03", name: "Assembly Line B" },
  { id: "cam-04", name: "Warehouse" },
  { id: "cam-05", name: "Parking Lot" },
]

const PRESET_ICONS: Record<string, React.ReactNode> = {
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
    preset: null,
  }
}

/* ── component ───────────────────────────────────────────────────── */

export function RuleEditor() {
  const { ruleId } = useParams<{ ruleId: string }>()
  const navigate = useNavigate()
  const location = useLocation()

  const isNew = !ruleId || ruleId === "new"
  const existing = !isNew ? mockRules.find((r) => r.id === ruleId) : null

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
  const [preset, setPreset] = useState<string | null>(null)
  const [showPresets, setShowPresets] = useState(isNew)

  // load existing rule
  useEffect(() => {
    if (existing) {
      setName(existing.name)
      setDescription(existing.description)
      setTrigger(existing.trigger)
      setCameras(existing.cameras)
      setAllCameras(existing.cameras.length === 0)
      setConditions(existing.conditions)
      setThenActions(existing.thenActions)
      setElseActions(existing.elseActions)
      setShowElse(existing.elseActions.length > 0)
      setCooldown(existing.cooldownSeconds)
      setPriority(existing.priority)
      setPreset(existing.preset)
      setShowPresets(false)
    }
  }, [existing])

  // check for preset passed via location state
  useEffect(() => {
    const state = location.state as { preset?: string } | null
    if (state?.preset) {
      applyPreset(state.preset)
    }
  }, [location.state])

  function applyPreset(key: string) {
    const tpl = presetTemplates.find((p) => p.key === key)
    if (!tpl) return
    const t = tpl.template
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
    setPreset(t.preset)
    setShowPresets(false)
  }

  function handleSave() {
    const rule: EngineRule = {
      id: isNew ? `rule-${Date.now()}` : ruleId!,
      name,
      description,
      enabled: true,
      trigger,
      cameras: allCameras ? [] : cameras,
      conditions,
      thenActions,
      elseActions: showElse ? elseActions : [],
      cooldownSeconds: cooldown,
      priority,
      lastTriggered: existing?.lastTriggered ?? null,
      preset,
    }
    console.log("Saving rule:", rule)
    navigate("/configure/rules")
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
    setList: React.Dispatch<React.SetStateAction<RuleAction[]>>,
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
    setList: React.Dispatch<React.SetStateAction<RuleAction[]>>
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
    setList: React.Dispatch<React.SetStateAction<RuleAction[]>>
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
            {isNew ? "Create Rule" : `Edit Rule: ${existing?.name ?? ""}`}
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
              {CAMERA_LIST.map((cam) => (
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
        <Button variant="secondary" onClick={() => navigate("/configure/rules")}>
          Cancel
        </Button>
        <Button onClick={handleSave}>Save Rule</Button>
      </div>
    </div>
  )
}
