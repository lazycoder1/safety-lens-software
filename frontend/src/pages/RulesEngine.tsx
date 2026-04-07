import { useState, useEffect, useCallback } from "react"
import { useNavigate } from "react-router-dom"
import {
  Plus,
  Search,
  Pencil,
  Trash2,
  ArrowRight,
  ArrowLeftRight,
  Clock,
  Zap,
  X,
} from "lucide-react"
import * as Switch from "@radix-ui/react-switch"
import * as Tabs from "@radix-ui/react-tabs"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { SeverityBadge } from "@/components/ui/SeverityBadge"
import { cn } from "@/lib/utils"
import {
  mockRules,
  triggerOptions,
  conditionTypes,
  actionTypes,
  type EngineRule,
} from "@/data/mockRules"
import {
  getSafetyRules,
  createSafetyRule,
  updateSafetyRule,
  toggleSafetyRule,
  deleteSafetyRule,
  getCameras,
} from "@/lib/api"
import type { SafetyRule, Severity, Camera } from "@/types"
import { toast } from "sonner"

/* ── helpers ─────────────────────────────────────────────────────── */

function triggerLabel(value: string): string {
  return triggerOptions.find((t) => t.value === value)?.label ?? value
}

function conditionLabel(c: { type: string; params: Record<string, string> }): string {
  switch (c.type) {
    case "plate_in_list":
      return `plate is in ${c.params.list}`
    case "face_in_group":
      return `face is in ${c.params.group}`
    case "confidence_above":
      return `confidence > ${Math.round(Number(c.params.value) * 100)}%`
    case "zone_is":
      return `zone is ${c.params.zone}`
    case "time_between":
      return `time between ${c.params.from} – ${c.params.to}`
    case "class_is":
      return `class is ${c.params.classes}`
    case "count_exceeds":
      return `count > ${c.params.count}`
    default:
      return c.type
  }
}

function actionLabel(a: { type: string; params: Record<string, string> }): string {
  const meta = actionTypes.find((t) => t.value === a.type)
  const base = meta?.label ?? a.type
  switch (a.type) {
    case "create_alert":
      return `Create ${a.params.severity} alert`
    case "open_gate":
    case "close_gate":
    case "trigger_plc":
      return `${base} (${a.params.device})`
    case "webhook":
      return `Webhook → ${a.params.url}`
    default:
      return base
  }
}

function presetBadgeVariant(preset: string | null): "default" | "info" | "warning" | "success" | "critical" {
  switch (preset) {
    case "fire":
      return "critical"
    case "ppe":
      return "warning"
    case "gate_entry":
      return "info"
    case "after_hours":
      return "critical"
    case "overcrowding":
      return "warning"
    default:
      return "default"
  }
}

function formatLastTriggered(iso: string | null): string {
  if (!iso) return "Never triggered"
  const d = new Date(iso)
  const now = new Date()
  const diffMs = now.getTime() - d.getTime()
  const diffMin = Math.floor(diffMs / 60000)
  if (diffMin < 1) return "Just now"
  if (diffMin < 60) return `${diffMin}m ago`
  const diffH = Math.floor(diffMin / 60)
  if (diffH < 24) return `${diffH}h ago`
  return d.toLocaleDateString()
}

/* ── Safety Rules Tab ───────────────────────────────────────────── */

const SEVERITY_OPTIONS: Severity[] = ["P1", "P2", "P3", "P4"]

function SafetyRulesTab() {
  const [rules, setRules] = useState<SafetyRule[]>([])
  const [cameras, setCameras] = useState<Camera[]>([])
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)

  // form state
  const [formName, setFormName] = useState("")
  const [formType, setFormType] = useState<"ppe" | "alert">("ppe")
  const [formModel, setFormModel] = useState<"yolo" | "yoloe">("yoloe")
  const [formClasses, setFormClasses] = useState<string[]>([])
  const [formClassInput, setFormClassInput] = useState("")
  const [formSeverity, setFormSeverity] = useState<Severity>("P2")

  const fetchData = useCallback(async () => {
    try {
      const [safetyRules, cams] = await Promise.all([getSafetyRules(), getCameras()])
      setRules(safetyRules)
      setCameras(cams)
    } catch {
      toast.error("Failed to load safety rules")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  function getCameraCount(ruleId: string): number {
    return cameras.filter((c) => (c.safety_rule_ids || []).includes(ruleId)).length
  }

  function resetForm() {
    setFormName("")
    setFormType("ppe")
    setFormModel("yoloe")
    setFormClasses([])
    setFormClassInput("")
    setFormSeverity("P2")
    setShowForm(false)
    setEditingId(null)
  }

  function startEdit(rule: SafetyRule) {
    setEditingId(rule.id)
    setFormName(rule.name)
    setFormType(rule.type)
    setFormModel(rule.model)
    setFormClasses([...rule.classes])
    setFormClassInput("")
    setFormSeverity(rule.severity)
    setShowForm(true)
  }

  function addChip() {
    const val = formClassInput.trim().toLowerCase()
    if (val && !formClasses.includes(val)) {
      setFormClasses((prev) => [...prev, val])
    }
    setFormClassInput("")
  }

  function removeChip(cls: string) {
    setFormClasses((prev) => prev.filter((c) => c !== cls))
  }

  async function handleSave() {
    if (!formName.trim()) {
      toast.error("Rule name is required")
      return
    }
    if (formClasses.length === 0) {
      toast.error("At least one detection class is required")
      return
    }
    try {
      if (editingId) {
        const updated = await updateSafetyRule(editingId, {
          name: formName.trim(),
          type: formType,
          model: formModel,
          classes: formClasses,
          severity: formSeverity,
        })
        setRules((prev) => prev.map((r) => (r.id === editingId ? updated : r)))
        toast.success("Rule updated")
      } else {
        const created = await createSafetyRule({
          name: formName.trim(),
          type: formType,
          model: formModel,
          classes: formClasses,
          severity: formSeverity,
        })
        setRules((prev) => [...prev, created])
        toast.success("Rule created")
      }
      resetForm()
    } catch (err: any) {
      toast.error(err.message || "Failed to save rule")
    }
  }

  async function handleToggle(id: string) {
    try {
      const updated = await toggleSafetyRule(id)
      setRules((prev) => prev.map((r) => (r.id === id ? updated : r)))
    } catch (err: any) {
      toast.error(err.message || "Failed to toggle rule")
    }
  }

  async function confirmDelete() {
    if (!deleteTarget) return
    try {
      await deleteSafetyRule(deleteTarget)
      setRules((prev) => prev.filter((r) => r.id !== deleteTarget))
      toast.success("Rule deleted")
    } catch (err: any) {
      toast.error(err.message || "Failed to delete rule")
    }
    setDeleteTarget(null)
  }

  if (loading) {
    return (
      <div className="py-12 text-center text-sm text-[var(--color-text-tertiary)]">
        Loading safety rules...
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Safety Rules</h2>
          <p className="text-sm text-[var(--color-text-secondary)]">
            Define PPE requirements and alert detection rules for your cameras.
          </p>
        </div>
        {!showForm && (
          <Button onClick={() => { resetForm(); setShowForm(true) }}>
            <Plus className="h-4 w-4" />
            Add Rule
          </Button>
        )}
      </div>

      {/* inline form */}
      {showForm && (
        <Card className="space-y-4">
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
            {editingId ? "Edit Rule" : "New Safety Rule"}
          </h3>
          <div className="space-y-3">
            <div>
              <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Name</label>
              <input
                type="text"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
                placeholder="e.g. Hard Hat Required"
                className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Type</label>
              <select
                value={formType}
                onChange={(e) => setFormType(e.target.value as "ppe" | "alert")}
                className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer"
              >
                <option value="ppe">PPE — Missing item</option>
                <option value="alert">Alert — Item detected</option>
              </select>
              <p className="mt-1 text-xs text-[var(--color-text-tertiary)]">
                {formType === "ppe"
                  ? "Fires when a person is detected WITHOUT this item"
                  : "Fires when this item IS detected in frame"}
              </p>
            </div>
            <div>
              <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Model</label>
              <select
                value={formModel}
                onChange={(e) => setFormModel(e.target.value as "yolo" | "yoloe")}
                className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer"
              >
                <option value="yoloe">YOLOe Open-Vocab</option>
                <option value="yolo">YOLO COCO 80-class</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Detection Classes</label>
              <div className="flex flex-wrap gap-1.5 mb-2">
                {formClasses.map((cls) => (
                  <span
                    key={cls}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-[var(--color-info-bg)] text-[#1e40af]"
                  >
                    {cls}
                    <button
                      onClick={() => removeChip(cls)}
                      className="hover:text-[var(--color-critical)] cursor-pointer"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </span>
                ))}
              </div>
              <input
                type="text"
                value={formClassInput}
                onChange={(e) => setFormClassInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault()
                    addChip()
                  }
                }}
                placeholder="Type a class name and press Enter"
                className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Severity</label>
              <select
                value={formSeverity}
                onChange={(e) => setFormSeverity(e.target.value as Severity)}
                className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer"
              >
                {SEVERITY_OPTIONS.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="flex gap-2 pt-1">
            <Button onClick={handleSave}>
              {editingId ? "Update" : "Save"}
            </Button>
            <Button variant="secondary" onClick={resetForm}>
              Cancel
            </Button>
          </div>
        </Card>
      )}

      {/* table */}
      {rules.length === 0 && !showForm ? (
        <p className="py-12 text-center text-sm text-[var(--color-text-tertiary)]">
          No safety rules configured yet. Click "Add Rule" to create one.
        </p>
      ) : (
        <div className="border border-[var(--color-border-default)] rounded-[var(--radius-lg)] overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-[var(--color-bg-secondary)] border-b border-[var(--color-border-default)]">
                <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Name</th>
                <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Type</th>
                <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Detection Classes</th>
                <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Model</th>
                <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Severity</th>
                <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Cameras</th>
                <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Enabled</th>
                <th className="text-right px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody>
              {rules.map((rule) => (
                <tr key={rule.id} className="border-b border-[var(--color-border-default)] last:border-b-0">
                  <td className="px-4 py-3 font-medium text-[var(--color-text-primary)]">{rule.name}</td>
                  <td className="px-4 py-3">
                    <Badge variant={rule.type === "ppe" ? "info" : "warning"}>
                      {rule.type === "ppe" ? "PPE" : "Alert"}
                    </Badge>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap gap-1">
                      {rule.classes.map((cls) => (
                        <span
                          key={cls}
                          className="inline-flex px-2 py-0.5 rounded-full text-xs font-medium bg-[var(--color-info-bg)] text-[#1e40af]"
                        >
                          {cls}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant={rule.model === "yoloe" ? "success" : "default"}>
                      {rule.model === "yoloe" ? "YOLOe" : "YOLO"}
                    </Badge>
                  </td>
                  <td className="px-4 py-3">
                    <SeverityBadge severity={rule.severity} />
                  </td>
                  <td className="px-4 py-3 text-[var(--color-text-secondary)]">{getCameraCount(rule.id)}</td>
                  <td className="px-4 py-3">
                    <Switch.Root
                      checked={rule.enabled}
                      onCheckedChange={() => handleToggle(rule.id)}
                      className={cn(
                        "relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full transition-colors",
                        rule.enabled ? "bg-[var(--color-success)]" : "bg-[var(--color-border-default)]"
                      )}
                    >
                      <Switch.Thumb
                        className={cn(
                          "pointer-events-none block h-4 w-4 rounded-full bg-white shadow-sm transition-transform",
                          rule.enabled ? "translate-x-[18px]" : "translate-x-[2px]",
                          "mt-[2px]"
                        )}
                      />
                    </Switch.Root>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end gap-1">
                      <Button variant="ghost" size="sm" onClick={() => startEdit(rule)}>
                        <Pencil className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="hover:text-[var(--color-critical)] hover:bg-[var(--color-critical-bg)]"
                        onClick={() => setDeleteTarget(rule.id)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* delete confirmation */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
          <Card className="w-full max-w-sm space-y-4 shadow-lg">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">Delete Safety Rule</h2>
              <button
                onClick={() => setDeleteTarget(null)}
                className="text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <p className="text-sm text-[var(--color-text-secondary)]">
              Are you sure you want to delete{" "}
              <span className="font-medium">{rules.find((r) => r.id === deleteTarget)?.name}</span>?
              This action cannot be undone.
            </p>
            <div className="flex items-center justify-end gap-2">
              <Button variant="secondary" size="sm" onClick={() => setDeleteTarget(null)}>Cancel</Button>
              <Button variant="danger" size="sm" onClick={confirmDelete}>Delete</Button>
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}

/* ── Automation Rules Tab (existing content) ────────────────────── */

function AutomationRulesTab() {
  const navigate = useNavigate()
  const [rules, setRules] = useState<EngineRule[]>(mockRules)
  const [search, setSearch] = useState("")
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)

  const filtered = rules.filter(
    (r) =>
      r.name.toLowerCase().includes(search.toLowerCase()) ||
      r.description.toLowerCase().includes(search.toLowerCase())
  )

  function toggleRule(id: string) {
    setRules((prev) => prev.map((r) => (r.id === id ? { ...r, enabled: !r.enabled } : r)))
  }

  function confirmDelete() {
    if (!deleteTarget) return
    setRules((prev) => prev.filter((r) => r.id !== deleteTarget))
    setDeleteTarget(null)
  }

  return (
    <div className="space-y-4">
      {/* header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Automation Rules</h2>
          <p className="text-sm text-[var(--color-text-secondary)]">
            {rules.length} rule{rules.length !== 1 && "s"} configured
          </p>
        </div>
        <Button onClick={() => navigate("/configure/rules/new")}>
          <Plus className="h-4 w-4" />
          New Rule
        </Button>
      </div>

      {/* search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--color-text-tertiary)]" />
        <input
          type="text"
          placeholder="Search rules..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full pl-9 pr-3 py-2 text-sm rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
        />
      </div>

      {/* list */}
      <div className="space-y-3">
        {filtered.length === 0 && (
          <p className="py-12 text-center text-sm text-[var(--color-text-tertiary)]">
            No rules match your search.
          </p>
        )}

        {filtered.map((rule) => (
          <Card key={rule.id} className="flex items-start gap-4">
            {/* toggle */}
            <div className="pt-0.5">
              <Switch.Root
                checked={rule.enabled}
                onCheckedChange={() => toggleRule(rule.id)}
                className={cn(
                  "relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full transition-colors",
                  rule.enabled ? "bg-[var(--color-success)]" : "bg-[var(--color-border-default)]"
                )}
              >
                <Switch.Thumb
                  className={cn(
                    "pointer-events-none block h-4 w-4 rounded-full bg-white shadow-sm transition-transform",
                    rule.enabled ? "translate-x-[18px]" : "translate-x-[2px]",
                    "mt-[2px]"
                  )}
                />
              </Switch.Root>
            </div>

            {/* body */}
            <div className="flex-1 min-w-0 space-y-2">
              <div>
                <span className="font-semibold text-sm text-[var(--color-text-primary)]">
                  {rule.name}
                </span>
                <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                  {rule.description}
                </p>
              </div>

              {/* summary */}
              <div className="text-xs text-[var(--color-text-secondary)] space-y-1">
                <p className="flex items-center gap-1.5 flex-wrap">
                  <Zap className="h-3.5 w-3.5 text-[var(--color-info)]" />
                  <span className="font-medium">When</span>{" "}
                  {triggerLabel(rule.trigger).toLowerCase()}
                  {rule.cameras.length > 0 && (
                    <>
                      {" "}on{" "}
                      <span className="font-medium">
                        {rule.cameras.join(", ")}
                      </span>
                    </>
                  )}
                  {rule.conditions.length > 0 && (
                    <>
                      {" "}and{" "}
                      {rule.conditions.map((c, i) => (
                        <span key={i}>
                          {i > 0 && " + "}
                          {conditionLabel(c)}
                        </span>
                      ))}
                    </>
                  )}
                </p>
                <p className="flex items-center gap-1.5 flex-wrap">
                  <ArrowRight className="h-3.5 w-3.5 text-[var(--color-success)]" />
                  <span className="font-medium">Then</span>{" "}
                  {rule.thenActions.map((a, i) => (
                    <span key={i}>
                      {i > 0 && " → "}
                      {actionLabel(a)}
                    </span>
                  ))}
                </p>
                {rule.elseActions.length > 0 && (
                  <p className="flex items-center gap-1.5 flex-wrap">
                    <ArrowLeftRight className="h-3.5 w-3.5 text-[var(--color-critical)]" />
                    <span className="font-medium">Otherwise</span>{" "}
                    {rule.elseActions.map((a, i) => (
                      <span key={i}>
                        {i > 0 && " → "}
                        {actionLabel(a)}
                      </span>
                    ))}
                  </p>
                )}
              </div>

              {/* footer badges */}
              <div className="flex items-center gap-2 flex-wrap">
                <Badge>
                  <Clock className="h-3 w-3" />
                  {rule.cooldownSeconds}s cooldown
                </Badge>
                <Badge>{formatLastTriggered(rule.lastTriggered)}</Badge>
                {rule.preset && (
                  <Badge variant={presetBadgeVariant(rule.preset)}>
                    {rule.preset.replace("_", " ")}
                  </Badge>
                )}
              </div>
            </div>

            {/* actions */}
            <div className="flex items-center gap-1 shrink-0">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate(`/configure/rules/${rule.id}`)}
              >
                <Pencil className="h-3.5 w-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="hover:text-[var(--color-critical)] hover:bg-[var(--color-critical-bg)]"
                onClick={() => setDeleteTarget(rule.id)}
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </div>
          </Card>
        ))}
      </div>

      {/* delete confirmation dialog */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
          <Card className="w-full max-w-sm space-y-4 shadow-lg">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">
                Delete Rule
              </h2>
              <button
                onClick={() => setDeleteTarget(null)}
                className="text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <p className="text-sm text-[var(--color-text-secondary)]">
              Are you sure you want to delete{" "}
              <span className="font-medium">
                {rules.find((r) => r.id === deleteTarget)?.name}
              </span>
              ? This action cannot be undone.
            </p>
            <div className="flex items-center justify-end gap-2">
              <Button variant="secondary" size="sm" onClick={() => setDeleteTarget(null)}>
                Cancel
              </Button>
              <Button variant="danger" size="sm" onClick={confirmDelete}>
                Delete
              </Button>
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}

/* ── main component ─────────────────────────────────────────────── */

export function RulesEngine() {
  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-lg font-semibold text-[var(--color-text-primary)]">Rules Engine</h1>
        <p className="text-sm text-[var(--color-text-secondary)]">
          Configure safety rules and automation workflows.
        </p>
      </div>

      <Tabs.Root defaultValue="safety" className="space-y-4">
        <Tabs.List className="flex gap-0 border-b border-[var(--color-border-default)]">
          <Tabs.Trigger
            value="safety"
            className="px-4 py-2 text-sm font-medium text-[var(--color-text-secondary)] border-b-2 border-transparent data-[state=active]:text-[var(--color-text-primary)] data-[state=active]:border-[var(--color-info)] transition-colors cursor-pointer"
          >
            Safety Rules
          </Tabs.Trigger>
          <Tabs.Trigger
            value="automation"
            className="px-4 py-2 text-sm font-medium text-[var(--color-text-secondary)] border-b-2 border-transparent data-[state=active]:text-[var(--color-text-primary)] data-[state=active]:border-[var(--color-info)] transition-colors cursor-pointer"
          >
            Automation Rules
          </Tabs.Trigger>
        </Tabs.List>

        <Tabs.Content value="safety">
          <SafetyRulesTab />
        </Tabs.Content>

        <Tabs.Content value="automation">
          <AutomationRulesTab />
        </Tabs.Content>
      </Tabs.Root>
    </div>
  )
}
