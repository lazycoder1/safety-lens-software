import { useState } from "react"
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
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import {
  mockRules,
  triggerOptions,
  conditionTypes,
  actionTypes,
  type EngineRule,
} from "@/data/mockRules"

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

/* ── component ───────────────────────────────────────────────────── */

export function RulesEngine() {
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
    <div className="space-y-6 p-6">
      {/* header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-[var(--color-text-primary)]">Rules Engine</h1>
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
