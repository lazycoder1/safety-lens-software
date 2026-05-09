import { useState, useEffect, useMemo, useCallback } from "react"
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  X,
  Eye,
  CheckCircle2,
  Clock,
  AlertTriangle,
  XCircle,
  Loader2,
} from "lucide-react"
import { severityConfig, severityOrder } from "@/lib/constants"
import type { Severity, AlertStatus, Alert } from "@/types"
import { useAlertStore } from "@/stores/alertStore"
import { useAuthStore } from "@/stores/authStore"
import { cn, timeAgo, formatTime } from "@/lib/utils"
import { getCameras, reportFalseNegative } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { SeverityBadge } from "@/components/ui/SeverityBadge"
import { StatusBadge } from "@/components/ui/StatusBadge"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"
import { Skeleton } from "@/components/ui/skeleton"
import { API_BASE } from "@/lib/api"

const severities: Severity[] = ["P1", "P2", "P3", "P4"]
const statuses: AlertStatus[] = ["active", "acknowledged", "resolved", "snoozed"]

const statusIcons: Record<AlertStatus, typeof AlertTriangle> = {
  active: AlertTriangle,
  acknowledged: CheckCircle2,
  resolved: CheckCircle2,
  snoozed: Clock,
}

type SortField = "timestamp" | "severity" | "confidence"
type SortDir = "asc" | "desc"

export function AlertCenter() {
  const { alerts, loading, fetchAlerts, acknowledge, snooze, resolve, markFalsePositive } = useAlertStore()
  const userRole = useAuthStore((s) => s.user?.role)
  const canAct = userRole === "admin" || userRole === "operator"

  const [sevFilter, setSevFilter] = useState<Set<Severity>>(new Set())
  const [statusFilter, setStatusFilter] = useState<Set<AlertStatus>>(new Set())
  const [sortField, setSortField] = useState<SortField>("timestamp")
  const [sortDir, setSortDir] = useState<SortDir>("desc")
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null)
  const [showFnModal, setShowFnModal] = useState(false)
  const [cameras, setCameras] = useState<{ id: string; name: string; zone: string }[]>([])

  // Fetch alerts and cameras on mount
  useEffect(() => {
    fetchAlerts()
    getCameras()
      .then((cams: any[]) => setCameras(cams.map((c) => ({ id: c.id, name: c.name || c.id, zone: c.zone || "" }))))
      .catch(() => {})
  }, [fetchAlerts])

  const toggleSev = useCallback((s: Severity) => {
    setSevFilter((prev) => {
      const next = new Set(prev)
      next.has(s) ? next.delete(s) : next.add(s)
      return next
    })
  }, [])

  const toggleStatus = useCallback((s: AlertStatus) => {
    setStatusFilter((prev) => {
      const next = new Set(prev)
      next.has(s) ? next.delete(s) : next.add(s)
      return next
    })
  }, [])

  const clearFilters = useCallback(() => {
    setSevFilter(new Set())
    setStatusFilter(new Set())
  }, [])

  const hasFilters = sevFilter.size > 0 || statusFilter.size > 0

  const filtered = useMemo(() => {
    let result = alerts
    if (sevFilter.size > 0) result = result.filter((a) => sevFilter.has(a.severity))
    if (statusFilter.size > 0) result = result.filter((a) => statusFilter.has(a.status))

    result = [...result].sort((a, b) => {
      let cmp = 0
      switch (sortField) {
        case "timestamp":
          cmp = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
          break
        case "severity":
          cmp = severityOrder[a.severity] - severityOrder[b.severity]
          break
        case "confidence":
          cmp = a.confidence - b.confidence
          break
      }
      return sortDir === "asc" ? cmp : -cmp
    })

    return result
  }, [alerts, sevFilter, statusFilter, sortField, sortDir])

  function handleSort(field: SortField) {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"))
    } else {
      setSortField(field)
      setSortDir("desc")
    }
  }

  function SortIcon({ field }: { field: SortField }) {
    if (sortField !== field) return <ArrowUpDown size={12} className="opacity-30" />
    return sortDir === "desc" ? <ArrowDown size={12} /> : <ArrowUp size={12} />
  }

  async function handleAcknowledge(id: string) {
    try {
      await acknowledge(id)
      toast.success("Alert acknowledged")
      if (selectedAlert?.id === id) {
        setSelectedAlert((prev) => prev ? { ...prev, status: "acknowledged", acknowledgedBy: "Admin", acknowledgedAt: new Date().toISOString() } : null)
      }
    } catch { toast.error("Failed to acknowledge alert") }
  }

  async function handleSnooze(id: string, minutes: number = 15) {
    try {
      await snooze(id, minutes)
      toast.info(`Alert snoozed for ${minutes}m`)
      if (selectedAlert?.id === id) {
        setSelectedAlert((prev) => prev ? { ...prev, status: "snoozed" } : null)
      }
    } catch { toast.error("Failed to snooze alert") }
  }

  async function handleResolve(id: string) {
    try {
      await resolve(id)
      toast.success("Alert resolved")
      if (selectedAlert?.id === id) {
        setSelectedAlert((prev) => prev ? { ...prev, status: "resolved" } : null)
      }
    } catch { toast.error("Failed to resolve alert") }
  }

  async function handleFalsePositive(id: string) {
    try {
      await markFalsePositive(id)
      toast("Marked as false positive", { description: "Alert has been resolved as false positive." })
      if (selectedAlert?.id === id) {
        setSelectedAlert((prev) => prev ? { ...prev, status: "resolved", falsePositive: true } : null)
      }
    } catch { toast.error("Failed to mark as false positive") }
  }

  const liveSelectedAlert = useMemo(() => {
    if (!selectedAlert) return null
    return alerts.find((a) => a.id === selectedAlert.id) ?? selectedAlert
  }, [alerts, selectedAlert])

  if (loading) {
    return (
      <div className="flex h-full overflow-hidden">
        <div className="flex-1 flex flex-col min-w-0">
          <div className="px-6 pt-6 pb-4">
            <div className="flex items-center justify-between mb-4">
              <Skeleton className="h-7 w-32" />
            </div>
            <div className="flex flex-wrap items-center gap-3 mb-3">
              <div className="flex items-center gap-1.5">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-7 w-14 rounded-full" />
                ))}
              </div>
              <div className="flex items-center gap-1.5">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-7 w-24 rounded-full" />
                ))}
              </div>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto px-6">
            <div className="border border-[var(--color-border-default)] rounded-[var(--radius-lg)] overflow-hidden">
              <div className="bg-[var(--color-bg-secondary)] border-b border-[var(--color-border-default)] px-4 py-3 flex gap-4">
                {["w-10", "w-16", "w-28", "w-20", "w-32", "w-12", "w-24", "w-16", "w-20"].map((w, i) => (
                  <Skeleton key={i} className={`h-3.5 ${w}`} />
                ))}
              </div>
              {Array.from({ length: 10 }).map((_, i) => (
                <div key={i} className="flex items-center gap-4 px-4 py-3.5 border-b border-[var(--color-border-default)] last:border-0">
                  <Skeleton className="h-3 w-3 rounded-full" />
                  <Skeleton className="h-4 w-16" />
                  <Skeleton className="h-4 w-28" />
                  <Skeleton className="h-4 w-20" />
                  <Skeleton className="h-4 w-32" />
                  <Skeleton className="h-4 w-10" />
                  <Skeleton className="h-4 w-24" />
                  <Skeleton className="h-5 w-16 rounded-full" />
                  <Skeleton className="h-4 w-20" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="px-6 pt-6 pb-4">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Alert Center</h1>
            {canAct && (
              <Button
                variant="secondary"
                size="sm"
                className="text-xs gap-1.5"
                onClick={() => setShowFnModal(true)}
              >
                <AlertTriangle size={12} />
                Report Missed Detection
              </Button>
            )}
          </div>

          {/* Filter bar */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Severity filters */}
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-medium text-[var(--color-text-secondary)] mr-1">Severity</span>
              {severities.map((s) => {
                const cfg = severityConfig[s]
                const active = sevFilter.has(s)
                return (
                  <button
                    key={s}
                    onClick={() => toggleSev(s)}
                    className={cn(
                      "inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-all cursor-pointer border",
                      active
                        ? "border-current"
                        : "border-transparent opacity-60 hover:opacity-100"
                    )}
                    style={{
                      backgroundColor: active ? cfg.bg : undefined,
                      color: cfg.textColor,
                    }}
                  >
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: cfg.color }}
                    />
                    {s}
                  </button>
                )
              })}
            </div>

            <div className="w-px h-5 bg-[var(--color-border-default)]" />

            {/* Status filters */}
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-medium text-[var(--color-text-secondary)] mr-1">Status</span>
              {statuses.map((s) => {
                const active = statusFilter.has(s)
                return (
                  <button
                    key={s}
                    onClick={() => toggleStatus(s)}
                    className={cn(
                      "inline-flex items-center rounded-md px-2 py-1 text-xs font-medium transition-all cursor-pointer border capitalize",
                      active
                        ? "bg-[var(--color-bg-tertiary)] text-[var(--color-text-primary)] border-[var(--color-border-active)]"
                        : "border-transparent text-[var(--color-text-secondary)] opacity-60 hover:opacity-100"
                    )}
                  >
                    {s}
                  </button>
                )
              })}
            </div>

            {hasFilters && (
              <>
                <div className="w-px h-5 bg-[var(--color-border-default)]" />
                <Button variant="ghost" size="sm" onClick={clearFilters} className="text-xs gap-1">
                  <X size={12} />
                  Clear
                </Button>
              </>
            )}

            <div className="ml-auto text-xs text-[var(--color-text-secondary)]">
              {filtered.length} alert{filtered.length !== 1 ? "s" : ""}
            </div>
          </div>
        </div>

        {/* Table */}
        <div className="flex-1 overflow-auto px-6 pb-6">
          <div className="border rounded-[var(--radius-lg)] overflow-hidden bg-white">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-[var(--color-bg-secondary)]">
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] w-16">
                    Sev
                  </th>
                  <th
                    className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] cursor-pointer select-none hover:text-[var(--color-text-primary)]"
                    onClick={() => handleSort("timestamp")}
                  >
                    <span className="inline-flex items-center gap-1">
                      Time <SortIcon field="timestamp" />
                    </span>
                  </th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">
                    Camera
                  </th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">
                    Zone
                  </th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">
                    Rule
                  </th>
                  <th
                    className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] cursor-pointer select-none hover:text-[var(--color-text-primary)]"
                    onClick={() => handleSort("confidence")}
                  >
                    <span className="inline-flex items-center gap-1">
                      Conf <SortIcon field="confidence" />
                    </span>
                  </th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">
                    Source
                  </th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">
                    Status
                  </th>
                  <th className="text-right px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((alert) => {
                  const sev = severityConfig[alert.severity]
                  const StatusIcon = statusIcons[alert.status]
                  const isSelected = liveSelectedAlert?.id === alert.id
                  return (
                    <tr
                      key={alert.id}
                      className={cn(
                        "border-b last:border-b-0 transition-colors hover:bg-[var(--color-bg-secondary)] cursor-pointer",
                        isSelected && "bg-[var(--color-bg-tertiary)]"
                      )}
                      onClick={() => setSelectedAlert(alert)}
                    >
                      <td className="px-4 py-2.5">
                        <span
                          className="inline-block w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: sev.color }}
                          title={`${alert.severity} - ${sev.label}`}
                        />
                      </td>
                      <td className="px-4 py-2.5 text-xs text-[var(--color-text-secondary)] whitespace-nowrap">
                        <span title={new Date(alert.timestamp).toLocaleString()}>
                          {formatTime(alert.timestamp)}
                        </span>
                        <span className="block text-[10px] opacity-60">{timeAgo(alert.timestamp)}</span>
                      </td>
                      <td className="px-4 py-2.5 text-xs font-medium text-[var(--color-text-primary)] max-w-[180px] truncate">
                        {alert.cameraName}
                      </td>
                      <td className="px-4 py-2.5 text-xs text-[var(--color-text-secondary)]">
                        {alert.zone}
                      </td>
                      <td className="px-4 py-2.5 text-xs font-medium text-[var(--color-text-primary)]">
                        {alert.rule}
                      </td>
                      <td className="px-4 py-2.5 text-xs text-[var(--color-text-secondary)] tabular-nums">
                        {Math.round(alert.confidence * 100)}%
                      </td>
                      <td className="px-4 py-2.5 text-xs text-[var(--color-text-secondary)]">
                        {alert.source}
                      </td>
                      <td className="px-4 py-2.5">
                        <StatusBadge status={alert.status} className="gap-1">
                          <StatusIcon size={10} />
                          {alert.status}
                        </StatusBadge>
                      </td>
                      <td className="px-4 py-2.5 text-right" onClick={(e) => e.stopPropagation()}>
                        <div className="inline-flex items-center gap-1">
                          {canAct && alert.status === "active" && (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 px-2 text-xs"
                              onClick={() => handleAcknowledge(alert.id)}
                            >
                              Ack
                            </Button>
                          )}
                          {canAct && (alert.status === "active" || alert.status === "acknowledged") && (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 px-2 text-xs"
                              onClick={() => handleSnooze(alert.id)}
                            >
                              <Clock size={12} />
                            </Button>
                          )}
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 px-2 text-xs"
                            onClick={() => setSelectedAlert(alert)}
                          >
                            <Eye size={12} />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  )
                })}

                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={9} className="px-4 py-12 text-center text-sm text-[var(--color-text-secondary)]">
                      {alerts.length === 0
                        ? "No alerts yet. Alerts will appear here as detections trigger violations."
                        : "No alerts match the current filters."}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* False Negative Report Modal */}
      {showFnModal && (
        <FalseNegativeModal
          cameras={cameras}
          onClose={() => setShowFnModal(false)}
          onSubmitted={() => {
            setShowFnModal(false)
            toast.success("Missed detection reported")
          }}
        />
      )}

      {/* Detail slide-over panel */}
      {liveSelectedAlert && (
        <div className="w-[480px] border-l bg-white flex flex-col shrink-0 overflow-hidden">
          {/* Panel header */}
          <div className="flex items-center justify-between px-5 py-4 border-b">
            <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">Alert Detail</h2>
            <button
              onClick={() => setSelectedAlert(null)}
              className="p-1 rounded hover:bg-[var(--color-bg-tertiary)] transition-colors cursor-pointer text-[var(--color-text-secondary)]"
            >
              <X size={16} />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto">
            {/* Snapshot */}
            <div className="p-5">
              {liveSelectedAlert.snapshotUrl ? (
                <img
                  src={`${API_BASE}${liveSelectedAlert.snapshotUrl}`}
                  alt="Alert snapshot"
                  className="w-full rounded-[var(--radius-md)] border"
                  style={{ aspectRatio: "16/9", objectFit: "cover" }}
                />
              ) : (
                <div
                  className="w-full rounded-[var(--radius-md)] border bg-[var(--color-bg-tertiary)] flex items-center justify-center text-[var(--color-text-tertiary)] text-xs"
                  style={{ aspectRatio: "16/9" }}
                >
                  No snapshot available
                </div>
              )}
            </div>

            {/* Metadata */}
            <div className="px-5 pb-5 space-y-4">
              {/* Rule + Severity */}
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <h3 className="text-base font-semibold text-[var(--color-text-primary)]">
                    {liveSelectedAlert.rule}
                  </h3>
                  <SeverityBadge severity={liveSelectedAlert.severity} />
                </div>
                <div className="flex items-center gap-2">
                  <StatusBadge status={liveSelectedAlert.status} />
                  {liveSelectedAlert.falsePositive && (
                    <Badge variant="default" className="text-[10px]">False positive</Badge>
                  )}
                </div>
              </div>

              {/* Description */}
              {liveSelectedAlert.description && (
                <p className="text-xs text-[var(--color-text-secondary)] leading-relaxed bg-[var(--color-bg-secondary)] rounded-[var(--radius-md)] p-3">
                  {liveSelectedAlert.description}
                </p>
              )}

              {/* Details grid */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <span className="text-[11px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">
                    Camera
                  </span>
                  <p className="text-sm text-[var(--color-text-primary)] mt-0.5">{liveSelectedAlert.cameraName}</p>
                </div>
                <div>
                  <span className="text-[11px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">
                    Zone
                  </span>
                  <p className="text-sm text-[var(--color-text-primary)] mt-0.5">{liveSelectedAlert.zone}</p>
                </div>
                <div>
                  <span className="text-[11px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">
                    Confidence
                  </span>
                  <p className="text-sm text-[var(--color-text-primary)] mt-0.5 tabular-nums">
                    {Math.round(liveSelectedAlert.confidence * 100)}%
                  </p>
                </div>
                <div>
                  <span className="text-[11px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">
                    Source
                  </span>
                  <p className="text-sm text-[var(--color-text-primary)] mt-0.5">{liveSelectedAlert.source}</p>
                </div>
                <div>
                  <span className="text-[11px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">
                    Detected
                  </span>
                  <p className="text-sm text-[var(--color-text-primary)] mt-0.5">
                    {formatTime(liveSelectedAlert.timestamp)}
                  </p>
                </div>
                <div>
                  <span className="text-[11px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">
                    Alert ID
                  </span>
                  <p className="text-sm text-[var(--color-text-primary)] mt-0.5 font-mono">{liveSelectedAlert.id}</p>
                </div>
              </div>

              {/* Timeline */}
              <div>
                <h4 className="text-xs font-semibold text-[var(--color-text-primary)] mb-2 uppercase tracking-wider">
                  Timeline
                </h4>
                <div className="space-y-0">
                  <div className="flex gap-3">
                    <div className="flex flex-col items-center">
                      <div className="w-2 h-2 rounded-full bg-red-500 mt-1.5" />
                      <div className="w-px flex-1 bg-[var(--color-border-default)]" />
                    </div>
                    <div className="pb-3">
                      <p className="text-xs font-medium text-[var(--color-text-primary)]">Detection triggered</p>
                      <p className="text-[11px] text-[var(--color-text-secondary)]">
                        {formatTime(liveSelectedAlert.timestamp)} ({timeAgo(liveSelectedAlert.timestamp)})
                      </p>
                    </div>
                  </div>

                  {liveSelectedAlert.acknowledgedBy && liveSelectedAlert.acknowledgedAt && (
                    <div className="flex gap-3">
                      <div className="flex flex-col items-center">
                        <div className="w-2 h-2 rounded-full bg-blue-500 mt-1.5" />
                        <div className="w-px flex-1 bg-[var(--color-border-default)]" />
                      </div>
                      <div className="pb-3">
                        <p className="text-xs font-medium text-[var(--color-text-primary)]">
                          Acknowledged by {liveSelectedAlert.acknowledgedBy}
                        </p>
                        <p className="text-[11px] text-[var(--color-text-secondary)]">
                          {formatTime(liveSelectedAlert.acknowledgedAt)} ({timeAgo(liveSelectedAlert.acknowledgedAt)})
                        </p>
                      </div>
                    </div>
                  )}

                  {liveSelectedAlert.resolvedAt && (
                    <div className="flex gap-3">
                      <div className="flex flex-col items-center">
                        <div className="w-2 h-2 rounded-full bg-emerald-500 mt-1.5" />
                      </div>
                      <div className="pb-3">
                        <p className="text-xs font-medium text-[var(--color-text-primary)]">
                          {liveSelectedAlert.falsePositive ? "Marked as false positive" : "Resolved"}
                        </p>
                        <p className="text-[11px] text-[var(--color-text-secondary)]">
                          {formatTime(liveSelectedAlert.resolvedAt)} ({timeAgo(liveSelectedAlert.resolvedAt)})
                        </p>
                      </div>
                    </div>
                  )}

                  {liveSelectedAlert.status === "snoozed" && liveSelectedAlert.snoozedUntil && (
                    <div className="flex gap-3">
                      <div className="flex flex-col items-center">
                        <div className="w-2 h-2 rounded-full bg-amber-500 mt-1.5" />
                      </div>
                      <div className="pb-3">
                        <p className="text-xs font-medium text-[var(--color-text-primary)]">Snoozed</p>
                        <p className="text-[11px] text-[var(--color-text-secondary)]">
                          Until {formatTime(liveSelectedAlert.snoozedUntil)}
                        </p>
                      </div>
                    </div>
                  )}

                  {liveSelectedAlert.status === "active" && (
                    <div className="flex gap-3">
                      <div className="flex flex-col items-center">
                        <div className="w-2 h-2 rounded-full bg-red-500 mt-1.5 animate-pulse" />
                      </div>
                      <div className="pb-3">
                        <p className="text-xs font-medium text-red-600">Awaiting response</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Action buttons */}
          {canAct && <div className="border-t px-5 py-4 space-y-2">
            {liveSelectedAlert.status === "active" && (
              <div className="flex gap-2">
                <Button
                  variant="primary"
                  size="sm"
                  className="flex-1"
                  onClick={() => handleAcknowledge(liveSelectedAlert.id)}
                >
                  <CheckCircle2 size={14} />
                  Acknowledge
                </Button>
              </div>
            )}

            <div className="flex gap-2">
              {(liveSelectedAlert.status === "active" || liveSelectedAlert.status === "acknowledged") && (
                <>
                  <Button
                    variant="secondary"
                    size="sm"
                    className="flex-1 text-xs"
                    onClick={() => handleSnooze(liveSelectedAlert.id, 15)}
                  >
                    <Clock size={12} />
                    Snooze 15m
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    className="flex-1 text-xs"
                    onClick={() => handleSnooze(liveSelectedAlert.id, 60)}
                  >
                    Snooze 1h
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    className="flex-1 text-xs"
                    onClick={() => handleSnooze(liveSelectedAlert.id, 480)}
                  >
                    Snooze Shift
                  </Button>
                </>
              )}
            </div>

            <div className="flex gap-2">
              {liveSelectedAlert.status !== "resolved" && (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="flex-1 text-xs"
                    onClick={() => handleFalsePositive(liveSelectedAlert.id)}
                  >
                    <XCircle size={12} />
                    False Positive
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="flex-1 text-xs"
                    onClick={() => handleResolve(liveSelectedAlert.id)}
                  >
                    <CheckCircle2 size={12} />
                    Resolve
                  </Button>
                </>
              )}
            </div>
          </div>}
        </div>
      )}
    </div>
  )
}

const PPE_TYPES = [
  "Helmet",
  "Safety Vest",
  "Safety Goggles",
  "Safety Boots",
  "Face Mask",
  "Gloves",
  "Safety Harness",
  "Hairnet",
  "Apron",
  "Other",
]

function FalseNegativeModal({
  cameras,
  onClose,
  onSubmitted,
}: {
  cameras: { id: string; name: string; zone: string }[]
  onClose: () => void
  onSubmitted: () => void
}) {
  const [cameraId, setCameraId] = useState(cameras[0]?.id || "")
  const [missedType, setMissedType] = useState(PPE_TYPES[0])
  const [notes, setNotes] = useState("")
  const [submitting, setSubmitting] = useState(false)

  const selectedCam = cameras.find((c) => c.id === cameraId)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setSubmitting(true)
    try {
      await reportFalseNegative({
        cameraId,
        zone: selectedCam?.zone || "Unknown",
        missedType,
        timestamp: new Date().toISOString(),
        notes,
      })
      onSubmitted()
    } catch {
      setSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
      <div className="bg-white rounded-[var(--radius-lg)] shadow-xl w-full max-w-md mx-4">
        <div className="flex items-center justify-between px-5 py-4 border-b">
          <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">Report Missed Detection</h2>
          <button onClick={onClose} className="p-1 rounded hover:bg-[var(--color-bg-tertiary)] cursor-pointer">
            <X size={16} />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          <p className="text-xs text-[var(--color-text-secondary)]">
            Report a violation that the system failed to detect (false negative). This feedback helps improve detection accuracy.
          </p>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-[var(--color-text-primary)]">Camera</label>
            <select
              value={cameraId}
              onChange={(e) => setCameraId(e.target.value)}
              className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white cursor-pointer focus:outline-2 focus:outline-[var(--color-info)]"
              required
            >
              {cameras.map((cam) => (
                <option key={cam.id} value={cam.id}>
                  {cam.name} ({cam.zone})
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-[var(--color-text-primary)]">Missed PPE / Violation Type</label>
            <select
              value={missedType}
              onChange={(e) => setMissedType(e.target.value)}
              className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white cursor-pointer focus:outline-2 focus:outline-[var(--color-info)]"
              required
            >
              {PPE_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-[var(--color-text-primary)]">Notes (optional)</label>
            <textarea
              rows={3}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Describe the missed detection..."
              className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] resize-y"
            />
          </div>

          <div className="flex gap-2 pt-2">
            <Button type="button" variant="ghost" size="sm" className="flex-1" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" variant="primary" size="sm" className="flex-1" disabled={submitting}>
              {submitting ? <Loader2 size={14} className="animate-spin" /> : null}
              Submit Report
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
