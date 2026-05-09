import { useState, useEffect, useMemo, useCallback } from "react"
import {
  Download,
  FileText,
  Filter,
  X,
  ChevronLeft,
  ChevronRight,
  Eye,
  Calendar,
} from "lucide-react"
import { useNavigate } from "react-router-dom"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { SeverityBadge } from "@/components/ui/SeverityBadge"
import { StatusBadge } from "@/components/ui/StatusBadge"
import { Skeleton } from "@/components/ui/skeleton"
import { severityConfig } from "@/lib/constants"
import { getAlerts, getCameras, getAlertTimeSeries, API_BASE, downloadPdfReport } from "@/lib/api"
import type { Severity, AlertStatus, Alert } from "@/types"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts"

const PAGE_SIZE = 25

type DatePreset = "today" | "7d" | "30d" | "90d" | "custom"

const DATE_PRESETS: { value: DatePreset; label: string }[] = [
  { value: "today", label: "Today" },
  { value: "7d", label: "Last 7 days" },
  { value: "30d", label: "Last 30 days" },
  { value: "90d", label: "Last 90 days" },
  { value: "custom", label: "Custom" },
]

function presetToRange(preset: DatePreset): { from: string; to: string } {
  const now = new Date()
  const to = now.toISOString().slice(0, 10)
  let from: Date
  switch (preset) {
    case "today":
      from = new Date(now.getFullYear(), now.getMonth(), now.getDate())
      break
    case "7d":
      from = new Date(now.getTime() - 7 * 86400000)
      break
    case "30d":
      from = new Date(now.getTime() - 30 * 86400000)
      break
    case "90d":
      from = new Date(now.getTime() - 90 * 86400000)
      break
    default:
      from = new Date(now.getTime() - 30 * 86400000)
  }
  return { from: from.toISOString().slice(0, 10), to }
}

function formatDate(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" })
}

function formatDateTime(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })
}

function downloadCSV(alerts: Alert[], filename: string) {
  const headers = [
    "Alert ID",
    "Timestamp",
    "Severity",
    "Status",
    "Rule",
    "Camera",
    "Zone",
    "Confidence",
    "Source",
    "Description",
    "False Positive",
  ]
  const rows = alerts.map((a) => [
    a.id,
    a.timestamp,
    a.severity,
    a.status,
    a.rule,
    a.cameraName,
    a.zone,
    Math.round(a.confidence * 100) + "%",
    a.source,
    `"${(a.description || "").replace(/"/g, '""')}"`,
    a.falsePositive ? "Yes" : "No",
  ])
  const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n")
  const blob = new Blob([csv], { type: "text/csv" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

const SEVERITY_COLORS: Record<Severity, string> = { P1: "#dc2626", P2: "#f97316", P3: "#f59e0b", P4: "#2563eb" }

export function Reports() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [allAlerts, setAllAlerts] = useState<Alert[]>([])
  const [cameras, setCameras] = useState<{ id: string; name: string }[]>([])
  const [timeSeries, setTimeSeries] = useState<any[]>([])

  // Filters
  const [datePreset, setDatePreset] = useState<DatePreset>("30d")
  const [customFrom, setCustomFrom] = useState("")
  const [customTo, setCustomTo] = useState("")
  const [sevFilter, setSevFilter] = useState<Set<Severity>>(new Set())
  const [statusFilter, setStatusFilter] = useState<Set<AlertStatus>>(new Set())
  const [cameraFilter, setCameraFilter] = useState("")
  const [ruleFilter, setRuleFilter] = useState("")

  // Pagination
  const [page, setPage] = useState(0)

  const dateRange = useMemo(() => {
    if (datePreset === "custom" && customFrom && customTo) {
      return { from: customFrom, to: customTo }
    }
    return presetToRange(datePreset)
  }, [datePreset, customFrom, customTo])

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const [alertData, camData] = await Promise.all([
        getAlerts({ limit: 1000 }),
        getCameras(),
      ])
      setAllAlerts(alertData)
      const camList = Array.isArray(camData) ? camData : []
      setCameras(camList.map((c: any) => ({ id: c.id, name: c.name || c.id })))

      const hours = Math.max(
        24,
        Math.ceil((new Date().getTime() - new Date(dateRange.from).getTime()) / 3600000)
      )
      const ts = await getAlertTimeSeries(hours)
      setTimeSeries(
        (ts || []).map((d: any) => ({
          ...d,
          hour: d.hour ? d.hour.slice(5, 16).replace("T", " ") : "",
        }))
      )
    } catch {
      // fail silently
    } finally {
      setLoading(false)
    }
  }, [dateRange.from])

  useEffect(() => {
    loadData()
  }, [loadData])

  // Apply filters
  const filtered = useMemo(() => {
    const fromTs = new Date(dateRange.from + "T00:00:00").getTime()
    const toTs = new Date(dateRange.to + "T23:59:59").getTime()

    return allAlerts.filter((a) => {
      const ts = new Date(a.timestamp).getTime()
      if (ts < fromTs || ts > toTs) return false
      if (sevFilter.size > 0 && !sevFilter.has(a.severity)) return false
      if (statusFilter.size > 0 && !statusFilter.has(a.status)) return false
      if (cameraFilter && a.cameraId !== cameraFilter) return false
      if (ruleFilter && a.rule !== ruleFilter) return false
      return true
    })
  }, [allAlerts, dateRange, sevFilter, statusFilter, cameraFilter, ruleFilter])

  // Reset page when filters change
  useEffect(() => { setPage(0) }, [filtered.length])

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const paged = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  // Unique rules for filter dropdown
  const uniqueRules = useMemo(
    () => [...new Set(allAlerts.map((a) => a.rule))].sort(),
    [allAlerts]
  )

  // Summary stats
  const summary = useMemo(() => {
    const bySev: Record<string, number> = { P1: 0, P2: 0, P3: 0, P4: 0 }
    const byStatus: Record<string, number> = {}
    const byCamera: Record<string, number> = {}
    const byRule: Record<string, number> = {}
    let fpCount = 0

    for (const a of filtered) {
      bySev[a.severity] = (bySev[a.severity] || 0) + 1
      byStatus[a.status] = (byStatus[a.status] || 0) + 1
      byCamera[a.cameraName] = (byCamera[a.cameraName] || 0) + 1
      byRule[a.rule] = (byRule[a.rule] || 0) + 1
      if (a.falsePositive) fpCount++
    }

    return { bySev, byStatus, byCamera, byRule, fpCount, total: filtered.length }
  }, [filtered])

  const sevChartData = useMemo(
    () =>
      (["P1", "P2", "P3", "P4"] as Severity[]).map((s) => ({
        name: `${s} - ${severityConfig[s].label}`,
        value: summary.bySev[s] || 0,
        severity: s,
      })),
    [summary]
  )

  const topCamerasData = useMemo(
    () =>
      Object.entries(summary.byCamera)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([name, count]) => ({ name: name.length > 20 ? name.slice(0, 18) + "..." : name, count })),
    [summary]
  )

  const topRulesData = useMemo(
    () =>
      Object.entries(summary.byRule)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([name, count]) => ({ name: name.length > 25 ? name.slice(0, 23) + "..." : name, count })),
    [summary]
  )

  const hasFilters = sevFilter.size > 0 || statusFilter.size > 0 || !!cameraFilter || !!ruleFilter
  const clearFilters = () => {
    setSevFilter(new Set())
    setStatusFilter(new Set())
    setCameraFilter("")
    setRuleFilter("")
  }

  if (loading) {
    return (
      <div className="space-y-6 p-6">
        <div className="flex items-center justify-between">
          <Skeleton className="h-7 w-32" />
          <Skeleton className="h-9 w-36" />
        </div>
        <Skeleton className="h-16 w-full rounded-[var(--radius-md)]" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-24 rounded-[var(--radius-md)]" />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Skeleton className="h-72 rounded-[var(--radius-md)]" />
          <Skeleton className="h-72 rounded-[var(--radius-md)]" />
        </div>
        <Skeleton className="h-96 rounded-[var(--radius-md)]" />
      </div>
    )
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Reports</h1>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            className="gap-1.5 text-xs"
            onClick={() => downloadPdfReport(dateRange.from, dateRange.to)}
            disabled={filtered.length === 0}
          >
            <FileText size={14} />
            Download PDF
          </Button>
          <Button
            variant="secondary"
            size="sm"
            className="gap-1.5 text-xs"
            onClick={() => downloadCSV(filtered, `safety-report-${dateRange.from}-to-${dateRange.to}.csv`)}
            disabled={filtered.length === 0}
          >
            <Download size={14} />
            Export CSV
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <div className="flex flex-wrap items-end gap-4">
          {/* Date range */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[var(--color-text-secondary)] flex items-center gap-1">
              <Calendar size={12} /> Date Range
            </label>
            <div className="flex items-center gap-1.5">
              {DATE_PRESETS.map((p) => (
                <button
                  key={p.value}
                  onClick={() => setDatePreset(p.value)}
                  className={`px-2.5 py-1.5 text-xs font-medium rounded-[var(--radius-md)] border cursor-pointer transition-colors ${
                    datePreset === p.value
                      ? "bg-[var(--color-text-primary)] text-white border-transparent"
                      : "bg-white text-[var(--color-text-secondary)] border-[var(--color-border-default)] hover:border-[var(--color-border-active)]"
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
            {datePreset === "custom" && (
              <div className="flex items-center gap-2 mt-1.5">
                <input
                  type="date"
                  value={customFrom}
                  onChange={(e) => setCustomFrom(e.target.value)}
                  className="px-2 py-1 text-xs rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)]"
                />
                <span className="text-xs text-[var(--color-text-tertiary)]">to</span>
                <input
                  type="date"
                  value={customTo}
                  onChange={(e) => setCustomTo(e.target.value)}
                  className="px-2 py-1 text-xs rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)]"
                />
              </div>
            )}
          </div>

          {/* Severity filter */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[var(--color-text-secondary)]">Severity</label>
            <div className="flex items-center gap-1">
              {(["P1", "P2", "P3", "P4"] as Severity[]).map((s) => {
                const active = sevFilter.has(s)
                return (
                  <button
                    key={s}
                    onClick={() => {
                      const next = new Set(sevFilter)
                      next.has(s) ? next.delete(s) : next.add(s)
                      setSevFilter(next)
                    }}
                    className={`px-2 py-1 text-xs font-medium rounded-[var(--radius-sm)] border cursor-pointer transition-colors ${
                      active
                        ? "border-current"
                        : "border-transparent opacity-60 hover:opacity-100"
                    }`}
                    style={{
                      backgroundColor: active ? severityConfig[s].bg : undefined,
                      color: severityConfig[s].textColor,
                    }}
                  >
                    {s}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Status filter */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[var(--color-text-secondary)]">Status</label>
            <div className="flex items-center gap-1">
              {(["active", "acknowledged", "resolved", "snoozed"] as AlertStatus[]).map((s) => {
                const active = statusFilter.has(s)
                return (
                  <button
                    key={s}
                    onClick={() => {
                      const next = new Set(statusFilter)
                      next.has(s) ? next.delete(s) : next.add(s)
                      setStatusFilter(next)
                    }}
                    className={`px-2 py-1 text-xs font-medium rounded-[var(--radius-sm)] border cursor-pointer transition-colors capitalize ${
                      active
                        ? "bg-[var(--color-bg-tertiary)] text-[var(--color-text-primary)] border-[var(--color-border-active)]"
                        : "border-transparent text-[var(--color-text-secondary)] opacity-60 hover:opacity-100"
                    }`}
                  >
                    {s}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Camera filter */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[var(--color-text-secondary)]">Camera</label>
            <select
              value={cameraFilter}
              onChange={(e) => setCameraFilter(e.target.value)}
              className="text-xs border rounded-[var(--radius-md)] px-2 py-1.5 bg-white cursor-pointer focus:outline-2 focus:outline-[var(--color-info)]"
            >
              <option value="">All Cameras</option>
              {cameras.map((cam) => (
                <option key={cam.id} value={cam.id}>
                  {cam.name}
                </option>
              ))}
            </select>
          </div>

          {/* Rule filter */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-[var(--color-text-secondary)]">Rule</label>
            <select
              value={ruleFilter}
              onChange={(e) => setRuleFilter(e.target.value)}
              className="text-xs border rounded-[var(--radius-md)] px-2 py-1.5 bg-white cursor-pointer focus:outline-2 focus:outline-[var(--color-info)]"
            >
              <option value="">All Rules</option>
              {uniqueRules.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>

          {hasFilters && (
            <Button variant="ghost" size="sm" onClick={clearFilters} className="text-xs gap-1 mb-0.5">
              <X size={12} /> Clear filters
            </Button>
          )}

          <div className="ml-auto text-xs text-[var(--color-text-secondary)] self-end pb-1">
            {filtered.length} violation{filtered.length !== 1 ? "s" : ""} &middot;{" "}
            {formatDate(dateRange.from)} &ndash; {formatDate(dateRange.to)}
          </div>
        </div>
      </Card>

      {/* Summary KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {(["P1", "P2", "P3", "P4"] as Severity[]).map((s) => {
          const count = summary.bySev[s] || 0
          return (
            <div
              key={s}
              className="rounded-[var(--radius-md)] p-4 text-center border-l-4"
              style={{ backgroundColor: severityConfig[s].bg, borderLeftColor: severityConfig[s].color }}
            >
              <p className="text-2xl font-bold" style={{ color: severityConfig[s].color }}>
                {count}
              </p>
              <p className="text-xs font-medium mt-0.5" style={{ color: severityConfig[s].textColor }}>
                {s} - {severityConfig[s].label}
              </p>
            </div>
          )
        })}
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Violations by Camera */}
        <Card>
          <CardHeader>
            <CardTitle>Violations by Camera</CardTitle>
          </CardHeader>
          <CardContent>
            {topCamerasData.length === 0 ? (
              <p className="text-sm text-[var(--color-text-tertiary)] py-8 text-center">No data</p>
            ) : (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={topCamerasData} layout="vertical" margin={{ left: 0, right: 16, top: 4, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-default)" horizontal={false} />
                    <XAxis type="number" tick={{ fontSize: 11, fill: "var(--color-text-tertiary)" }} allowDecimals={false} />
                    <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} width={120} />
                    <Tooltip contentStyle={{ fontSize: 12, borderRadius: "var(--radius-md)" }} />
                    <Bar dataKey="count" fill="var(--color-warning)" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Violations by Rule */}
        <Card>
          <CardHeader>
            <CardTitle>Violations by Rule</CardTitle>
          </CardHeader>
          <CardContent>
            {topRulesData.length === 0 ? (
              <p className="text-sm text-[var(--color-text-tertiary)] py-8 text-center">No data</p>
            ) : (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={topRulesData} layout="vertical" margin={{ left: 0, right: 16, top: 4, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-default)" horizontal={false} />
                    <XAxis type="number" tick={{ fontSize: 11, fill: "var(--color-text-tertiary)" }} allowDecimals={false} />
                    <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} width={160} />
                    <Tooltip contentStyle={{ fontSize: 12, borderRadius: "var(--radius-md)" }} />
                    <Bar dataKey="count" fill="var(--color-critical)" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Severity distribution + status breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Severity Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            {summary.total === 0 ? (
              <p className="text-sm text-[var(--color-text-tertiary)] py-8 text-center">No data</p>
            ) : (
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={sevChartData.filter((d) => d.value > 0)}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`}
                    >
                      {sevChartData
                        .filter((d) => d.value > 0)
                        .map((d) => (
                          <Cell key={d.severity} fill={SEVERITY_COLORS[d.severity]} />
                        ))}
                    </Pie>
                    <Tooltip contentStyle={{ fontSize: 12, borderRadius: "var(--radius-md)" }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-[var(--color-text-secondary)]">Total Violations</span>
                <span className="font-semibold text-[var(--color-text-primary)]">{summary.total}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-[var(--color-text-secondary)]">Active</span>
                <span className="font-semibold text-[var(--color-critical)]">{summary.byStatus["active"] || 0}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-[var(--color-text-secondary)]">Acknowledged</span>
                <span className="font-semibold text-[var(--color-info)]">{summary.byStatus["acknowledged"] || 0}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-[var(--color-text-secondary)]">Resolved</span>
                <span className="font-semibold text-[var(--color-success)]">{summary.byStatus["resolved"] || 0}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-[var(--color-text-secondary)]">Snoozed</span>
                <span className="font-semibold text-[var(--color-warning)]">{summary.byStatus["snoozed"] || 0}</span>
              </div>
              <div className="border-t pt-3 flex justify-between text-sm">
                <span className="text-[var(--color-text-secondary)]">False Positives</span>
                <span className="font-semibold text-[var(--color-text-primary)]">{summary.fpCount}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-[var(--color-text-secondary)]">Cameras Involved</span>
                <span className="font-semibold text-[var(--color-text-primary)]">
                  {Object.keys(summary.byCamera).length}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-[var(--color-text-secondary)]">Unique Rules Triggered</span>
                <span className="font-semibold text-[var(--color-text-primary)]">
                  {Object.keys(summary.byRule).length}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Violations table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Violation History</CardTitle>
            <div className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]">
              <span>
                Page {page + 1} of {totalPages}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0"
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
              >
                <ChevronLeft size={14} />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0"
                disabled={page >= totalPages - 1}
                onClick={() => setPage((p) => p + 1)}
              >
                <ChevronRight size={14} />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-[var(--color-bg-secondary)]">
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">Snapshot</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">Sev</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">Time</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">Camera</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">Zone</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">Rule</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">Conf</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">Status</th>
                  <th className="text-right px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)]">Actions</th>
                </tr>
              </thead>
              <tbody>
                {paged.map((alert) => (
                  <tr
                    key={alert.id}
                    className="border-b last:border-b-0 hover:bg-[var(--color-bg-secondary)] transition-colors"
                  >
                    <td className="px-4 py-2">
                      {alert.snapshotUrl ? (
                        <img
                          src={`${API_BASE}${alert.snapshotUrl}`}
                          alt=""
                          className="w-16 h-10 object-cover rounded border"
                        />
                      ) : (
                        <div className="w-16 h-10 rounded border bg-[var(--color-bg-tertiary)] flex items-center justify-center">
                          <FileText size={12} className="text-[var(--color-text-tertiary)]" />
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-2">
                      <SeverityBadge severity={alert.severity} />
                    </td>
                    <td className="px-4 py-2 text-xs text-[var(--color-text-secondary)] whitespace-nowrap">
                      {formatDateTime(alert.timestamp)}
                    </td>
                    <td className="px-4 py-2 text-xs font-medium text-[var(--color-text-primary)] max-w-[160px] truncate">
                      {alert.cameraName}
                    </td>
                    <td className="px-4 py-2 text-xs text-[var(--color-text-secondary)]">
                      {alert.zone}
                    </td>
                    <td className="px-4 py-2 text-xs font-medium text-[var(--color-text-primary)]">
                      {alert.rule}
                    </td>
                    <td className="px-4 py-2 text-xs text-[var(--color-text-secondary)] tabular-nums">
                      {Math.round(alert.confidence * 100)}%
                    </td>
                    <td className="px-4 py-2">
                      <StatusBadge status={alert.status} />
                    </td>
                    <td className="px-4 py-2 text-right">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => navigate(`/alerts/${alert.id}`)}
                      >
                        <Eye size={12} />
                      </Button>
                    </td>
                  </tr>
                ))}
                {paged.length === 0 && (
                  <tr>
                    <td colSpan={9} className="px-4 py-12 text-center text-sm text-[var(--color-text-secondary)]">
                      No violations found for the selected filters and date range.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
