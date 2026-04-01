import { useState, useEffect, useMemo } from "react"
import { ArrowUp, ArrowDown, Minus, RefreshCw } from "lucide-react"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useAlertStore } from "@/stores/alertStore"
import { getAlertStats, getCameras, getAlertTimeSeries } from "@/lib/api"
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts"

const PIE_COLORS = ["#dc2626", "#f97316", "#2563eb", "#f59e0b", "#059669", "#8b5cf6", "#a3a3a3"]

interface Stats {
  total: number
  active: number
  acknowledged: number
  resolved: number
  bySeverity: Record<string, number>
  byRule: Record<string, number>
  byZone: Record<string, number>
  byCamera: Record<string, number>
}

function complianceCellColor(value: number): string {
  if (value > 95) return "bg-[var(--color-success-bg)] text-[#065f46]"
  if (value >= 85) return "bg-[var(--color-warning-bg)] text-[#92400e]"
  return "bg-[var(--color-critical-bg)] text-[#991b1b]"
}

export function Dashboard() {
  const { alerts, fetchAlerts } = useAlertStore()
  const [stats, setStats] = useState<Stats | null>(null)
  const [cameraCount, setCameraCount] = useState({ total: 0, online: 0 })
  const [timeSeries, setTimeSeries] = useState<any[]>([])
  const [refreshing, setRefreshing] = useState(false)

  const loadData = async () => {
    setRefreshing(true)
    try {
      await fetchAlerts()
      const [s, cams, ts] = await Promise.all([getAlertStats(), getCameras(), getAlertTimeSeries(24)])
      setStats(s)
      setTimeSeries(
        (ts || []).map((d: any) => ({
          ...d,
          hour: d.hour ? d.hour.slice(11, 16) : "",
        }))
      )
      const camList = Array.isArray(cams) ? cams : []
      const online = camList.filter((c: any) => c.status === "online").length
      setCameraCount({ total: camList.length, online })
    } catch {
      // stats may not be available yet
    }
    setRefreshing(false)
  }

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, 30000)
    return () => clearInterval(interval)
  }, [])

  const activeAlerts = useMemo(() => alerts.filter((a) => a.status === "active"), [alerts])

  const kpis = useMemo(() => {
    if (!stats) return []
    const resolvedPct = stats.total > 0 ? Math.round((stats.resolved / stats.total) * 100) : 0
    return [
      {
        label: "Total Alerts",
        value: stats.total,
        change: 0,
        changeLabel: "all time",
      },
      {
        label: "Active Alerts",
        value: stats.active,
        change: stats.active > 0 ? -1 : 0,
        changeLabel: stats.active > 0 ? "needs attention" : "all clear",
        invertColor: true,
      },
      {
        label: "Active Cameras",
        value: `${cameraCount.online} / ${cameraCount.total}`,
        change: cameraCount.total === cameraCount.online ? 1 : -1,
        changeLabel: cameraCount.total === cameraCount.online ? "all online" : `${cameraCount.total - cameraCount.online} offline`,
      },
      {
        label: "Resolved Rate",
        value: `${resolvedPct}%`,
        change: resolvedPct >= 80 ? 1 : resolvedPct > 0 ? -1 : 0,
        changeLabel: "of total alerts",
      },
    ]
  }, [stats, cameraCount])

  const violationsByZone = useMemo(() => {
    if (!stats?.byZone) return []
    return Object.entries(stats.byZone)
      .map(([zone, count]) => ({ zone, count }))
      .sort((a, b) => b.count - a.count)
  }, [stats])

  const violationsByType = useMemo(() => {
    if (!stats?.byRule) return []
    return Object.entries(stats.byRule)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
  }, [stats])

  const severityTrend = timeSeries

  const maxZoneCount = violationsByZone[0]?.count || 1

  if (!stats) {
    return (
      <div className="flex items-center justify-center h-64 text-sm text-[var(--color-text-tertiary)]">
        Loading dashboard data...
      </div>
    )
  }

  return (
    <div className="space-y-6 p-6">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Dashboard</h1>
        <Button
          variant="ghost"
          size="sm"
          onClick={loadData}
          disabled={refreshing}
        >
          <RefreshCw className={`w-4 h-4 mr-1.5 ${refreshing ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {kpis.map((kpi) => {
          const invertColor = "invertColor" in kpi && kpi.invertColor
          const isPositive = invertColor
            ? kpi.change < 0
            : kpi.change > 0
          const isNeutral = kpi.change === 0
          const borderColor = isNeutral
            ? "border-l-[var(--color-border-strong)]"
            : isPositive
              ? "border-l-[var(--color-success)]"
              : "border-l-[var(--color-critical)]"

          return (
            <Card key={kpi.label} className={`border-l-4 ${borderColor}`}>
              <p className="text-xs font-medium text-[var(--color-text-secondary)] mb-1">
                {kpi.label}
              </p>
              <p className="text-2xl font-bold text-[var(--color-text-primary)] mb-1">
                {kpi.value}
              </p>
              <div className="flex items-center gap-1 text-xs">
                {isNeutral ? (
                  <Minus className="w-3 h-3 text-[var(--color-text-tertiary)]" />
                ) : isPositive ? (
                  <ArrowUp className="w-3 h-3 text-[var(--color-success)]" />
                ) : (
                  <ArrowDown className="w-3 h-3 text-[var(--color-critical)]" />
                )}
                <span className="text-[var(--color-text-tertiary)]">{kpi.changeLabel}</span>
              </div>
            </Card>
          )
        })}
      </div>

      {/* Severity breakdown */}
      <Card>
        <CardHeader>
          <CardTitle>Alert Severity Breakdown</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-3">
            {(["P1", "P2", "P3", "P4"] as const).map((sev) => {
              const count = stats.bySeverity[sev] || 0
              const config: Record<string, { label: string; color: string; bg: string }> = {
                P1: { label: "Critical", color: "#dc2626", bg: "#fef2f2" },
                P2: { label: "High", color: "#f97316", bg: "#fff7ed" },
                P3: { label: "Medium", color: "#f59e0b", bg: "#fffbeb" },
                P4: { label: "Low", color: "#2563eb", bg: "#eff6ff" },
              }
              const c = config[sev]
              return (
                <div
                  key={sev}
                  className="rounded-[var(--radius-md)] p-3 text-center"
                  style={{ backgroundColor: c.bg }}
                >
                  <p className="text-2xl font-bold" style={{ color: c.color }}>
                    {count}
                  </p>
                  <p className="text-xs font-medium mt-0.5" style={{ color: c.color }}>
                    {sev} - {c.label}
                  </p>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Alerts by hour */}
      {severityTrend.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Alerts by Hour</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={severityTrend} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-default)" />
                  <XAxis
                    dataKey="hour"
                    tick={{ fontSize: 11, fill: "var(--color-text-tertiary)" }}
                    tickLine={false}
                    axisLine={{ stroke: "var(--color-border-default)" }}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "var(--color-text-tertiary)" }}
                    tickLine={false}
                    axisLine={false}
                    allowDecimals={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "white",
                      border: "1px solid var(--color-border-default)",
                      borderRadius: "var(--radius-md)",
                      fontSize: 12,
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                  <Area type="monotone" dataKey="P1" stackId="1" stroke="#dc2626" fill="#dc2626" fillOpacity={0.3} />
                  <Area type="monotone" dataKey="P2" stackId="1" stroke="#f97316" fill="#f97316" fillOpacity={0.3} />
                  <Area type="monotone" dataKey="P3" stackId="1" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.3} />
                  <Area type="monotone" dataKey="P4" stackId="1" stroke="#2563eb" fill="#2563eb" fillOpacity={0.3} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Bottom row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Top Violating Zones */}
        <Card>
          <CardHeader>
            <CardTitle>Alerts by Zone</CardTitle>
          </CardHeader>
          <CardContent>
            {violationsByZone.length === 0 ? (
              <p className="text-sm text-[var(--color-text-tertiary)] py-4 text-center">
                No alerts yet. Violations will appear here as they are detected.
              </p>
            ) : (
              <div className="space-y-3">
                {violationsByZone.map((item) => (
                  <div key={item.zone} className="flex items-center gap-3">
                    <span className="text-sm font-medium text-[var(--color-text-primary)] w-32 shrink-0 truncate">
                      {item.zone}
                    </span>
                    <div className="flex-1 h-6 bg-[var(--color-bg-tertiary)] rounded-[var(--radius-sm)] overflow-hidden">
                      <div
                        className="h-full bg-[var(--color-critical)] rounded-[var(--radius-sm)] transition-all"
                        style={{
                          width: `${(item.count / maxZoneCount) * 100}%`,
                          opacity: 0.3 + (item.count / maxZoneCount) * 0.7,
                        }}
                      />
                    </div>
                    <span className="text-sm font-semibold text-[var(--color-text-primary)] w-8 text-right">
                      {item.count}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Violation Types */}
        <Card>
          <CardHeader>
            <CardTitle>Alert Types</CardTitle>
          </CardHeader>
          <CardContent>
            {violationsByType.length === 0 ? (
              <p className="text-sm text-[var(--color-text-tertiary)] py-4 text-center">
                No alerts yet.
              </p>
            ) : (
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={violationsByType}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={80}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {violationsByType.map((_, idx) => (
                        <Cell key={idx} fill={PIE_COLORS[idx % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "white",
                        border: "1px solid var(--color-border-default)",
                        borderRadius: "var(--radius-md)",
                        fontSize: 12,
                      }}
                    />
                    <Legend
                      wrapperStyle={{ fontSize: 11 }}
                      formatter={(value: string) => (
                        <span className="text-[var(--color-text-secondary)]">{value}</span>
                      )}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
