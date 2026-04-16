import { useState, useEffect, useMemo } from "react"
import { Link } from "react-router-dom"
import { AlertTriangle, CheckCircle2, ShieldAlert, RefreshCw } from "lucide-react"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useAlertStore } from "@/stores/alertStore"
import { getAlertStats, getCameras, getAlertTimeSeries, getComplianceMetrics } from "@/lib/api"
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

interface Compliance {
  safety_compliance_pct: number
  ppe_compliance_pct: number
  mtta_seconds: number | null
  active_p1_count: number
  active_p2_count: number
  window_hours: number
}

type TrafficLight = "green" | "amber" | "red"

function lightFromThresholds(value: number, green: number, amber: number): TrafficLight {
  if (value >= green) return "green"
  if (value >= amber) return "amber"
  return "red"
}

function lightClasses(light: TrafficLight): { border: string; value: string } {
  if (light === "green")
    return { border: "border-l-[var(--color-success)]", value: "text-[var(--color-success)]" }
  if (light === "amber")
    return { border: "border-l-[var(--color-warning)]", value: "text-[var(--color-warning)]" }
  return { border: "border-l-[var(--color-critical)]", value: "text-[var(--color-critical)]" }
}

function formatMtta(seconds: number | null): string {
  if (seconds == null) return "—"
  if (seconds < 60) return `${Math.round(seconds)}s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return s === 0 ? `${m}m` : `${m}m ${s}s`
}


export function Dashboard() {
  const { fetchAlerts } = useAlertStore()
  const [stats, setStats] = useState<Stats | null>(null)
  const [compliance, setCompliance] = useState<Compliance | null>(null)
  const [cameraCount, setCameraCount] = useState({ total: 0, online: 0 })
  const [timeSeries, setTimeSeries] = useState<any[]>([])
  const [refreshing, setRefreshing] = useState(false)

  const loadData = async () => {
    setRefreshing(true)
    try {
      await fetchAlerts()
      const [s, cams, ts, comp] = await Promise.all([
        getAlertStats(),
        getCameras(),
        getAlertTimeSeries(24),
        getComplianceMetrics(24),
      ])
      setStats(s)
      setCompliance(comp)
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

  const kpis = useMemo(() => {
    if (!compliance) return []
    const uptimePct =
      cameraCount.total > 0 ? Math.round((cameraCount.online / cameraCount.total) * 100) : 0
    const offline = cameraCount.total - cameraCount.online
    return [
      {
        label: "Safety Compliance",
        value: `${compliance.safety_compliance_pct}%`,
        caption: "last 24h · no P1/P2 violations",
        light: lightFromThresholds(compliance.safety_compliance_pct, 98, 90),
      },
      {
        label: "PPE Compliance",
        value: `${compliance.ppe_compliance_pct}%`,
        caption: "last 24h · helmet + vest",
        light: lightFromThresholds(compliance.ppe_compliance_pct, 95, 85),
      },
      {
        label: "Camera Uptime",
        value: `${cameraCount.online} / ${cameraCount.total}`,
        caption:
          cameraCount.total === 0
            ? "no cameras configured"
            : offline === 0
              ? "all online"
              : `${offline} offline (${uptimePct}%)`,
        light:
          cameraCount.total === 0
            ? ("amber" as TrafficLight)
            : lightFromThresholds(uptimePct, 100, 80),
      },
      {
        label: "Mean Time to Acknowledge",
        value: formatMtta(compliance.mtta_seconds),
        caption:
          compliance.mtta_seconds == null
            ? "no alerts acked in 24h"
            : "avg ack time · last 24h",
        light:
          compliance.mtta_seconds == null
            ? ("amber" as TrafficLight)
            : compliance.mtta_seconds <= 120
              ? "green"
              : compliance.mtta_seconds <= 600
                ? "amber"
                : "red",
      },
    ]
  }, [compliance, cameraCount])

  const safetyBanner = useMemo(() => {
    if (!compliance) return null
    const offline = cameraCount.total - cameraCount.online
    if (compliance.active_p1_count > 0) {
      return {
        light: "red" as TrafficLight,
        icon: ShieldAlert,
        title: `${compliance.active_p1_count} active P1 alert${compliance.active_p1_count === 1 ? "" : "s"} — critical violations need attention`,
        detail:
          compliance.active_p2_count > 0
            ? `${compliance.active_p2_count} P2 alert${compliance.active_p2_count === 1 ? "" : "s"} also active`
            : offline > 0
              ? `${offline} camera${offline === 1 ? "" : "s"} offline`
              : "All cameras online",
      }
    }
    if (compliance.active_p2_count > 0 || offline > 0) {
      const bits: string[] = []
      if (compliance.active_p2_count > 0)
        bits.push(
          `${compliance.active_p2_count} active P2 alert${compliance.active_p2_count === 1 ? "" : "s"}`,
        )
      if (offline > 0)
        bits.push(`${offline} camera${offline === 1 ? "" : "s"} offline`)
      return {
        light: "amber" as TrafficLight,
        icon: AlertTriangle,
        title: "Attention needed",
        detail: bits.join(" · "),
      }
    }
    return {
      light: "green" as TrafficLight,
      icon: CheckCircle2,
      title: "Safe — no active violations",
      detail: `${cameraCount.online}/${cameraCount.total} cameras online`,
    }
  }, [compliance, cameraCount])

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

  if (!stats || !compliance) {
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

      {/* Safety status banner */}
      {safetyBanner && (() => {
        const { icon: Icon, light, title, detail } = safetyBanner
        const bgVar =
          light === "red"
            ? "var(--color-critical-bg)"
            : light === "amber"
              ? "var(--color-warning-bg)"
              : "var(--color-success-bg)"
        const fgVar =
          light === "red"
            ? "var(--color-critical)"
            : light === "amber"
              ? "var(--color-warning)"
              : "var(--color-success)"
        return (
          <div
            className="flex items-center gap-3 rounded-[var(--radius-md)] border-l-4 px-4 py-3"
            style={{ backgroundColor: bgVar, borderLeftColor: fgVar }}
          >
            <Icon className="w-5 h-5 shrink-0" style={{ color: fgVar }} />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold" style={{ color: fgVar }}>
                {title}
              </p>
              <p className="text-xs text-[var(--color-text-secondary)]">{detail}</p>
            </div>
            {(compliance.active_p1_count > 0 || compliance.active_p2_count > 0) && (
              <Link to="/alerts" className="text-xs font-medium underline" style={{ color: fgVar }}>
                View alerts
              </Link>
            )}
          </div>
        )
      })()}

      {/* KPI row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {kpis.map((kpi) => {
          const { border, value } = lightClasses(kpi.light)
          return (
            <Card key={kpi.label} className={`border-l-4 ${border}`}>
              <p className="text-xs font-medium text-[var(--color-text-secondary)] mb-1">
                {kpi.label}
              </p>
              <p className={`text-2xl font-bold mb-1 ${value}`}>{kpi.value}</p>
              <p className="text-xs text-[var(--color-text-tertiary)]">{kpi.caption}</p>
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
