import { useState, useEffect, useMemo } from "react"
import { Link, useNavigate } from "react-router-dom"
import { AlertTriangle, CheckCircle2, ShieldAlert, TrendingUp, TrendingDown, Minus } from "lucide-react"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { useAlertStore } from "@/stores/alertStore"
import { getAlertStats, getCameras, getAlertTimeSeries, getComplianceMetrics, getAlerts, getZoneTimeHeatmap, getSpatialHeatmap } from "@/lib/api"
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"

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
  prev_safety_compliance_pct?: number
  prev_ppe_compliance_pct?: number
  prev_mtta_seconds?: number | null
}

type TrafficLight = "green" | "amber" | "red"
type TimeRange = 24 | 168 | 720

const TIME_RANGES: { value: TimeRange; label: string }[] = [
  { value: 24, label: "24h" },
  { value: 168, label: "7d" },
  { value: 720, label: "30d" },
]

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

function TrendBadge({ current, previous, inverted }: { current: number; previous: number | undefined | null; inverted?: boolean }) {
  if (previous == null || previous === 0) return null
  const delta = current - previous
  if (Math.abs(delta) < 0.1) return (
    <span className="inline-flex items-center gap-0.5 text-[10px] text-[var(--color-text-tertiary)]">
      <Minus className="w-3 h-3" /> 0%
    </span>
  )
  const pct = Math.abs(delta).toFixed(1)
  const improving = inverted ? delta > 0 : delta < 0
  const color = improving ? "text-[var(--color-success)]" : "text-[var(--color-critical)]"
  const Icon = delta > 0 ? TrendingUp : TrendingDown
  return (
    <span className={`inline-flex items-center gap-0.5 text-[10px] font-medium ${color}`}>
      <Icon className="w-3 h-3" /> {pct}%
    </span>
  )
}

function MttaTrendBadge({ current, previous }: { current: number | null; previous: number | null | undefined }) {
  if (current == null || previous == null) return null
  const delta = current - previous
  if (Math.abs(delta) < 1) return null
  // Lower MTTA is better
  const improving = delta < 0
  const pct = Math.abs(Math.round((delta / previous) * 100))
  const color = improving ? "text-[var(--color-success)]" : "text-[var(--color-critical)]"
  const Icon = improving ? TrendingDown : TrendingUp
  return (
    <span className={`inline-flex items-center gap-0.5 text-[10px] font-medium ${color}`}>
      <Icon className="w-3 h-3" /> {pct}%
    </span>
  )
}


export function Dashboard() {
  const { fetchAlerts } = useAlertStore()
  const navigate = useNavigate()
  const [stats, setStats] = useState<Stats | null>(null)
  const [compliance, setCompliance] = useState<Compliance | null>(null)
  const [cameraCount, setCameraCount] = useState({ total: 0, online: 0 })
  const [cameras, setCameras] = useState<{ id: string; name: string }[]>([])
  const [timeSeries, setTimeSeries] = useState<any[]>([])
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [timeRange, setTimeRange] = useState<TimeRange>(24)
  const [selectedCamera, setSelectedCamera] = useState<string>("")
  const [accuracy, setAccuracy] = useState<{ total: number; fp: number; rate: number } | null>(null)
  const [heatmapData, setHeatmapData] = useState<{ zones: string[]; buckets: string[]; cells: { zone: string; bucket: string; count: number }[]; maxCount: number } | null>(null)
  const [spatialData, setSpatialData] = useState<{ gridSize: number; cells: number[][]; maxCount: number; totalDetections: number } | null>(null)

  const loadData = async (hours: TimeRange = timeRange, camId: string = selectedCamera) => {
    try {
      await fetchAlerts()
      const cameraFilter = camId || undefined
      const [s, cams, ts, comp] = await Promise.all([
        getAlertStats(cameraFilter),
        getCameras(),
        getAlertTimeSeries(hours, cameraFilter),
        getComplianceMetrics(hours, cameraFilter),
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
      setCameras(camList.map((c: any) => ({ id: c.id, name: c.name || c.id })))

      // Compute detection accuracy from false positive data
      try {
        const allAlerts = await getAlerts({ limit: 1000 })
        const total = allAlerts.length
        const fp = allAlerts.filter((a: any) => a.falsePositive).length
        const rate = total > 0 ? Math.round(((total - fp) / total) * 1000) / 10 : 100
        setAccuracy({ total, fp, rate })
      } catch {
        // alerts endpoint may not be available
      }

      // Heatmap data
      try {
        const bucketSize = hours <= 24 ? "hour" : "day"
        const hm = await getZoneTimeHeatmap({ hours, cameraId: cameraFilter, bucket: bucketSize as any })
        setHeatmapData(hm)
      } catch {
        setHeatmapData(null)
      }

      // Spatial heatmap (only when a specific camera is selected)
      if (cameraFilter) {
        try {
          const sp = await getSpatialHeatmap({ cameraId: cameraFilter, hours })
          setSpatialData(sp)
        } catch {
          setSpatialData(null)
        }
      } else {
        setSpatialData(null)
      }

      setLastUpdated(new Date())
    } catch {
      // stats may not be available yet
    }
  }

  useEffect(() => {
    loadData(timeRange, selectedCamera)
    const interval = setInterval(() => loadData(timeRange, selectedCamera), 30000)
    return () => clearInterval(interval)
  }, [timeRange, selectedCamera])

  const kpis = useMemo(() => {
    if (!compliance) return []
    const uptimePct =
      cameraCount.total > 0 ? Math.round((cameraCount.online / cameraCount.total) * 100) : 0
    const offline = cameraCount.total - cameraCount.online
    const rangeLabel = timeRange === 24 ? "24h" : timeRange === 168 ? "7d" : "30d"
    return [
      {
        label: "Safety Compliance",
        value: `${compliance.safety_compliance_pct}%`,
        caption: `last ${rangeLabel} · no P1/P2 violations`,
        light: lightFromThresholds(compliance.safety_compliance_pct, 98, 90),
        trend: <TrendBadge current={compliance.safety_compliance_pct} previous={compliance.prev_safety_compliance_pct} />,
      },
      {
        label: "PPE Compliance",
        value: `${compliance.ppe_compliance_pct}%`,
        caption: `last ${rangeLabel} · helmet + vest`,
        light: lightFromThresholds(compliance.ppe_compliance_pct, 95, 85),
        trend: <TrendBadge current={compliance.ppe_compliance_pct} previous={compliance.prev_ppe_compliance_pct} />,
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
        trend: null,
      },
      {
        label: "Mean Time to Acknowledge",
        value: formatMtta(compliance.mtta_seconds),
        caption:
          compliance.mtta_seconds == null
            ? `no alerts acked in ${rangeLabel}`
            : `avg ack time · last ${rangeLabel}`,
        light:
          compliance.mtta_seconds == null
            ? ("amber" as TrafficLight)
            : compliance.mtta_seconds <= 120
              ? "green"
              : compliance.mtta_seconds <= 600
                ? "amber"
                : "red",
        trend: <MttaTrendBadge current={compliance.mtta_seconds} previous={compliance.prev_mtta_seconds} />,
      },
    ]
  }, [compliance, cameraCount, timeRange])

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

  const violationsByCamera = useMemo(() => {
    if (!stats?.byCamera) return []
    return Object.entries(stats.byCamera)
      .map(([camera, count]) => ({ camera, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 7)
  }, [stats])

  const severityTrend = timeSeries

  const maxZoneCount = violationsByZone[0]?.count || 1
  const maxCameraCount = violationsByCamera[0]?.count || 1

  if (!stats || !compliance) {
    return (
      <div className="space-y-6 p-6">
        {/* Header skeleton */}
        <div className="flex items-center justify-between">
          <Skeleton className="h-7 w-32" />
          <div className="flex items-center gap-3">
            <Skeleton className="h-8 w-28 rounded-[var(--radius-md)]" />
            <Skeleton className="h-8 w-36 rounded-[var(--radius-md)]" />
          </div>
        </div>

        {/* Safety banner skeleton */}
        <Skeleton className="h-14 w-full rounded-[var(--radius-md)]" />

        {/* KPI cards skeleton */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="border-l-4 border-l-[var(--color-bg-tertiary)]">
              <Skeleton className="h-3.5 w-28 mb-2" />
              <Skeleton className="h-8 w-20 mb-2" />
              <Skeleton className="h-3 w-36" />
            </Card>
          ))}
        </div>

        {/* Severity breakdown skeleton */}
        <Card>
          <CardHeader><Skeleton className="h-5 w-44" /></CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-20 rounded-[var(--radius-md)]" />
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Chart skeleton */}
        <Card>
          <CardHeader><Skeleton className="h-5 w-32" /></CardHeader>
          <CardContent><Skeleton className="h-72 w-full" /></CardContent>
        </Card>

        {/* Bottom row skeleton */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {Array.from({ length: 2 }).map((_, i) => (
            <Card key={i}>
              <CardHeader><Skeleton className="h-5 w-32" /></CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {Array.from({ length: 4 }).map((_, j) => (
                    <div key={j} className="flex items-center gap-3">
                      <Skeleton className="h-4 w-32 shrink-0" />
                      <Skeleton className="h-6 flex-1" />
                      <Skeleton className="h-4 w-10" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 p-6">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Dashboard</h1>
        <div className="flex items-center gap-3">
          {/* Camera selector */}
          <select
            value={selectedCamera}
            onChange={(e) => setSelectedCamera(e.target.value)}
            className="text-xs font-medium border rounded-[var(--radius-md)] px-2.5 py-1.5 bg-white text-[var(--color-text-primary)] cursor-pointer outline-none focus:ring-1 focus:ring-[var(--color-info)]"
          >
            <option value="">All Cameras</option>
            {cameras.map((cam) => (
              <option key={cam.id} value={cam.id}>{cam.name}</option>
            ))}
          </select>

          {/* Time range selector */}
          <div className="flex items-center bg-[var(--color-bg-tertiary)] rounded-[var(--radius-md)] p-0.5">
            {TIME_RANGES.map((r) => (
              <button
                key={r.value}
                onClick={() => setTimeRange(r.value)}
                className={`px-3 py-1 text-xs font-medium rounded-[var(--radius-sm)] transition-colors cursor-pointer ${
                  timeRange === r.value
                    ? "bg-white text-[var(--color-text-primary)] shadow-sm"
                    : "text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)]"
                }`}
              >
                {r.label}
              </button>
            ))}
          </div>

          {/* Last updated */}
          {lastUpdated && (
            <span className="text-[10px] text-[var(--color-text-tertiary)]">
              {lastUpdated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </span>
          )}
        </div>
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
              <div className="flex items-baseline gap-2 mb-1">
                <p className={`text-2xl font-bold ${value}`}>{kpi.value}</p>
                {kpi.trend}
              </div>
              <p className="text-xs text-[var(--color-text-tertiary)]">{kpi.caption}</p>
            </Card>
          )
        })}
      </div>

      {/* Severity breakdown — clickable */}
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
                  className="rounded-[var(--radius-md)] p-3 text-center cursor-pointer hover:opacity-80 transition-opacity"
                  style={{ backgroundColor: c.bg }}
                  onClick={() => navigate(`/alerts?severity=${sev}`)}
                  title={`View ${sev} alerts`}
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

      {/* Detection Accuracy */}
      {accuracy && accuracy.total > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Detection Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="text-center p-3 rounded-[var(--radius-md)] bg-[var(--color-bg-secondary)]">
                <p className={`text-3xl font-bold ${
                  accuracy.rate >= 90
                    ? "text-[var(--color-success)]"
                    : accuracy.rate >= 80
                      ? "text-[var(--color-warning)]"
                      : "text-[var(--color-critical)]"
                }`}>
                  {accuracy.rate}%
                </p>
                <p className="text-xs text-[var(--color-text-secondary)] mt-1">Overall Accuracy</p>
                <p className="text-[10px] text-[var(--color-text-tertiary)] mt-0.5">
                  Target: {">"}90% ideal, {">"}80% min
                </p>
              </div>
              <div className="text-center p-3 rounded-[var(--radius-md)] bg-[var(--color-bg-secondary)]">
                <p className="text-3xl font-bold text-[var(--color-text-primary)]">{accuracy.total}</p>
                <p className="text-xs text-[var(--color-text-secondary)] mt-1">Total Detections</p>
              </div>
              <div className="text-center p-3 rounded-[var(--radius-md)] bg-[var(--color-bg-secondary)]">
                <p className="text-3xl font-bold text-[var(--color-warning)]">{accuracy.fp}</p>
                <p className="text-xs text-[var(--color-text-secondary)] mt-1">False Positives</p>
                <p className="text-[10px] text-[var(--color-text-tertiary)] mt-0.5">
                  FP Rate: {accuracy.total > 0 ? ((accuracy.fp / accuracy.total) * 100).toFixed(1) : 0}%
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

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

      {/* Zone × Time Heatmap */}
      {heatmapData && heatmapData.zones.length > 0 && heatmapData.buckets.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Compliance Heatmap</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <div
                className="grid gap-px"
                style={{
                  gridTemplateColumns: `120px repeat(${heatmapData.buckets.length}, minmax(28px, 1fr))`,
                }}
              >
                {/* Header row */}
                <div />
                {heatmapData.buckets.map((b) => {
                  const d = new Date(b)
                  const label = timeRange <= 24
                    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                    : d.toLocaleDateString([], { month: "short", day: "numeric" })
                  return (
                    <div key={b} className="text-[9px] text-[var(--color-text-tertiary)] text-center truncate px-0.5" title={b}>
                      {label}
                    </div>
                  )
                })}

                {/* Data rows */}
                {heatmapData.zones.map((zone) => (
                  <>
                    <div key={`label-${zone}`} className="text-xs font-medium text-[var(--color-text-secondary)] truncate pr-2 flex items-center" title={zone}>
                      {zone}
                    </div>
                    {heatmapData.buckets.map((bucket) => {
                      const cell = heatmapData.cells.find((c) => c.zone === zone && c.bucket === bucket)
                      const count = cell?.count || 0
                      const max = heatmapData.maxCount || 1
                      const ratio = count / max
                      let bg = "var(--color-bg-tertiary)"
                      if (count > 0) {
                        if (ratio < 0.25) bg = "#dcfce7"
                        else if (ratio < 0.5) bg = "#fef9c3"
                        else if (ratio < 0.75) bg = "#fed7aa"
                        else bg = "#fecaca"
                      }
                      const d = new Date(bucket)
                      const timeLabel = timeRange <= 24
                        ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                        : d.toLocaleDateString([], { month: "short", day: "numeric" })
                      return (
                        <div
                          key={`${zone}-${bucket}`}
                          className="h-7 rounded-[2px] transition-colors cursor-default"
                          style={{ backgroundColor: bg }}
                          title={`${zone} · ${timeLabel} · ${count} violation${count !== 1 ? "s" : ""}`}
                        />
                      )
                    })}
                  </>
                ))}
              </div>

              {/* Legend */}
              <div className="flex items-center gap-3 mt-3 text-[10px] text-[var(--color-text-tertiary)]">
                <span>Less</span>
                {["var(--color-bg-tertiary)", "#dcfce7", "#fef9c3", "#fed7aa", "#fecaca"].map((c, i) => (
                  <div key={i} className="w-4 h-4 rounded-[2px]" style={{ backgroundColor: c }} />
                ))}
                <span>More</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Spatial Detection Density (only when camera selected) */}
      {spatialData && spatialData.totalDetections > 0 && selectedCamera && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Detection Density Map</CardTitle>
              <span className="text-xs text-[var(--color-text-tertiary)]">
                {spatialData.totalDetections} detections
              </span>
            </div>
          </CardHeader>
          <CardContent>
            <div className="max-w-md mx-auto">
              <div
                className="grid gap-px aspect-square"
                style={{
                  gridTemplateColumns: `repeat(${spatialData.gridSize}, 1fr)`,
                  gridTemplateRows: `repeat(${spatialData.gridSize}, 1fr)`,
                }}
              >
                {spatialData.cells.flatMap((row, rowIdx) =>
                  row.map((count, colIdx) => {
                    const max = spatialData.maxCount || 1
                    const ratio = count / max
                    let bg = "var(--color-bg-tertiary)"
                    if (count > 0) {
                      if (ratio < 0.25) bg = "#dcfce7"
                      else if (ratio < 0.5) bg = "#fef9c3"
                      else if (ratio < 0.75) bg = "#fed7aa"
                      else bg = "#fecaca"
                    }
                    return (
                      <div
                        key={`${rowIdx}-${colIdx}`}
                        className="rounded-[2px] transition-colors"
                        style={{ backgroundColor: bg }}
                        title={`Row ${rowIdx + 1}, Col ${colIdx + 1}: ${count} detection${count !== 1 ? "s" : ""}`}
                      />
                    )
                  })
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Bottom row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Alerts by Zone */}
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
                    <span className="text-sm font-semibold text-[var(--color-text-primary)] w-12 text-right">
                      {item.count}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Top Cameras */}
        <Card>
          <CardHeader>
            <CardTitle>Top Cameras</CardTitle>
          </CardHeader>
          <CardContent>
            {violationsByCamera.length === 0 ? (
              <p className="text-sm text-[var(--color-text-tertiary)] py-4 text-center">
                No alerts yet.
              </p>
            ) : (
              <div className="space-y-3">
                {violationsByCamera.map((item) => (
                  <div key={item.camera} className="flex items-center gap-3">
                    <span className="text-sm font-medium text-[var(--color-text-primary)] w-40 shrink-0 truncate">
                      {item.camera}
                    </span>
                    <div className="flex-1 h-6 bg-[var(--color-bg-tertiary)] rounded-[var(--radius-sm)] overflow-hidden">
                      <div
                        className="h-full bg-[var(--color-warning)] rounded-[var(--radius-sm)] transition-all"
                        style={{
                          width: `${(item.count / maxCameraCount) * 100}%`,
                          opacity: 0.3 + (item.count / maxCameraCount) * 0.7,
                        }}
                      />
                    </div>
                    <span className="text-sm font-semibold text-[var(--color-text-primary)] w-12 text-right">
                      {item.count}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
