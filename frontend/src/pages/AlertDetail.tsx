import { useState, useEffect } from "react"
import { useParams, useNavigate, Link } from "react-router-dom"
import { ArrowLeft, Clock, Copy, Check } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { SeverityBadge } from "@/components/ui/SeverityBadge"
import { StatusBadge } from "@/components/ui/StatusBadge"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { severityConfig } from "@/lib/constants"
import { getAlerts, API_BASE } from "@/lib/api"
import type { Alert } from "@/types"
import { timeAgo, formatTime } from "@/lib/utils"

export function AlertDetail() {
  const { alertId } = useParams<{ alertId: string }>()
  const navigate = useNavigate()
  const [alert, setAlert] = useState<Alert | null>(null)
  const [loading, setLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!alertId) return
    setLoading(true)
    getAlerts({ limit: 1000 })
      .then((alerts) => {
        const found = alerts.find((a: Alert) => a.id === alertId)
        if (found) {
          setAlert(found)
        } else {
          setNotFound(true)
        }
      })
      .catch(() => setNotFound(true))
      .finally(() => setLoading(false))
  }, [alertId])

  function copyLink() {
    navigator.clipboard.writeText(window.location.href)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (loading) {
    return (
      <div className="max-w-3xl mx-auto p-6 space-y-6">
        <Skeleton className="h-7 w-48" />
        <Skeleton className="h-80 w-full rounded-[var(--radius-md)]" />
        <div className="grid grid-cols-2 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-16 rounded-[var(--radius-md)]" />
          ))}
        </div>
      </div>
    )
  }

  if (notFound || !alert) {
    return (
      <div className="max-w-3xl mx-auto p-6 text-center space-y-4">
        <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Alert Not Found</h1>
        <p className="text-sm text-[var(--color-text-secondary)]">
          The alert with ID &ldquo;{alertId}&rdquo; could not be found. It may have been archived.
        </p>
        <Button variant="secondary" onClick={() => navigate("/alerts")}>
          <ArrowLeft size={14} /> Back to Alert Center
        </Button>
      </div>
    )
  }

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={() => navigate(-1)} className="gap-1">
            <ArrowLeft size={14} /> Back
          </Button>
          <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Alert Detail</h1>
        </div>
        <Button variant="secondary" size="sm" onClick={copyLink} className="gap-1.5 text-xs">
          {copied ? <Check size={14} /> : <Copy size={14} />}
          {copied ? "Copied" : "Copy Link"}
        </Button>
      </div>

      {/* Snapshot */}
      <Card className="overflow-hidden p-0">
        {alert.snapshotUrl ? (
          <img
            src={`${API_BASE}${alert.snapshotUrl}`}
            alt="Violation snapshot"
            className="w-full"
            style={{ aspectRatio: "16/9", objectFit: "cover" }}
          />
        ) : (
          <div
            className="w-full bg-[var(--color-bg-tertiary)] flex items-center justify-center text-[var(--color-text-tertiary)] text-sm"
            style={{ aspectRatio: "16/9" }}
          >
            No snapshot available
          </div>
        )}
      </Card>

      {/* Rule + Severity header */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">{alert.rule}</h2>
          <SeverityBadge severity={alert.severity} />
          <StatusBadge status={alert.status} />
          {alert.falsePositive && (
            <Badge variant="default" className="text-[10px]">False positive</Badge>
          )}
        </div>
        {alert.description && (
          <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed bg-[var(--color-bg-secondary)] rounded-[var(--radius-md)] p-3">
            {alert.description}
          </p>
        )}
      </div>

      {/* Details grid */}
      <Card>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-5">
          <DetailField label="Camera" value={alert.cameraName} />
          <DetailField label="Zone" value={alert.zone} />
          <DetailField label="Confidence" value={`${Math.round(alert.confidence * 100)}%`} />
          <DetailField label="Source" value={alert.source} />
          <DetailField label="Detected At" value={new Date(alert.timestamp).toLocaleString("en-IN")} />
          <DetailField label="Alert ID" value={alert.id} mono />
        </div>
      </Card>

      {/* Timeline */}
      <Card>
        <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3 uppercase tracking-wider">
          Timeline
        </h3>
        <div className="space-y-0">
          <TimelineItem color="bg-red-500" title="Detection triggered">
            {formatTime(alert.timestamp)} ({timeAgo(alert.timestamp)})
          </TimelineItem>

          {alert.acknowledgedBy && alert.acknowledgedAt && (
            <TimelineItem color="bg-blue-500" title={`Acknowledged by ${alert.acknowledgedBy}`}>
              {formatTime(alert.acknowledgedAt)} ({timeAgo(alert.acknowledgedAt)})
            </TimelineItem>
          )}

          {alert.resolvedAt && (
            <TimelineItem color="bg-emerald-500" title={alert.falsePositive ? "Marked as false positive" : "Resolved"}>
              {formatTime(alert.resolvedAt)} ({timeAgo(alert.resolvedAt)})
            </TimelineItem>
          )}

          {alert.status === "snoozed" && alert.snoozedUntil && (
            <TimelineItem color="bg-amber-500" title="Snoozed">
              Until {formatTime(alert.snoozedUntil)}
            </TimelineItem>
          )}

          {alert.status === "active" && (
            <TimelineItem color="bg-red-500 animate-pulse" title="Awaiting response" />
          )}
        </div>
      </Card>

      {alert.deliveryResults && alert.deliveryResults.length > 0 && (
        <Card>
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3 uppercase tracking-wider">
            Delivery
          </h3>
          <div className="space-y-2">
            {alert.deliveryResults.map((result) => (
              <div key={`${result.outputId}-${result.timestamp}`} className="flex items-center justify-between gap-3 text-sm">
                <div>
                  <p className="font-medium text-[var(--color-text-primary)]">{result.outputName}</p>
                  <p className="text-xs text-[var(--color-text-secondary)]">{result.message}</p>
                </div>
                <Badge variant={result.status === "failed" ? "critical" : result.status === "simulated" ? "info" : "success"} className="capitalize">
                  {result.status}
                </Badge>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Bounding boxes info */}
      {alert.bboxes && alert.bboxes.length > 0 && (
        <Card>
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3 uppercase tracking-wider">
            Detections
          </h3>
          <div className="space-y-2">
            {alert.bboxes.map((b, i) => (
              <div key={i} className="flex items-center justify-between text-sm">
                <span className="font-medium text-[var(--color-text-primary)]">{b.label}</span>
                <span className="text-xs text-[var(--color-text-secondary)] tabular-nums">
                  {Math.round(b.confidence * 100)}% confidence
                </span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

function DetailField({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <span className="text-[11px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">
        {label}
      </span>
      <p className={`text-sm text-[var(--color-text-primary)] mt-0.5 ${mono ? "font-mono" : ""}`}>
        {value}
      </p>
    </div>
  )
}

function TimelineItem({ color, title, children }: { color: string; title: string; children?: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <div className="flex flex-col items-center">
        <div className={`w-2 h-2 rounded-full ${color} mt-1.5`} />
        <div className="w-px flex-1 bg-[var(--color-border-default)]" />
      </div>
      <div className="pb-3">
        <p className="text-xs font-medium text-[var(--color-text-primary)]">{title}</p>
        {children && (
          <p className="text-[11px] text-[var(--color-text-secondary)]">{children}</p>
        )}
      </div>
    </div>
  )
}
