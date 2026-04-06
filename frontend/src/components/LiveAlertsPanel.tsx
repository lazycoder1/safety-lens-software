import { useState, useEffect, useCallback } from "react"
import { CheckCircle2, Brain, PanelRightOpen, PanelRightClose } from "lucide-react"
import { severityConfig } from "@/data/mock"
import type { Severity } from "@/data/mock"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"
import { useAlertStore } from "@/stores/alertStore"
import { useAuthStore } from "@/stores/authStore"
import { useViolationModal } from "@/components/ViolationModal"
import { playP1AlertSound } from "@/lib/alertSound"

const severityVariantMap: Record<Severity, "critical" | "high" | "warning" | "info"> = {
  P1: "critical",
  P2: "high",
  P3: "warning",
  P4: "info",
}

function timeAgo(isoString: string): string {
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(isoString).getTime()) / 1000))
  if (seconds < 5) return "just now"
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  return `${Math.floor(minutes / 60)}h ago`
}

export function LiveAlertsPanel() {
  const [panelOpen, setPanelOpen] = useState(true)
  const alerts = useAlertStore((s) => s.alerts)
  const acknowledge = useAlertStore((s) => s.acknowledge)
  const openModal = useViolationModal((s) => s.open)
  const userRole = useAuthStore((s) => s.user?.role)
  const canAct = userRole === "admin" || userRole === "operator"
  const [, setTick] = useState(0)

  // Force re-render every second for timeAgo updates
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 1000)
    return () => clearInterval(id)
  }, [])

  // Play sound on new P1 alerts
  const [lastAlertId, setLastAlertId] = useState<string | null>(null)
  useEffect(() => {
    if (alerts.length > 0 && alerts[0].id !== lastAlertId) {
      setLastAlertId(alerts[0].id)
      if (alerts[0].severity === "P1" && alerts[0].status === "active") {
        playP1AlertSound()
      }
    }
  }, [alerts, lastAlertId])

  const handleAcknowledge = useCallback(
    (alertId: string) => {
      acknowledge(alertId)
      toast.success("Alert acknowledged")
    },
    [acknowledge],
  )

  const liveAlerts = alerts.slice(0, 100)
  const activeAlerts = liveAlerts.filter((a) => a.status === "active")

  if (!panelOpen) {
    return (
      <div className="border-l bg-white flex flex-col items-center py-3 px-1 shrink-0">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setPanelOpen(true)}
          title="Show alerts panel"
          className="mb-2"
        >
          <PanelRightOpen size={16} />
        </Button>
        {activeAlerts.length > 0 && (
          <span className="inline-flex items-center justify-center min-w-[20px] h-5 rounded-full bg-red-600 text-white text-[10px] font-bold px-1.5 animate-pulse">
            {activeAlerts.length}
          </span>
        )}
      </div>
    )
  }

  return (
    <div className="w-80 border-l bg-white flex flex-col shrink-0">
      <div className="flex items-center justify-between px-4 border-b bg-white h-[41px] shrink-0">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">Live Alerts</h2>
          {activeAlerts.length > 0 && (
            <span className="inline-flex items-center justify-center min-w-[20px] h-5 rounded-full bg-red-600 text-white text-[10px] font-bold px-1.5">
              {activeAlerts.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-[var(--color-text-tertiary)]">
            {liveAlerts.length} total
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPanelOpen(false)}
            title="Hide alerts panel"
          >
            <PanelRightClose size={16} />
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {activeAlerts.length === 0 && liveAlerts.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-[var(--color-text-secondary)]">
            <CheckCircle2 size={32} className="mb-2 opacity-40" />
            <span className="text-sm">No alerts yet</span>
            <span className="text-xs text-[var(--color-text-tertiary)] mt-1">Waiting for detections...</span>
          </div>
        )}

        {liveAlerts.map((alert) => {
          const sev = severityConfig[alert.severity as Severity]
          const isActive = alert.status === "active"
          return (
            <div
              key={alert.id}
              className={cn(
                "border-b last:border-b-0 transition-colors cursor-pointer",
                isActive ? "bg-white hover:bg-[var(--color-bg-secondary)]" : "bg-[var(--color-bg-secondary)] opacity-60"
              )}
              onClick={() => openModal(alert)}
            >
              <div className="flex gap-0">
                <div className="w-1 shrink-0" style={{ backgroundColor: sev?.color || "#999" }} />
                <div className="flex-1 px-3 py-2.5">
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <div className="min-w-0">
                      <p className="text-xs font-semibold text-[var(--color-text-primary)] truncate">
                        {alert.rule}
                      </p>
                      <p className="text-[11px] text-[var(--color-text-secondary)] truncate">
                        {alert.cameraName}
                      </p>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      {alert.source.includes("VLM") && (
                        <Brain size={12} className="text-purple-500" />
                      )}
                      <Badge variant={severityVariantMap[alert.severity as Severity] || "default"}>
                        {alert.severity}
                      </Badge>
                    </div>
                  </div>

                  {alert.description && (
                    <p className="text-[10px] text-[var(--color-text-tertiary)] line-clamp-2 mb-1.5">
                      {alert.description}
                    </p>
                  )}

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-[10px] text-[var(--color-text-tertiary)]">
                      <span>{timeAgo(alert.timestamp)}</span>
                      <span>|</span>
                      <span>{alert.source}</span>
                      <span>|</span>
                      <span>{Math.round(alert.confidence * 100)}%</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-[10px] text-blue-500">Click to view</span>
                      {isActive && canAct && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-[10px] h-5 px-1.5"
                          onClick={(e) => { e.stopPropagation(); handleAcknowledge(alert.id) }}
                        >
                          Ack
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
