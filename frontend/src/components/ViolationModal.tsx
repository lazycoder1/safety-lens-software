import { useEffect, useCallback } from "react"
import { create } from "zustand"
import { X, ShieldAlert } from "lucide-react"
import { severityConfig } from "@/lib/constants"
import { SeverityBadge } from "@/components/ui/SeverityBadge"
import type { Alert, Severity } from "@/types"
import { API_BASE } from "@/lib/api"

interface ViolationModalStore {
  alert: Alert | null
  open: (alert: Alert) => void
  close: () => void
}

export const useViolationModal = create<ViolationModalStore>((set) => ({
  alert: null,
  open: (alert) => set({ alert }),
  close: () => set({ alert: null }),
}))

export function ViolationModal() {
  const { alert, close } = useViolationModal()

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") close()
    },
    [close],
  )

  useEffect(() => {
    if (alert) {
      document.addEventListener("keydown", handleKeyDown)
      return () => document.removeEventListener("keydown", handleKeyDown)
    }
  }, [alert, handleKeyDown])

  if (!alert) return null

  const sev = severityConfig[alert.severity as Severity]
  const imageUrl = alert.cleanSnapshotUrl
    ? `${API_BASE}${alert.cleanSnapshotUrl}`
    : alert.snapshotUrl
      ? `${API_BASE}${alert.snapshotUrl}`
      : null
  const bboxes = alert.bboxes || []
  const ts = new Date(alert.timestamp)
  const timeStr = ts.toLocaleString()

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={close}
    >
      <div
        className="bg-white rounded-xl shadow-2xl max-w-3xl w-full mx-4 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-neutral-200">
          <div className="flex items-center gap-3">
            <ShieldAlert size={20} style={{ color: sev?.color || "#666" }} />
            <span className="font-semibold text-[var(--color-text-primary)]">{alert.rule}</span>
            <SeverityBadge severity={alert.severity as Severity} />
          </div>
          <button
            onClick={close}
            className="p-1.5 rounded-lg hover:bg-neutral-100 text-neutral-400 hover:text-neutral-600 transition-colors cursor-pointer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Image with bbox overlays */}
        <div className="px-5 py-4">
          {imageUrl ? (
            <div className="relative w-full">
              <img
                src={imageUrl}
                alt={`Violation: ${alert.rule}`}
                className="w-full rounded-lg"
                draggable={false}
              />
              {bboxes.map((b, i) => {
                const [x1, y1, x2, y2] = b.bbox
                return (
                  <div key={i}>
                    {/* Bounding box */}
                    <div
                      className="absolute pointer-events-none"
                      style={{
                        left: `${x1 * 100}%`,
                        top: `${y1 * 100}%`,
                        width: `${(x2 - x1) * 100}%`,
                        height: `${(y2 - y1) * 100}%`,
                        border: "2.5px solid #dc2626",
                        borderRadius: "3px",
                      }}
                    />
                    {/* Label */}
                    <div
                      className="absolute pointer-events-none flex items-center"
                      style={{
                        left: `${x1 * 100}%`,
                        top: `${y1 * 100}%`,
                        transform: "translateY(-100%)",
                      }}
                    >
                      <span
                        className="text-[11px] font-semibold text-white px-1.5 py-0.5 rounded-t-sm whitespace-nowrap"
                        style={{ backgroundColor: "#dc2626" }}
                      >
                        {b.label} {Math.round(b.confidence * 100)}%
                      </span>
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="flex items-center justify-center h-48 bg-neutral-100 rounded-lg text-neutral-400">
              No snapshot available
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-neutral-200 bg-neutral-50 flex flex-wrap items-center gap-x-5 gap-y-1 text-xs text-neutral-500">
          <span><strong>Camera:</strong> {alert.cameraName}</span>
          <span><strong>Zone:</strong> {alert.zone}</span>
          <span><strong>Confidence:</strong> {Math.round(alert.confidence * 100)}%</span>
          <span><strong>Source:</strong> {alert.source}</span>
          <span><strong>Time:</strong> {timeStr}</span>
          {alert.description && (
            <span className="basis-full text-neutral-600 mt-1">{alert.description}</span>
          )}
        </div>
      </div>
    </div>
  )
}
