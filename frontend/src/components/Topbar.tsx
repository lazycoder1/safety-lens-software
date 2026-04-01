import { useEffect, useState } from "react"
import { Bell, User } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { getAlerts } from "@/lib/api"

export function Topbar() {
  const [activeCount, setActiveCount] = useState(0)
  const navigate = useNavigate()

  useEffect(() => {
    // Fetch only today's active alerts for the badge count
    const load = () => {
      getAlerts({ status: "active", limit: 100 })
        .then((alerts: any[]) => setActiveCount(alerts.length))
        .catch(() => {})
    }
    load()
    const interval = setInterval(load, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <header className="flex items-center justify-between h-14 px-6 border-b bg-white shrink-0">
      <div className="flex items-center gap-4">
        <span className="text-sm font-semibold text-[var(--color-text-primary)] tracking-tight">SafetyLens</span>
        <span className="text-xs text-[var(--color-text-tertiary)]">Industrial Safety Monitoring</span>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={() => navigate("/alerts")}
          className="relative p-2 rounded-[var(--radius-md)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] transition-colors cursor-pointer"
        >
          <Bell className="w-4.5 h-4.5" />
          {activeCount > 0 && (
            <span className="absolute -top-0.5 -right-0.5 min-w-[18px] h-[18px] flex items-center justify-center rounded-full bg-[var(--color-critical)] text-white text-[10px] font-semibold px-1">
              {activeCount > 99 ? "99+" : activeCount}
            </span>
          )}
        </button>

        <div className="w-px h-6 bg-[var(--color-border-default)]" />

        <div className="flex items-center gap-2 pl-2 cursor-pointer">
          <div className="w-7 h-7 rounded-full bg-[var(--color-info-bg)] flex items-center justify-center">
            <User className="w-3.5 h-3.5 text-[var(--color-info)]" />
          </div>
          <div className="text-sm">
            <span className="font-medium">Admin</span>
          </div>
        </div>
      </div>
    </header>
  )
}
