import { useState, useMemo, useEffect } from "react"
import { NavLink } from "react-router-dom"
import { create } from "zustand"
import { cn } from "@/lib/utils"
import { useAuthStore } from "@/stores/authStore"
import { getUsers } from "@/lib/api"

interface PendingCountStore {
  count: number
  refresh: () => Promise<void>
}

export const usePendingCount = create<PendingCountStore>((set) => ({
  count: 0,
  refresh: async () => {
    try {
      const data = await getUsers()
      const users = data.users ?? data
      set({ count: users.filter((u: any) => u.status === "pending").length })
    } catch {
      // ignore
    }
  },
}))
import {
  Monitor,
  AlertTriangle,
  LayoutDashboard,
  FileText,
  Camera,
  Scan,
  Map,
  Bell,
  Link2,
  ChevronLeft,
  ChevronRight,
  Shield,
  Settings,
  Search,
  Car,
  Users,
  Key,
} from "lucide-react"

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const user = useAuthStore((s) => s.user)
  const pendingCount = usePendingCount((s) => s.count)
  const refreshPending = usePendingCount((s) => s.refresh)

  useEffect(() => {
    if (user?.role !== "admin") return
    refreshPending()
    const interval = setInterval(refreshPending, 30000)
    return () => clearInterval(interval)
  }, [user?.role, refreshPending])

  const navGroups = useMemo(() => {
    const role = user?.role
    const groups = [
      {
        label: "Monitoring",
        items: [
          { to: "/live", icon: Monitor, label: "Live View" },
          { to: "/alerts", icon: AlertTriangle, label: "Alert Center" },
        ],
      },
      {
        label: "Analytics",
        items: [
          { to: "/dashboard", icon: LayoutDashboard, label: "Dashboard" },
          { to: "/search", icon: Search, label: "AI Search" },
        ],
      },
    ]

    // Configuration — admin only
    if (role === "admin") {
      groups.push({
        label: "Configuration",
        items: [
          { to: "/configure/cameras", icon: Camera, label: "Cameras" },
          { to: "/configure/rules", icon: Scan, label: "Rules Engine" },
          { to: "/configure/alerts", icon: Bell, label: "Alert Routing" },
          { to: "/configure/plates", icon: Car, label: "Vehicle Plates" },
          { to: "/configure/faces", icon: Users, label: "Faces" },
        ],
      })
    }

    // System — admin and operator (User Management admin only)
    if (role === "admin" || role === "operator") {
      const systemItems = [
        { to: "/system/settings", icon: Settings, label: "Settings" },
        { to: "/system/license", icon: Key, label: "License" },
      ]
      if (role === "admin") {
        systemItems.push({ to: "/system/users", icon: Shield, label: "User Management", badge: pendingCount })
      }
      groups.push({ label: "System", items: systemItems })
    }

    return groups
  }, [user?.role, pendingCount])

  return (
    <aside
      className={cn(
        "flex flex-col border-r bg-[var(--color-bg-secondary)] transition-all duration-200 shrink-0",
        collapsed ? "w-[56px]" : "w-[240px]"
      )}
    >
      <div className={cn("flex items-center gap-2.5 px-4 h-14 border-b", collapsed && "justify-center px-0")}>
        <div className="w-7 h-7 rounded-[var(--radius-md)] bg-[var(--color-text-primary)] flex items-center justify-center shrink-0">
          <Shield className="w-4 h-4 text-white" />
        </div>
        {!collapsed && <span className="font-semibold text-sm tracking-tight">SafetyLens</span>}
      </div>

      <nav className="flex-1 overflow-y-auto py-3">
        {navGroups.map((group) => (
          <div key={group.label} className="mb-4">
            {!collapsed && (
              <div className="px-4 mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-tertiary)]">
                {group.label}
              </div>
            )}
            {group.items.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-2.5 px-4 py-1.5 mx-2 rounded-[var(--radius-md)] text-sm transition-colors",
                    collapsed && "justify-center px-0 mx-1",
                    isActive
                      ? "bg-[var(--color-bg-tertiary)] text-[var(--color-text-primary)] font-medium"
                      : "text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-tertiary)]"
                  )
                }
                title={collapsed ? item.label : undefined}
              >
                <item.icon className="w-4 h-4 shrink-0" />
                {!collapsed && <span className="flex-1">{item.label}</span>}
                {!collapsed && "badge" in item && (item as any).badge > 0 && (
                  <span className="inline-flex items-center justify-center min-w-[18px] h-[18px] rounded-full bg-amber-500 text-white text-[10px] font-bold px-1">
                    {(item as any).badge}
                  </span>
                )}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center justify-center h-10 border-t text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] transition-colors cursor-pointer"
      >
        {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </button>
    </aside>
  )
}
