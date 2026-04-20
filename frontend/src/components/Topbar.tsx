import { useEffect, useState, useRef } from "react"
import { User, LogOut, KeyRound, ChevronDown } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { useAuthStore } from "@/stores/authStore"
import { Badge } from "@/components/ui/badge"

const roleBadgeVariant: Record<string, "default" | "info" | "warning"> = {
  admin: "warning",
  operator: "info",
  viewer: "default",
}

export function Topbar() {
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()
  const { user, logout } = useAuthStore()

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClick)
    return () => document.removeEventListener("mousedown", handleClick)
  }, [])

  const handleLogout = () => {
    logout()
    navigate("/login")
  }

  return (
    <header className="flex items-center justify-between h-14 px-6 border-b bg-white shrink-0">
      <div />

      <div className="flex items-center gap-3">
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setMenuOpen(!menuOpen)}
            className="flex items-center gap-2 pl-2 pr-1 py-1 rounded-[var(--radius-md)] hover:bg-[var(--color-bg-tertiary)] transition-colors cursor-pointer"
          >
            <div className="w-7 h-7 rounded-full bg-[var(--color-info-bg)] flex items-center justify-center">
              <User className="w-3.5 h-3.5 text-[var(--color-info)]" />
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-sm font-medium">{user?.username || "User"}</span>
              <Badge variant={roleBadgeVariant[user?.role || "viewer"] || "default"} className="text-[10px]">
                {user?.role || "viewer"}
              </Badge>
            </div>
            <ChevronDown className="w-3.5 h-3.5 text-[var(--color-text-tertiary)]" />
          </button>

          {menuOpen && (
            <div className="absolute right-0 top-full mt-1 w-48 bg-white border rounded-lg shadow-lg py-1 z-50">
              <button
                onClick={() => { setMenuOpen(false); navigate("/change-password") }}
                className="flex items-center gap-2 w-full px-3 py-2 text-sm text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-secondary)] transition-colors cursor-pointer"
              >
                <KeyRound size={14} />
                Change Password
              </button>
              <div className="border-t my-1" />
              <button
                onClick={() => { setMenuOpen(false); handleLogout() }}
                className="flex items-center gap-2 w-full px-3 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors cursor-pointer"
              >
                <LogOut size={14} />
                Logout
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}
