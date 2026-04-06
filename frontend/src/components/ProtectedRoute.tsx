import { type ReactNode, useEffect } from "react"
import { Navigate } from "react-router-dom"
import { useAuthStore } from "@/stores/authStore"
import { Loader2 } from "lucide-react"

interface Props {
  children: ReactNode
  requiredRole?: string[]
}

export function ProtectedRoute({ children, requiredRole }: Props) {
  const { user, loading, token, checkAuth } = useAuthStore()

  useEffect(() => {
    if (token && !user && loading) {
      checkAuth()
    }
  }, [token, user, loading, checkAuth])

  if (loading && token) {
    return (
      <div className="flex items-center justify-center h-screen bg-neutral-100">
        <Loader2 className="h-8 w-8 animate-spin text-neutral-400" />
      </div>
    )
  }

  if (!token || !user) {
    return <Navigate to="/login" replace />
  }

  if (user.mustChangePassword) {
    return <Navigate to="/change-password" replace />
  }

  if (requiredRole && !requiredRole.includes(user.role)) {
    return <Navigate to="/live" replace />
  }

  return <>{children}</>
}
