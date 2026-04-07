import { Badge } from "@/components/ui/badge"
import { statusVariantMap } from "@/lib/constants"
import type { AlertStatus } from "@/types"

interface StatusBadgeProps {
  status: AlertStatus
  className?: string
  children?: React.ReactNode
}

export function StatusBadge({ status, className, children }: StatusBadgeProps) {
  return (
    <Badge variant={statusVariantMap[status]} className={`capitalize ${className || ""}`}>
      {children || status}
    </Badge>
  )
}
