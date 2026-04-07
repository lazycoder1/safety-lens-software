import { Badge } from "@/components/ui/badge"
import { severityConfig, severityVariantMap } from "@/lib/constants"
import type { Severity } from "@/types"

interface SeverityBadgeProps {
  severity: Severity
  className?: string
}

export function SeverityBadge({ severity, className }: SeverityBadgeProps) {
  const config = severityConfig[severity]
  return (
    <Badge variant={severityVariantMap[severity]} className={className}>
      {severity} {config.label}
    </Badge>
  )
}
