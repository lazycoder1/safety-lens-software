import { cn } from "@/lib/utils"

interface BadgeProps {
  children: React.ReactNode
  variant?: "default" | "critical" | "high" | "warning" | "success" | "info"
  className?: string
}

const variantStyles: Record<string, string> = {
  default: "bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]",
  critical: "bg-[var(--color-critical-bg)] text-[#991b1b]",
  high: "bg-[var(--color-high-bg)] text-[#9a3412]",
  warning: "bg-[var(--color-warning-bg)] text-[#92400e]",
  success: "bg-[var(--color-success-bg)] text-[#065f46]",
  info: "bg-[var(--color-info-bg)] text-[#1e40af]",
}

export function Badge({ children, variant = "default", className }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium",
        variantStyles[variant],
        className
      )}
    >
      {children}
    </span>
  )
}
