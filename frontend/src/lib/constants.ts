import type { Severity, AlertStatus } from "@/types"

export const severityConfig: Record<Severity, { label: string; color: string; bg: string; textColor: string }> = {
  P1: { label: "Critical", color: "#dc2626", bg: "#fef2f2", textColor: "#991b1b" },
  P2: { label: "High", color: "#f97316", bg: "#fff7ed", textColor: "#9a3412" },
  P3: { label: "Medium", color: "#f59e0b", bg: "#fffbeb", textColor: "#92400e" },
  P4: { label: "Low", color: "#2563eb", bg: "#eff6ff", textColor: "#1e40af" },
}

export const severityVariantMap: Record<Severity, "critical" | "high" | "warning" | "info"> = {
  P1: "critical",
  P2: "high",
  P3: "warning",
  P4: "info",
}

export const statusVariantMap: Record<AlertStatus, "critical" | "high" | "warning" | "success" | "info" | "default"> = {
  active: "critical",
  acknowledged: "info",
  resolved: "success",
  snoozed: "warning",
}

export const severityOrder: Record<Severity, number> = { P1: 0, P2: 1, P3: 2, P4: 3 }
