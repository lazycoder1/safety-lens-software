import type { SafetyRule } from "@/types"

export function inferDetectionMode(
  selectedRuleIds: string[],
  allRules: SafetyRule[]
): { mode: string; reason: string } {
  const selected = allRules.filter(r => selectedRuleIds.includes(r.id) && r.enabled)
  const hasPPE = selected.some(r => r.type === "ppe")
  const hasAlert = selected.some(r => r.type === "alert")
  if (hasPPE) return { mode: "yoloe", reason: "Required by PPE rules (open-vocabulary detection)" }
  if (hasAlert) return { mode: "yolo", reason: "Standard object detection for alert rules" }
  return { mode: "yolo", reason: "Default detection mode" }
}

export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium text-[var(--color-text-primary)]">{label}</label>
      {children}
    </div>
  )
}

export function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs font-medium text-[var(--color-text-secondary)]">{label}</span>
      <span className="text-sm text-[var(--color-text-primary)]">{value}</span>
    </div>
  )
}
