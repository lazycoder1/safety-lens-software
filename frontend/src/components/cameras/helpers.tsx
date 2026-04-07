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
