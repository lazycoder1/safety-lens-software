interface EmptyStateProps {
  icon?: React.ReactNode
  message: string
  action?: React.ReactNode
  className?: string
}

export function EmptyState({ icon, message, action, className }: EmptyStateProps) {
  return (
    <div className={`flex flex-col items-center justify-center py-12 text-center ${className || ""}`}>
      {icon && <div className="mb-3 text-[var(--color-text-tertiary)]">{icon}</div>}
      <p className="text-sm text-[var(--color-text-tertiary)]">{message}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}
