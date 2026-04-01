import { Construction } from "lucide-react"

export function Placeholder({ title }: { title: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-3 text-[var(--color-text-tertiary)]">
      <Construction className="w-10 h-10" />
      <h1 className="text-lg font-semibold text-[var(--color-text-primary)]">{title}</h1>
      <p className="text-sm">This page is under construction.</p>
    </div>
  )
}
