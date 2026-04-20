import { Badge } from "@/components/ui/badge"
import type { SafetyRule } from "@/types"
import {
  getDetectionCatalog,
  getDetectionOptions,
  type CameraDetectionKey,
} from "./detectionCatalog"

interface DetectionChecklistProps {
  safetyRules: SafetyRule[]
  selectedKeys: CameraDetectionKey[]
  onChange: (keys: CameraDetectionKey[]) => void
}

const GROUP_ORDER = ["Core Monitoring", "Safety Events", "PPE", "Advanced"] as const

export function DetectionChecklist({
  safetyRules,
  selectedKeys,
  onChange,
}: DetectionChecklistProps) {
  const selected = new Set(selectedKeys)
  const options = getDetectionOptions(safetyRules)
  const availableKeys = options.filter((option) => option.available).map((option) => option.key)

  function toggleKey(key: CameraDetectionKey) {
    if (!availableKeys.includes(key)) return
    const next = selected.has(key)
      ? selectedKeys.filter((value) => value !== key)
      : [...selectedKeys, key]
    onChange(next)
  }

  function selectAll() {
    onChange(availableKeys)
  }

  function clearAll() {
    onChange([])
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-[var(--color-text-primary)]">
            Choose one, many, or all detections for this camera.
          </p>
          <p className="text-xs text-[var(--color-text-secondary)] mt-1">
            These options map directly to the capability planner and the installed model stack.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={selectAll}
            className="text-xs text-[var(--color-info)] hover:underline cursor-pointer"
          >
            Select all
          </button>
          <button
            type="button"
            onClick={clearAll}
            className="text-xs text-[var(--color-text-secondary)] hover:underline cursor-pointer"
          >
            Clear
          </button>
        </div>
      </div>

      {GROUP_ORDER.map((group) => {
        const groupOptions = options.filter((option) => option.group === group)
        if (groupOptions.length === 0) return null

        return (
          <div key={group} className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">
              {group}
            </p>
            <div className="grid gap-2 md:grid-cols-2">
              {groupOptions.map((option) => {
                const checked = selected.has(option.key)
                return (
                  <label
                    key={option.key}
                    className={`rounded-[var(--radius-lg)] border p-3 transition-colors ${
                      option.available
                        ? checked
                          ? "border-[var(--color-info)] bg-[var(--color-info-bg)] cursor-pointer"
                          : "border-[var(--color-border-default)] hover:border-[var(--color-border-active)] cursor-pointer"
                        : "border-[var(--color-border-default)] bg-[var(--color-bg-secondary)] opacity-60 cursor-not-allowed"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={!option.available}
                        onChange={() => toggleKey(option.key)}
                        className="mt-0.5 accent-[var(--color-info)]"
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 flex-wrap">
                          <p className="text-sm font-semibold text-[var(--color-text-primary)]">
                            {option.label}
                          </p>
                          {!option.available && (
                            <Badge variant="warning">Unavailable</Badge>
                          )}
                        </div>
                        <p className="mt-1 text-xs leading-5 text-[var(--color-text-secondary)]">
                          {option.description}
                        </p>
                      </div>
                    </div>
                  </label>
                )
              })}
            </div>
          </div>
        )
      })}

      {getDetectionCatalog().length === 0 && (
        <p className="text-sm text-[var(--color-text-tertiary)]">
          No detections available.
        </p>
      )}
    </div>
  )
}
