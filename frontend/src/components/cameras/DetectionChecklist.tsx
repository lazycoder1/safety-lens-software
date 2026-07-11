import { Badge } from "@/components/ui/badge"
import { Check } from "lucide-react"
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
  allowCustomDetection?: boolean
}

const GROUP_ORDER = ["Core Monitoring", "Safety Events", "PPE", "Advanced"] as const

export function DetectionChecklist({
  safetyRules,
  selectedKeys,
  onChange,
  allowCustomDetection = false,
}: DetectionChecklistProps) {
  const selected = new Set(selectedKeys)
  const options = getDetectionOptions(safetyRules).filter(
    (option) => allowCustomDetection || option.key !== "custom_long_tail"
  )
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
            The camera runs only the detections selected here.
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
                  <button
                    type="button"
                    key={option.key}
                    onClick={() => toggleKey(option.key)}
                    disabled={!option.available}
                    aria-pressed={checked}
                    aria-label={`${checked ? "Disable" : "Enable"} ${option.label}`}
                    className={`rounded-[var(--radius-lg)] border p-3 transition-colors ${
                      option.available
                        ? checked
                          ? "border-[var(--color-info)] bg-[var(--color-info-bg)] cursor-pointer"
                          : "border-[var(--color-border-default)] hover:border-[var(--color-border-active)] cursor-pointer"
                        : "border-[var(--color-border-default)] bg-[var(--color-bg-secondary)] opacity-60 cursor-not-allowed"
                    } text-left`}
                  >
                    <div className="flex items-start gap-3">
                      <span
                        aria-hidden="true"
                        className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-[4px] border ${
                          checked
                            ? "border-[var(--color-info)] bg-[var(--color-info)] text-white"
                            : "border-[var(--color-border-default)] bg-white text-transparent"
                        }`}
                      >
                        <Check className="h-3 w-3" />
                      </span>
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
                  </button>
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
