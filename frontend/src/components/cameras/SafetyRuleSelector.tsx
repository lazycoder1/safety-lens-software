import { SeverityBadge } from "@/components/ui/SeverityBadge"
import type { SafetyRule } from "@/types"

interface SafetyRuleSelectorProps {
  safetyRules: SafetyRule[]
  selectedIds: string[]
  onToggle: (ruleId: string) => void
  readOnly?: boolean
  prominent?: boolean
}

function RuleCard({
  rule,
  selected,
  onToggle,
  readOnly,
}: {
  rule: SafetyRule
  selected: boolean
  onToggle: () => void
  readOnly?: boolean
}) {
  const Tag = readOnly ? "div" : "label"
  return (
    <Tag
      className={`flex items-center gap-3 p-3 rounded-[var(--radius-md)] border transition-colors ${
        readOnly ? "" : "cursor-pointer"
      } ${
        selected
          ? "border-[var(--color-info)] bg-blue-50"
          : "border-[var(--color-border)] hover:border-[var(--color-border-strong)]"
      }`}
    >
      {!readOnly && (
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggle}
          className="accent-[var(--color-info)]"
        />
      )}
      <div className="flex-1 min-w-0">
        <span className="text-sm font-medium text-[var(--color-text-primary)]">{rule.name}</span>
        <span className="text-xs text-[var(--color-text-tertiary)] ml-2">{rule.classes.join(", ")}</span>
      </div>
      <SeverityBadge severity={rule.severity} />
    </Tag>
  )
}

export function SafetyRuleSelector({
  safetyRules,
  selectedIds,
  onToggle,
  readOnly = false,
  prominent = false,
}: SafetyRuleSelectorProps) {
  const ppeRules = safetyRules.filter((r) => r.type === "ppe" && r.enabled)
  const alertRules = safetyRules.filter((r) => r.type === "alert" && r.enabled)

  if (ppeRules.length === 0 && alertRules.length === 0) {
    return (
      <p className="text-xs text-[var(--color-text-tertiary)] italic py-2">
        No safety rules available. Create rules in the Rules Engine first.
      </p>
    )
  }

  return (
    <div className={prominent ? "border-l-4 border-l-[var(--color-info)] pl-3" : ""}>
      {ppeRules.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">PPE Rules</p>
          <div className="space-y-2">
            {ppeRules.map((rule) => (
              <RuleCard
                key={rule.id}
                rule={rule}
                selected={selectedIds.includes(rule.id)}
                onToggle={() => onToggle(rule.id)}
                readOnly={readOnly}
              />
            ))}
          </div>
        </div>
      )}
      {alertRules.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">Alert Rules</p>
          <div className="space-y-2">
            {alertRules.map((rule) => (
              <RuleCard
                key={rule.id}
                rule={rule}
                selected={selectedIds.includes(rule.id)}
                onToggle={() => onToggle(rule.id)}
                readOnly={readOnly}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
