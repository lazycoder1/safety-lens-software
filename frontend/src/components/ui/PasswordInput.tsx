import { useState } from "react"
import { Eye, EyeOff, Check, X } from "lucide-react"
import { evaluatePassword, type PasswordStrength } from "@/lib/passwordValidation"
import { cn } from "@/lib/utils"

interface PasswordInputProps {
  value: string
  onChange: (value: string) => void
  label?: string
  placeholder?: string
  showStrengthMeter?: boolean
  showRequirements?: boolean
  autoComplete?: string
  disabled?: boolean
}

const meterColors: Record<string, string> = {
  "too-weak": "bg-red-500",
  weak: "bg-orange-500",
  fair: "bg-yellow-500",
  good: "bg-emerald-500",
  strong: "bg-emerald-600",
}

const meterTextColors: Record<string, string> = {
  "too-weak": "text-red-600",
  weak: "text-orange-600",
  fair: "text-yellow-600",
  good: "text-emerald-600",
  strong: "text-emerald-700",
}

export function PasswordInput({
  value,
  onChange,
  label = "Password",
  placeholder = "Enter password",
  showStrengthMeter = true,
  showRequirements = true,
  autoComplete = "new-password",
  disabled = false,
}: PasswordInputProps) {
  const [visible, setVisible] = useState(false)
  const strength: PasswordStrength = evaluatePassword(value)
  const hasValue = value.length > 0

  return (
    <div>
      <label className="block text-sm font-medium text-neutral-700 mb-1">{label}</label>
      <div className="relative">
        <input
          type={visible ? "text" : "password"}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full rounded-lg border border-neutral-300 px-3 py-2 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
          placeholder={placeholder}
          autoComplete={autoComplete}
          disabled={disabled}
        />
        <button
          type="button"
          onClick={() => setVisible(!visible)}
          className="absolute right-2.5 top-1/2 -translate-y-1/2 text-neutral-400 hover:text-neutral-600 cursor-pointer"
          tabIndex={-1}
        >
          {visible ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
        </button>
      </div>

      {hasValue && showStrengthMeter && (
        <div className="mt-2 flex items-center gap-2">
          <div className="flex-1 h-1.5 rounded-full bg-neutral-200 overflow-hidden">
            <div
              className={cn("h-full rounded-full transition-all duration-300", meterColors[strength.level])}
              style={{ width: `${(strength.score / 5) * 100}%` }}
            />
          </div>
          <span className={cn("text-xs font-medium whitespace-nowrap", meterTextColors[strength.level])}>
            {strength.label}
          </span>
        </div>
      )}

      {hasValue && showRequirements && (
        <ul className="mt-2 space-y-0.5">
          {strength.checks.map((check) => (
            <li key={check.label} className="flex items-center gap-1.5">
              {check.met ? (
                <Check className="w-3 h-3 text-emerald-500" />
              ) : (
                <X className="w-3 h-3 text-neutral-400" />
              )}
              <span className={cn("text-xs", check.met ? "text-emerald-600" : "text-neutral-400")}>
                {check.label}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
