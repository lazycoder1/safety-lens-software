// Password validation — rules must match backend/auth_store.py validate_password()

export interface PasswordCheck {
  label: string
  met: boolean
}

export type StrengthLevel = "too-weak" | "weak" | "fair" | "good" | "strong"

export interface PasswordStrength {
  score: number
  level: StrengthLevel
  label: string
  checks: PasswordCheck[]
  isValid: boolean
}

const SPECIAL_CHARS = /[!@#$%^&*()\-_+=[\]{}|;:,.<>?]/

export function evaluatePassword(password: string): PasswordStrength {
  const checks: PasswordCheck[] = [
    { label: "At least 8 characters", met: password.length >= 8 },
    { label: "Uppercase letter", met: /[A-Z]/.test(password) },
    { label: "Lowercase letter", met: /[a-z]/.test(password) },
    { label: "Number", met: /[0-9]/.test(password) },
    { label: "Special character", met: SPECIAL_CHARS.test(password) },
  ]

  const metCount = checks.filter((c) => c.met).length

  let score: number
  if (password.length === 0) score = 0
  else if (!checks[0].met) score = 1
  else if (metCount <= 2) score = 2
  else if (metCount <= 3) score = 3
  else if (metCount <= 4) score = 4
  else score = password.length >= 12 ? 5 : 4

  const levels: [StrengthLevel, string][] = [
    ["too-weak", "Too Weak"],
    ["too-weak", "Too Weak"],
    ["weak", "Weak"],
    ["fair", "Fair"],
    ["good", "Good"],
    ["strong", "Strong"],
  ]

  const [level, label] = levels[score]
  const isValid = metCount === 5

  return { score, level, label, checks, isValid }
}

export function isPasswordValid(password: string): boolean {
  return evaluatePassword(password).isValid
}

export function generateStrongPassword(length = 16): string {
  const specials = "!@#$%^&*()-_+=[]{}|;:,.<>?"
  const upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  const lower = "abcdefghijklmnopqrstuvwxyz"
  const digits = "0123456789"
  const all = upper + lower + digits + specials

  const pick = (s: string) => s[Math.floor(Math.random() * s.length)]
  const chars = [pick(upper), pick(lower), pick(digits), pick(specials)]
  for (let i = chars.length; i < length; i++) chars.push(pick(all))

  // Fisher-Yates shuffle
  for (let i = chars.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[chars[i], chars[j]] = [chars[j], chars[i]]
  }
  return chars.join("")
}
