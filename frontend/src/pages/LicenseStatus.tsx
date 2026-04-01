import { useState, useEffect, useCallback } from "react"
import { Check, X, Upload } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { toast } from "sonner"

type LicenseState = "active" | "expired" | "disabled"

interface Feature {
  name: string
  description: string
  licensed: boolean
}

const FEATURES: Feature[] = [
  { name: "Base Detections (10 features)", description: "Person, vehicle, fire, PPE, phone usage, and more", licensed: true },
  { name: "ANPR (Vehicle Plate Recognition)", description: "Automatic number plate recognition at entry/exit points", licensed: true },
  { name: "Face Recognition", description: "Face enrollment and matching for access control", licensed: true },
  { name: "AI Search", description: "Semantic search across all camera footage", licensed: false },
]

const STATUS_CONFIG: Record<LicenseState, { color: string; label: string; dotClass: string; badgeVariant: "success" | "critical" | "default" }> = {
  active: { color: "var(--color-success)", label: "Active", dotClass: "bg-[var(--color-success)]", badgeVariant: "success" },
  expired: { color: "var(--color-critical)", label: "Expired", dotClass: "bg-[var(--color-critical)]", badgeVariant: "critical" },
  disabled: { color: "var(--color-text-tertiary)", label: "Disabled", dotClass: "bg-[var(--color-text-tertiary)]", badgeVariant: "default" },
}

export function LicenseStatus() {
  const [licenseState, setLicenseState] = useState<LicenseState>("active")
  const [devMode, setDevMode] = useState(false)

  // Dev toggle: Shift+L pressed 5 times within 2 seconds
  const handleDevToggle = useCallback(() => {
    const presses: number[] = []

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "L" && e.shiftKey) {
        const now = Date.now()
        presses.push(now)
        // Keep only presses within last 2 seconds
        while (presses.length > 0 && now - presses[0] > 2000) {
          presses.shift()
        }
        if (presses.length >= 5) {
          setDevMode((prev) => !prev)
          presses.length = 0
        }
      }
    }

    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [])

  useEffect(() => {
    return handleDevToggle()
  }, [handleDevToggle])

  const status = STATUS_CONFIG[licenseState]
  const isNormal = licenseState === "active"

  const daysRemaining = 363
  const camerasUsed = 7
  const camerasTotal = 10

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <h1 className="text-xl font-bold text-[var(--color-text-primary)]">License</h1>

      {/* Banner for expired/disabled */}
      {!isNormal && (
        <div
          className={cn(
            "rounded-[var(--radius-md)] px-4 py-3 text-sm font-medium",
            licenseState === "expired"
              ? "bg-[var(--color-critical-bg)] text-[#991b1b] border border-[var(--color-critical)]"
              : "bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] border"
          )}
        >
          {licenseState === "expired"
            ? "License expired. Detection features are disabled."
            : "License is disabled."}
        </div>
      )}

      {/* Status card */}
      <Card>
        <div className="flex items-center gap-2 mb-5">
          <span className={cn("inline-block w-2.5 h-2.5 rounded-full", status.dotClass)} />
          <Badge variant={status.badgeVariant}>{status.label}</Badge>
        </div>

        <div className="grid gap-3">
          <InfoRow label="Customer" value="TMEIC Jamshedpur" />
          <InfoRow label="License ID" value="SL-2026-0001" mono />
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[var(--color-text-secondary)]">Cameras</span>
            <div className="flex items-center gap-3">
              <span className="text-sm text-[var(--color-text-primary)]">
                {camerasUsed} / {camerasTotal} used
              </span>
              <div className="w-24 h-1.5 rounded-full bg-[var(--color-bg-tertiary)] overflow-hidden">
                <div
                  className="h-full rounded-full bg-[var(--color-info)]"
                  style={{ width: `${(camerasUsed / camerasTotal) * 100}%` }}
                />
              </div>
            </div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-[var(--color-text-secondary)]">Expires</span>
            <span className="text-sm text-[var(--color-text-primary)]">
              2027-03-26{" "}
              {licenseState === "expired" ? (
                <span className="text-[var(--color-critical)] text-xs">(Expired)</span>
              ) : (
                <span className="text-[var(--color-success)] text-xs">
                  ({daysRemaining} days remaining)
                </span>
              )}
            </span>
          </div>
          <InfoRow label="Issued" value="2026-03-26" />
        </div>
      </Card>

      {/* Licensed Features */}
      <div>
        <h2 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">
          Licensed Features
        </h2>
        <div className="space-y-2">
          {FEATURES.map((feat) => {
            const enabled = isNormal && feat.licensed
            return (
              <div
                key={feat.name}
                className="flex items-start gap-3 px-4 py-3 rounded-[var(--radius-md)] border bg-white"
              >
                {enabled ? (
                  <Check className="w-4 h-4 text-[var(--color-success)] mt-0.5 shrink-0" />
                ) : (
                  <X className="w-4 h-4 text-[var(--color-text-tertiary)] mt-0.5 shrink-0" />
                )}
                <div>
                  <p
                    className={cn(
                      "text-sm font-medium",
                      enabled
                        ? "text-[var(--color-text-primary)]"
                        : "text-[var(--color-text-tertiary)]"
                    )}
                  >
                    {feat.name}
                  </p>
                  <p className="text-xs text-[var(--color-text-tertiary)]">
                    {enabled ? feat.description : !feat.licensed && isNormal ? "Not licensed" : feat.description}
                  </p>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Upload License */}
      <div>
        <h2 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">
          Upload License
        </h2>
        <div
          onClick={() =>
            toast("License upload available in production deployment")
          }
          className={cn(
            "flex flex-col items-center justify-center gap-2 py-10 rounded-[var(--radius-lg)] border-2 border-dashed cursor-pointer transition-colors hover:bg-[var(--color-bg-secondary)]",
            licenseState === "expired"
              ? "border-[var(--color-warning)]"
              : "border-[var(--color-border-default)]"
          )}
        >
          <Upload className="w-6 h-6 text-[var(--color-text-tertiary)]" />
          <p className="text-sm text-[var(--color-text-secondary)]">
            Drop .lic file here or click to browse
          </p>
        </div>
      </div>

      {/* Dev mode panel */}
      {devMode && (
        <div className="fixed bottom-4 right-4 z-50 bg-white border rounded-[var(--radius-lg)] shadow-lg p-3 space-y-2">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-tertiary)]">
            Dev mode
          </p>
          <div className="flex gap-1.5">
            <Button
              size="sm"
              variant={licenseState === "active" ? "primary" : "secondary"}
              onClick={() => setLicenseState("active")}
            >
              Active
            </Button>
            <Button
              size="sm"
              variant={licenseState === "expired" ? "danger" : "secondary"}
              onClick={() => setLicenseState("expired")}
            >
              Expired
            </Button>
            <Button
              size="sm"
              variant={licenseState === "disabled" ? "primary" : "secondary"}
              onClick={() => setLicenseState("disabled")}
            >
              Disabled
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

function InfoRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs font-medium text-[var(--color-text-secondary)]">{label}</span>
      <span className={cn("text-sm text-[var(--color-text-primary)]", mono && "font-mono")}>
        {value}
      </span>
    </div>
  )
}
