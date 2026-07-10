import { useState, useEffect, useCallback, useRef } from "react"
import { Check, X, Upload, Loader2, AlertTriangle } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { toast } from "sonner"
import {
  getLicenseStatus,
  uploadLicense,
  uploadHeartbeat,
  type LicenseStatusResponse,
  type LicenseState,
} from "@/lib/api"

// Catalog of features Rakshak Lens supports. We render every entry — those
// not present in the license's `features` array show as "Not licensed".
// Order matches the rollout: base is v1, the rest are post-v1 add-ons.
const FEATURE_CATALOG: { key: string; name: string; description: string }[] = [
  {
    key: "base",
    name: "Base Detections",
    description: "Person, vehicle, fire, PPE, phone usage, and more",
  },
  {
    key: "anpr",
    name: "ANPR (Vehicle Plate Recognition)",
    description: "Automatic number plate recognition at entry/exit points",
  },
  {
    key: "face",
    name: "Face Recognition",
    description: "Face enrollment and matching for access control",
  },
  {
    key: "ai_search",
    name: "AI Search",
    description: "Semantic search across all camera footage",
  },
]

const STATUS_CONFIG: Record<
  LicenseState,
  {
    label: string
    dotClass: string
    badgeVariant: "success" | "critical" | "warning" | "default"
  }
> = {
  valid: { label: "Active", dotClass: "bg-[var(--color-success)]", badgeVariant: "success" },
  warning: { label: "Renewal Due", dotClass: "bg-[var(--color-warning)]", badgeVariant: "warning" },
  grace: { label: "Grace Period", dotClass: "bg-[var(--color-warning)]", badgeVariant: "warning" },
  suspended: { label: "Suspended", dotClass: "bg-[var(--color-critical)]", badgeVariant: "critical" },
}

function formatDate(iso: string | undefined | null): string {
  if (!iso) return "—"
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    })
  } catch {
    return iso
  }
}

function daysUntil(iso: string): number {
  const milliseconds = new Date(iso).getTime() - Date.now()
  return Math.max(0, Math.floor(milliseconds / 86_400_000))
}

export function LicenseStatus() {
  const [status, setStatus] = useState<LicenseStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const refresh = useCallback(async () => {
    try {
      const next = await getLicenseStatus()
      setStatus(next)
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to load license status"
      toast.error(msg)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const handleFile = useCallback(
    async (file: File) => {
      setUploading(true)
      try {
        // Pick the right endpoint based on filename — .lic is the long-lived
        // license, anything else (typically .json) is treated as a heartbeat.
        const isLicense = file.name.toLowerCase().endsWith(".lic")
        const next = isLicense ? await uploadLicense(file) : await uploadHeartbeat(file)
        setStatus(next)
        toast.success(
          isLicense
            ? `License installed for ${next.license?.customer_name ?? "this site"}`
            : "Heartbeat token installed",
        )
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Upload failed"
        toast.error(msg)
      } finally {
        setUploading(false)
      }
    },
    [],
  )

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setDragActive(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile],
  )

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
      // Allow re-uploading the same file by clearing the input
      e.target.value = ""
    },
    [handleFile],
  )

  if (loading) {
    return (
      <div className="space-y-6 p-6">
        <Skeleton className="h-7 w-24" />
        <Card>
          <div className="flex items-center gap-2 mb-5">
            <Skeleton className="w-2.5 h-2.5 rounded-full" />
            <Skeleton className="h-5 w-16 rounded-full" />
          </div>
          <div className="grid gap-3">
            {["Customer", "License ID", "Issued by", "Cameras", "Expires", "Issued"].map((_, i) => (
              <div key={i} className="flex items-center justify-between">
                <Skeleton className="h-3.5 w-24" />
                <Skeleton className="h-4 w-40" />
              </div>
            ))}
          </div>
        </Card>
        <div>
          <Skeleton className="h-4 w-32 mb-3" />
          <div className="space-y-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="flex items-start gap-3 px-4 py-3 rounded-[var(--radius-md)] border bg-white">
                <Skeleton className="w-4 h-4 mt-0.5 rounded" />
                <div className="flex-1">
                  <Skeleton className="h-4 w-48 mb-1" />
                  <Skeleton className="h-3 w-64" />
                </div>
              </div>
            ))}
          </div>
        </div>
        <div>
          <Skeleton className="h-4 w-28 mb-3" />
          <Skeleton className="h-32 w-full rounded-[var(--radius-lg)]" />
        </div>
      </div>
    )
  }

  if (!status) {
    return (
      <div className="p-6">
        <p className="text-sm text-[var(--color-critical)]">Failed to load license status.</p>
      </div>
    )
  }

  const cfg = STATUS_CONFIG[status.state]
  const isHealthy = status.state === "valid"
  const license = status.license

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <h1 className="text-xl font-bold text-[var(--color-text-primary)]">License</h1>

      {/* Banner for non-healthy states */}
      {!isHealthy && (
        <div
          className={cn(
            "rounded-[var(--radius-md)] px-4 py-3 text-sm font-medium flex items-start gap-2",
            status.state === "suspended"
              ? "bg-[var(--color-critical-bg)] text-[#991b1b] border border-[var(--color-critical)]"
              : "bg-[var(--color-warning-bg)] text-[#92400e] border border-[var(--color-warning)]",
          )}
        >
          <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{status.reason}</span>
        </div>
      )}

      {/* Status card */}
      <Card>
        <div className="flex items-center gap-2 mb-5">
          <span className={cn("inline-block w-2.5 h-2.5 rounded-full", cfg.dotClass)} />
          <Badge variant={cfg.badgeVariant}>{cfg.label}</Badge>
        </div>

        {license ? (
          <div className="grid gap-3">
            <InfoRow label="Customer" value={license.customer_name} />
            <InfoRow label="License ID" value={license.license_id} mono />
            <InfoRow label="Issued by" value={license.issued_by_partner} />
            <InfoRow label="Cameras allowed" value={`${license.max_cameras}`} />
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-[var(--color-text-secondary)]">Expires</span>
              <span className="text-sm text-[var(--color-text-primary)]">
                {formatDate(license.expires_at)}
                {status.state === "valid" && (
                  <span className="text-[var(--color-success)] text-xs ml-2">
                    ({daysUntil(license.expires_at)} days remaining)
                  </span>
                )}
                {status.state === "warning" && (
                  <span className="text-[var(--color-warning)] text-xs ml-2">(renewal due)</span>
                )}
                {status.state === "grace" && (
                  <span className="text-[var(--color-critical)] text-xs ml-2">(in grace period)</span>
                )}
                {status.state === "suspended" && (
                  <span className="text-[var(--color-critical)] text-xs ml-2">(suspended)</span>
                )}
              </span>
            </div>
            <InfoRow label="Issued" value={formatDate(license.issued_at)} />
            {status.heartbeat && (
              <InfoRow
                label="Heartbeat valid until"
                value={formatDate(status.heartbeat.valid_until)}
              />
            )}
          </div>
        ) : (
          <p className="text-sm text-[var(--color-text-secondary)]">
            No license installed. Upload a .lic file from your implementation partner to activate
            Rakshak Lens.
          </p>
        )}
      </Card>

      {/* Licensed Features */}
      <div>
        <h2 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">
          Licensed Features
        </h2>
        <div className="space-y-2">
          {FEATURE_CATALOG.map((feat) => {
            const enabled = isHealthy && license?.features.includes(feat.key)
            return (
              <div
                key={feat.key}
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
                        : "text-[var(--color-text-tertiary)]",
                    )}
                  >
                    {feat.name}
                  </p>
                  <p className="text-xs text-[var(--color-text-tertiary)]">
                    {enabled ? feat.description : license?.features.includes(feat.key)
                      ? "Suspended — see banner above"
                      : "Not licensed"}
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
        <input
          ref={fileInputRef}
          type="file"
          accept=".lic,.json"
          className="hidden"
          onChange={onFileChange}
        />
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragActive(true)
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={onDrop}
          className={cn(
            "flex flex-col items-center justify-center gap-2 py-10 rounded-[var(--radius-lg)] border-2 border-dashed cursor-pointer transition-colors",
            uploading && "opacity-50 pointer-events-none",
            dragActive
              ? "border-[var(--color-info)] bg-[var(--color-info-bg)]"
              : status.state === "suspended" || status.state === "grace"
                ? "border-[var(--color-warning)] hover:bg-[var(--color-bg-secondary)]"
                : "border-[var(--color-border-default)] hover:bg-[var(--color-bg-secondary)]",
          )}
        >
          {uploading ? (
            <Loader2 className="w-6 h-6 animate-spin text-[var(--color-text-tertiary)]" />
          ) : (
            <Upload className="w-6 h-6 text-[var(--color-text-tertiary)]" />
          )}
          <p className="text-sm text-[var(--color-text-secondary)]">
            {uploading
              ? "Uploading..."
              : "Drop a .lic or heartbeat file here, or click to browse"}
          </p>
        </div>
      </div>
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
