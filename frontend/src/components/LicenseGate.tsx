import { useState, useEffect, useCallback, useRef } from "react"
import { useNavigate } from "react-router-dom"
import { AlertTriangle, ShieldX, Upload, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { toast } from "sonner"
import {
  getLicenseStatus,
  uploadLicense,
  type LicenseStatusResponse,
} from "@/lib/api"

/**
 * Full-screen modal that appears immediately after login when the license
 * is not in a healthy state. Blocks access to the app until a valid license
 * is uploaded, or lets the user dismiss warnings/grace states.
 */
export function LicenseGate() {
  const [status, setStatus] = useState<LicenseStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [dismissed, setDismissed] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  useEffect(() => {
    getLicenseStatus()
      .then(setStatus)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".lic")) {
      toast.error("Please upload a .lic license file")
      return
    }
    setUploading(true)
    try {
      const next = await uploadLicense(file)
      setStatus(next)
      if (next.state === "valid") {
        toast.success(`License activated for ${next.license?.customer_name ?? "this site"}`)
        setDismissed(true)
      } else {
        toast.info(`License uploaded — status: ${next.state}`)
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Upload failed")
    } finally {
      setUploading(false)
    }
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragActive(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile],
  )

  // Don't show anything while loading, if license is valid, or if user dismissed
  if (loading || dismissed) return null
  if (!status || status.state === "valid") return null

  const isSuspended = status.state === "suspended"
  const canDismiss = !isSuspended // warning and grace can be dismissed

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full mx-4 overflow-hidden">
        {/* Header */}
        <div
          className={cn(
            "px-6 py-5 flex items-center gap-3",
            isSuspended
              ? "bg-red-50 border-b border-red-200"
              : "bg-amber-50 border-b border-amber-200",
          )}
        >
          {isSuspended ? (
            <ShieldX className="w-7 h-7 text-red-600 shrink-0" />
          ) : (
            <AlertTriangle className="w-7 h-7 text-amber-600 shrink-0" />
          )}
          <div>
            <h2
              className={cn(
                "text-lg font-bold",
                isSuspended ? "text-red-900" : "text-amber-900",
              )}
            >
              {isSuspended
                ? "License Required"
                : status.state === "grace"
                  ? "License Grace Period"
                  : "License Renewal Due"}
            </h2>
            <p
              className={cn(
                "text-sm mt-0.5",
                isSuspended ? "text-red-700" : "text-amber-700",
              )}
            >
              {status.reason}
            </p>
          </div>
        </div>

        {/* Body */}
        <div className="px-6 py-5 space-y-4">
          {isSuspended && (
            <p className="text-sm text-gray-600">
              Inference is paused and cameras are not processing. Upload a valid
              license file to restore operation.
            </p>
          )}

          {status.state === "grace" && status.days_until_suspension !== null && (
            <p className="text-sm text-gray-600">
              Cameras are still running, but the system will suspend in{" "}
              <span className="font-semibold text-amber-700">
                {status.days_until_suspension} days
              </span>
              . Please upload a renewed license.
            </p>
          )}

          {status.state === "warning" && status.days_until_suspension !== null && (
            <p className="text-sm text-gray-600">
              Your license expires in{" "}
              <span className="font-semibold text-amber-700">
                {status.days_until_suspension} days
              </span>
              . Contact your implementation partner for renewal.
            </p>
          )}

          {/* Upload area */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".lic"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) handleFile(file)
              e.target.value = ""
            }}
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
              "flex flex-col items-center justify-center gap-2 py-8 rounded-xl border-2 border-dashed cursor-pointer transition-colors",
              uploading && "opacity-50 pointer-events-none",
              dragActive
                ? "border-blue-400 bg-blue-50"
                : "border-gray-300 hover:border-gray-400 hover:bg-gray-50",
            )}
          >
            {uploading ? (
              <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
            ) : (
              <Upload className="w-6 h-6 text-gray-400" />
            )}
            <p className="text-sm text-gray-500">
              {uploading ? "Uploading..." : "Drop a .lic file here, or click to browse"}
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-gray-50 border-t flex items-center justify-between">
          <button
            onClick={() => navigate("/system/license")}
            className="text-sm text-blue-600 hover:text-blue-800 font-medium"
          >
            Go to License page
          </button>
          {canDismiss ? (
            <button
              onClick={() => setDismissed(true)}
              className="px-4 py-2 text-sm font-medium rounded-lg bg-gray-200 hover:bg-gray-300 text-gray-700 transition-colors"
            >
              Continue anyway
            </button>
          ) : (
            <span className="text-xs text-red-500 font-medium">
              Upload a license to continue
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
