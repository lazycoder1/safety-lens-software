import { useEffect, useMemo, useState } from "react"
import { create } from "zustand"
import { getModelInstallJob, retryModelInstallJob } from "@/lib/api"
import type { ModelInstallJob } from "@/types"
import { Button } from "@/components/ui/button"

interface ModelInstallModalState {
  isOpen: boolean
  jobId: string | null
  title: string
  subtitle: string
  onReady: (() => void) | null
  open: (jobId: string, opts?: { title?: string; subtitle?: string; onReady?: (() => void) | null }) => void
  setJobId: (jobId: string) => void
  close: () => void
}

const SESSION_KEY = "rakshak_lens_active_model_install_job"

export const useModelInstallModal = create<ModelInstallModalState>((set) => ({
  isOpen: false,
  jobId: null,
  title: "Setting up required models",
  subtitle: "The system is downloading and preparing model files for this workflow.",
  onReady: null,
  open: (jobId, opts) => {
    sessionStorage.setItem(SESSION_KEY, jobId)
    set({
      isOpen: true,
      jobId,
      title: opts?.title || "Setting up required models",
      subtitle: opts?.subtitle || "The system is downloading and preparing model files for this workflow.",
      onReady: opts?.onReady || null,
    })
  },
  setJobId: (jobId) => {
    sessionStorage.setItem(SESSION_KEY, jobId)
    set({ jobId })
  },
  close: () => {
    sessionStorage.removeItem(SESSION_KEY)
    set({ isOpen: false, jobId: null, onReady: null })
  },
}))

const STAGE_LABELS: Record<ModelInstallJob["stage"], string> = {
  queued: "Queued",
  checking_disk: "Checking local files",
  downloading: "Downloading model",
  verifying: "Verifying files",
  preparing_assets: "Preparing assets",
  loading: "Loading model",
  warming_up: "Warming up model",
  ready: "Ready",
  failed: "Failed",
}

function formatBytes(value: number | null) {
  if (!value || value <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB"]
  let current = value
  let unitIndex = 0
  while (current >= 1024 && unitIndex < units.length - 1) {
    current /= 1024
    unitIndex += 1
  }
  return `${current.toFixed(current >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`
}

export function ModelInstallModal() {
  const { isOpen, jobId, title, subtitle, onReady, close, open, setJobId } = useModelInstallModal()
  const [job, setJob] = useState<ModelInstallJob | null>(null)
  const [retrying, setRetrying] = useState(false)
  const [bootstrapped, setBootstrapped] = useState(false)

  useEffect(() => {
    if (bootstrapped) return
    setBootstrapped(true)
    const persistedJobId = sessionStorage.getItem(SESSION_KEY)
    if (persistedJobId && !isOpen) {
      open(persistedJobId, {
        title: "Setting up required models",
        subtitle: "The system is resuming the active model setup job.",
      })
    }
  }, [bootstrapped, isOpen, open])

  useEffect(() => {
    if (!isOpen || !jobId) return
    let cancelled = false
    const activeJobId = jobId

    async function poll() {
      try {
        const current = await getModelInstallJob(activeJobId)
        if (cancelled) return
        setJob(current)
        if (current.status === "ready") {
          window.setTimeout(() => {
            if (!cancelled) {
              const callback = onReady
              close()
              callback?.()
            }
          }, 700)
          return
        }
      } catch {
        // Leave the modal open with the last known state.
      }
      if (!cancelled) {
        window.setTimeout(poll, 1000)
      }
    }

    poll()
    return () => {
      cancelled = true
    }
  }, [close, isOpen, jobId, onReady])

  const currentStageLabel = job ? STAGE_LABELS[job.stage] : "Checking status"
  const currentModelLabel = useMemo(() => {
    if (!job) return "Preparing"
    if (job.current_model_key === "coco_primary") return "COCO Primary"
    if (job.current_model_key === "ppe_specialist") return "PPE Specialist"
    return "YOLOE Long-Tail"
  }, [job])

  async function handleRetry() {
    if (!jobId) return
    setRetrying(true)
    try {
      const nextJob = await retryModelInstallJob(jobId)
      setJob(nextJob)
      setJobId(nextJob.id)
    } finally {
      setRetrying(false)
    }
  }

  if (!isOpen || !jobId) return null

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/55">
      <div className="w-full max-w-lg rounded-[var(--radius-xl)] border bg-white p-6 shadow-2xl">
        <div className="space-y-1">
          <p className="text-lg font-semibold text-[var(--color-text-primary)]">{title}</p>
          <p className="text-sm text-[var(--color-text-secondary)]">{subtitle}</p>
        </div>

        <div className="mt-6 space-y-4">
          <div className="rounded-[var(--radius-lg)] border bg-[var(--color-bg-secondary)] p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">
                  Current Stage
                </p>
                <p className="mt-1 text-sm font-medium text-[var(--color-text-primary)]">
                  {currentStageLabel}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">
                  Model
                </p>
                <p className="mt-1 text-sm font-medium text-[var(--color-text-primary)]">
                  {currentModelLabel}
                </p>
              </div>
            </div>

            <div className="mt-4 h-2 overflow-hidden rounded-full bg-[var(--color-bg-tertiary)]">
              <div
                className={`h-full transition-all ${job?.status === "failed" ? "bg-[var(--color-critical)]" : "bg-[var(--color-info)]"}`}
                style={{ width: `${Math.max(6, job?.progress_percent || 8)}%` }}
              />
            </div>

            <div className="mt-3 flex items-center justify-between text-xs text-[var(--color-text-secondary)]">
              <span>{Math.round(job?.progress_percent || 0)}%</span>
              <span>
                {formatBytes(job?.bytes_downloaded || 0)}
                {job?.total_bytes ? ` / ${formatBytes(job.total_bytes)}` : ""}
              </span>
            </div>
          </div>

          <div className="rounded-[var(--radius-lg)] border p-4">
            <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">
              Stage Checklist
            </p>
            <div className="mt-3 grid gap-2">
              {Object.entries(STAGE_LABELS)
                .filter(([key]) => key !== "failed")
                .map(([key, label]) => {
                  const reached = job
                    ? [
                        "queued",
                        "checking_disk",
                        "downloading",
                        "verifying",
                        "preparing_assets",
                        "loading",
                        "warming_up",
                        "ready",
                      ].indexOf(job.stage) >=
                      [
                        "queued",
                        "checking_disk",
                        "downloading",
                        "verifying",
                        "preparing_assets",
                        "loading",
                        "warming_up",
                        "ready",
                      ].indexOf(key)
                    : key === "queued"
                  return (
                    <div key={key} className="flex items-center justify-between gap-3 text-sm">
                      <span className="text-[var(--color-text-primary)]">{label}</span>
                      <span className={reached ? "text-[var(--color-success)]" : "text-[var(--color-text-tertiary)]"}>
                        {reached ? "Done" : "Pending"}
                      </span>
                    </div>
                  )
                })}
            </div>
          </div>

          {job?.status === "failed" && (
            <div className="rounded-[var(--radius-lg)] border border-[var(--color-critical)] bg-[var(--color-critical-bg)] p-4">
              <p className="text-sm font-semibold text-[var(--color-text-primary)]">Model setup failed</p>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                {job.error || "The install job failed before the model became ready."}
              </p>
              <div className="mt-4">
                <Button onClick={handleRetry} disabled={retrying}>
                  {retrying ? "Retrying…" : "Retry Setup"}
                </Button>
              </div>
            </div>
          )}

          {job?.status === "ready" && (
            <div className="rounded-[var(--radius-lg)] border border-[var(--color-success)] bg-[var(--color-success-bg)] p-4">
              <p className="text-sm font-semibold text-[var(--color-text-primary)]">Model ready</p>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                Setup completed successfully. Continuing automatically.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
