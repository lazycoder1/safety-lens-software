import { useCallback, useEffect, useMemo, useState } from "react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { getModels, installModels } from "@/lib/api"
import type { ModelStatus } from "@/types"
import { useModelInstallModal } from "@/components/ModelInstallModal"

function prettyStatus(model: ModelStatus) {
  if (model.status === "ready") return "Ready"
  if (model.status === "installing") return "Installing"
  if (model.status === "failed") return "Failed"
  return "Not Downloaded"
}

export function SystemModels() {
  const [models, setModels] = useState<ModelStatus[]>([])
  const [loading, setLoading] = useState(true)
  const openModal = useModelInstallModal((state) => state.open)

  const refresh = useCallback(async () => {
    try {
      const data = await getModels()
      setModels(data.models)
    } catch (error: any) {
      toast.error(error.message || "Failed to load models")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    const interval = window.setInterval(refresh, 5000)
    return () => window.clearInterval(interval)
  }, [refresh])

  const missingModels = useMemo(
    () => models.filter((model) => !model.is_ready).map((model) => model.model_key),
    [models]
  )

  async function handleInstall(modelKeys: string[], title: string) {
    try {
      const job = await installModels(modelKeys)
      openModal(job.id, {
        title,
        subtitle: "The system is downloading and preparing the selected model assets. This modal will stay open until setup completes.",
        onReady: () => {
          refresh()
          toast.success("Models are ready")
        },
      })
      await refresh()
    } catch (error: any) {
      toast.error(error.message || "Failed to start model setup")
    }
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Models</h1>
          <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
            Check model readiness, shared asset paths, and start downloads before camera setup needs them.
          </p>
        </div>
        <Button
          onClick={() => handleInstall(missingModels, "Setting up missing models")}
          disabled={loading || missingModels.length === 0}
        >
          Download All Missing
        </Button>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        {models.map((model) => (
          <Card key={model.model_key} className="space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-base font-semibold text-[var(--color-text-primary)]">{model.display_name}</p>
                <p className="mt-1 text-sm text-[var(--color-text-secondary)]">{model.filename}</p>
              </div>
              <span
                className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                  model.status === "ready"
                    ? "bg-[var(--color-success-bg)] text-[var(--color-success)]"
                    : model.status === "failed"
                      ? "bg-[var(--color-critical-bg)] text-[var(--color-critical)]"
                      : "bg-[var(--color-info-bg)] text-[var(--color-info)]"
                }`}
              >
                {prettyStatus(model)}
              </span>
            </div>

            <div className="space-y-2 text-sm text-[var(--color-text-secondary)]">
              <p><span className="font-medium text-[var(--color-text-primary)]">Warmup:</span> {model.warmup_behavior}</p>
              <p className="break-all"><span className="font-medium text-[var(--color-text-primary)]">Shared Asset:</span> {model.shared_asset_key}</p>
              <p className="break-all"><span className="font-medium text-[var(--color-text-primary)]">Resolved Path:</span> {model.active_path || "Not downloaded yet"}</p>
              {model.error && (
                <p className="text-[var(--color-critical)]">
                  <span className="font-medium">Error:</span> {model.error}
                </p>
              )}
            </div>

            <div className="flex gap-2">
              <Button
                onClick={() => handleInstall([model.model_key], `Setting up ${model.display_name}`)}
                disabled={model.status === "installing"}
              >
                {model.is_ready ? "Reinstall / Verify" : "Download / Setup"}
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}

