import { useState, useEffect, useRef, useCallback } from "react"
import { Card } from "@/components/ui/card"
import { getConfig, updateGlobalConfig, updateVlmConfig, updateTelegramConfig, testTelegramConfig } from "@/lib/api"

export function SystemSettings() {
  const [global, setGlobal] = useState({
    target_fps: 6,
    yolo_conf: 0.35,
    jpeg_quality: 60,
    inference_width: 640,
    alert_cooldown: 8,
  })

  const [vlm, setVlm] = useState({
    enabled: true,
    model: "qwen3-vl:8b",
    interval: 45,
    temperature: 0.1,
    max_tokens: 300,
    prompt: "",
    violation_keywords: "",
  })

  const [telegram, setTelegram] = useState({
    enabled: false,
    bot_token: "",
    chat_id: "",
    severities: ["P1", "P2"] as string[],
  })
  const [telegramTest, setTelegramTest] = useState<{ status: string; message?: string } | null>(null)

  const [saved, setSaved] = useState(false)
  const globalTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const vlmTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const telegramTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    getConfig()
      .then((cfg: any) => {
        if (cfg.global) {
          setGlobal((prev) => ({ ...prev, ...cfg.global }))
        }
        if (cfg.vlm) {
          setVlm((prev) => ({
            ...prev,
            ...cfg.vlm,
            violation_keywords: Array.isArray(cfg.vlm.violation_keywords)
              ? cfg.vlm.violation_keywords.join(", ")
              : cfg.vlm.violation_keywords || "",
          }))
        }
        if (cfg.telegram) {
          setTelegram((prev) => ({ ...prev, ...cfg.telegram }))
        }
      })
      .catch(() => {})
  }, [])

  const flash = useCallback(() => {
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }, [])

  function patchGlobal(patch: Partial<typeof global>) {
    const next = { ...global, ...patch }
    setGlobal(next)
    clearTimeout(globalTimer.current)
    globalTimer.current = setTimeout(() => {
      updateGlobalConfig(next).then(flash).catch(() => {})
    }, 500)
  }

  function patchVlm(patch: Partial<typeof vlm>) {
    const next = { ...vlm, ...patch }
    setVlm(next)
    clearTimeout(vlmTimer.current)
    vlmTimer.current = setTimeout(() => {
      const payload: any = { ...next }
      if (typeof payload.violation_keywords === "string") {
        payload.violation_keywords = payload.violation_keywords
          .split(",")
          .map((k: string) => k.trim())
          .filter(Boolean)
      }
      updateVlmConfig(payload).then(flash).catch(() => {})
    }, 500)
  }

  function patchTelegram(patch: Partial<typeof telegram>) {
    const next = { ...telegram, ...patch }
    setTelegram(next)
    clearTimeout(telegramTimer.current)
    telegramTimer.current = setTimeout(() => {
      updateTelegramConfig(next).then(flash).catch(() => {})
    }, 500)
  }

  function toggleSeverity(sev: string) {
    const current = telegram.severities
    const next = current.includes(sev)
      ? current.filter((s) => s !== sev)
      : [...current, sev]
    patchTelegram({ severities: next })
  }

  async function handleTestTelegram() {
    setTelegramTest({ status: "testing" })
    try {
      const result = await testTelegramConfig()
      setTelegramTest(result.ok ? { status: "ok" } : { status: "error", message: result.error })
    } catch (e: any) {
      setTelegramTest({ status: "error", message: e.message })
    }
  }

  return (
    <div className="space-y-6 max-w-2xl p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-[var(--color-text-primary)]">System Settings</h1>
        {saved && (
          <span className="text-xs font-medium text-[var(--color-success)] animate-pulse">
            Saved
          </span>
        )}
      </div>

      {/* Global Settings */}
      <Card>
        <h2 className="text-sm font-semibold text-[var(--color-text-primary)] mb-4">Global Settings</h2>
        <div className="space-y-5">
          <RangeField
            label="Target FPS"
            value={global.target_fps}
            min={1}
            max={15}
            step={1}
            onChange={(v) => patchGlobal({ target_fps: v })}
          />
          <RangeField
            label="YOLO Confidence"
            value={global.yolo_conf}
            min={0.1}
            max={0.9}
            step={0.05}
            onChange={(v) => patchGlobal({ yolo_conf: v })}
          />
          <RangeField
            label="JPEG Quality"
            value={global.jpeg_quality}
            min={20}
            max={95}
            step={1}
            onChange={(v) => patchGlobal({ jpeg_quality: v })}
          />
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-[var(--color-text-primary)]">Inference Width</label>
              <span className="text-xs text-[var(--color-text-tertiary)]">{global.inference_width}px</span>
            </div>
            <select
              value={global.inference_width}
              onChange={(e) => patchGlobal({ inference_width: Number(e.target.value) })}
              className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer"
            >
              {[320, 480, 640, 854].map((w) => (
                <option key={w} value={w}>{w}px</option>
              ))}
            </select>
          </div>
          <NumberField
            label="Alert Cooldown (seconds)"
            value={global.alert_cooldown}
            min={0}
            onChange={(v) => patchGlobal({ alert_cooldown: v })}
          />
        </div>
      </Card>

      {/* VLM Settings */}
      <Card>
        <h2 className="text-sm font-semibold text-[var(--color-text-primary)] mb-4">VLM Settings</h2>
        <div className="space-y-5">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-[var(--color-text-primary)]">Enabled</label>
            <button
              onClick={() => patchVlm({ enabled: !vlm.enabled })}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors cursor-pointer ${
                vlm.enabled ? "bg-[var(--color-success)]" : "bg-[var(--color-bg-tertiary)] border"
              }`}
            >
              <span
                className={`inline-block h-4 w-4 rounded-full bg-white shadow transition-transform ${
                  vlm.enabled ? "translate-x-6" : "translate-x-1"
                }`}
              />
            </button>
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-[var(--color-text-primary)]">Model</label>
            <input
              type="text"
              value={vlm.model}
              onChange={(e) => patchVlm({ model: e.target.value })}
              className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
            />
          </div>

          <NumberField
            label="Interval (seconds)"
            value={vlm.interval}
            min={5}
            onChange={(v) => patchVlm({ interval: v })}
          />

          <RangeField
            label="Temperature"
            value={vlm.temperature}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => patchVlm({ temperature: v })}
          />

          <NumberField
            label="Max Tokens"
            value={vlm.max_tokens}
            min={32}
            onChange={(v) => patchVlm({ max_tokens: v })}
          />

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-[var(--color-text-primary)]">Prompt</label>
            <textarea
              rows={4}
              value={vlm.prompt}
              onChange={(e) => patchVlm({ prompt: e.target.value })}
              className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 resize-y"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-[var(--color-text-primary)]">Violation Keywords (comma-separated)</label>
            <input
              type="text"
              value={vlm.violation_keywords}
              onChange={(e) => patchVlm({ violation_keywords: e.target.value })}
              placeholder="e.g. violation, unsafe, missing, not wearing"
              className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
            />
          </div>
        </div>
      </Card>

      {/* Telegram Alerts */}
      <Card>
        <h2 className="text-sm font-semibold text-[var(--color-text-primary)] mb-4">Telegram Alerts</h2>
        <div className="space-y-5">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-[var(--color-text-primary)]">Enabled</label>
            <button
              onClick={() => patchTelegram({ enabled: !telegram.enabled })}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors cursor-pointer ${
                telegram.enabled ? "bg-[var(--color-success)]" : "bg-[var(--color-bg-tertiary)] border"
              }`}
            >
              <span
                className={`inline-block h-4 w-4 rounded-full bg-white shadow transition-transform ${
                  telegram.enabled ? "translate-x-6" : "translate-x-1"
                }`}
              />
            </button>
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-[var(--color-text-primary)]">Bot Token</label>
            <input
              type="password"
              value={telegram.bot_token}
              onChange={(e) => patchTelegram({ bot_token: e.target.value })}
              placeholder="123456:ABC-DEF..."
              className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-[var(--color-text-primary)]">Chat ID</label>
            <input
              type="text"
              value={telegram.chat_id}
              onChange={(e) => patchTelegram({ chat_id: e.target.value })}
              placeholder="-1001234567890"
              className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-[var(--color-text-primary)]">Alert Severities</label>
            <div className="flex gap-2">
              {["P1", "P2", "P3", "P4"].map((sev) => (
                <button
                  key={sev}
                  onClick={() => toggleSeverity(sev)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-[var(--radius-md)] border cursor-pointer transition-colors ${
                    telegram.severities.includes(sev)
                      ? "bg-[var(--color-text-primary)] text-white border-transparent"
                      : "bg-white text-[var(--color-text-secondary)] border-[var(--color-border)]"
                  }`}
                >
                  {sev}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleTestTelegram}
              disabled={!telegram.bot_token || !telegram.chat_id}
              className="px-4 py-2 text-sm font-medium rounded-[var(--radius-md)] bg-[var(--color-info)] text-white disabled:opacity-40 cursor-pointer hover:opacity-90 transition-opacity"
            >
              {telegramTest?.status === "testing" ? "Testing..." : "Test Connection"}
            </button>
            {telegramTest?.status === "ok" && (
              <span className="text-xs text-[var(--color-success)] font-medium">Connected</span>
            )}
            {telegramTest?.status === "error" && (
              <span className="text-xs text-[var(--color-critical)] font-medium">
                {telegramTest.message || "Failed"}
              </span>
            )}
          </div>
        </div>
      </Card>
    </div>
  )
}

function RangeField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  const display = step < 1 ? value.toFixed(2) : value
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-[var(--color-text-primary)]">{label}</label>
        <span className="text-xs font-mono text-[var(--color-text-tertiary)]">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[var(--color-text-primary)]"
      />
    </div>
  )
}

function NumberField({
  label,
  value,
  min,
  onChange,
}: {
  label: string
  value: number
  min?: number
  onChange: (v: number) => void
}) {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium text-[var(--color-text-primary)]">{label}</label>
      <input
        type="number"
        value={value}
        min={min}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
      />
    </div>
  )
}
