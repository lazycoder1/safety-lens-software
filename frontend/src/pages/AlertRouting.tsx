import { useEffect, useMemo, useState } from "react"
import type { ElementType, ReactNode } from "react"
import {
  Bell,
  CheckCircle2,
  CircleOff,
  Clock,
  ExternalLink,
  Mail,
  Megaphone,
  MessageCircle,
  Phone,
  PlugZap,
  Radio,
  RefreshCw,
  Save,
  Send,
  Settings2,
  Smartphone,
  TestTube2,
  Volume2,
  Webhook,
  X,
  XCircle,
  Zap,
} from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn, timeAgo } from "@/lib/utils"
import { severityConfig } from "@/lib/constants"
import {
  getAlertOutputs,
  getCameras,
  testAlertOutput,
  testAlertRouting,
  updateAlertOutput,
} from "@/lib/api"
import { playP1AlertSound } from "@/lib/alertSound"
import type { AlertDeliveryResult, AlertOutput, AlertOutputType, Severity } from "@/types"

type Tab = "overview" | "mobile" | "email" | "bridges" | "audio" | "physical" | "escalation" | "templates"

const severities: Severity[] = ["P1", "P2", "P3", "P4"]

const tabs: Array<{ key: Tab; label: string; icon: ElementType }> = [
  { key: "overview", label: "Overview", icon: Bell },
  { key: "mobile", label: "Mobile", icon: Smartphone },
  { key: "email", label: "Email", icon: Mail },
  { key: "bridges", label: "Webhooks", icon: Webhook },
  { key: "audio", label: "Audio", icon: Volume2 },
  { key: "physical", label: "Physical", icon: PlugZap },
  { key: "escalation", label: "Escalation", icon: RefreshCw },
  { key: "templates", label: "Templates", icon: Settings2 },
]

const outputMeta: Record<AlertOutputType, { label: string; icon: ElementType; group: Tab; description: string }> = {
  in_app: { label: "In-App", icon: Bell, group: "overview", description: "Rakshak Lens browser alerts and WebSocket updates" },
  browser_sound: { label: "Browser Sound", icon: Volume2, group: "audio", description: "Local alarm tone in armed browser clients" },
  telegram: { label: "Telegram", icon: MessageCircle, group: "mobile", description: "Bot messages to a Telegram group" },
  email: { label: "Email", icon: Mail, group: "email", description: "SMTP or SendGrid alert emails" },
  webhook: { label: "Webhook", icon: Webhook, group: "bridges", description: "POST alert JSON to another system or bridge" },
  pushover: { label: "Pushover", icon: Smartphone, group: "mobile", description: "Loud iPhone push notifications for testing and escalation" },
  ip_speaker: { label: "Speaker", icon: Megaphone, group: "audio", description: "IP speaker, AudioRelay, or dry-run announcement output" },
  relay: { label: "Relay / Buzzer", icon: PlugZap, group: "physical", description: "Dry-run, HTTP relay, or MQTT buzzer output" },
  plc: { label: "PLC", icon: Radio, group: "physical", description: "PLC/Modbus channel for sirens, strobes, and plant controls" },
}

const statusStyles: Record<string, string> = {
  ready: "bg-emerald-50 text-emerald-700 border-emerald-200",
  needs_setup: "bg-amber-50 text-amber-700 border-amber-200",
  simulated: "bg-sky-50 text-sky-700 border-sky-200",
  failed: "bg-red-50 text-red-700 border-red-200",
  disabled: "bg-neutral-100 text-neutral-600 border-neutral-200",
  not_implemented: "bg-neutral-100 text-neutral-500 border-neutral-200",
}

function statusLabel(output: AlertOutput) {
  if (!output.enabled) return "Disabled"
  return output.status.replace("_", " ")
}

function outputHealth(output: AlertOutput) {
  if (!output.enabled) return "disabled"
  return output.status
}

function formatWhen(value?: string | null) {
  return value ? timeAgo(value) : "Never"
}

function parseCsv(value: string) {
  return value.split(",").map((item) => item.trim()).filter(Boolean)
}

function stringifyCsv(value: string[] | undefined) {
  return (value || []).join(", ")
}

function parseRecipientList(value: string) {
  return value.split(/[\s,;]+/).map((item) => item.trim()).filter(Boolean)
}

function stringifyRecipientList(value: string[] | undefined) {
  return (value || []).join("\n")
}

export function AlertRouting() {
  const [activeTab, setActiveTab] = useState<Tab>("overview")
  const [outputs, setOutputs] = useState<AlertOutput[]>([])
  const [zones, setZones] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [savingId, setSavingId] = useState<string | null>(null)
  const [testingId, setTestingId] = useState<string | null>(null)
  const [testAllRunning, setTestAllRunning] = useState(false)
  const [selectedOutput, setSelectedOutput] = useState<AlertOutput | null>(null)
  const [draft, setDraft] = useState<AlertOutput | null>(null)
  const [lastResults, setLastResults] = useState<AlertDeliveryResult[]>([])

  useEffect(() => {
    void load()
  }, [])

  async function load() {
    setLoading(true)
    try {
      const [nextOutputs, cams] = await Promise.all([
        getAlertOutputs(),
        getCameras().catch(() => []),
      ])
      setOutputs(nextOutputs)
      const nextZones = Array.from(new Set<string>((cams || []).map((cam: any) => String(cam.zone || "")).filter(Boolean))).sort()
      setZones(nextZones)
    } catch {
      toast.error("Could not load alert outputs")
    } finally {
      setLoading(false)
    }
  }

  const visibleOutputs = useMemo(() => {
    if (activeTab === "overview") return outputs
    return outputs.filter((output) => outputMeta[output.type]?.group === activeTab)
  }, [activeTab, outputs])

  const summary = useMemo(() => {
    const enabled = outputs.filter((output) => output.enabled)
    return {
      total: outputs.length,
      enabled: enabled.length,
      failed: outputs.filter((output) => output.enabled && output.status === "failed").length,
      simulated: outputs.filter((output) => output.enabled && output.status === "simulated").length,
      hardwareReady: outputs.filter((output) => output.enabled && ["ip_speaker", "relay", "plc"].includes(output.type) && output.status === "ready").length,
    }
  }, [outputs])

  function beginEdit(output: AlertOutput) {
    const copy = JSON.parse(JSON.stringify(output))
    setSelectedOutput(output)
    setDraft(copy)
  }

  function updateDraft(updates: Partial<AlertOutput>) {
    setDraft((prev) => prev ? { ...prev, ...updates } : prev)
  }

  function updateSettings(updates: Record<string, any>) {
    setDraft((prev) => prev ? { ...prev, settings: { ...prev.settings, ...updates } } : prev)
  }

  function toggleSeverity(severity: Severity) {
    if (!draft) return
    const current = new Set(draft.severities)
    current.has(severity) ? current.delete(severity) : current.add(severity)
    updateDraft({ severities: Array.from(current) as Severity[] })
  }

  async function saveDraft() {
    if (!draft) return
    setSavingId(draft.id)
    try {
      const saved = await updateAlertOutput(draft)
      setOutputs((prev) => prev.map((output) => output.id === saved.id ? saved : output))
      setSelectedOutput(saved)
      setDraft(saved)
      toast.success("Alert output saved")
    } catch {
      toast.error("Failed to save alert output")
    } finally {
      setSavingId(null)
    }
  }

  async function toggleEnabled(output: AlertOutput) {
    const next = { ...output, enabled: !output.enabled }
    setSavingId(output.id)
    try {
      const saved = await updateAlertOutput(next)
      setOutputs((prev) => prev.map((item) => item.id === saved.id ? saved : item))
      if (draft?.id === saved.id) setDraft(saved)
    } catch {
      toast.error("Failed to update output")
    } finally {
      setSavingId(null)
    }
  }

  async function runOutputTest(output: AlertOutput) {
    setTestingId(output.id)
    try {
      if (output.type === "browser_sound" || output.id === "audio_relay") {
        playP1AlertSound()
      }
      const response = await testAlertOutput(output.id)
      const result = response.result as AlertDeliveryResult
      setLastResults((prev) => [result, ...prev].slice(0, 12))
      await load()
      if (result.status === "failed") toast.error(result.message)
      else if (result.status === "simulated") toast.info(result.message)
      else toast.success(result.message)
    } catch {
      toast.error("Test failed")
    } finally {
      setTestingId(null)
    }
  }

  async function runP1Test() {
    setTestAllRunning(true)
    try {
      playP1AlertSound()
      const response = await testAlertRouting({
        severity: "P1",
        rule: "Factory Alert Test",
        cameraName: "Test Camera",
        zone: "Test Zone",
        outputIds: outputs.filter((output) => output.enabled).map((output) => output.id),
      })
      const results = (response.results || []) as AlertDeliveryResult[]
      setLastResults(results)
      await load()
      const failed = results.filter((result) => result.status === "failed")
      if (failed.length) toast.error(`${failed.length} output${failed.length === 1 ? "" : "s"} failed`)
      else toast.success(`Test sent to ${results.length} configured output${results.length === 1 ? "" : "s"}`)
    } catch {
      toast.error("P1 output test failed")
    } finally {
      setTestAllRunning(false)
    }
  }

  return (
    <div className="h-full overflow-auto bg-[var(--color-bg-secondary)]">
      <div className="max-w-7xl mx-auto p-6">
        <div className="flex items-start justify-between gap-4 mb-5">
          <div>
            <h1 className="text-xl font-bold text-[var(--color-text-primary)]">Alert Delivery</h1>
            <p className="text-sm text-[var(--color-text-secondary)] mt-1">
              Configure, test, and monitor every notification, speaker, buzzer, and bridge output.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="secondary" size="sm" onClick={load} disabled={loading}>
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
              Refresh
            </Button>
            <Button size="sm" onClick={runP1Test} disabled={testAllRunning}>
              {testAllRunning ? <RefreshCw size={14} className="animate-spin" /> : <TestTube2 size={14} />}
              Test P1 Outputs
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 mb-5">
          <SummaryTile label="Outputs" value={summary.total} icon={Bell} />
          <SummaryTile label="Enabled" value={summary.enabled} icon={CheckCircle2} />
          <SummaryTile label="Simulated" value={summary.simulated} icon={TestTube2} />
          <SummaryTile label="Hardware Ready" value={summary.hardwareReady} icon={PlugZap} />
          <SummaryTile label="Failed" value={summary.failed} icon={XCircle} danger={summary.failed > 0} />
        </div>

        <div className="flex items-center gap-1 border-b mb-5 overflow-x-auto">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={cn(
                  "flex items-center gap-1.5 px-3 py-2.5 text-sm font-medium border-b-2 transition-colors -mb-px cursor-pointer whitespace-nowrap",
                  activeTab === tab.key
                    ? "border-[var(--color-text-primary)] text-[var(--color-text-primary)]"
                    : "border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
                )}
              >
                <Icon size={14} />
                {tab.label}
              </button>
            )
          })}
        </div>

        {activeTab === "escalation" ? (
          <PlaceholderPanel
            icon={RefreshCw}
            title="Escalation"
            text="Escalation still uses the existing saved chain. Output delivery now provides the channels that escalation can target."
          />
        ) : activeTab === "templates" ? (
          <PlaceholderPanel
            icon={Settings2}
            title="Templates"
            text="Message templates are applied by each output adapter. Email, speaker, webhook, and Pushover outputs expose their main message fields in Configure."
          />
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-[1fr_360px] gap-5">
            <div className="space-y-3">
              {loading ? (
                <Card className="p-6 text-sm text-[var(--color-text-secondary)]">Loading alert outputs...</Card>
              ) : visibleOutputs.length === 0 ? (
                <Card className="p-6 text-sm text-[var(--color-text-secondary)]">No outputs in this category.</Card>
              ) : (
                visibleOutputs.map((output) => (
                  <OutputCard
                    key={output.id}
                    output={output}
                    selected={selectedOutput?.id === output.id}
                    saving={savingId === output.id}
                    testing={testingId === output.id}
                    onEdit={() => beginEdit(output)}
                    onToggle={() => toggleEnabled(output)}
                    onTest={() => runOutputTest(output)}
                  />
                ))
              )}
            </div>

            <div className="space-y-4">
              <LastResults results={lastResults} />
              <GuidanceCard />
            </div>
          </div>
        )}
      </div>

      {draft && (
        <ConfigPanel
          output={draft}
          zones={zones}
          saving={savingId === draft.id}
          onClose={() => { setDraft(null); setSelectedOutput(null) }}
          onSave={saveDraft}
          onChange={updateDraft}
          onSettingsChange={updateSettings}
          onToggleSeverity={toggleSeverity}
        />
      )}
    </div>
  )
}

function SummaryTile({ label, value, icon: Icon, danger = false }: { label: string; value: number; icon: ElementType; danger?: boolean }) {
  return (
    <Card className="p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs text-[var(--color-text-secondary)]">{label}</p>
          <p className={cn("text-2xl font-semibold mt-1", danger ? "text-red-600" : "text-[var(--color-text-primary)]")}>{value}</p>
        </div>
        <Icon size={18} className={danger ? "text-red-500" : "text-[var(--color-text-tertiary)]"} />
      </div>
    </Card>
  )
}

function OutputCard({
  output,
  selected,
  saving,
  testing,
  onEdit,
  onToggle,
  onTest,
}: {
  output: AlertOutput
  selected: boolean
  saving: boolean
  testing: boolean
  onEdit: () => void
  onToggle: () => void
  onTest: () => void
}) {
  const meta = outputMeta[output.type]
  const Icon = meta?.icon || Bell
  const health = outputHealth(output)
  return (
    <Card className={cn("p-0 overflow-hidden", selected && "ring-2 ring-[var(--color-info)]")}>
      <div className="p-4 flex items-start gap-4">
        <div className="w-10 h-10 rounded-[var(--radius-md)] bg-[var(--color-bg-secondary)] border flex items-center justify-center shrink-0">
          <Icon size={18} className="text-[var(--color-text-secondary)]" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">{output.name}</h3>
            <span className={cn("inline-flex items-center px-2 py-0.5 rounded-full border text-[10px] font-medium capitalize", statusStyles[health] || statusStyles.disabled)}>
              {statusLabel(output)}
            </span>
            <Badge variant="default" className="text-[10px]">{output.mode}</Badge>
          </div>
          <p className="text-xs text-[var(--color-text-secondary)] mt-1">{meta?.description}</p>
          <div className="flex flex-wrap items-center gap-2 mt-3">
            {output.severities.map((sev) => {
              const cfg = severityConfig[sev]
              return (
                <span key={sev} className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border" style={{ backgroundColor: cfg.bg, color: cfg.textColor }}>
                  <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: cfg.color }} />
                  {sev}
                </span>
              )
            })}
            <span className="text-[11px] text-[var(--color-text-tertiary)]">
              Zones: {output.zones.length ? output.zones.join(", ") : "All"}
            </span>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 text-[11px] text-[var(--color-text-tertiary)]">
            <span>Last test: {formatWhen(output.lastTestAt)}</span>
            <span>Last fired: {formatWhen(output.lastFiredAt)}</span>
            {output.lastError && <span className="text-red-600">Error: {output.lastError}</span>}
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button variant="secondary" size="sm" onClick={onTest} disabled={testing || !output.enabled || output.status === "not_implemented"}>
            {testing ? <RefreshCw size={14} className="animate-spin" /> : <Zap size={14} />}
            Test
          </Button>
          <Button variant="secondary" size="sm" onClick={onEdit}>
            <Settings2 size={14} />
            Configure
          </Button>
          <button
            onClick={onToggle}
            disabled={saving || output.status === "not_implemented"}
            className={cn(
              "relative inline-flex h-6 w-11 items-center rounded-full transition-colors cursor-pointer disabled:cursor-not-allowed disabled:opacity-50",
              output.enabled ? "bg-[var(--color-success)]" : "bg-[var(--color-bg-tertiary)] border"
            )}
            title={output.enabled ? "Disable output" : "Enable output"}
          >
            <span className={cn("inline-block h-4 w-4 rounded-full bg-white shadow transition-transform", output.enabled ? "translate-x-6" : "translate-x-1")} />
          </button>
        </div>
      </div>
    </Card>
  )
}

function ConfigPanel({
  output,
  zones,
  saving,
  onClose,
  onSave,
  onChange,
  onSettingsChange,
  onToggleSeverity,
}: {
  output: AlertOutput
  zones: string[]
  saving: boolean
  onClose: () => void
  onSave: () => void
  onChange: (updates: Partial<AlertOutput>) => void
  onSettingsChange: (updates: Record<string, any>) => void
  onToggleSeverity: (severity: Severity) => void
}) {
  const meta = outputMeta[output.type]
  const Icon = meta?.icon || Bell
  const settings = output.settings || {}

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/30" onClick={onClose} />
      <div className="relative w-full max-w-xl bg-white shadow-xl h-full overflow-y-auto">
        <div className="sticky top-0 bg-white border-b px-5 py-4 flex items-center justify-between z-10">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-[var(--radius-md)] bg-[var(--color-bg-secondary)] border flex items-center justify-center">
              <Icon size={17} />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">Configure {output.name}</h2>
              <p className="text-xs text-[var(--color-text-secondary)]">{meta?.description}</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1 rounded hover:bg-[var(--color-bg-tertiary)] cursor-pointer">
            <X size={16} />
          </button>
        </div>

        <div className="p-5 space-y-5">
          <Section title="Delivery">
            <Field label="Name">
              <input value={output.name} onChange={(e) => onChange({ name: e.target.value })} className="input" />
            </Field>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Mode">
                <ModeSelect output={output} onChange={(mode) => onChange({ mode })} />
              </Field>
              <Field label="Status">
                <select value={output.status} onChange={(e) => onChange({ status: e.target.value as any })} className="input">
                  <option value="ready">Ready</option>
                  <option value="needs_setup">Needs setup</option>
                  <option value="simulated">Simulated</option>
                  <option value="failed">Failed</option>
                  <option value="not_implemented">Not implemented</option>
                </select>
              </Field>
            </div>
            <div>
              <label className="text-sm font-medium text-[var(--color-text-primary)]">Severities</label>
              <div className="flex flex-wrap gap-2 mt-2">
                {severities.map((sev) => {
                  const active = output.severities.includes(sev)
                  const cfg = severityConfig[sev]
                  return (
                    <button
                      key={sev}
                      onClick={() => onToggleSeverity(sev)}
                      className={cn("inline-flex items-center gap-1.5 px-3 py-1.5 rounded-[var(--radius-md)] text-xs font-medium border cursor-pointer", active ? "border-current" : "border-transparent bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]")}
                      style={active ? { backgroundColor: cfg.bg, color: cfg.textColor } : undefined}
                    >
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: cfg.color }} />
                      {sev}
                    </button>
                  )
                })}
              </div>
            </div>
            <Field label="Zones">
              <input
                value={stringifyCsv(output.zones)}
                onChange={(e) => onChange({ zones: parseCsv(e.target.value) })}
                placeholder={zones.length ? zones.join(", ") : "Blank means all zones"}
                className="input"
              />
            </Field>
          </Section>

          {output.type === "telegram" && (
            <Section title="Telegram Bot">
              <Field label="Bot Token"><input type="password" value={settings.bot_token || ""} onChange={(e) => onSettingsChange({ bot_token: e.target.value })} className="input" /></Field>
              <Field label="Chat ID"><input value={settings.chat_id || ""} onChange={(e) => onSettingsChange({ chat_id: e.target.value })} className="input" /></Field>
            </Section>
          )}

          {output.type === "email" && (
            <Section title="Email Provider">
              <Field label="Provider">
                <select
                  value={settings.provider || "smtp"}
                  onChange={(e) => {
                    onSettingsChange({ provider: e.target.value })
                    onChange({ mode: e.target.value })
                  }}
                  className="input"
                >
                  <option value="smtp">SMTP</option>
                  <option value="sendgrid">SendGrid</option>
                </select>
              </Field>
              {settings.provider === "sendgrid" ? (
                <>
                  <Field label="SendGrid API Key"><input type="password" value={settings.sendgrid_api_key || ""} onChange={(e) => onSettingsChange({ sendgrid_api_key: e.target.value })} placeholder="SG..." className="input" /></Field>
                  <Field label="Template ID (optional)"><input value={settings.sendgrid_template_id || ""} onChange={(e) => onSettingsChange({ sendgrid_template_id: e.target.value })} className="input" /></Field>
                </>
              ) : (
                <div className="grid grid-cols-2 gap-3">
                  <Field label="SMTP Host"><input value={settings.smtp_host || ""} onChange={(e) => onSettingsChange({ smtp_host: e.target.value })} placeholder="smtp.gmail.com" className="input" /></Field>
                  <Field label="SMTP Port"><input type="number" value={settings.smtp_port || 587} onChange={(e) => onSettingsChange({ smtp_port: Number(e.target.value) })} className="input" /></Field>
                  <Field label="SMTP User"><input value={settings.smtp_user || ""} onChange={(e) => onSettingsChange({ smtp_user: e.target.value })} className="input" /></Field>
                  <Field label="SMTP Password"><input type="password" value={settings.smtp_pass || ""} onChange={(e) => onSettingsChange({ smtp_pass: e.target.value })} className="input" /></Field>
                </div>
              )}
              <div className="grid grid-cols-2 gap-3">
                <Field label="From Email"><input value={settings.from_address || ""} onChange={(e) => onSettingsChange({ from_address: e.target.value })} className="input" /></Field>
                <Field label="From Name"><input value={settings.from_name || ""} onChange={(e) => onSettingsChange({ from_name: e.target.value })} className="input" /></Field>
              </div>
              <Field label="Recipients">
                <textarea
                  rows={4}
                  value={stringifyRecipientList(settings.to_addresses)}
                  onChange={(e) => onSettingsChange({ to_addresses: parseRecipientList(e.target.value) })}
                  placeholder={"safety@example.com\nmanager@example.com"}
                  className="input resize-y"
                />
                <p className="text-xs text-[var(--color-text-secondary)]">
                  Add one address per line, or paste a comma-separated list. {(settings.to_addresses || []).length} recipient{(settings.to_addresses || []).length === 1 ? "" : "s"} configured.
                </p>
              </Field>
            </Section>
          )}

          {output.type === "webhook" && (
            <Section title="Webhook">
              <Field label="URL"><input value={settings.url || ""} onChange={(e) => onSettingsChange({ url: e.target.value })} placeholder="http://localhost:9009/alert" className="input" /></Field>
              <Field label="Headers JSON"><textarea rows={4} value={JSON.stringify(settings.headers || {}, null, 2)} onChange={(e) => safeJson(e.target.value, (headers) => onSettingsChange({ headers }))} className="input font-mono" /></Field>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={!!settings.include_snapshot} onChange={(e) => onSettingsChange({ include_snapshot: e.target.checked })} />
                Include base64 snapshot
              </label>
            </Section>
          )}

          {output.type === "pushover" && (
            <Section title="Pushover">
              <Field label="App API Token"><input type="password" value={settings.app_token || ""} onChange={(e) => onSettingsChange({ app_token: e.target.value })} className="input" /></Field>
              <Field label="User / Group Key"><input type="password" value={settings.user_key || ""} onChange={(e) => onSettingsChange({ user_key: e.target.value })} className="input" /></Field>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Device (optional)"><input value={settings.device || ""} onChange={(e) => onSettingsChange({ device: e.target.value })} className="input" /></Field>
                <Field label="Sound"><input value={settings.sound || "siren"} onChange={(e) => onSettingsChange({ sound: e.target.value })} className="input" /></Field>
                <Field label="Priority"><input type="number" value={settings.priority ?? 1} onChange={(e) => onSettingsChange({ priority: Number(e.target.value) })} className="input" /></Field>
                <Field label="Emergency Retry (s)"><input type="number" value={settings.emergency_retry ?? 60} onChange={(e) => onSettingsChange({ emergency_retry: Number(e.target.value) })} className="input" /></Field>
              </div>
            </Section>
          )}

          {output.type === "ip_speaker" && (
            <Section title="Speaker Output">
              {output.mode === "http" && (
                <>
                  <Field label="Speaker URL"><input value={settings.url || ""} onChange={(e) => onSettingsChange({ url: e.target.value })} className="input" /></Field>
                  <Field label="Method"><select value={settings.method || "POST"} onChange={(e) => onSettingsChange({ method: e.target.value })} className="input"><option>POST</option><option>GET</option><option>PUT</option></select></Field>
                </>
              )}
              <Field label="Announcement"><textarea rows={3} value={settings.message || ""} onChange={(e) => onSettingsChange({ message: e.target.value })} placeholder="Safety alert in {zone}." className="input" /></Field>
            </Section>
          )}

          {output.type === "relay" && (
            <Section title="Relay / Buzzer">
              <Field label="Pulse Seconds"><input type="number" value={settings.pulseSeconds || 5} onChange={(e) => onSettingsChange({ pulseSeconds: Number(e.target.value) })} className="input" /></Field>
              {output.mode === "http" && <Field label="HTTP Relay URL"><input value={settings.url || ""} onChange={(e) => onSettingsChange({ url: e.target.value })} className="input" /></Field>}
              {output.mode === "mqtt" && <Field label="MQTT Topic"><input value={settings.mqtt_topic || ""} onChange={(e) => onSettingsChange({ mqtt_topic: e.target.value })} className="input" /></Field>}
            </Section>
          )}

          {output.type === "plc" && (
            <Section title="PLC / Modbus">
              <div className="grid grid-cols-2 gap-3">
                <Field label="Host"><input value={settings.host || ""} onChange={(e) => onSettingsChange({ host: e.target.value })} className="input" /></Field>
                <Field label="Port"><input type="number" value={settings.port || 502} onChange={(e) => onSettingsChange({ port: Number(e.target.value) })} className="input" /></Field>
                <Field label="Register"><input value={settings.register || ""} onChange={(e) => onSettingsChange({ register: e.target.value })} className="input" /></Field>
                <Field label="Coil"><input value={settings.coil || ""} onChange={(e) => onSettingsChange({ coil: e.target.value })} className="input" /></Field>
              </div>
            </Section>
          )}

          <div className="sticky bottom-0 bg-white border-t py-4 flex items-center justify-end gap-2">
            <Button variant="ghost" size="sm" onClick={onClose}>Cancel</Button>
            <Button size="sm" onClick={onSave} disabled={saving}>
              {saving ? <RefreshCw size={14} className="animate-spin" /> : <Save size={14} />}
              Save Output
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function ModeSelect({ output, onChange }: { output: AlertOutput; onChange: (mode: string) => void }) {
  const options: Record<AlertOutputType, string[]> = {
    in_app: ["websocket"],
    browser_sound: ["local_browser"],
    telegram: ["bot"],
    email: ["smtp", "sendgrid"],
    webhook: ["http"],
    pushover: ["mobile_push"],
    ip_speaker: ["audio_relay", "dry_run", "http", "sip", "multicast"],
    relay: ["dry_run", "http", "mqtt"],
    plc: ["modbus"],
  }
  return (
    <select value={output.mode} onChange={(e) => onChange(e.target.value)} className="input">
      {(options[output.type] || [output.mode]).map((mode) => <option key={mode} value={mode}>{mode}</option>)}
    </select>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Card className="p-4 space-y-3">
      <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">{title}</h3>
      {children}
    </Card>
  )
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block space-y-1.5">
      <span className="text-sm font-medium text-[var(--color-text-primary)]">{label}</span>
      {children}
    </label>
  )
}

function LastResults({ results }: { results: AlertDeliveryResult[] }) {
  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">Latest Test Results</h3>
        <Clock size={14} className="text-[var(--color-text-tertiary)]" />
      </div>
      {results.length === 0 ? (
        <p className="text-xs text-[var(--color-text-secondary)]">Run a single output test or a P1 test to see delivery status here.</p>
      ) : (
        <div className="space-y-2">
          {results.map((result) => (
            <div key={`${result.outputId}-${result.timestamp}`} className="border rounded-[var(--radius-md)] p-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-medium text-[var(--color-text-primary)]">{result.outputName}</span>
                <ResultBadge status={result.status} />
              </div>
              <p className="text-[11px] text-[var(--color-text-secondary)] mt-1">{result.message}</p>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}

function ResultBadge({ status }: { status: string }) {
  const icon = status === "failed" ? XCircle : status === "skipped" ? CircleOff : status === "simulated" ? TestTube2 : CheckCircle2
  const Icon = icon
  const cls = status === "failed" ? "text-red-700 bg-red-50 border-red-200" : status === "simulated" ? "text-sky-700 bg-sky-50 border-sky-200" : status === "skipped" ? "text-neutral-600 bg-neutral-100 border-neutral-200" : "text-emerald-700 bg-emerald-50 border-emerald-200"
  return (
    <span className={cn("inline-flex items-center gap-1 border rounded-full px-2 py-0.5 text-[10px] capitalize", cls)}>
      <Icon size={10} />
      {status}
    </span>
  )
}

function GuidanceCard() {
  return (
    <Card className="p-4">
      <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-2">Setup Order</h3>
      <div className="space-y-2 text-xs text-[var(--color-text-secondary)]">
        <p><strong>1.</strong> Test Pushover and SendGrid for phone/email delivery.</p>
        <p><strong>2.</strong> Use webhook or dry-run relay to validate factory actions without hardware.</p>
        <p><strong>3.</strong> Move speakers and buzzers from simulated mode to HTTP/MQTT/SIP when hardware is selected.</p>
        <a href="https://pushover.net/api" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-[var(--color-info)] hover:underline">
          Pushover API <ExternalLink size={11} />
        </a>
      </div>
    </Card>
  )
}

function PlaceholderPanel({ icon: Icon, title, text }: { icon: ElementType; title: string; text: string }) {
  return (
    <Card className="p-8 text-center">
      <Icon size={24} className="mx-auto text-[var(--color-text-tertiary)] mb-3" />
      <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">{title}</h2>
      <p className="text-sm text-[var(--color-text-secondary)] mt-1 max-w-xl mx-auto">{text}</p>
    </Card>
  )
}

function safeJson(value: string, onParsed: (value: any) => void) {
  try {
    onParsed(JSON.parse(value || "{}"))
  } catch {
    // Keep typing forgiving; save/test will expose invalid state if left bad.
  }
}
