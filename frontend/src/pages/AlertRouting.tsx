import { useState, useEffect, useRef } from "react"
import {
  Bell,
  Send,
  Clock,
  ArrowUpRight,
  Plus,
  Trash2,
  X,
  CheckCircle2,
  XCircle,
  Loader2,
  Zap,
  Eye,
  EyeOff,
  Settings2,
  HelpCircle,
  RefreshCw,
  ExternalLink,
} from "lucide-react"
import { cameras, alertRoutingRules } from "@/data/mock"
import { severityConfig, severityVariantMap } from "@/lib/constants"
import type { Severity } from "@/types"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { toast } from "sonner"
import {
  getConfig,
  updateTelegramConfig,
  testTelegramConfig,
  fetchTelegramGroups,
  updateEmailConfig,
  testEmailConfig,
  updateWebhookConfig,
  testWebhookConfig,
  updateAlertRouting,
  testAlertRouting,
} from "@/lib/api"

/* ------------------------------------------------------------------ */
/*  Types & constants                                                  */
/* ------------------------------------------------------------------ */

type Tab = "channels" | "timeouts" | "escalation" | "templates"

type Channel = "inApp" | "telegram" | "email" | "webhook"

const channelLabels: Record<Channel, string> = {
  inApp: "In-App",
  telegram: "Telegram",
  email: "Email",
  webhook: "Webhook",
}

const channels: Channel[] = ["inApp", "telegram", "email", "webhook"]

const severities: Severity[] = ["P1", "P2", "P3", "P4"]

type ChannelMatrix = Record<Severity, Record<Channel, boolean>>

const defaultChannelMatrix: ChannelMatrix = {
  P1: { inApp: true, telegram: true, email: true, webhook: true },
  P2: { inApp: true, telegram: true, email: true, webhook: true },
  P3: { inApp: true, telegram: true, email: false, webhook: false },
  P4: { inApp: true, telegram: false, email: false, webhook: false },
}

function normalizeChannelMatrix(value: unknown): ChannelMatrix {
  const source = value && typeof value === "object"
    ? value as Record<string, unknown>
    : {}

  return severities.reduce((matrix, severity) => {
    const valueForSeverity = source[severity]
    const sourceRow = valueForSeverity && typeof valueForSeverity === "object"
      ? valueForSeverity as Record<string, unknown>
      : {}
    const row = { ...defaultChannelMatrix[severity] }

    for (const channel of channels) {
      if (typeof sourceRow[channel] === "boolean") {
        row[channel] = sourceRow[channel]
      }
    }
    if (typeof sourceRow.email !== "boolean" && typeof sourceRow.emailDigest === "boolean") {
      row.email = sourceRow.emailDigest
    }

    matrix[severity] = row
    return matrix
  }, {} as ChannelMatrix)
}

/* Timeouts */
interface TimeoutRow {
  category: string
  dedupWindow: number
  maxAlertsPerHr: number
  autoResolve: number
  toastDuration: number
}

const defaultTimeouts: TimeoutRow[] = [
  { category: "Fire/Smoke", dedupWindow: 0, maxAlertsPerHr: 999, autoResolve: 300, toastDuration: 0 },
  { category: "Person Fall", dedupWindow: 0, maxAlertsPerHr: 999, autoResolve: 300, toastDuration: 0 },
  { category: "Zone Intrusion", dedupWindow: 30, maxAlertsPerHr: 60, autoResolve: 120, toastDuration: 10 },
  { category: "No Helmet", dedupWindow: 60, maxAlertsPerHr: 30, autoResolve: 180, toastDuration: 8 },
  { category: "No Safety Vest", dedupWindow: 60, maxAlertsPerHr: 30, autoResolve: 180, toastDuration: 8 },
  { category: "Gangway Blocked", dedupWindow: 120, maxAlertsPerHr: 10, autoResolve: 600, toastDuration: 10 },
  { category: "Animal Detected", dedupWindow: 60, maxAlertsPerHr: 20, autoResolve: 300, toastDuration: 10 },
  { category: "Camera Offline", dedupWindow: 300, maxAlertsPerHr: 5, autoResolve: 0, toastDuration: 15 },
]

/* Escalation */
interface EscalationStep {
  id: number
  afterMinutes: number
  role: string
  channel: string
}

function normalizeEscalationChannel(channel: unknown): "telegram" | "email" | "webhook" {
  const normalized = String(channel ?? "").trim().toLowerCase()
  if (normalized === "email" || normalized === "webhook") return normalized
  return "telegram"
}

const defaultEscalation: EscalationStep[] = [
  { id: 1, afterMinutes: 3, role: "Floor Manager", channel: "telegram" },
  { id: 2, afterMinutes: 10, role: "Plant Manager", channel: "email" },
]

/* Test alert */
interface TestAlertConfig {
  severity: Severity
  cameraId: string
  ruleId: string
  channels: Record<Channel, boolean>
}

interface TestResult {
  channel: string
  success: boolean
  status?: string
  message?: string
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function AlertRouting() {
  const [activeTab, setActiveTab] = useState<Tab>("channels")
  const [channelMatrix, setChannelMatrix] = useState<ChannelMatrix>(defaultChannelMatrix)
  const [timeouts, setTimeouts] = useState<TimeoutRow[]>(defaultTimeouts)
  const [escalationSteps, setEscalationSteps] = useState<EscalationStep[]>(defaultEscalation)
  const [showTestModal, setShowTestModal] = useState(false)
  const [testConfig, setTestConfig] = useState<TestAlertConfig>({
    severity: "P2",
    cameraId: cameras[0].id,
    ruleId: alertRoutingRules[0].id,
    channels: { inApp: false, telegram: true, email: false, webhook: false },
  })
  const [testResults, setTestResults] = useState<TestResult[] | null>(null)
  const [testRunning, setTestRunning] = useState(false)

  /* Routing config dirty/save state */
  const [routingDirty, setRoutingDirty] = useState(false)
  const [routingSaving, setRoutingSaving] = useState(false)

  /* Email config */
  const [emEnabled, setEmEnabled] = useState(false)
  const [emHost, setEmHost] = useState("")
  const [emPort, setEmPort] = useState(587)
  const [emUser, setEmUser] = useState("")
  const [emPass, setEmPass] = useState("")
  const [emFrom, setEmFrom] = useState("")
  const [emTo, setEmTo] = useState("")
  const [emSeverities, setEmSeverities] = useState<string[]>(["P1", "P2"])
  const [emSaving, setEmSaving] = useState(false)
  const [emTesting, setEmTesting] = useState(false)
  const [emDirty, setEmDirty] = useState(false)

  /* Webhook config */
  const [whEnabled, setWhEnabled] = useState(false)
  const [whUrl, setWhUrl] = useState("")
  const [whHeaders, setWhHeaders] = useState("")
  const [whSeverities, setWhSeverities] = useState<string[]>(["P1", "P2"])
  const [whIncludeSnapshot, setWhIncludeSnapshot] = useState(false)
  const [whSaving, setWhSaving] = useState(false)
  const [whTesting, setWhTesting] = useState(false)
  const [whDirty, setWhDirty] = useState(false)

  /* Telegram config */
  const [tgEnabled, setTgEnabled] = useState(false)
  const [tgBotToken, setTgBotToken] = useState("")
  const [tgChatId, setTgChatId] = useState("")
  const [tgSeverities, setTgSeverities] = useState<string[]>(["P1", "P2"])
  const [tgShowToken, setTgShowToken] = useState(false)
  const [tgSaving, setTgSaving] = useState(false)
  const [tgTesting, setTgTesting] = useState(false)
  const [tgDirty, setTgDirty] = useState(false)
  const tgLoaded = useRef(false)
  const [tgGroups, setTgGroups] = useState<{ chat_id: string; title: string; type: string }[]>([])
  const [tgFetchingGroups, setTgFetchingGroups] = useState(false)
  const [showTgGuide, setShowTgGuide] = useState(false)

  /* Load all routing config from backend */
  useEffect(() => {
    getConfig()
      .then((cfg) => {
        const tg = cfg.telegram || {}
        setTgEnabled(tg.enabled ?? false)
        setTgBotToken(tg.bot_token ?? "")
        setTgChatId(tg.chat_id ?? "")
        setTgSeverities(tg.severities ?? ["P1", "P2"])
        tgLoaded.current = true

        // Email config
        const em = cfg.email || {}
        setEmEnabled(em.enabled ?? false)
        setEmHost(em.smtp_host ?? "")
        setEmPort(em.smtp_port ?? 587)
        setEmUser(em.smtp_user ?? "")
        setEmPass(em.smtp_pass ?? "")
        setEmFrom(em.from_address ?? "")
        setEmTo((em.to_addresses ?? []).join(", "))
        setEmSeverities(em.severities ?? ["P1", "P2"])

        // Webhook config
        const wh = cfg.webhook || {}
        setWhEnabled(wh.enabled ?? false)
        setWhUrl(wh.url ?? "")
        setWhHeaders(wh.headers ? JSON.stringify(wh.headers, null, 2) : "")
        setWhSeverities(wh.severities ?? ["P1", "P2"])
        setWhIncludeSnapshot(wh.include_snapshot ?? false)

        // Alert routing
        const ar = cfg.alert_routing || {}
        if (ar.channel_matrix) setChannelMatrix(normalizeChannelMatrix(ar.channel_matrix))
        if (ar.timeouts) {
          // Convert dict-based timeouts to array for the UI
          const arr = Object.entries(ar.timeouts).map(([category, vals]: [string, any]) => ({
            category,
            ...vals,
          }))
          if (arr.length > 0) setTimeouts(arr)
        }
        if (Array.isArray(ar.escalation_steps)) {
          setEscalationSteps(
            ar.escalation_steps.map((step: EscalationStep) => ({
              ...step,
              channel: normalizeEscalationChannel(step.channel),
            }))
          )
        }
      })
      .catch(() => {
        /* config endpoint unavailable — keep defaults */
      })
  }, [])

  function updateTgField<T>(setter: (v: T) => void) {
    return (value: T) => {
      setter(value)
      setTgDirty(true)
    }
  }

  function toggleTgSeverity(sev: string) {
    setTgSeverities((prev) =>
      prev.includes(sev) ? prev.filter((s) => s !== sev) : [...prev, sev]
    )
    setTgDirty(true)
  }

  async function saveTelegramConfig() {
    setTgSaving(true)
    try {
      await updateTelegramConfig({
        enabled: tgEnabled,
        bot_token: tgBotToken,
        chat_id: tgChatId,
        severities: tgSeverities,
      })
      setTgDirty(false)
      toast.success("Telegram configuration saved")
    } catch {
      toast.error("Failed to save Telegram configuration")
    } finally {
      setTgSaving(false)
    }
  }

  async function handleTestTelegram() {
    if (!tgBotToken || !tgChatId) {
      toast.error("Enter Bot Token and Chat ID before testing")
      return
    }
    // Save first if dirty so the backend uses the latest credentials
    if (tgDirty) {
      await saveTelegramConfig()
    }
    setTgTesting(true)
    try {
      const result = await testTelegramConfig()
      if (result.ok) {
        toast.success("Test message sent — check your Telegram")
      } else {
        toast.error(`Telegram test failed: ${result.error}`)
      }
    } catch {
      toast.error("Could not reach the server")
    } finally {
      setTgTesting(false)
    }
  }

  async function handleFetchGroups() {
    if (!tgBotToken) {
      toast.error("Enter a Bot Token first")
      return
    }
    setTgFetchingGroups(true)
    try {
      const result = await fetchTelegramGroups(tgBotToken)
      if (result.ok) {
        setTgGroups(result.groups)
        if (result.groups.length === 0) {
          toast("No groups found — make sure the bot is added to a group and someone has sent a message", { duration: 5000 })
        } else {
          toast.success(`Found ${result.groups.length} group${result.groups.length > 1 ? "s" : ""}`)
        }
      } else {
        toast.error(`Failed to fetch groups: ${result.error}`)
      }
    } catch {
      toast.error("Could not reach the server")
    } finally {
      setTgFetchingGroups(false)
    }
  }

  /* Channel matrix */
  function toggleChannel(severity: Severity, channel: Channel) {
    setChannelMatrix((prev) => ({
      ...prev,
      [severity]: {
        ...prev[severity],
        [channel]: !prev[severity][channel],
      },
    }))
    setRoutingDirty(true)
  }

  /* Timeouts */
  function updateTimeout(index: number, field: keyof TimeoutRow, value: number) {
    setTimeouts((prev) =>
      prev.map((row, i) => (i === index ? { ...row, [field]: value } : row))
    )
    setRoutingDirty(true)
  }

  /* Escalation */
  function addEscalationStep() {
    const newId = Math.max(0, ...escalationSteps.map((s) => s.id)) + 1
    const lastStep = escalationSteps[escalationSteps.length - 1]
    setEscalationSteps((prev) => [
      ...prev,
      {
        id: newId,
        afterMinutes: lastStep ? lastStep.afterMinutes + 10 : 5,
        role: "Safety Officer",
        channel: "telegram",
      },
    ])
    setRoutingDirty(true)
  }

  function removeEscalationStep(id: number) {
    setEscalationSteps((prev) => prev.filter((s) => s.id !== id))
    setRoutingDirty(true)
  }

  function updateEscalationStep(id: number, field: keyof EscalationStep, value: string | number) {
    setEscalationSteps((prev) =>
      prev.map((s) => (s.id === id ? { ...s, [field]: value } : s))
    )
    setRoutingDirty(true)
  }

  /* Save routing config (channel matrix, timeouts, escalation) */
  async function saveRoutingConfig() {
    setRoutingSaving(true)
    try {
      // Convert timeouts array to dict keyed by category
      const timeoutsDict: Record<string, any> = {}
      for (const t of timeouts) {
        timeoutsDict[t.category] = {
          dedupWindow: t.dedupWindow,
          maxAlertsPerHr: t.maxAlertsPerHr,
          autoResolve: t.autoResolve,
          toastDuration: t.toastDuration,
        }
      }
      await updateAlertRouting({
        channel_matrix: channelMatrix,
        timeouts: timeoutsDict,
        escalation_steps: escalationSteps,
      })
      setRoutingDirty(false)
      toast.success("Alert routing configuration saved")
    } catch {
      toast.error("Failed to save alert routing configuration")
    } finally {
      setRoutingSaving(false)
    }
  }

  /* Save email config */
  async function saveEmailConfig() {
    setEmSaving(true)
    try {
      await updateEmailConfig({
        enabled: emEnabled,
        smtp_host: emHost,
        smtp_port: emPort,
        smtp_user: emUser,
        smtp_pass: emPass,
        from_address: emFrom,
        to_addresses: emTo.split(",").map((s) => s.trim()).filter(Boolean),
        severities: emSeverities,
      })
      setEmDirty(false)
      toast.success("Email configuration saved")
    } catch {
      toast.error("Failed to save email configuration")
    } finally {
      setEmSaving(false)
    }
  }

  async function handleTestEmail() {
    if (!emHost || !emFrom || !emTo) {
      toast.error("Configure SMTP host, from address, and recipients first")
      return
    }
    if (emDirty) await saveEmailConfig()
    setEmTesting(true)
    try {
      const result = await testEmailConfig()
      if (result.ok) {
        toast.success("Test email sent — check your inbox")
      } else {
        toast.error(`Email test failed: ${result.error}`)
      }
    } catch {
      toast.error("Could not reach the server")
    } finally {
      setEmTesting(false)
    }
  }

  /* Save webhook config */
  async function saveWebhookConfig() {
    setWhSaving(true)
    try {
      let headers = {}
      if (whHeaders.trim()) {
        try {
          headers = JSON.parse(whHeaders)
        } catch {
          toast.error("Webhook headers must be valid JSON")
          setWhSaving(false)
          return
        }
      }
      await updateWebhookConfig({
        enabled: whEnabled,
        url: whUrl,
        headers,
        severities: whSeverities,
        include_snapshot: whIncludeSnapshot,
      })
      setWhDirty(false)
      toast.success("Webhook configuration saved")
    } catch {
      toast.error("Failed to save webhook configuration")
    } finally {
      setWhSaving(false)
    }
  }

  async function handleTestWebhook() {
    if (!whUrl) {
      toast.error("Configure a webhook URL first")
      return
    }
    if (whDirty) await saveWebhookConfig()
    setWhTesting(true)
    try {
      const result = await testWebhookConfig()
      if (result.ok) {
        toast.success("Test payload sent to webhook")
      } else {
        toast.error(`Webhook test failed: ${result.error}`)
      }
    } catch {
      toast.error("Could not reach the server")
    } finally {
      setWhTesting(false)
    }
  }

  /* Test alert */
  function toggleTestChannel(channel: Channel) {
    setTestConfig((prev) => ({
      ...prev,
      channels: { ...prev.channels, [channel]: !prev.channels[channel] },
    }))
  }

  async function fireTestAlert() {
    setTestRunning(true)
    setTestResults(null)

    const selectedChannels = (Object.entries(testConfig.channels) as [Channel, boolean][])
      .filter(([, v]) => v)
      .map(([k]) => k)

    try {
      const result = await testAlertRouting({
        severity: testConfig.severity,
        rule: "Test Alert",
        cameraName: "Test Camera",
        zone: "Test Zone",
        channels: selectedChannels,
      })
      const results: TestResult[] = Array.isArray(result.results)
        ? result.results.map((entry: TestResult) => ({
            channel: channelLabels[entry.channel as Channel] ?? entry.channel,
            success: Boolean(entry.success),
            status: entry.status,
            message: entry.message,
          }))
        : []
      setTestResults(results)

      if (result.ok) {
        toast.success("Test alert delivered to every selected channel")
      } else {
        toast.error("One or more selected channels did not accept the test alert")
      }
    } catch {
      toast.error("Could not reach the server")
    } finally {
      setTestRunning(false)
    }
  }

  const tabs: { key: Tab; label: string; icon: React.ElementType }[] = [
    { key: "channels", label: "Channels", icon: Send },
    { key: "timeouts", label: "Timeouts", icon: Clock },
    { key: "escalation", label: "Escalation", icon: ArrowUpRight },
    { key: "templates", label: "Templates", icon: Settings2 },
  ]

  return (
    <div className="h-full overflow-auto bg-[var(--color-bg-secondary)]">
      <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-xl font-bold text-[var(--color-text-primary)]">
              Alert Routing
            </h1>
            <p className="text-sm text-[var(--color-text-secondary)] mt-0.5">
              Configure how alerts are delivered, deduplicated, and escalated
            </p>
          </div>
          <Button size="sm" onClick={() => setShowTestModal(true)}>
            <Zap size={14} />
            Test Alert
          </Button>
        </div>

        {/* Tab navigation */}
        <div className="flex items-center gap-1 border-b mb-6">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={cn(
                  "flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors -mb-px cursor-pointer",
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

        {/* ============================================================ */}
        {/*  CHANNELS TAB                                                 */}
        {/* ============================================================ */}
        {activeTab === "channels" && (
          <div className="space-y-6">
            <Card className="p-0 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-[var(--color-bg-secondary)]">
                      <th className="text-left px-4 py-3 text-xs font-medium text-[var(--color-text-secondary)]">
                        Severity
                      </th>
                      {channels.map((ch) => (
                        <th
                          key={ch}
                          className="text-center px-4 py-3 text-xs font-medium text-[var(--color-text-secondary)]"
                        >
                          {channelLabels[ch]}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {severities.map((sev) => {
                      const config = severityConfig[sev]
                      return (
                        <tr
                          key={sev}
                          className="border-b last:border-b-0 hover:bg-[var(--color-bg-secondary)] transition-colors"
                        >
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              <span
                                className="w-2.5 h-2.5 rounded-full shrink-0"
                                style={{ backgroundColor: config.color }}
                              />
                              <span className="font-medium text-[var(--color-text-primary)]">
                                {sev}
                              </span>
                              <span className="text-[var(--color-text-secondary)]">
                                {config.label}
                              </span>
                            </div>
                          </td>
                          {channels.map((ch) => (
                            <td key={ch} className="text-center px-4 py-3">
                              <label className="inline-flex items-center justify-center cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={channelMatrix[sev][ch]}
                                  onChange={() => toggleChannel(sev, ch)}
                                  className="w-4 h-4 rounded border-[var(--color-border-default)] text-[var(--color-info)] focus:ring-[var(--color-info)] focus:ring-offset-0 cursor-pointer"
                                />
                              </label>
                            </td>
                          ))}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
              <div className="flex items-center gap-3 px-4 py-3 border-t">
                <Button
                  size="sm"
                  onClick={saveRoutingConfig}
                  disabled={routingSaving || !routingDirty}
                >
                  {routingSaving ? (
                    <>
                      <Loader2 size={14} className="animate-spin" />
                      Saving...
                    </>
                  ) : (
                    "Save Channel Matrix"
                  )}
                </Button>
                {routingDirty && (
                  <span className="text-xs text-[var(--color-warning)]">Unsaved changes</span>
                )}
              </div>
            </Card>

            {/* Telegram Setup */}
            <Card>
              <div className="flex items-center gap-2 mb-4">
                <Settings2 size={16} className="text-[var(--color-text-secondary)]" />
                <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
                  Telegram Setup
                </h3>
                <span className="text-xs text-[var(--color-text-tertiary)]">
                  Connect a Telegram bot to receive alert notifications
                </span>
                <button
                  onClick={() => setShowTgGuide(true)}
                  className="ml-auto flex items-center gap-1 text-xs text-[var(--color-info)] hover:underline cursor-pointer"
                >
                  <HelpCircle size={14} />
                  How to set up
                </button>
              </div>

              <div className="space-y-4">
                {/* Enable toggle */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">
                      Enabled
                    </label>
                    <p className="text-xs text-[var(--color-text-tertiary)]">
                      Send alerts to Telegram when triggered
                    </p>
                  </div>
                  <button
                    onClick={() => {
                      setTgEnabled((v) => !v)
                      setTgDirty(true)
                    }}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors cursor-pointer ${
                      tgEnabled ? "bg-[var(--color-success)]" : "bg-[var(--color-bg-tertiary)] border"
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 rounded-full bg-white shadow transition-transform ${
                        tgEnabled ? "translate-x-6" : "translate-x-1"
                      }`}
                    />
                  </button>
                </div>

                {/* Bot Token */}
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-[var(--color-text-primary)]">
                    Bot Token
                  </label>
                  <p className="text-xs text-[var(--color-text-tertiary)]">
                    Create a bot via @BotFather on Telegram and paste the token here
                  </p>
                  <div className="relative">
                    <input
                      type={tgShowToken ? "text" : "password"}
                      value={tgBotToken}
                      onChange={(e) => updateTgField(setTgBotToken)(e.target.value)}
                      placeholder="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
                      className="w-full px-3 py-2 pr-10 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
                    />
                    <button
                      type="button"
                      onClick={() => setTgShowToken((v) => !v)}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] cursor-pointer"
                    >
                      {tgShowToken ? <EyeOff size={14} /> : <Eye size={14} />}
                    </button>
                  </div>
                </div>

                {/* Group / Chat ID */}
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-[var(--color-text-primary)]">
                    Group
                  </label>
                  <p className="text-xs text-[var(--color-text-tertiary)]">
                    Select the Telegram group where alerts will be sent
                  </p>
                  <div className="flex gap-2">
                    {tgGroups.length > 0 ? (
                      <select
                        value={tgChatId}
                        onChange={(e) => {
                          setTgChatId(e.target.value)
                          setTgDirty(true)
                        }}
                        className="flex-1 px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white text-[var(--color-text-primary)] focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 cursor-pointer"
                      >
                        <option value="">Select a group...</option>
                        {tgGroups.map((g) => (
                          <option key={g.chat_id} value={g.chat_id}>
                            {g.title}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="text"
                        value={tgChatId}
                        onChange={(e) => updateTgField(setTgChatId)(e.target.value)}
                        placeholder="Click Fetch Groups or enter ID manually"
                        className="flex-1 px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
                        readOnly={false}
                      />
                    )}
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={handleFetchGroups}
                      disabled={tgFetchingGroups || !tgBotToken}
                    >
                      {tgFetchingGroups ? (
                        <Loader2 size={14} className="animate-spin" />
                      ) : (
                        <RefreshCw size={14} />
                      )}
                      Fetch Groups
                    </Button>
                  </div>
                  {tgGroups.length > 0 && tgChatId && (
                    <p className="text-xs text-[var(--color-text-tertiary)]">
                      Chat ID: {tgChatId}
                    </p>
                  )}
                </div>

                {/* Severity filter */}
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-[var(--color-text-primary)]">
                    Alert Severities
                  </label>
                  <p className="text-xs text-[var(--color-text-tertiary)]">
                    Only send alerts matching these severity levels
                  </p>
                  <div className="flex gap-2">
                    {severities.map((sev) => {
                      const sevCfg = severityConfig[sev]
                      const active = tgSeverities.includes(sev)
                      return (
                        <button
                          key={sev}
                          onClick={() => toggleTgSeverity(sev)}
                          className={cn(
                            "flex items-center gap-1.5 px-3 py-1.5 rounded-[var(--radius-md)] text-xs font-medium transition-colors cursor-pointer border",
                            active
                              ? "border-current"
                              : "border-transparent bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]"
                          )}
                          style={
                            active
                              ? { backgroundColor: sevCfg.bg, color: sevCfg.textColor }
                              : undefined
                          }
                        >
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: sevCfg.color }}
                          />
                          {sev}
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-3 pt-2 border-t">
                  <Button
                    size="sm"
                    onClick={saveTelegramConfig}
                    disabled={tgSaving || !tgDirty}
                  >
                    {tgSaving ? (
                      <>
                        <Loader2 size={14} className="animate-spin" />
                        Saving...
                      </>
                    ) : (
                      "Save"
                    )}
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={handleTestTelegram}
                    disabled={tgTesting || !tgBotToken || !tgChatId}
                  >
                    {tgTesting ? (
                      <>
                        <Loader2 size={14} className="animate-spin" />
                        Testing...
                      </>
                    ) : (
                      <>
                        <Send size={14} />
                        Test Connection
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </Card>

            {/* Email Setup */}
            <Card>
              <div className="flex items-center gap-2 mb-4">
                <Settings2 size={16} className="text-[var(--color-text-secondary)]" />
                <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
                  Email Setup
                </h3>
                <span className="text-xs text-[var(--color-text-tertiary)]">
                  Send alert emails via SMTP
                </span>
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">Enabled</label>
                    <p className="text-xs text-[var(--color-text-tertiary)]">Send alerts via email when triggered</p>
                  </div>
                  <button
                    onClick={() => { setEmEnabled((v) => !v); setEmDirty(true) }}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors cursor-pointer ${emEnabled ? "bg-[var(--color-success)]" : "bg-[var(--color-bg-tertiary)] border"}`}
                  >
                    <span className={`inline-block h-4 w-4 rounded-full bg-white shadow transition-transform ${emEnabled ? "translate-x-6" : "translate-x-1"}`} />
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">SMTP Host</label>
                    <input type="text" value={emHost} onChange={(e) => { setEmHost(e.target.value); setEmDirty(true) }} placeholder="smtp.gmail.com" className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">SMTP Port</label>
                    <input type="number" value={emPort} onChange={(e) => { setEmPort(Number(e.target.value)); setEmDirty(true) }} className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">SMTP User</label>
                    <input type="text" value={emUser} onChange={(e) => { setEmUser(e.target.value); setEmDirty(true) }} placeholder="user@example.com" className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">SMTP Password</label>
                    <input type="password" value={emPass} onChange={(e) => { setEmPass(e.target.value); setEmDirty(true) }} placeholder="App password" className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
                  </div>
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-[var(--color-text-primary)]">From Address</label>
                  <input type="email" value={emFrom} onChange={(e) => { setEmFrom(e.target.value); setEmDirty(true) }} placeholder="alerts@yourcompany.com" className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-[var(--color-text-primary)]">Recipients</label>
                  <input type="text" value={emTo} onChange={(e) => { setEmTo(e.target.value); setEmDirty(true) }} placeholder="safety@company.com, manager@company.com" className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
                  <p className="text-[11px] text-[var(--color-text-tertiary)]">Comma-separated email addresses</p>
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-[var(--color-text-primary)]">Alert Severities</label>
                  <div className="flex gap-2">
                    {severities.map((sev) => {
                      const sevCfg = severityConfig[sev]
                      const active = emSeverities.includes(sev)
                      return (
                        <button key={sev} onClick={() => { setEmSeverities((prev) => prev.includes(sev) ? prev.filter((s) => s !== sev) : [...prev, sev]); setEmDirty(true) }}
                          className={cn("flex items-center gap-1.5 px-3 py-1.5 rounded-[var(--radius-md)] text-xs font-medium transition-colors cursor-pointer border", active ? "border-current" : "border-transparent bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]")}
                          style={active ? { backgroundColor: sevCfg.bg, color: sevCfg.textColor } : undefined}
                        >
                          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: sevCfg.color }} />
                          {sev}
                        </button>
                      )
                    })}
                  </div>
                </div>
                <div className="flex items-center gap-3 pt-2 border-t">
                  <Button size="sm" onClick={saveEmailConfig} disabled={emSaving || !emDirty}>
                    {emSaving ? (<><Loader2 size={14} className="animate-spin" />Saving...</>) : "Save"}
                  </Button>
                  <Button variant="secondary" size="sm" onClick={handleTestEmail} disabled={emTesting || !emHost || !emFrom}>
                    {emTesting ? (<><Loader2 size={14} className="animate-spin" />Testing...</>) : (<><Send size={14} />Test Connection</>)}
                  </Button>
                </div>
              </div>
            </Card>

            {/* Webhook Setup */}
            <Card>
              <div className="flex items-center gap-2 mb-4">
                <Settings2 size={16} className="text-[var(--color-text-secondary)]" />
                <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
                  Webhook Setup
                </h3>
                <span className="text-xs text-[var(--color-text-tertiary)]">
                  POST alert JSON to an external endpoint
                </span>
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">Enabled</label>
                    <p className="text-xs text-[var(--color-text-tertiary)]">Send alert payloads to webhook URL</p>
                  </div>
                  <button
                    onClick={() => { setWhEnabled((v) => !v); setWhDirty(true) }}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors cursor-pointer ${whEnabled ? "bg-[var(--color-success)]" : "bg-[var(--color-bg-tertiary)] border"}`}
                  >
                    <span className={`inline-block h-4 w-4 rounded-full bg-white shadow transition-transform ${whEnabled ? "translate-x-6" : "translate-x-1"}`} />
                  </button>
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-[var(--color-text-primary)]">Webhook URL</label>
                  <input type="url" value={whUrl} onChange={(e) => { setWhUrl(e.target.value); setWhDirty(true) }} placeholder="https://hooks.example.com/safetylens" className="w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0" />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-[var(--color-text-primary)]">Headers (JSON)</label>
                  <textarea rows={3} value={whHeaders} onChange={(e) => { setWhHeaders(e.target.value); setWhDirty(true) }} placeholder='{"Authorization": "Bearer token"}' className="w-full px-3 py-2 text-sm font-mono rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 resize-y" />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-[var(--color-text-primary)]">Include Snapshot</label>
                    <p className="text-xs text-[var(--color-text-tertiary)]">Attach base64-encoded snapshot in payload</p>
                  </div>
                  <button
                    onClick={() => { setWhIncludeSnapshot((v) => !v); setWhDirty(true) }}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors cursor-pointer ${whIncludeSnapshot ? "bg-[var(--color-success)]" : "bg-[var(--color-bg-tertiary)] border"}`}
                  >
                    <span className={`inline-block h-4 w-4 rounded-full bg-white shadow transition-transform ${whIncludeSnapshot ? "translate-x-6" : "translate-x-1"}`} />
                  </button>
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-[var(--color-text-primary)]">Alert Severities</label>
                  <div className="flex gap-2">
                    {severities.map((sev) => {
                      const sevCfg = severityConfig[sev]
                      const active = whSeverities.includes(sev)
                      return (
                        <button key={sev} onClick={() => { setWhSeverities((prev) => prev.includes(sev) ? prev.filter((s) => s !== sev) : [...prev, sev]); setWhDirty(true) }}
                          className={cn("flex items-center gap-1.5 px-3 py-1.5 rounded-[var(--radius-md)] text-xs font-medium transition-colors cursor-pointer border", active ? "border-current" : "border-transparent bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]")}
                          style={active ? { backgroundColor: sevCfg.bg, color: sevCfg.textColor } : undefined}
                        >
                          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: sevCfg.color }} />
                          {sev}
                        </button>
                      )
                    })}
                  </div>
                </div>
                <div className="flex items-center gap-3 pt-2 border-t">
                  <Button size="sm" onClick={saveWebhookConfig} disabled={whSaving || !whDirty}>
                    {whSaving ? (<><Loader2 size={14} className="animate-spin" />Saving...</>) : "Save"}
                  </Button>
                  <Button variant="secondary" size="sm" onClick={handleTestWebhook} disabled={whTesting || !whUrl}>
                    {whTesting ? (<><Loader2 size={14} className="animate-spin" />Testing...</>) : (<><Send size={14} />Test Connection</>)}
                  </Button>
                </div>
              </div>
            </Card>
          </div>
        )}

        {/* ============================================================ */}
        {/*  TIMEOUTS TAB                                                 */}
        {/* ============================================================ */}
        {activeTab === "timeouts" && (
          <Card className="p-0 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-[var(--color-bg-secondary)]">
                    <th className="text-left px-4 py-3 text-xs font-medium text-[var(--color-text-secondary)]">
                      Rule Category
                    </th>
                    <th className="text-center px-4 py-3 text-xs font-medium text-[var(--color-text-secondary)]">
                      Dedup Window (s)
                    </th>
                    <th className="text-center px-4 py-3 text-xs font-medium text-[var(--color-text-secondary)]">
                      Max Alerts/Hr
                    </th>
                    <th className="text-center px-4 py-3 text-xs font-medium text-[var(--color-text-secondary)]">
                      Auto-Resolve (s)
                    </th>
                    <th className="text-center px-4 py-3 text-xs font-medium text-[var(--color-text-secondary)]">
                      Toast Duration (s)
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {timeouts.map((row, idx) => (
                    <tr
                      key={row.category}
                      className="border-b last:border-b-0 hover:bg-[var(--color-bg-secondary)] transition-colors"
                    >
                      <td className="px-4 py-3 font-medium text-[var(--color-text-primary)]">
                        {row.category}
                      </td>
                      <td className="px-4 py-3">
                        <input
                          type="number"
                          min={0}
                          value={row.dedupWindow}
                          onChange={(e) =>
                            updateTimeout(idx, "dedupWindow", Number(e.target.value))
                          }
                          className="w-20 mx-auto block text-center px-2 py-1 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent"
                        />
                      </td>
                      <td className="px-4 py-3">
                        <input
                          type="number"
                          min={0}
                          value={row.maxAlertsPerHr}
                          onChange={(e) =>
                            updateTimeout(idx, "maxAlertsPerHr", Number(e.target.value))
                          }
                          className="w-20 mx-auto block text-center px-2 py-1 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent"
                        />
                      </td>
                      <td className="px-4 py-3">
                        <input
                          type="number"
                          min={0}
                          value={row.autoResolve}
                          onChange={(e) =>
                            updateTimeout(idx, "autoResolve", Number(e.target.value))
                          }
                          className="w-20 mx-auto block text-center px-2 py-1 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent"
                        />
                      </td>
                      <td className="px-4 py-3">
                        <input
                          type="number"
                          min={0}
                          value={row.toastDuration}
                          onChange={(e) =>
                            updateTimeout(idx, "toastDuration", Number(e.target.value))
                          }
                          className="w-20 mx-auto block text-center px-2 py-1 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent"
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="flex items-center gap-3 px-4 py-3 border-t">
              <Button size="sm" onClick={saveRoutingConfig} disabled={routingSaving || !routingDirty}>
                {routingSaving ? (<><Loader2 size={14} className="animate-spin" />Saving...</>) : "Save Timeouts"}
              </Button>
              {routingDirty && (
                <span className="text-xs text-[var(--color-warning)]">Unsaved changes</span>
              )}
            </div>
          </Card>
        )}

        {/* ============================================================ */}
        {/*  ESCALATION TAB                                               */}
        {/* ============================================================ */}
        {activeTab === "escalation" && (
          <div className="space-y-4">
            <Card>
              <div className="flex items-center gap-2 mb-4">
                <ArrowUpRight size={16} className="text-[var(--color-text-secondary)]" />
                <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
                  Escalation Chain
                </h3>
                <span className="text-xs text-[var(--color-text-tertiary)]">
                  Define who gets notified if alerts remain unacknowledged
                </span>
              </div>

              <div className="space-y-3">
                {escalationSteps.map((step, idx) => (
                  <div
                    key={step.id}
                    className="flex items-center gap-3 p-3 bg-[var(--color-bg-secondary)] rounded-[var(--radius-md)]"
                  >
                    {/* Step number */}
                    <span className="flex items-center justify-center w-6 h-6 rounded-full bg-[var(--color-text-primary)] text-white text-xs font-bold shrink-0">
                      {idx + 1}
                    </span>

                    {/* "If unacknowledged after" */}
                    <span className="text-sm text-[var(--color-text-secondary)] shrink-0">
                      If unacknowledged after
                    </span>
                    <input
                      type="number"
                      min={1}
                      value={step.afterMinutes}
                      onChange={(e) =>
                        updateEscalationStep(step.id, "afterMinutes", Number(e.target.value))
                      }
                      className="w-16 text-center px-2 py-1 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent"
                    />
                    <span className="text-sm text-[var(--color-text-secondary)] shrink-0">
                      min, notify
                    </span>

                    {/* Role */}
                    <select
                      value={step.role}
                      onChange={(e) =>
                        updateEscalationStep(step.id, "role", e.target.value)
                      }
                      className="px-2 py-1 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent cursor-pointer"
                    >
                      <option value="Floor Manager">Floor Manager</option>
                      <option value="Plant Manager">Plant Manager</option>
                      <option value="Safety Officer">Safety Officer</option>
                    </select>

                    <span className="text-sm text-[var(--color-text-secondary)] shrink-0">
                      via
                    </span>

                    {/* Channel */}
                    <select
                      value={step.channel}
                      onChange={(e) =>
                        updateEscalationStep(step.id, "channel", e.target.value)
                      }
                      className="px-2 py-1 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent cursor-pointer"
                    >
                      <option value="telegram">Telegram</option>
                      <option value="email">Email</option>
                      <option value="webhook">Webhook</option>
                    </select>

                    {/* Remove */}
                    {escalationSteps.length > 1 && (
                      <button
                        onClick={() => removeEscalationStep(step.id)}
                        className="ml-auto text-[var(--color-text-tertiary)] hover:text-red-500 transition-colors cursor-pointer"
                        title="Remove step"
                      >
                        <Trash2 size={14} />
                      </button>
                    )}
                  </div>
                ))}
              </div>

              <div className="mt-4 flex items-center gap-3">
                <Button variant="secondary" size="sm" onClick={addEscalationStep}>
                  <Plus size={14} />
                  Add Step
                </Button>
                <Button size="sm" onClick={saveRoutingConfig} disabled={routingSaving || !routingDirty}>
                  {routingSaving ? (<><Loader2 size={14} className="animate-spin" />Saving...</>) : "Save Escalation"}
                </Button>
                {routingDirty && (
                  <span className="text-xs text-[var(--color-warning)]">Unsaved changes</span>
                )}
              </div>
            </Card>
          </div>
        )}

        {/* ============================================================ */}
        {/*  TEMPLATES TAB                                                */}
        {/* ============================================================ */}
        {activeTab === "templates" && (
          <TemplatesTab />
        )}
      </div>

      {/* ============================================================ */}
      {/*  TEST ALERT MODAL                                             */}
      {/* ============================================================ */}
      {showTestModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            onClick={() => {
              setShowTestModal(false)
              setTestResults(null)
            }}
          />

          {/* Modal */}
          <div className="relative bg-white rounded-[var(--radius-lg)] shadow-xl border w-full max-w-lg mx-4 max-h-[90vh] overflow-y-auto">
            {/* Modal header */}
            <div className="flex items-center justify-between px-5 py-4 border-b">
              <div className="flex items-center gap-2">
                <Bell size={16} className="text-[var(--color-text-secondary)]" />
                <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">
                  Send Test Alert
                </h2>
              </div>
              <button
                onClick={() => {
                  setShowTestModal(false)
                  setTestResults(null)
                }}
                className="text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] cursor-pointer"
              >
                <X size={16} />
              </button>
            </div>

            <div className="p-5 space-y-4">
              {/* Severity */}
              <div>
                <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1.5">
                  Severity
                </label>
                <div className="flex gap-2">
                  {severities.map((sev) => {
                    const config = severityConfig[sev]
                    return (
                      <button
                        key={sev}
                        onClick={() =>
                          setTestConfig((prev) => ({ ...prev, severity: sev }))
                        }
                        className={cn(
                          "flex items-center gap-1.5 px-3 py-1.5 rounded-[var(--radius-md)] text-xs font-medium transition-colors cursor-pointer border",
                          testConfig.severity === sev
                            ? "border-current"
                            : "border-transparent bg-[var(--color-bg-tertiary)]"
                        )}
                        style={
                          testConfig.severity === sev
                            ? { backgroundColor: config.bg, color: config.textColor }
                            : undefined
                        }
                      >
                        <span
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: config.color }}
                        />
                        {sev}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Camera */}
              <div>
                <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1.5">
                  Camera
                </label>
                <select
                  value={testConfig.cameraId}
                  onChange={(e) =>
                    setTestConfig((prev) => ({ ...prev, cameraId: e.target.value }))
                  }
                  className="w-full px-3 py-1.5 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent cursor-pointer"
                >
                  {cameras.map((cam) => (
                    <option key={cam.id} value={cam.id}>
                      {cam.name} ({cam.zone})
                    </option>
                  ))}
                </select>
              </div>

              {/* Rule */}
              <div>
                <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1.5">
                  Rule
                </label>
                <select
                  value={testConfig.ruleId}
                  onChange={(e) =>
                    setTestConfig((prev) => ({ ...prev, ruleId: e.target.value }))
                  }
                  className="w-full px-3 py-1.5 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent cursor-pointer"
                >
                  {alertRoutingRules.map((rule) => (
                    <option key={rule.id} value={rule.id}>
                      {rule.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Channels */}
              <div>
                <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1.5">
                  Channels
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {channels.map((ch) => (
                    <label
                      key={ch}
                      className={cn(
                        "flex items-center gap-2 px-3 py-2 rounded-[var(--radius-md)] border cursor-pointer transition-colors",
                        testConfig.channels[ch]
                          ? "border-[var(--color-info)] bg-[var(--color-info-bg)]"
                          : "border-[var(--color-border-default)] hover:bg-[var(--color-bg-secondary)]"
                      )}
                    >
                      <input
                        type="checkbox"
                        checked={testConfig.channels[ch]}
                        onChange={() => toggleTestChannel(ch)}
                        className="w-3.5 h-3.5 rounded border-[var(--color-border-default)] text-[var(--color-info)] focus:ring-[var(--color-info)] cursor-pointer"
                      />
                      <span className="text-xs font-medium text-[var(--color-text-primary)]">
                        {channelLabels[ch]}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Fire button */}
              <Button
                className="w-full"
                onClick={fireTestAlert}
                disabled={
                  testRunning ||
                  !Object.values(testConfig.channels).some(Boolean)
                }
              >
                {testRunning ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    Sending...
                  </>
                ) : (
                  <>
                    <Zap size={14} />
                    Fire Test Alert
                  </>
                )}
              </Button>

              {/* Results */}
              {testResults && (
                <div className="space-y-2 pt-2 border-t">
                  <p className="text-xs font-medium text-[var(--color-text-secondary)]">
                    Results
                  </p>
                  {testResults.map((result, index) => (
                    <div
                      key={`${result.channel}-${index}`}
                      className={cn(
                        "flex items-center justify-between px-3 py-2 rounded-[var(--radius-md)] text-xs font-medium",
                        result.success
                          ? "bg-[var(--color-success-bg)] text-[#065f46]"
                          : "bg-[var(--color-critical-bg)] text-[#991b1b]"
                      )}
                    >
                      <div className="min-w-0">
                        <div>{result.channel}</div>
                        {result.message && (
                          <p className="mt-0.5 text-[11px] font-normal opacity-80">
                            {result.message}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-1">
                        {result.success ? (
                          <>
                            <CheckCircle2 size={12} />
                            {result.status ?? "Delivered"}
                          </>
                        ) : (
                          <>
                            <XCircle size={12} />
                            {result.status ?? "Failed"}
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ============================================================ */}
      {/*  TELEGRAM SETUP GUIDE MODAL                                   */}
      {/* ============================================================ */}
      {showTgGuide && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            onClick={() => setShowTgGuide(false)}
          />
          <div className="relative bg-white rounded-[var(--radius-lg)] shadow-xl border w-full max-w-lg mx-4 max-h-[90vh] overflow-y-auto">
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b">
              <div className="flex items-center gap-2">
                <HelpCircle size={16} className="text-[var(--color-info)]" />
                <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">
                  Setting up Telegram Alerts
                </h2>
              </div>
              <button
                onClick={() => setShowTgGuide(false)}
                className="text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] cursor-pointer"
              >
                <X size={16} />
              </button>
            </div>

            <div className="p-5 space-y-5">
              {/* Step 1 */}
              <div className="flex gap-3">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-[var(--color-info)] text-white text-xs font-bold shrink-0">
                  1
                </span>
                <div>
                  <p className="text-sm font-medium text-[var(--color-text-primary)]">
                    Create a Telegram Bot
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                    Open Telegram, search for <strong>@BotFather</strong>, and
                    send <code className="px-1 py-0.5 bg-[var(--color-bg-tertiary)] rounded text-xs">/newbot</code>.
                    Follow the prompts to name your bot. BotFather will give you a <strong>Bot Token</strong>.
                  </p>
                </div>
              </div>

              {/* Step 2 */}
              <div className="flex gap-3">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-[var(--color-info)] text-white text-xs font-bold shrink-0">
                  2
                </span>
                <div>
                  <p className="text-sm font-medium text-[var(--color-text-primary)]">
                    Disable Group Privacy mode
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                    In @BotFather, send <code className="px-1 py-0.5 bg-[var(--color-bg-tertiary)] rounded text-xs">/mybots</code>,
                    select your bot, then go to <strong>Bot Settings</strong> → <strong>Group Privacy</strong> → <strong>Turn off</strong>.
                    This allows the bot to detect the group when you send a message.
                  </p>
                </div>
              </div>

              {/* Step 3 */}
              <div className="flex gap-3">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-[var(--color-info)] text-white text-xs font-bold shrink-0">
                  3
                </span>
                <div>
                  <p className="text-sm font-medium text-[var(--color-text-primary)]">
                    Add the bot to your group
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                    Create a Telegram group (or use an existing one) for your team to receive alerts.
                    Open group settings, tap <strong>Add Members</strong>, and search for your bot by its username.
                  </p>
                </div>
              </div>

              {/* Step 4 */}
              <div className="flex gap-3">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-[var(--color-info)] text-white text-xs font-bold shrink-0">
                  4
                </span>
                <div>
                  <p className="text-sm font-medium text-[var(--color-text-primary)]">
                    Send a message in the group
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                    Send any message in the group so the bot can detect it. This is needed for the
                    bot to discover the group.
                  </p>
                </div>
              </div>

              {/* Step 5 */}
              <div className="flex gap-3">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-[var(--color-info)] text-white text-xs font-bold shrink-0">
                  5
                </span>
                <div>
                  <p className="text-sm font-medium text-[var(--color-text-primary)]">
                    Configure in SafetyLens
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                    Paste the <strong>Bot Token</strong> above, click <strong>Fetch Groups</strong> to
                    find your group, select it from the dropdown, choose which alert severities to
                    forward, and hit <strong>Save</strong>.
                  </p>
                </div>
              </div>

              {/* Step 6 */}
              <div className="flex gap-3">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-[var(--color-info)] text-white text-xs font-bold shrink-0">
                  6
                </span>
                <div>
                  <p className="text-sm font-medium text-[var(--color-text-primary)]">
                    Test the connection
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                    Click <strong>Test Connection</strong> to send a test message to your group.
                    If it arrives, you're all set! Enable the toggle and alerts will flow to Telegram.
                  </p>
                </div>
              </div>

              {/* Tip */}
              <div className="p-3 bg-[var(--color-info-bg)] rounded-[var(--radius-md)]">
                <p className="text-xs text-[var(--color-text-secondary)]">
                  <strong>Tip:</strong> You can add the bot to multiple groups. Use
                  Fetch Groups to see all groups the bot belongs to, and pick the one
                  you want alerts sent to.
                </p>
              </div>

              <Button
                className="w-full"
                variant="secondary"
                onClick={() => setShowTgGuide(false)}
              >
                Got it
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}


/* ================================================================== */
/*  Templates Tab                                                      */
/* ================================================================== */

const TEMPLATE_VARIABLES = [
  { var: "{violation_type}", desc: "Type of violation (e.g. No Helmet)" },
  { var: "{severity}", desc: "Alert severity (P1-P4)" },
  { var: "{camera}", desc: "Camera name" },
  { var: "{zone}", desc: "Zone name" },
  { var: "{timestamp}", desc: "Detection timestamp" },
  { var: "{confidence}", desc: "Detection confidence %" },
  { var: "{alert_id}", desc: "Unique alert ID" },
  { var: "{alert_link}", desc: "Link to alert detail page" },
]

function TemplatesTab() {
  const [emailSubject, setEmailSubject] = useState(
    "[SafetyLens {severity}] {violation_type} detected at {zone}"
  )
  const [emailBody, setEmailBody] = useState(
    `A safety violation has been detected:\n\nViolation: {violation_type}\nSeverity: {severity}\nCamera: {camera}\nZone: {zone}\nTime: {timestamp}\nConfidence: {confidence}\n\nView details: {alert_link}\n\nThis is an automated alert from SafetyLens PPE Compliance System.`
  )
  const [telegramTemplate, setTelegramTemplate] = useState(
    `🚨 *{severity} Alert*\n{violation_type} at {zone}\nCamera: {camera}\nTime: {timestamp}\n[View Alert]({alert_link})`
  )
  const [saved, setSaved] = useState(false)

  async function handleSave() {
    try {
      await updateAlertRouting({
        templates: {
          email_subject: emailSubject,
          email_body: emailBody,
          telegram_template: telegramTemplate,
        },
      })
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
      toast.success("Templates saved")
    } catch {
      toast.error("Failed to save templates")
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <div className="flex items-center gap-2 mb-4">
          <Settings2 size={16} className="text-[var(--color-text-secondary)]" />
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
            Message Templates
          </h3>
          <p className="text-xs text-[var(--color-text-tertiary)] ml-auto">
            Customize alert messages sent via email and Telegram
          </p>
        </div>

        {/* Available variables */}
        <div className="mb-5 p-3 rounded-[var(--radius-md)] bg-[var(--color-bg-secondary)] border">
          <p className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">
            Available Variables
          </p>
          <div className="flex flex-wrap gap-1.5">
            {TEMPLATE_VARIABLES.map((v) => (
              <span
                key={v.var}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-white border text-xs font-mono text-[var(--color-text-primary)]"
                title={v.desc}
              >
                {v.var}
              </span>
            ))}
          </div>
        </div>

        <div className="space-y-5">
          {/* Email subject */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium text-[var(--color-text-primary)]">
              Email Subject
            </label>
            <input
              type="text"
              value={emailSubject}
              onChange={(e) => setEmailSubject(e.target.value)}
              className="w-full px-3 py-2 text-sm font-mono rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0"
            />
          </div>

          {/* Email body */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium text-[var(--color-text-primary)]">
              Email Body
            </label>
            <textarea
              rows={8}
              value={emailBody}
              onChange={(e) => setEmailBody(e.target.value)}
              className="w-full px-3 py-2 text-sm font-mono rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 resize-y"
            />
          </div>

          {/* Telegram template */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium text-[var(--color-text-primary)]">
              Telegram Message
            </label>
            <textarea
              rows={5}
              value={telegramTemplate}
              onChange={(e) => setTelegramTemplate(e.target.value)}
              className="w-full px-3 py-2 text-sm font-mono rounded-[var(--radius-md)] border bg-white focus:outline-2 focus:outline-[var(--color-info)] focus:outline-offset-0 resize-y"
            />
            <p className="text-[11px] text-[var(--color-text-tertiary)]">
              Supports Markdown formatting: *bold*, _italic_, [link](url)
            </p>
          </div>

          <div className="flex items-center gap-3 pt-2">
            <Button variant="primary" size="sm" onClick={handleSave}>
              Save Templates
            </Button>
            {saved && (
              <span className="text-xs text-[var(--color-success)] font-medium animate-pulse">
                Saved
              </span>
            )}
          </div>
        </div>
      </Card>
    </div>
  )
}
