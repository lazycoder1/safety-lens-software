import { useState } from "react"
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
} from "lucide-react"
import { cameras, detectionRules } from "@/data/mock"
import { severityConfig, severityVariantMap } from "@/lib/constants"
import type { Severity } from "@/types"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { toast } from "sonner"

/* ------------------------------------------------------------------ */
/*  Types & constants                                                  */
/* ------------------------------------------------------------------ */

type Tab = "channels" | "timeouts" | "escalation"

type Channel = "inApp" | "telegram" | "whatsapp" | "sms" | "plc" | "emailDigest"

const channelLabels: Record<Channel, string> = {
  inApp: "In-App",
  telegram: "Telegram",
  whatsapp: "WhatsApp",
  sms: "SMS",
  plc: "PLC",
  emailDigest: "Email Digest",
}

const channels: Channel[] = ["inApp", "telegram", "whatsapp", "sms", "plc", "emailDigest"]

const severities: Severity[] = ["P1", "P2", "P3", "P4"]

type ChannelMatrix = Record<Severity, Record<Channel, boolean>>

const defaultChannelMatrix: ChannelMatrix = {
  P1: { inApp: true, telegram: true, whatsapp: true, sms: true, plc: true, emailDigest: true },
  P2: { inApp: true, telegram: true, whatsapp: true, sms: false, plc: true, emailDigest: true },
  P3: { inApp: true, telegram: true, whatsapp: false, sms: false, plc: false, emailDigest: true },
  P4: { inApp: true, telegram: false, whatsapp: false, sms: false, plc: false, emailDigest: false },
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

const defaultEscalation: EscalationStep[] = [
  { id: 1, afterMinutes: 3, role: "Floor Manager", channel: "Telegram" },
  { id: 2, afterMinutes: 10, role: "Plant Manager", channel: "SMS" },
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
    ruleId: detectionRules[0].id,
    channels: { inApp: true, telegram: true, whatsapp: false, sms: false, plc: false, emailDigest: false },
  })
  const [testResults, setTestResults] = useState<TestResult[] | null>(null)
  const [testRunning, setTestRunning] = useState(false)

  /* Channel matrix */
  function toggleChannel(severity: Severity, channel: Channel) {
    setChannelMatrix((prev) => ({
      ...prev,
      [severity]: {
        ...prev[severity],
        [channel]: !prev[severity][channel],
      },
    }))
  }

  /* Timeouts */
  function updateTimeout(index: number, field: keyof TimeoutRow, value: number) {
    setTimeouts((prev) =>
      prev.map((row, i) => (i === index ? { ...row, [field]: value } : row))
    )
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
        channel: "WhatsApp",
      },
    ])
  }

  function removeEscalationStep(id: number) {
    setEscalationSteps((prev) => prev.filter((s) => s.id !== id))
  }

  function updateEscalationStep(id: number, field: keyof EscalationStep, value: string | number) {
    setEscalationSteps((prev) =>
      prev.map((s) => (s.id === id ? { ...s, [field]: value } : s))
    )
  }

  /* Test alert */
  function toggleTestChannel(channel: Channel) {
    setTestConfig((prev) => ({
      ...prev,
      channels: { ...prev.channels, [channel]: !prev.channels[channel] },
    }))
  }

  function fireTestAlert() {
    setTestRunning(true)
    setTestResults(null)

    const selectedChannels = (Object.entries(testConfig.channels) as [Channel, boolean][])
      .filter(([, v]) => v)
      .map(([k]) => k)

    // Simulate async test
    setTimeout(() => {
      const results: TestResult[] = selectedChannels.map((ch) => ({
        channel: channelLabels[ch],
        success: true, // simulated delivery for demo
      }))
      setTestResults(results)
      setTestRunning(false)

      const allOk = results.every((r) => r.success)
      if (allOk) {
        toast.success("Test alert delivered to all channels")
      } else {
        toast.error("Some channels failed delivery")
      }
    }, 1500)
  }

  const tabs: { key: Tab; label: string; icon: React.ElementType }[] = [
    { key: "channels", label: "Channels", icon: Send },
    { key: "timeouts", label: "Timeouts", icon: Clock },
    { key: "escalation", label: "Escalation", icon: ArrowUpRight },
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
          </Card>
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
                      <option value="Telegram">Telegram</option>
                      <option value="SMS">SMS</option>
                      <option value="WhatsApp">WhatsApp</option>
                      <option value="Email">Email</option>
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

              <div className="mt-4">
                <Button variant="secondary" size="sm" onClick={addEscalationStep}>
                  <Plus size={14} />
                  Add Step
                </Button>
              </div>
            </Card>
          </div>
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
                  {detectionRules.map((rule) => (
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
                  {testResults.map((result) => (
                    <div
                      key={result.channel}
                      className={cn(
                        "flex items-center justify-between px-3 py-2 rounded-[var(--radius-md)] text-xs font-medium",
                        result.success
                          ? "bg-[var(--color-success-bg)] text-[#065f46]"
                          : "bg-[var(--color-critical-bg)] text-[#991b1b]"
                      )}
                    >
                      <span>{result.channel}</span>
                      <div className="flex items-center gap-1">
                        {result.success ? (
                          <>
                            <CheckCircle2 size={12} />
                            Delivered
                          </>
                        ) : (
                          <>
                            <XCircle size={12} />
                            Failed
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
    </div>
  )
}
