import { useEffect, useMemo, useState } from "react"
import { Link, useNavigate } from "react-router-dom"
import { ArrowLeft, Camera, CheckCircle2, Loader2, Network, RefreshCw, SearchCheck, ShieldAlert } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { SearchInput } from "@/components/ui/SearchInput"
import { DetectionChecklist } from "@/components/cameras/DetectionChecklist"
import { Field } from "@/components/cameras/helpers"
import { PROFILE_DEFAULTS, PROFILE_OPTIONS } from "@/components/cameras/profileOptions"
import { getDetectionLabelsFromKeys, usesConfiguredZones, type CameraDetectionKey } from "@/components/cameras/detectionCatalog"
import { discoverCameras, getSafetyRules, importDiscoveredCameras, testDiscoveredCamera } from "@/lib/api"
import type {
  CameraProfile,
  DiscoveredCamera,
  DiscoveryImportResult,
  DiscoveryRowOverride,
  DiscoveryTestResult,
  SafetyRule,
} from "@/types"

type Phase = "scan" | "review" | "testing" | "importing" | "done"

const STEP_ORDER: Phase[] = ["scan", "review", "testing", "importing", "done"]
const STEP_LABELS: Record<Phase, string> = {
  scan: "Scan",
  review: "Review",
  testing: "Test",
  importing: "Import",
  done: "Done",
}

function defaultDisplayName(device: DiscoveredCamera) {
  if (device.name && device.name !== device.host) return device.name
  if (device.model) return `${device.model} ${device.ip}`
  return `Camera ${device.ip}`
}

export function CameraDiscoverPage() {
  const navigate = useNavigate()
  const [phase, setPhase] = useState<Phase>("scan")
  const [loadingRules, setLoadingRules] = useState(true)
  const [scanLoading, setScanLoading] = useState(false)
  const [cidrInput, setCidrInput] = useState("")
  const [showAdvancedScan, setShowAdvancedScan] = useState(false)
  const [warnings, setWarnings] = useState<string[]>([])
  const [search, setSearch] = useState("")
  const [safetyRules, setSafetyRules] = useState<SafetyRule[]>([])
  const [sharedProfile, setSharedProfile] = useState<CameraProfile>("general_safety")
  const [sharedCapabilities, setSharedCapabilities] = useState<CameraDetectionKey[]>(PROFILE_DEFAULTS.general_safety)
  const [sharedZone, setSharedZone] = useState("")
  const [sharedUsername, setSharedUsername] = useState("")
  const [sharedPassword, setSharedPassword] = useState("")
  const [devices, setDevices] = useState<DiscoveredCamera[]>([])
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [overridesById, setOverridesById] = useState<Record<string, DiscoveryRowOverride>>({})
  const [testResults, setTestResults] = useState<Record<string, DiscoveryTestResult>>({})
  const [importResult, setImportResult] = useState<DiscoveryImportResult | null>(null)

  useEffect(() => {
    let cancelled = false
    getSafetyRules()
      .then((rules) => {
        if (!cancelled) setSafetyRules(rules)
      })
      .catch(() => {
        if (!cancelled) toast.error("Failed to load safety rules")
      })
      .finally(() => {
        if (!cancelled) setLoadingRules(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const filteredDevices = useMemo(() => {
    const query = search.trim().toLowerCase()
    if (!query) return devices
    return devices.filter((device) =>
      [device.name, device.ip, device.vendor, device.model, device.host]
        .filter(Boolean)
        .some((value) => value.toLowerCase().includes(query))
    )
  }, [devices, search])

  const selectedDevices = useMemo(() => devices.filter((device) => selectedIds.includes(device.fingerprint)), [devices, selectedIds])

  const selectableFilteredFingerprints = useMemo(
    () => filteredDevices
      .filter((device) => device.duplicate_state !== "exact")
      .map((device) => device.fingerprint),
    [filteredDevices]
  )

  const allVisibleSelected = selectableFilteredFingerprints.length > 0 &&
    selectableFilteredFingerprints.every((fingerprint) => selectedIds.includes(fingerprint))

  const readyFingerprints = useMemo(
    () => selectedIds.filter((fingerprint) => testResults[fingerprint]?.ok),
    [selectedIds, testResults]
  )

  const missingRequiredSetupFingerprints = useMemo(
    () => readyFingerprints.filter((fingerprint) => {
      const override = overridesById[fingerprint]
      return !override?.displayName.trim() || !override?.zone.trim()
    }),
    [readyFingerprints, overridesById]
  )

  const exactDuplicateCount = useMemo(
    () => devices.filter((device) => device.duplicate_state === "exact").length,
    [devices]
  )

  const needsCredentialCount = useMemo(
    () => selectedIds.filter((fingerprint) => {
      const result = testResults[fingerprint]
      return result && !result.ok && result.error_code === "auth_failed"
    }).length,
    [selectedIds, testResults]
  )

  function initializeOverrides(nextDevices: DiscoveredCamera[]) {
    const nextOverrides: Record<string, DiscoveryRowOverride> = {}
    for (const device of nextDevices) {
      nextOverrides[device.fingerprint] = {
        displayName: defaultDisplayName(device),
        zone: sharedZone,
        profile: sharedProfile,
        capabilities: [...sharedCapabilities],
        preferredStream: (device.recommended_stream as DiscoveryRowOverride["preferredStream"]) || "main",
        streamPath: device.stream_path || "",
        credentialMode: "inherit",
        username: "",
        password: "",
      }
    }
    setOverridesById(nextOverrides)
  }

  async function handleScan() {
    setScanLoading(true)
    setPhase("scan")
    setImportResult(null)
    setTestResults({})
    try {
      const cidrs = cidrInput
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
      const result = await discoverCameras({
        cidrs: cidrs.length > 0 ? cidrs : undefined,
        timeout_seconds: 5,
      })
      const nextDevices = (result.devices || []) as DiscoveredCamera[]
      setDevices(nextDevices)
      setWarnings(result.warnings || [])
      setSelectedIds([])
      initializeOverrides(nextDevices)
      setPhase("review")
      if (nextDevices.length === 0) {
        toast.error("No cameras found on the scanned network")
      } else {
        toast.success(`Found ${nextDevices.length} camera candidate${nextDevices.length === 1 ? "" : "s"}`)
      }
    } catch (error: any) {
      toast.error(error.message || "Failed to scan the network")
    } finally {
      setScanLoading(false)
    }
  }

  function handleProfileChange(nextProfile: CameraProfile) {
    setSharedProfile(nextProfile)
    setSharedCapabilities(PROFILE_DEFAULTS[nextProfile])
  }

  function applySharedDefaults() {
    setOverridesById((current) => {
      const next = { ...current }
      for (const fingerprint of selectedIds) {
        const existing = next[fingerprint]
        if (!existing) continue
        next[fingerprint] = {
          ...existing,
          zone: sharedZone,
          profile: sharedProfile,
          capabilities: [...sharedCapabilities],
        }
      }
      return next
    })
    toast.success("Applied shared defaults to selected cameras")
  }

  function toggleSelection(fingerprint: string) {
    setSelectedIds((current) =>
      current.includes(fingerprint)
        ? current.filter((item) => item !== fingerprint)
        : [...current, fingerprint]
    )
  }

  function selectVisibleCameras() {
    setSelectedIds((current) => Array.from(new Set([...current, ...selectableFilteredFingerprints])))
  }

  function clearVisibleSelection() {
    setSelectedIds((current) => current.filter((fingerprint) => !selectableFilteredFingerprints.includes(fingerprint)))
  }

  function updateOverride(fingerprint: string, patch: Partial<DiscoveryRowOverride>) {
    setOverridesById((current) => ({
      ...current,
      [fingerprint]: {
        ...current[fingerprint],
        ...patch,
      },
    }))
  }

  async function handleTestSelected() {
    if (selectedDevices.length === 0) {
      toast.error("Select at least one camera to test")
      return
    }
    setPhase("testing")
    const nextResults: Record<string, DiscoveryTestResult> = {}
    await Promise.all(
      selectedDevices.map(async (device) => {
        const override = overridesById[device.fingerprint]
        const username = override.credentialMode === "override" ? override.username.trim() : sharedUsername.trim()
        const password = override.credentialMode === "override" ? override.password : sharedPassword
        try {
          const result = (await testDiscoveredCamera({
            fingerprint: device.fingerprint,
            host: device.host,
            ip: device.ip,
            name: device.name,
            vendor: device.vendor,
            model: device.model,
            onvif_uuid: device.onvif_uuid || "",
            onvif_xaddr: device.onvif_xaddr || "",
            onvif_port: device.onvif_port || null,
            rtsp_port: device.rtsp_port || null,
            preferred_stream: override.preferredStream,
            stream_path: override.streamPath,
            username,
            password,
          })) as DiscoveryTestResult
          nextResults[device.fingerprint] = result
        } catch (error: any) {
          nextResults[device.fingerprint] = {
            ok: false,
            auth_state: "unknown",
            error: error.message || "Camera test failed",
            error_code: "request_failed",
            host: device.host,
          }
        }
      })
    )
    setTestResults((current) => ({ ...current, ...nextResults }))
    setPhase("review")
    const successCount = Object.values(nextResults).filter((result) => result.ok).length
    toast.success(`Validated ${successCount} of ${selectedDevices.length} selected cameras`)
  }

  async function handleImport() {
    if (readyFingerprints.length === 0) {
      toast.error("Test at least one camera successfully before importing")
      return
    }
    if (missingRequiredSetupFingerprints.length > 0) {
      toast.error("Add a display name and area/location before importing")
      setPhase("review")
      return
    }
    setPhase("importing")
    try {
      const payload = readyFingerprints.map((fingerprint) => {
        const device = devices.find((item) => item.fingerprint === fingerprint)!
        const override = overridesById[fingerprint]
        const testResult = testResults[fingerprint]
        const username = override.credentialMode === "override" ? override.username.trim() : sharedUsername.trim()
        const password = override.credentialMode === "override" ? override.password : sharedPassword
        return {
          fingerprint,
          host: testResult.host || device.host,
          ip: device.ip,
          name: override.displayName.trim(),
          zone: override.zone.trim(),
          profile: override.profile,
          capabilities: override.capabilities,
          preferred_stream: testResult.preferred_stream || override.preferredStream,
          stream_path: testResult.stream_path || override.streamPath,
          username,
          password,
          onvif_uuid: testResult.onvif_uuid || device.onvif_uuid || "",
          onvif_xaddr: device.onvif_xaddr || "",
          onvif_port: testResult.onvif_port || device.onvif_port || null,
          rtsp_port: testResult.rtsp_port || device.rtsp_port || null,
        }
      })
      const result = (await importDiscoveredCameras({ devices: payload })) as DiscoveryImportResult
      setImportResult(result)
      setPhase("done")
      if (result.created.length > 0) {
        toast.success(`Imported ${result.created.length} camera${result.created.length === 1 ? "" : "s"}`)
      } else {
        toast.error("No cameras were imported")
      }
    } catch (error: any) {
      setPhase("review")
      toast.error(error.message || "Import failed")
    }
  }

  function retryFailedRows() {
    if (!importResult) return
    const failedFingerprints = importResult.failed.map((item) => item.fingerprint).filter(Boolean)
    if (failedFingerprints.length === 0) {
      setPhase("review")
      return
    }
    setSelectedIds(failedFingerprints)
    setPhase("review")
  }

  const stepIndex = STEP_ORDER.indexOf(phase)

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
            <Link to="/configure/cameras" className="hover:text-[var(--color-text-primary)]">Cameras</Link>
            <span>/</span>
            <span>Discover Cameras</span>
          </div>
          <h1 className="mt-2 text-2xl font-semibold text-[var(--color-text-primary)]">Discover Cameras</h1>
          <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
            Scan the local network, validate streams, then import cameras with shared credentials and per-camera exceptions.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={() => navigate("/configure/cameras/new")}>
            <Camera className="h-4 w-4" />
            Add Manually
          </Button>
          <Button variant="secondary" onClick={() => navigate("/configure/cameras")}>
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
        </div>
      </div>

      <Card className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          {STEP_ORDER.map((step, index) => {
            const active = index === stepIndex
            const completed = index < stepIndex
            return (
              <div key={step} className="flex items-center gap-3">
                <div className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-semibold ${
                  completed
                    ? "bg-[var(--color-success-bg)] text-[var(--color-success)]"
                    : active
                      ? "bg-[var(--color-info-bg)] text-[var(--color-info)]"
                      : "bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]"
                }`}>
                  {completed ? <CheckCircle2 className="h-4 w-4" /> : index + 1}
                </div>
                <span className={`text-sm font-medium ${active ? "text-[var(--color-text-primary)]" : "text-[var(--color-text-secondary)]"}`}>
                  {STEP_LABELS[step]}
                </span>
              </div>
            )
          })}
        </div>
      </Card>

      {phase === "scan" && (
        <Card className="space-y-5">
          <div>
            <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Scan local network</h2>
            <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
              Rakshak Lens will scan the server’s attached private subnets for ONVIF devices and reachable RTSP services.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Button variant="secondary" type="button" onClick={() => setShowAdvancedScan((current) => !current)}>
              <Network className="h-4 w-4" />
              {showAdvancedScan ? "Hide Scan Range" : "Advanced Scan Range"}
            </Button>
            <Button onClick={() => void handleScan()} disabled={scanLoading || loadingRules}>
              {scanLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <SearchCheck className="h-4 w-4" />}
              Scan Network
            </Button>
          </div>

          {showAdvancedScan && (
            <Field label="CIDR Override">
              <input
                type="text"
                value={cidrInput}
                onChange={(event) => setCidrInput(event.target.value)}
                placeholder="192.168.1.0/24, 10.0.0.0/24"
                className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 font-mono text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
              />
            </Field>
          )}

          {warnings.length > 0 && (
            <div className="space-y-2">
              {warnings.map((warning) => (
                <div key={warning} className="rounded-[var(--radius-md)] border border-[var(--color-warning)] bg-[var(--color-warning-bg)] px-3 py-2 text-sm text-[var(--color-text-secondary)]">
                  {warning}
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

      {(phase === "review" || phase === "testing" || phase === "importing") && (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
          <div className="space-y-4">
            <Card className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Discovered Devices</h2>
                  <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                    Choose only the cameras you want to configure. Unselected cameras will be ignored.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="info">{devices.length} found</Badge>
                  <Badge variant="default">{selectedIds.length} selected</Badge>
                  {exactDuplicateCount > 0 && <Badge variant="warning">{exactDuplicateCount} duplicates</Badge>}
                  {needsCredentialCount > 0 && <Badge variant="warning">{needsCredentialCount} need credentials</Badge>}
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2 rounded-[var(--radius-md)] border bg-[var(--color-bg-tertiary)] px-3 py-3">
                <Button
                  variant="secondary"
                  size="sm"
                  type="button"
                  onClick={selectVisibleCameras}
                  disabled={selectableFilteredFingerprints.length === 0 || allVisibleSelected}
                >
                  Select Visible Cameras
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  type="button"
                  onClick={clearVisibleSelection}
                  disabled={selectableFilteredFingerprints.every((fingerprint) => !selectedIds.includes(fingerprint))}
                >
                  Clear Visible Selection
                </Button>
                <span className="text-sm text-[var(--color-text-secondary)]">
                  {selectableFilteredFingerprints.length} configurable in this view
                </span>
              </div>

              <SearchInput
                value={search}
                onChange={setSearch}
                placeholder="Filter by IP, model, or camera name..."
              />

              {filteredDevices.length === 0 ? (
                <div className="rounded-[var(--radius-md)] border border-dashed px-4 py-10 text-center">
                  <p className="text-sm text-[var(--color-text-secondary)]">No cameras matched this search.</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {filteredDevices.map((device) => {
                    const selected = selectedIds.includes(device.fingerprint)
                    const override = overridesById[device.fingerprint]
                    const result = testResults[device.fingerprint]
                    return (
                      <Card key={device.fingerprint} className={`space-y-4 ${selected ? "border-[var(--color-info)]" : ""}`}>
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex items-start gap-3">
                            <label className={`mt-0.5 flex items-center gap-2 rounded-[var(--radius-md)] border px-3 py-2 text-sm font-medium ${
                              selected
                                ? "border-[var(--color-info)] bg-[var(--color-info-bg)] text-[var(--color-info)]"
                                : "border-[var(--color-border-default)] bg-white text-[var(--color-text-secondary)]"
                            } ${device.duplicate_state === "exact" ? "cursor-not-allowed opacity-60" : "cursor-pointer"}`}>
                              <input
                                type="checkbox"
                                checked={selected}
                                disabled={device.duplicate_state === "exact"}
                                onChange={() => toggleSelection(device.fingerprint)}
                                className="h-4 w-4 rounded border-[var(--color-border-default)]"
                              />
                              Configure
                            </label>
                            <div>
                              <div className="flex flex-wrap items-center gap-2">
                                <p className="text-sm font-semibold text-[var(--color-text-primary)]">{override?.displayName || defaultDisplayName(device)}</p>
                                <Badge variant="default">{device.ip}</Badge>
                                {device.vendor && <Badge variant="info">{device.vendor}</Badge>}
                                {device.model && <Badge variant="info">{device.model}</Badge>}
                              </div>
                              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                                {device.onvif_uuid ? "ONVIF discovered" : "RTSP service discovered"}
                              </p>
                            </div>
                          </div>
                          <div className="flex flex-wrap justify-end gap-2">
                            {device.duplicate_state === "exact" && <Badge variant="warning">Already configured</Badge>}
                            {device.duplicate_state === "potential" && <Badge variant="warning">Potential duplicate</Badge>}
                            {result?.ok && <Badge variant="success">Validated</Badge>}
                            {result && !result.ok && <Badge variant="critical">{result.error_code || "Failed"}</Badge>}
                          </div>
                        </div>

                        {selected && override && (
                          <div className="space-y-4 border-t pt-4">
                            {result?.preview_data_url && (
                              <img
                                src={result.preview_data_url}
                                alt={override.displayName}
                                className="aspect-video w-full rounded-[var(--radius-md)] border object-cover"
                              />
                            )}

                            <div className="grid gap-4 md:grid-cols-2">
                              <Field label="Display Name">
                                <input
                                  type="text"
                                  value={override.displayName}
                                  onChange={(event) => updateOverride(device.fingerprint, { displayName: event.target.value })}
                                  className={`w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)] ${
                                    result?.ok && !override.displayName.trim() ? "border-[var(--color-critical)]" : ""
                                  }`}
                                />
                              </Field>
                              <Field label="Area / Location">
                                <input
                                  type="text"
                                  value={override.zone}
                                  onChange={(event) => updateOverride(device.fingerprint, { zone: event.target.value })}
                                  className={`w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)] ${
                                    result?.ok && !override.zone.trim() ? "border-[var(--color-critical)]" : ""
                                  }`}
                                />
                              </Field>
                            </div>

                            {result?.ok && (!override.displayName.trim() || !override.zone.trim()) && (
                              <div className="rounded-[var(--radius-md)] border border-[var(--color-warning)] bg-[var(--color-warning-bg)] px-3 py-2 text-sm text-[var(--color-text-secondary)]">
                                Add a display name and area/location before importing this camera.
                              </div>
                            )}

                            <div className="grid gap-4 md:grid-cols-2">
                              <Field label="Camera Profile">
                                <select
                                  value={override.profile}
                                  onChange={(event) => updateOverride(device.fingerprint, {
                                    profile: event.target.value as CameraProfile,
                                    capabilities: [...PROFILE_DEFAULTS[event.target.value as CameraProfile]],
                                  })}
                                  className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)]"
                                >
                                  {PROFILE_OPTIONS.map((option) => (
                                    <option key={option.key} value={option.key}>{option.label}</option>
                                  ))}
                                </select>
                              </Field>
                              <Field label="Preferred Stream">
                                <select
                                  value={override.preferredStream}
                                  onChange={(event) => updateOverride(device.fingerprint, { preferredStream: event.target.value as DiscoveryRowOverride["preferredStream"] })}
                                  className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)]"
                                >
                                  <option value="main">Main stream</option>
                                  <option value="sub">Sub stream</option>
                                  <option value="mjpeg">MJPEG stream</option>
                                  <option value="custom">Custom path</option>
                                </select>
                              </Field>
                            </div>

                            {override.preferredStream === "custom" && (
                              <Field label="Custom RTSP Path">
                                <input
                                  type="text"
                                  value={override.streamPath}
                                  onChange={(event) => updateOverride(device.fingerprint, { streamPath: event.target.value })}
                                  placeholder="/Streaming/Channels/101"
                                  className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 font-mono text-sm focus:outline-2 focus:outline-[var(--color-info)]"
                                />
                              </Field>
                            )}

                            <div className="space-y-3 rounded-[var(--radius-md)] border px-3 py-3">
                              <div className="flex flex-wrap items-center justify-between gap-3">
                                <div>
                                  <p className="text-sm font-medium text-[var(--color-text-primary)]">Camera Login</p>
                                  <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                                    Use shared credentials, or enter a different login for this camera.
                                  </p>
                                </div>
                                <div className="flex rounded-[var(--radius-md)] border bg-white p-1">
                                  <button
                                    type="button"
                                    onClick={() => updateOverride(device.fingerprint, { credentialMode: "inherit" })}
                                    className={`rounded-[var(--radius-sm)] px-3 py-1.5 text-sm font-medium ${
                                      override.credentialMode === "inherit"
                                        ? "bg-[var(--color-info-bg)] text-[var(--color-info)]"
                                        : "text-[var(--color-text-secondary)]"
                                    }`}
                                  >
                                    Shared
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => updateOverride(device.fingerprint, { credentialMode: "override" })}
                                    className={`rounded-[var(--radius-sm)] px-3 py-1.5 text-sm font-medium ${
                                      override.credentialMode === "override"
                                        ? "bg-[var(--color-info-bg)] text-[var(--color-info)]"
                                        : "text-[var(--color-text-secondary)]"
                                    }`}
                                  >
                                    Custom
                                  </button>
                                </div>
                              </div>

                              {override.credentialMode === "inherit" && (
                                <p className="rounded-[var(--radius-md)] bg-[var(--color-bg-tertiary)] px-3 py-2 text-sm text-[var(--color-text-secondary)]">
                                  This camera will use the shared username and password from the setup panel.
                                </p>
                              )}

                              {override.credentialMode === "override" && (
                                <div className="grid gap-4 md:grid-cols-2">
                                  <Field label="Username">
                                    <input
                                      type="text"
                                      value={override.username}
                                      onChange={(event) => updateOverride(device.fingerprint, { username: event.target.value, credentialMode: "override" })}
                                      placeholder="admin"
                                      className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)]"
                                    />
                                  </Field>
                                  <Field label="Password">
                                    <input
                                      type="password"
                                      value={override.password}
                                      onChange={(event) => updateOverride(device.fingerprint, { password: event.target.value, credentialMode: "override" })}
                                      placeholder="Camera password"
                                      className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)]"
                                    />
                                  </Field>
                                </div>
                              )}
                            </div>

                            <details className="rounded-[var(--radius-md)] border px-3 py-3">
                              <summary className="cursor-pointer text-sm font-medium text-[var(--color-text-primary)]">
                                Advanced
                              </summary>
                              <div className="mt-4 space-y-4">
                                <div>
                                  <p className="mb-2 text-sm font-medium text-[var(--color-text-primary)]">Detections</p>
                                  <DetectionChecklist
                                    safetyRules={safetyRules}
                                    selectedKeys={override.capabilities}
                                    onChange={(keys) => updateOverride(device.fingerprint, { capabilities: keys })}
                                  />
                                </div>
                              </div>
                            </details>

                            {device.duplicate_state === "exact" && device.existing_camera_id && (
                              <div className="flex items-center justify-between rounded-[var(--radius-md)] border border-[var(--color-warning)] bg-[var(--color-warning-bg)] px-3 py-2">
                                <p className="text-sm text-[var(--color-text-secondary)]">
                                  This camera already exists in Rakshak Lens.
                                </p>
                                <Button variant="secondary" size="sm" onClick={() => navigate(`/configure/cameras/${device.existing_camera_id}`)}>
                                  Open Existing
                                </Button>
                              </div>
                            )}

                            {result && !result.ok && (
                              <div className="rounded-[var(--radius-md)] border border-[var(--color-critical)] bg-[var(--color-critical-bg)] px-3 py-2 text-sm text-[var(--color-text-secondary)]">
                                {result.error}
                              </div>
                            )}
                          </div>
                        )}
                      </Card>
                    )
                  })}
                </div>
              )}
            </Card>
          </div>

          <div className="space-y-6">
            <Card className="space-y-4 xl:sticky xl:top-20">
              <div>
                <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Shared Defaults</h2>
                <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                  Apply one credential pair and one default setup to the selected cameras, then override exceptions.
                </p>
              </div>

              <div className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-1">
                  <Field label="Shared Username">
                    <input
                      type="text"
                      value={sharedUsername}
                      onChange={(event) => setSharedUsername(event.target.value)}
                      placeholder="admin"
                      className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)]"
                    />
                  </Field>
                  <Field label="Shared Password">
                    <input
                      type="password"
                      value={sharedPassword}
                      onChange={(event) => setSharedPassword(event.target.value)}
                      placeholder="Camera password"
                      className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)]"
                    />
                  </Field>
                </div>

                <Field label="Default Area / Location">
                  <input
                    type="text"
                    value={sharedZone}
                    onChange={(event) => setSharedZone(event.target.value)}
                    placeholder="Main Gate"
                    className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)]"
                  />
                </Field>

                <Field label="Default Camera Profile">
                  <div className="grid gap-2">
                    {PROFILE_OPTIONS.map((option) => (
                      <button
                        key={option.key}
                        type="button"
                        onClick={() => handleProfileChange(option.key)}
                        className={`rounded-[var(--radius-lg)] border p-3 text-left transition-colors ${
                          sharedProfile === option.key
                            ? "border-[var(--color-info)] bg-[var(--color-info-bg)]"
                            : "border-[var(--color-border-default)] bg-white hover:border-[var(--color-border-active)]"
                        }`}
                      >
                        <p className="text-sm font-semibold text-[var(--color-text-primary)]">{option.label}</p>
                        <p className="mt-1 text-xs text-[var(--color-text-secondary)]">{option.description}</p>
                      </button>
                    ))}
                  </div>
                </Field>

                <div>
                  <p className="mb-2 text-sm font-medium text-[var(--color-text-primary)]">Default Detections</p>
                  <DetectionChecklist
                    safetyRules={safetyRules}
                    selectedKeys={sharedCapabilities}
                    onChange={setSharedCapabilities}
                  />
                </div>

                <Button variant="secondary" type="button" onClick={applySharedDefaults} disabled={selectedIds.length === 0}>
                  Apply Defaults to Selected
                </Button>
              </div>

              <div className="space-y-2 rounded-[var(--radius-md)] border bg-[var(--color-bg-tertiary)] p-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-[var(--color-text-secondary)]">Selected</span>
                  <span className="font-medium text-[var(--color-text-primary)]">{selectedIds.length}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-[var(--color-text-secondary)]">Validated</span>
                  <span className="font-medium text-[var(--color-text-primary)]">{readyFingerprints.length}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-[var(--color-text-secondary)]">Needs Credentials</span>
                  <span className="font-medium text-[var(--color-text-primary)]">{needsCredentialCount}</span>
                </div>
              </div>

              <div className="flex flex-col gap-2">
                <Button variant="secondary" onClick={() => void handleScan()} disabled={scanLoading || loadingRules}>
                  <RefreshCw className="h-4 w-4" />
                  Re-scan Network
                </Button>
                <Button onClick={() => void handleTestSelected()} disabled={phase === "testing" || selectedIds.length === 0 || loadingRules}>
                  {phase === "testing" ? <Loader2 className="h-4 w-4 animate-spin" /> : <ShieldAlert className="h-4 w-4" />}
                  Test Selected Cameras
                </Button>
                <Button variant="secondary" onClick={() => void handleImport()} disabled={phase === "importing" || readyFingerprints.length === 0}>
                  {phase === "importing" ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
                  Import Ready Cameras
                </Button>
              </div>

              {selectedDevices.some((device) => {
                const override = overridesById[device.fingerprint]
                return override && usesConfiguredZones(override.capabilities)
              }) && (
                <p className="text-xs text-[var(--color-text-secondary)]">
                  Cameras with zone-based detections enabled will still need polygon setup after import.
                </p>
              )}
            </Card>
          </div>
        </div>
      )}

      {phase === "done" && importResult && (
        <div className="space-y-6">
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <p className="text-sm text-[var(--color-text-secondary)]">Imported</p>
              <p className="mt-2 text-2xl font-semibold text-[var(--color-text-primary)]">{importResult.created.length}</p>
            </Card>
            <Card>
              <p className="text-sm text-[var(--color-text-secondary)]">Need Zone Setup</p>
              <p className="mt-2 text-2xl font-semibold text-[var(--color-text-primary)]">{importResult.needs_zone_setup.length}</p>
            </Card>
            <Card>
              <p className="text-sm text-[var(--color-text-secondary)]">Failed</p>
              <p className="mt-2 text-2xl font-semibold text-[var(--color-text-primary)]">{importResult.failed.length}</p>
            </Card>
          </div>

          <Card className="space-y-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Import Summary</h2>
                <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                  Open each created camera, or continue into zone setup where needed.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="secondary" onClick={() => navigate("/configure/cameras")}>
                  Back to Cameras
                </Button>
                {importResult.failed.length > 0 && (
                  <Button variant="secondary" onClick={retryFailedRows}>
                    Retry Failed Rows
                  </Button>
                )}
                <Button onClick={() => {
                  setPhase("scan")
                  setDevices([])
                  setSelectedIds([])
                  setImportResult(null)
                  setTestResults({})
                  setOverridesById({})
                  setWarnings([])
                  setSearch("")
                }}>
                  Discover More
                </Button>
              </div>
            </div>

            <div className="space-y-3">
              {importResult.created.map((camera) => (
                <div key={camera.camera_id} className="flex flex-wrap items-center justify-between gap-3 rounded-[var(--radius-md)] border px-3 py-3">
                  <div>
                    <p className="text-sm font-semibold text-[var(--color-text-primary)]">{camera.name}</p>
                    <p className="mt-1 text-sm text-[var(--color-text-secondary)]">{camera.zone}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {camera.needs_zone_setup && <Badge variant="warning">Zone setup needed</Badge>}
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => navigate(camera.needs_zone_setup ? `/configure/cameras/${camera.camera_id}/edit?focus=zones` : `/configure/cameras/${camera.camera_id}`)}
                    >
                      {camera.needs_zone_setup ? "Configure Zones" : "View Camera"}
                    </Button>
                  </div>
                </div>
              ))}

              {importResult.failed.length > 0 && (
                <div className="space-y-3 pt-2">
                  <p className="text-sm font-semibold text-[var(--color-text-primary)]">Failed Rows</p>
                  {importResult.failed.map((item) => (
                    <div key={`${item.fingerprint}-${item.error_code}`} className="rounded-[var(--radius-md)] border border-[var(--color-critical)] bg-[var(--color-critical-bg)] px-3 py-2">
                      <p className="text-sm font-medium text-[var(--color-text-primary)]">{item.error_code}</p>
                      <p className="mt-1 text-sm text-[var(--color-text-secondary)]">{item.error}</p>
                      {item.existing_camera_id && (
                        <Button variant="secondary" size="sm" className="mt-2" onClick={() => navigate(`/configure/cameras/${item.existing_camera_id}`)}>
                          Open Existing Camera
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
