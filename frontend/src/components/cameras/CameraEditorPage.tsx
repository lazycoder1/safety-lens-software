import { useEffect, useMemo, useRef, useState } from "react"
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom"
import { AlertTriangle, ArrowLeft, Trash2 } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import {
  addCamera,
  addZone,
  ApiError,
  createAutomationRule,
  deleteCamera,
  deleteZone,
  getAlertOutputs,
  getAutomationRules,
  getCameraById,
  getSafetyRules,
  getVideos,
  getZones,
  installModels,
  previewCameraPlan,
  STREAM_BASE,
  testRtspConnection,
  updateAutomationRule,
  updateCamera,
  updateZone,
} from "@/lib/api"
import type { AlertOutput, Camera, CameraProfile, EngineRule, ExecutionPlan, SafetyRule, Zone } from "@/types"
import {
  cameraNeedsLegacyNormalization,
  deriveCameraPurpose,
  getConfiguredDetectionKeys,
  getDetectionLabelsFromKeys,
  getDetectionOptions,
  getSelectedRuleIds,
  getUnmappedRuleIds,
  usesConfiguredZones,
  usesZoneIntrusion,
  type CameraDetectionKey,
} from "./detectionCatalog"
import { PROFILE_DEFAULTS, PROFILE_OPTIONS } from "./profileOptions"
import { DetectionChecklist } from "./DetectionChecklist"
import { Field } from "./helpers"
import { PolygonDrawer } from "@/components/zones/PolygonDrawer"
import { useModelInstallModal } from "@/components/ModelInstallModal"
import {
  buildCameraEventPolicyPayload,
  cameraEventPolicyDraftFromRule,
  CameraEventPolicyEditor,
  cameraScopedAutomationRules,
  defaultCameraEventPolicyDraft,
  findDefaultCameraEventRule,
  validateCameraEventPolicyDraft,
  type CameraEventPolicyDraft,
} from "./CameraEventPolicyPanel"
import {
  buildCapabilityEditorOptions,
  buildCapabilityModelOverridesPayload,
  buildCapabilityWindowsPayload,
  buildHelmetColourPolicyPayload,
  buildSafetyRuleOverridesPayload,
  CameraCapabilityConfigPanel,
  capabilityConfigDraftsFromCamera,
  reconcileCapabilityConfigDrafts,
  validateCapabilityConfigDrafts,
  type CameraCapabilityConfigDrafts,
} from "./CameraCapabilityConfigPanel"
import {
  buildDetectionPolicyPayload,
  CameraDetectionPolicyEditor,
  detectionPolicyDraftsFromRules,
  findDetectionPolicyRule,
  reconcileDetectionPolicyDrafts,
  validateDetectionPolicyDrafts,
  type CameraDetectionPolicyDrafts,
} from "./CameraDetectionPolicyPanel"

interface CameraEditorPageProps {
  mode: "create" | "edit"
}

function parseCustomDetectionTerms(value: string): string[] {
  return Array.from(
    new Set(
      value
        .split(/[\n,]/)
        .map((term) => term.trim())
        .filter(Boolean)
    )
  )
}

export function CameraEditorPage({ mode }: CameraEditorPageProps) {
  const navigate = useNavigate()
  const params = useParams()
  const [searchParams] = useSearchParams()
  const openInstallModal = useModelInstallModal((state) => state.open)
  const cameraId = params.cameraId || null
  const isEdit = mode === "edit"
  const showZoneFocus = searchParams.get("focus") === "zones"
  const zoneSectionRef = useRef<HTMLDivElement>(null)

  const [loading, setLoading] = useState(isEdit)
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [videos, setVideos] = useState<string[]>([])
  const [safetyRules, setSafetyRules] = useState<SafetyRule[]>([])
  const [alertOutputs, setAlertOutputs] = useState<AlertOutput[]>([])
  const [automationRules, setAutomationRules] = useState<EngineRule[]>([])
  const [eventPolicyRule, setEventPolicyRule] = useState<EngineRule | null>(null)
  const [eventPolicyDraft, setEventPolicyDraft] = useState<CameraEventPolicyDraft>(() => defaultCameraEventPolicyDraft())
  const [capabilityDrafts, setCapabilityDrafts] = useState<CameraCapabilityConfigDrafts>({})
  const [detectionPolicyDrafts, setDetectionPolicyDrafts] = useState<CameraDetectionPolicyDrafts>({})
  const [camera, setCamera] = useState<Camera | null>(null)
  const [zones, setZones] = useState<Zone[]>([])
  const [hasUnsavedZoneDraft, setHasUnsavedZoneDraft] = useState(false)
  const [planPreview, setPlanPreview] = useState<{ execution_plan: ExecutionPlan; missing_model_keys: string[] } | null>(null)

  const [name, setName] = useState("")
  const [zone, setZone] = useState("")
  const [profile, setProfile] = useState<CameraProfile>("general_safety")
  const [streamFps, setStreamFps] = useState("6")
  const [inferenceFps, setInferenceFps] = useState("2")
  const [streamType, setStreamType] = useState<"file" | "rtsp">("file")
  const [video, setVideo] = useState("")
  const [rtspUrl, setRtspUrl] = useState("")
  const [rtspHost, setRtspHost] = useState("")
  const [rtspPort, setRtspPort] = useState("")
  const [onvifPort, setOnvifPort] = useState("")
  const [streamPath, setStreamPath] = useState("")
  const [preferredStream, setPreferredStream] = useState("")
  const [onvifUuid, setOnvifUuid] = useState("")
  const [discoveryFingerprint, setDiscoveryFingerprint] = useState("")
  const [rtspUsername, setRtspUsername] = useState("")
  const [rtspPassword, setRtspPassword] = useState("")
  const [replaceStoredCredentials, setReplaceStoredCredentials] = useState(false)
  const [selectedDetections, setSelectedDetections] = useState<CameraDetectionKey[]>([])
  const [customDetectionTermsInput, setCustomDetectionTermsInput] = useState("")
  const [rtspTestStatus, setRtspTestStatus] = useState<"idle" | "testing" | "success" | "failed">("idle")
  const [rtspTestError, setRtspTestError] = useState("")
  const [rtspResolution, setRtspResolution] = useState<[number, number] | null>(null)

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const [ruleData, videoData, outputData, automationRuleData] = await Promise.all([
          getSafetyRules(),
          getVideos(),
          getAlertOutputs(),
          getAutomationRules(),
        ])
        if (cancelled) return
        setSafetyRules(ruleData)
        setVideos(videoData)
        setAlertOutputs(outputData)
        setAutomationRules(automationRuleData)

        if (!isEdit || !cameraId) {
          setEventPolicyRule(null)
          setEventPolicyDraft(defaultCameraEventPolicyDraft(outputData))
          setCapabilityDrafts({})
          setDetectionPolicyDrafts({})
          if (videoData.length > 0) {
            setVideo((current) => current || videoData[0])
          }
          setLoading(false)
          return
        }

        const existingCamera = await getCameraById(cameraId)
        if (cancelled) return
        const zoneData = await getZones(existingCamera.id).catch(() => existingCamera.zones || [])
        if (cancelled) return

        setCamera(existingCamera)
        setZones(zoneData)
        const existingEventPolicy = findDefaultCameraEventRule(automationRuleData, existingCamera.id)
        setEventPolicyRule(existingEventPolicy)
        setEventPolicyDraft(cameraEventPolicyDraftFromRule(existingEventPolicy, outputData))
        setName(existingCamera.name)
        setZone(existingCamera.zone)
        setProfile(existingCamera.profile || "general_safety")
        setStreamFps(String(existingCamera.fps || 6))
        setInferenceFps(String(existingCamera.inference_fps || 2))
        setStreamType((existingCamera.stream_type || "file") as "file" | "rtsp")
        setVideo(existingCamera.video || videoData[0] || "")
        setRtspUrl(existingCamera.connection_summary || existingCamera.rtsp_url || "")
        setRtspHost(existingCamera.host || "")
        setRtspPort(existingCamera.rtsp_port ? String(existingCamera.rtsp_port) : "")
        setOnvifPort(existingCamera.onvif_port ? String(existingCamera.onvif_port) : "")
        setStreamPath(existingCamera.stream_path || "")
        setPreferredStream(existingCamera.preferred_stream || "")
        setOnvifUuid(existingCamera.onvif_uuid || "")
        setDiscoveryFingerprint(existingCamera.discovery_fingerprint || "")
        setRtspUsername("")
        setRtspPassword("")
        setReplaceStoredCredentials(false)
        setCustomDetectionTermsInput((existingCamera.custom_long_tail_terms || []).join(", "))
        const configuredKeys = getConfiguredDetectionKeys(existingCamera, ruleData)
        const configuredOptions = buildCapabilityEditorOptions(
          configuredKeys,
          ruleData,
          getDetectionOptions(ruleData).map((option) => ({ key: option.key, label: option.label, ruleId: option.ruleId }))
        )
        setSelectedDetections(configuredKeys)
        setCapabilityDrafts(capabilityConfigDraftsFromCamera(existingCamera, configuredOptions))
        setDetectionPolicyDrafts(detectionPolicyDraftsFromRules(automationRuleData, existingCamera.id, configuredOptions, outputData))
      } catch (error: any) {
        toast.error(error.message || "Failed to load camera")
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [cameraId, isEdit])

  const requiresZone = usesConfiguredZones(selectedDetections)
  const requiresIntrusionZone = usesZoneIntrusion(selectedDetections)
  const usesOfficeOccupancy = selectedDetections.includes("office_occupancy")
  const usesQueueMonitoring = selectedDetections.includes("queue_monitoring")
  const usesRouteObstruction = selectedDetections.includes("route_obstruction")
  const usesObjectLifecycle = selectedDetections.includes("object_lifecycle")
  const usesCustomDetection = selectedDetections.includes("custom_long_tail")
  const customDetectionTerms = useMemo(
    () => parseCustomDetectionTerms(customDetectionTermsInput),
    [customDetectionTermsInput]
  )
  const defaultZoneType = requiresIntrusionZone
    ? "restricted"
    : usesQueueMonitoring
      ? "queue"
      : usesRouteObstruction
        ? "gangway"
        : usesObjectLifecycle
          ? "object_watch"
          : "workstation"
  const zoneSetupDescription = requiresIntrusionZone
    ? "Zone Intrusion is enabled. Draw at least one restricted area for this camera."
    : usesQueueMonitoring
      ? "Queue Monitoring is enabled. Draw each queue area and keep Queue selected as its analytics type."
      : usesRouteObstruction
        ? "Route Obstruction is enabled. Draw each gangway or keep-clear area and select the blocking object classes."
        : usesObjectLifecycle
          ? "Object Lifecycle is enabled. Draw each object-watch area and select the watched object classes."
          : "Office Occupancy is enabled. Draw one zone per chair or seat."
  const zoneSetupTarget = requiresIntrusionZone
    ? "a restricted zone"
    : usesQueueMonitoring
      ? "a queue zone"
      : usesRouteObstruction
        ? "a gangway or keep-clear zone"
        : usesObjectLifecycle
          ? "an object-watch zone"
          : usesOfficeOccupancy
            ? "chair zones"
            : "a zone"
  const selectedRuleIds = useMemo(() => getSelectedRuleIds(selectedDetections, safetyRules), [selectedDetections, safetyRules])
  const selectedLabels = useMemo(() => getDetectionLabelsFromKeys(selectedDetections), [selectedDetections])
  const detectionDefinitions = useMemo(
    () => getDetectionOptions(safetyRules).map((option) => ({ key: option.key, label: option.label, ruleId: option.ruleId })),
    [safetyRules]
  )
  const capabilityOptions = useMemo(
    () => buildCapabilityEditorOptions(selectedDetections, safetyRules, detectionDefinitions),
    [selectedDetections, safetyRules, detectionDefinitions]
  )
  const trainedPpeEnabled = useMemo(
    () => capabilityOptions.some((option) => capabilityDrafts[option.key]?.modelKey === "ppe_closed_set_candidate"),
    [capabilityOptions, capabilityDrafts]
  )
  const runtimeLimitedDetectionCount = useMemo(
    () => capabilityOptions.filter((option) => capabilityDrafts[option.key]?.limitRuntime).length,
    [capabilityOptions, capabilityDrafts]
  )
  const capabilityModelOverridesPayload = useMemo(
    () => buildCapabilityModelOverridesPayload(capabilityOptions, capabilityDrafts),
    [capabilityOptions, capabilityDrafts]
  )
  const preservedRuleIds = useMemo(
    () => (camera ? getUnmappedRuleIds(camera.safety_rule_ids || [], safetyRules) : []),
    [camera, safetyRules]
  )
  const purpose = preservedRuleIds.length > 0 ? "Custom monitoring" : deriveCameraPurpose(selectedDetections)
  const showLegacyNote = camera ? cameraNeedsLegacyNormalization(camera) : false
  const needsZoneSetup = requiresZone && zones.length === 0
  const capabilityWindowsPayload = useMemo(
    () => buildCapabilityWindowsPayload(capabilityOptions, capabilityDrafts),
    [capabilityOptions, capabilityDrafts]
  )
  const safetyRuleOverridesPayload = useMemo(
    () => buildSafetyRuleOverridesPayload(capabilityOptions, capabilityDrafts),
    [capabilityOptions, capabilityDrafts]
  )
  const helmetColourPolicyPayload = useMemo(
    () => buildHelmetColourPolicyPayload(capabilityOptions, capabilityDrafts),
    [capabilityOptions, capabilityDrafts]
  )
  const customDetectionPolicyCount = useMemo(
    () => capabilityOptions.filter((option) => detectionPolicyDrafts[option.key]?.enabled).length,
    [capabilityOptions, detectionPolicyDrafts]
  )
  const cameraEventRules = useMemo(
    () => (cameraId ? cameraScopedAutomationRules(automationRules, cameraId) : []),
    [automationRules, cameraId]
  )

  useEffect(() => {
    setCapabilityDrafts((current) => reconcileCapabilityConfigDrafts(current, capabilityOptions))
    setDetectionPolicyDrafts((current) => reconcileDetectionPolicyDrafts(current, capabilityOptions, alertOutputs))
  }, [capabilityOptions, alertOutputs])

  useEffect(() => {
    if (!showZoneFocus || !requiresZone || !zoneSectionRef.current) return
    zoneSectionRef.current.scrollIntoView({ behavior: "smooth", block: "start" })
  }, [showZoneFocus, requiresZone, zones.length])

  useEffect(() => {
    if (loading) return
    const shouldSendCredentialUpdate =
      streamType === "rtsp" &&
      (!isEdit || replaceStoredCredentials || Boolean(rtspUsername.trim()) || Boolean(rtspPassword))
    let cancelled = false
    const timer = window.setTimeout(async () => {
      setPreviewLoading(true)
      try {
        const preview = await previewCameraPlan({
          name,
          zone,
          profile,
          capabilities: selectedDetections,
          fps: Number(streamFps),
          inference_fps: Number(inferenceFps),
          stream_type: streamType,
          video: streamType === "file" ? video : "",
          rtsp_url: streamType === "rtsp" ? rtspUrl : "",
          host: streamType === "rtsp" ? rtspHost.trim() : "",
          rtsp_port: streamType === "rtsp" && rtspPort.trim() ? Number(rtspPort) : null,
          onvif_port: streamType === "rtsp" && onvifPort.trim() ? Number(onvifPort) : null,
          stream_path: streamType === "rtsp" ? streamPath.trim() : "",
          preferred_stream: streamType === "rtsp" ? preferredStream.trim() : "",
          onvif_uuid: streamType === "rtsp" ? onvifUuid.trim() : "",
          discovery_fingerprint: streamType === "rtsp" ? discoveryFingerprint.trim() : "",
          username: shouldSendCredentialUpdate ? rtspUsername.trim() : undefined,
          password: shouldSendCredentialUpdate ? rtspPassword : undefined,
          safety_rule_ids: [...selectedRuleIds, ...preservedRuleIds],
          safety_rule_overrides: safetyRuleOverridesPayload,
          helmet_colour_policy: helmetColourPolicyPayload,
          custom_long_tail_terms: usesCustomDetection ? customDetectionTerms : [],
          capability_windows: capabilityWindowsPayload,
          capability_model_overrides: capabilityModelOverridesPayload,
        })
        if (!cancelled) {
          setPlanPreview(preview)
        }
      } catch {
        if (!cancelled) setPlanPreview(null)
      } finally {
        if (!cancelled) setPreviewLoading(false)
      }
    }, 250)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [
    loading,
    name,
    zone,
    profile,
    streamFps,
    inferenceFps,
    streamType,
    video,
    rtspUrl,
    rtspHost,
    rtspPort,
    onvifPort,
    streamPath,
    preferredStream,
    onvifUuid,
    discoveryFingerprint,
    rtspUsername,
    rtspPassword,
    replaceStoredCredentials,
    selectedDetections,
    capabilityWindowsPayload,
    safetyRuleOverridesPayload,
    helmetColourPolicyPayload,
    capabilityModelOverridesPayload,
    usesCustomDetection,
    customDetectionTerms,
    selectedRuleIds,
    preservedRuleIds,
    isEdit,
  ])

  function handleProfileChange(nextProfile: CameraProfile) {
    setProfile(nextProfile)
    if (selectedDetections.length === 0) {
      setSelectedDetections(PROFILE_DEFAULTS[nextProfile])
    }
  }

  async function refreshZones(targetCameraId: string) {
    const updatedZones = await getZones(targetCameraId).catch(() => zones)
    setZones(updatedZones)
  }

  async function handleSaveZone(newZone: Omit<Zone, "id">) {
    if (!cameraId) return
    try {
      await addZone(cameraId, newZone)
      await refreshZones(cameraId)
      toast.success("Zone saved. Save camera changes too if you edited detections or profile.")
    } catch (error: any) {
      toast.error(error.message || "Failed to save zone")
    }
  }

  async function handleUpdateZone(zoneId: string, updates: Partial<Zone>) {
    if (!cameraId) return
    try {
      await updateZone(cameraId, zoneId, updates)
      await refreshZones(cameraId)
      toast.success("Zone updated")
    } catch (error: any) {
      toast.error(error.message || "Failed to update zone")
    }
  }

  async function handleDeleteZone(zoneId: string) {
    if (!cameraId) return
    try {
      await deleteZone(cameraId, zoneId)
      setZones((current) => current.filter((item) => item.id !== zoneId))
      toast.success("Zone removed")
    } catch (error: any) {
      toast.error(error.message || "Failed to remove zone")
    }
  }

  async function handleDeleteCamera() {
    if (!cameraId || !camera) return
    if (!window.confirm(`Delete camera "${camera.name}"? This cannot be undone.`)) return
    setDeleting(true)
    try {
      await deleteCamera(cameraId)
      toast.success("Camera deleted")
      navigate("/configure/cameras")
    } catch (error: any) {
      toast.error(error.message || "Failed to delete camera")
    } finally {
      setDeleting(false)
    }
  }

  async function handleRtspTest() {
    if (!rtspUrl.trim()) {
      toast.error("RTSP URL is required before testing")
      return
    }
    setRtspTestStatus("testing")
    setRtspTestError("")
    setRtspResolution(null)
    try {
      const result = await testRtspConnection(rtspUrl.trim())
      if (result.success) {
        setRtspTestStatus("success")
        setRtspResolution(result.resolution)
        toast.success("RTSP connection succeeded")
      } else {
        setRtspTestStatus("failed")
        setRtspTestError(result.error || "Could not read the stream")
      }
    } catch (error: any) {
      setRtspTestStatus("failed")
      setRtspTestError(error.message || "Could not test RTSP connection")
    }
  }

  async function saveCameraEventPolicy(savedCamera: Camera) {
    if (!savedCamera.id) return
    if (!eventPolicyDraft.enabled && !eventPolicyRule) return

    const payload = buildCameraEventPolicyPayload({
      cameraId: savedCamera.id,
      cameraName: savedCamera.name || name.trim(),
      cameraZone: savedCamera.zone || zone.trim(),
      draft: eventPolicyDraft,
      existingRule: eventPolicyRule,
    })

    if (eventPolicyRule) {
      const updated = await updateAutomationRule(eventPolicyRule.id, payload)
      setEventPolicyRule(updated)
      setAutomationRules((current) => current.map((rule) => (rule.id === updated.id ? updated : rule)))
    } else if (eventPolicyDraft.enabled) {
      const created = await createAutomationRule(payload)
      setEventPolicyRule(created)
      setAutomationRules((current) => [...current, created])
    }
  }

  async function saveDetectionAlertPolicies(savedCamera: Camera) {
    if (!savedCamera.id) return

    for (const option of capabilityOptions) {
      const draft = detectionPolicyDrafts[option.key]
      const existingRule = findDetectionPolicyRule(automationRules, savedCamera.id, option.key)
      if (!draft || (!draft.enabled && !existingRule)) continue

      const payload = buildDetectionPolicyPayload({
        cameraId: savedCamera.id,
        cameraName: savedCamera.name || name.trim(),
        cameraZone: savedCamera.zone || zone.trim(),
        option,
        draft,
        existingRule,
      })

      if (existingRule) {
        const updated = await updateAutomationRule(existingRule.id, payload)
        setAutomationRules((current) => current.map((rule) => (rule.id === updated.id ? updated : rule)))
      } else if (draft.enabled) {
        const created = await createAutomationRule(payload)
        setAutomationRules((current) => [...current, created])
      }
    }
  }

  async function submitCamera(isRetry = false) {
    if (!name.trim()) {
      toast.error("Camera name is required")
      return
    }
    if (!zone.trim()) {
      toast.error("Zone / location is required")
      return
    }
    if (streamType === "file" && !video) {
      toast.error("Choose a video file")
      return
    }
    if (streamType === "rtsp" && !rtspUrl.trim()) {
      toast.error("RTSP URL is required")
      return
    }
    const parsedStreamFps = Number(streamFps)
    const parsedInferenceFps = Number(inferenceFps)
    if (!Number.isFinite(parsedStreamFps) || parsedStreamFps < 1 || parsedStreamFps > 60) {
      toast.error("Stream processing FPS must be between 1 and 60")
      return
    }
    if (!Number.isFinite(parsedInferenceFps) || parsedInferenceFps <= 0 || parsedInferenceFps > 60) {
      toast.error("Primary inference FPS must be greater than 0 and at most 60")
      return
    }
    if (parsedInferenceFps > parsedStreamFps) {
      toast.error("Primary inference FPS cannot exceed stream processing FPS")
      return
    }
    if (usesCustomDetection && customDetectionTerms.length === 0) {
      toast.error("Add at least one visible object or condition for Custom Detection")
      return
    }
    if (requiresZone && hasUnsavedZoneDraft) {
      zoneSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })
      toast.error("Zone draft not saved yet. In the Zones section, click Save Zone or Cancel.")
      return
    }
    const eventPolicyError = validateCameraEventPolicyDraft(eventPolicyDraft)
    if (eventPolicyError) {
      toast.error(eventPolicyError)
      return
    }
    const capabilityConfigError = validateCapabilityConfigDrafts(capabilityOptions, capabilityDrafts)
    if (capabilityConfigError) {
      toast.error(capabilityConfigError)
      return
    }
    const detectionPolicyError = validateDetectionPolicyDrafts(capabilityOptions, detectionPolicyDrafts)
    if (detectionPolicyError) {
      toast.error(detectionPolicyError)
      return
    }

    const shouldSendCredentialUpdate =
      streamType === "rtsp" &&
      (!isEdit || replaceStoredCredentials || Boolean(rtspUsername.trim()) || Boolean(rtspPassword))

    const payload = {
      name: name.trim(),
      zone: zone.trim(),
      profile,
      capabilities: selectedDetections,
      demo: planPreview?.execution_plan?.derived_demo || "yolo",
      video: streamType === "file" ? video : "",
      rules: [],
      fps: parsedStreamFps,
      inference_fps: parsedInferenceFps,
      stream_type: streamType,
      rtsp_url: streamType === "rtsp" ? rtspUrl.trim() : "",
      host: streamType === "rtsp" ? rtspHost.trim() : "",
      rtsp_port: streamType === "rtsp" && rtspPort.trim() ? Number(rtspPort) : null,
      onvif_port: streamType === "rtsp" && onvifPort.trim() ? Number(onvifPort) : null,
      stream_path: streamType === "rtsp" ? streamPath.trim() : "",
      preferred_stream: streamType === "rtsp" ? preferredStream.trim() : "",
      onvif_uuid: streamType === "rtsp" ? onvifUuid.trim() : "",
      discovery_fingerprint: streamType === "rtsp" ? discoveryFingerprint.trim() : "",
      username: shouldSendCredentialUpdate ? rtspUsername.trim() : undefined,
      password: shouldSendCredentialUpdate ? rtspPassword : undefined,
      safety_rule_ids: [...selectedRuleIds, ...preservedRuleIds],
      safety_rule_overrides: safetyRuleOverridesPayload,
      helmet_colour_policy: helmetColourPolicyPayload,
      custom_long_tail_terms: usesCustomDetection ? customDetectionTerms : [],
      capability_windows: capabilityWindowsPayload,
      capability_model_overrides: capabilityModelOverridesPayload,
    }

    setSaving(true)
    try {
      const saved = isEdit && cameraId ? await updateCamera(cameraId, payload) : await addCamera(payload)
      await saveCameraEventPolicy(saved)
      await saveDetectionAlertPolicies(saved)

      if (requiresZone && !isEdit) {
        toast.success("Camera created. Draw at least one zone to finish setup.")
        navigate(`/configure/cameras/${saved.id}/edit?focus=zones`)
      } else {
        toast.success(isEdit ? "Camera updated" : "Camera created")
        navigate(`/configure/cameras/${saved.id}`)
      }
    } catch (error) {
      if (error instanceof ApiError && error.status === 409 && error.payload?.code === "missing_models" && !isRetry) {
        try {
          const job = await installModels(error.payload.missing_model_keys || [])
          openInstallModal(job.id, {
            title: "Setting up required models",
            subtitle: "The selected camera needs missing model files. This modal will stay open until download and setup finish.",
            onReady: () => {
              void submitCamera(true)
            },
          })
          return
        } catch (installError: any) {
          toast.error(installError.message || "Failed to start model setup")
          return
        }
      }
      toast.error(error instanceof Error ? error.message : "Failed to save camera")
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="p-6">
        <Card className="p-6">
          <p className="text-sm text-[var(--color-text-secondary)]">Loading camera setup…</p>
        </Card>
      </div>
    )
  }

  if (isEdit && !camera) {
    return (
      <div className="p-6">
        <Card className="space-y-4 p-6">
          <div>
            <h1 className="text-xl font-semibold text-[var(--color-text-primary)]">Camera not found</h1>
            <p className="mt-1 text-sm text-[var(--color-text-secondary)]">The requested camera could not be loaded.</p>
          </div>
          <Button variant="secondary" onClick={() => navigate("/configure/cameras")}>
            <ArrowLeft className="w-4 h-4" />
            Back to Cameras
          </Button>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
            <Link to="/configure/cameras" className="hover:text-[var(--color-text-primary)]">Cameras</Link>
            <span>/</span>
            <span>{isEdit ? "Edit Camera" : "New Camera"}</span>
          </div>
          <h1 className="mt-2 text-2xl font-semibold text-[var(--color-text-primary)]">
            {isEdit ? `Edit ${camera?.name || "Camera"}` : "Set Up Camera"}
          </h1>
          <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
            Configure the camera profile, detections, and execution plan in one place.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={() => navigate(isEdit && cameraId ? `/configure/cameras/${cameraId}` : "/configure/cameras")}>
            Cancel
          </Button>
          {isEdit && (
            <Button variant="danger" onClick={handleDeleteCamera} disabled={deleting}>
              <Trash2 className="w-4 h-4" />
              {deleting ? "Deleting…" : "Delete Camera"}
            </Button>
          )}
        </div>
      </div>

      {showZoneFocus && needsZoneSetup && (
        <Card className="border-[var(--color-warning)] bg-[var(--color-warning-bg)] p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-4 w-4 text-[var(--color-warning)]" />
            <div>
              <p className="text-sm font-semibold text-[var(--color-text-primary)]">
                One more step: draw {zoneSetupTarget}
              </p>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                This camera needs zones. Use this order:
                {" "}Draw Zone, Save Zone, then Save Camera Changes if you also changed detections or profile.
              </p>
            </div>
          </div>
        </Card>
      )}

      {isEdit && camera && showLegacyNote && (
        <Card className="border-[var(--color-warning)] bg-[var(--color-warning-bg)] p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-4 w-4 text-[var(--color-warning)]" />
            <div>
              <p className="text-sm font-semibold text-[var(--color-text-primary)]">This camera is still on a legacy setup</p>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                Saving this screen will normalize the camera into the capability-planned setup.
              </p>
            </div>
          </div>
        </Card>
      )}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-6">
          <Card className="space-y-4">
            <div>
              <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Basics</h2>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                Start with the camera name, location, and source.
              </p>
            </div>

            <Field label="Camera Name">
              <input
                type="text"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="e.g. Main Gate Camera 1"
                className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
              />
            </Field>

            <Field label="Zone / Location">
              <input
                type="text"
                value={zone}
                onChange={(event) => setZone(event.target.value)}
                placeholder="e.g. Main Gate"
                className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
              />
            </Field>

            <Field label="Camera Profile">
              <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
                {PROFILE_OPTIONS.map((option) => (
                  <button
                    key={option.key}
                    type="button"
                    onClick={() => handleProfileChange(option.key)}
                    className={`rounded-[var(--radius-lg)] border p-3 text-left transition-colors ${
                      profile === option.key
                        ? "border-[var(--color-info)] bg-[var(--color-info-bg)]"
                        : "border-[var(--color-border-default)] bg-white hover:border-[var(--color-border-active)]"
                    }`}
                  >
                    <p className="text-sm font-semibold text-[var(--color-text-primary)]">{option.label}</p>
                    <p className="mt-1 text-xs leading-5 text-[var(--color-text-secondary)]">{option.description}</p>
                  </button>
                ))}
              </div>
            </Field>

            <div className="grid gap-4 md:grid-cols-2">
              <Field label="Stream Processing FPS">
                <input
                  type="number"
                  min="1"
                  max="60"
                  step="1"
                  value={streamFps}
                  onChange={(event) => setStreamFps(event.target.value)}
                  className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)]"
                />
                <p className="mt-1 text-xs text-[var(--color-text-secondary)]">
                  Frames consumed from this source each second. Prefer a lower-FPS AI substream when the camera supports one.
                </p>
              </Field>
              <Field label="Primary Inference FPS">
                <input
                  type="number"
                  min="0.5"
                  max={streamFps || "60"}
                  step="0.5"
                  value={inferenceFps}
                  onChange={(event) => setInferenceFps(event.target.value)}
                  className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)]"
                />
                <p className="mt-1 text-xs text-[var(--color-text-secondary)]">
                  Target detector cadence before motion and specialist gating. Use 2–5 FPS for normal safety monitoring.
                </p>
              </Field>
            </div>

            <Field label="Source Type">
              <div className="flex gap-2">
                {(["file", "rtsp"] as const).map((value) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setStreamType(value)}
                    className={`rounded-[var(--radius-md)] border px-4 py-2 text-sm font-medium transition-colors ${
                      streamType === value
                        ? "border-transparent bg-[var(--color-text-primary)] text-white"
                        : "border-[var(--color-border-default)] bg-white text-[var(--color-text-secondary)] hover:border-[var(--color-border-active)]"
                    }`}
                  >
                    {value === "file" ? "Video File" : "RTSP Stream"}
                  </button>
                ))}
              </div>
            </Field>

            {streamType === "file" ? (
              <Field label="Video File">
                <select
                  value={video}
                  onChange={(event) => setVideo(event.target.value)}
                  className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)]"
                >
                  {videos.length === 0 && <option value="">No video files available</option>}
                  {videos.map((item) => (
                    <option key={item} value={item}>{item}</option>
                  ))}
                </select>
              </Field>
            ) : (
              <div className="space-y-4">
                <Field label="RTSP URL">
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={rtspUrl}
                      onChange={(event) => setRtspUrl(event.target.value)}
                      placeholder="rtsp://192.168.1.100:554/stream1"
                      className="min-w-0 flex-1 rounded-[var(--radius-md)] border bg-white px-3 py-2 font-mono text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
                    />
                    <Button type="button" variant="secondary" onClick={() => void handleRtspTest()} disabled={rtspTestStatus === "testing"}>
                      {rtspTestStatus === "testing" ? "Testing..." : "Test"}
                    </Button>
                  </div>
                </Field>

                {rtspTestStatus !== "idle" && (
                  <div className={`rounded-[var(--radius-md)] border px-3 py-2 text-sm ${
                    rtspTestStatus === "success"
                      ? "border-[var(--color-success)] bg-[var(--color-success-bg)] text-[var(--color-text-primary)]"
                      : rtspTestStatus === "failed"
                        ? "border-[var(--color-danger)] bg-[var(--color-danger-bg)] text-[var(--color-text-primary)]"
                        : "border-[var(--color-border-default)] bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)]"
                  }`}>
                    {rtspTestStatus === "success"
                      ? `Connection OK${rtspResolution ? ` · ${rtspResolution[0]}x${rtspResolution[1]}` : ""}`
                      : rtspTestStatus === "failed"
                        ? rtspTestError || "Connection failed"
                        : "Testing stream..."}
                  </div>
                )}

                <details className="rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-white p-3">
                  <summary className="cursor-pointer text-sm font-semibold text-[var(--color-text-primary)]">
                    Stream discovery metadata
                  </summary>
                  <div className="mt-4 grid gap-4 md:grid-cols-2">
                    <Field label="Host / IP">
                      <input
                        type="text"
                        value={rtspHost}
                        onChange={(event) => setRtspHost(event.target.value)}
                        placeholder="192.168.1.100"
                        className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
                      />
                    </Field>
                    <Field label="Preferred Stream">
                      <select
                        value={preferredStream}
                        onChange={(event) => setPreferredStream(event.target.value)}
                        className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm focus:outline-2 focus:outline-[var(--color-info)]"
                      >
                        <option value="">Auto</option>
                        <option value="main">Main</option>
                        <option value="sub">Sub</option>
                        <option value="mjpeg">MJPEG</option>
                        <option value="custom">Custom</option>
                      </select>
                    </Field>
                    <Field label="RTSP Port">
                      <input
                        type="number"
                        min="1"
                        value={rtspPort}
                        onChange={(event) => setRtspPort(event.target.value)}
                        placeholder="554"
                        className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
                      />
                    </Field>
                    <Field label="ONVIF Port">
                      <input
                        type="number"
                        min="1"
                        value={onvifPort}
                        onChange={(event) => setOnvifPort(event.target.value)}
                        placeholder="80"
                        className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
                      />
                    </Field>
                    <Field label="Stream Path">
                      <input
                        type="text"
                        value={streamPath}
                        onChange={(event) => setStreamPath(event.target.value)}
                        placeholder="/stream1"
                        className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 font-mono text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
                      />
                    </Field>
                    <Field label="ONVIF UUID">
                      <input
                        type="text"
                        value={onvifUuid}
                        onChange={(event) => setOnvifUuid(event.target.value)}
                        placeholder="urn:uuid:..."
                        className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 font-mono text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
                      />
                    </Field>
                    <div className="md:col-span-2">
                      <Field label="Discovery Fingerprint">
                        <input
                          type="text"
                          value={discoveryFingerprint}
                          onChange={(event) => setDiscoveryFingerprint(event.target.value)}
                          placeholder="camera discovery fingerprint"
                          className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 font-mono text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
                        />
                      </Field>
                    </div>
                  </div>
                </details>

                {isEdit && camera?.credentials_configured && !replaceStoredCredentials ? (
                  <div className="rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-[var(--color-bg-tertiary)] px-3 py-3">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-sm font-medium text-[var(--color-text-primary)]">Stored credentials are configured</p>
                        <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                          Existing camera credentials will be preserved unless you choose to replace them.
                        </p>
                      </div>
                      <Button variant="secondary" size="sm" type="button" onClick={() => setReplaceStoredCredentials(true)}>
                        Replace Credentials
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="grid gap-4 md:grid-cols-2">
                    <Field label={isEdit ? "Username Override" : "Username"}>
                      <input
                        type="text"
                        value={rtspUsername}
                        onChange={(event) => setRtspUsername(event.target.value)}
                        placeholder="admin"
                        className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
                      />
                    </Field>
                    <Field label={isEdit ? "Password Override" : "Password"}>
                      <input
                        type="password"
                        value={rtspPassword}
                        onChange={(event) => setRtspPassword(event.target.value)}
                        placeholder={isEdit ? "Leave blank to keep current unless replacing" : "Camera password"}
                        className="w-full rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
                      />
                    </Field>
                  </div>
                )}
              </div>
            )}
          </Card>

          <Card className="space-y-4">
            <div>
              <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Enabled Detections</h2>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                Choose the capabilities this camera should run.
              </p>
            </div>
            <DetectionChecklist
              safetyRules={safetyRules}
              selectedKeys={selectedDetections}
              onChange={setSelectedDetections}
              allowCustomDetection
            />
            {usesCustomDetection && (
              <div className="rounded-[var(--radius-md)] border border-[var(--color-border-default)] bg-[var(--color-bg-tertiary)] p-3">
                <Field label="Custom detection terms">
                  <textarea
                    rows={3}
                    value={customDetectionTermsInput}
                    onChange={(event) => setCustomDetectionTermsInput(event.target.value)}
                    placeholder="Example: snake, medicine packet, specific toolcase"
                    className="w-full resize-y rounded-[var(--radius-md)] border bg-white px-3 py-2 text-sm placeholder:text-[var(--color-text-tertiary)] focus:outline-2 focus:outline-[var(--color-info)]"
                  />
                </Field>
                <p className="mt-2 text-xs text-[var(--color-text-secondary)]">
                  Enter comma-separated visible objects or conditions. Validate each term against representative site footage before operational use.
                </p>
              </div>
            )}
          </Card>

          <CameraCapabilityConfigPanel
            options={capabilityOptions}
            drafts={capabilityDrafts}
            onChange={setCapabilityDrafts}
          />

          <CameraEventPolicyEditor
            draft={eventPolicyDraft}
            onChange={setEventPolicyDraft}
            alertOutputs={alertOutputs}
            cameraScopedRules={cameraEventRules}
          />

          <CameraDetectionPolicyEditor
            options={capabilityOptions}
            drafts={detectionPolicyDrafts}
            onChange={setDetectionPolicyDrafts}
            alertOutputs={alertOutputs}
          />

          {isEdit && requiresZone && cameraId && (
            <div ref={zoneSectionRef}>
              <Card className="space-y-4">
                <div>
                  <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Zones</h2>
                  <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                    {zoneSetupDescription}
                  </p>
                </div>

                {needsZoneSetup && (
                  <div className="rounded-[var(--radius-md)] border border-[var(--color-warning)] bg-[var(--color-warning-bg)] px-3 py-2">
                    <p className="text-sm font-medium text-[var(--color-text-primary)]">Required sequence</p>
                    <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                      1. Click <strong>Draw Zone</strong> 2. Draw the polygon 3. Click <strong>Save Zone</strong>
                      4. Click <strong>Save Camera Changes</strong> only if you edited other camera settings.
                    </p>
                  </div>
                )}

                <PolygonDrawer
                  imageUrl={`${STREAM_BASE}/api/stream/${cameraId}`}
                  existingZones={zones}
                  defaultZoneType={defaultZoneType}
                  onSave={handleSaveZone}
                  onDelete={handleDeleteZone}
                  onUpdate={handleUpdateZone}
                  onDraftStateChange={setHasUnsavedZoneDraft}
                />

                <div className="space-y-2">
                  {hasUnsavedZoneDraft && (
                    <div className="rounded-[var(--radius-md)] border border-[var(--color-warning)] bg-[var(--color-warning-bg)] px-3 py-2">
                      <p className="text-sm font-medium text-[var(--color-text-primary)]">Zone draft not saved yet</p>
                      <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                        Click <strong>Save Zone</strong> inside the drawing panel before saving the camera.
                      </p>
                    </div>
                  )}
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-[var(--color-text-primary)]">Configured Zones</p>
                    <span className="text-xs text-[var(--color-text-secondary)]">
                      {zones.length} zone{zones.length !== 1 ? "s" : ""}
                    </span>
                  </div>
                  {zones.length === 0 ? (
                    <p className="text-sm text-[var(--color-text-tertiary)]">No zones configured yet.</p>
                  ) : (
                    <div className="space-y-2">
                      {zones.map((item) => (
                        <div key={item.id} className="flex items-center justify-between rounded-[var(--radius-md)] border px-3 py-2">
                          <div className="flex items-center gap-2">
                            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                            <div>
                              <p className="text-sm font-medium text-[var(--color-text-primary)]">{item.name}</p>
                              <p className="text-xs text-[var(--color-text-secondary)]">{item.type} · {item.points.length} points</p>
                            </div>
                          </div>
                          <Button variant="ghost" size="sm" onClick={() => handleDeleteZone(item.id)}>
                            <Trash2 className="w-3.5 h-3.5" />
                            Remove
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </Card>
            </div>
          )}
        </div>

        <div className="space-y-6">
          <Card className="space-y-4 xl:sticky xl:top-20">
            <div>
              <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Execution Plan</h2>
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                This is the grouped runtime plan the backend will compile for this camera.
              </p>
            </div>

            <div className="space-y-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">Purpose</p>
                <p className="mt-1 text-sm font-medium text-[var(--color-text-primary)]">{purpose}</p>
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">Selected Detections</p>
                {selectedLabels.length > 0 ? (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {selectedLabels.map((label) => (
                      <Badge key={label} variant="info">{label}</Badge>
                    ))}
                  </div>
                ) : (
                  <p className="mt-1 text-sm text-[var(--color-text-secondary)]">Monitoring only</p>
                )}
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">Profile</p>
                <p className="mt-1 text-sm font-medium text-[var(--color-text-primary)]">
                  {PROFILE_OPTIONS.find((option) => option.key === profile)?.label || profile}
                </p>
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">Detection Runtime</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  <Badge variant={selectedDetections.length > 0 ? "success" : "default"}>
                    {selectedDetections.length > 0 ? "Runs selected detections" : "Monitoring only"}
                  </Badge>
                  <Badge variant="info">Primary {inferenceFps || "—"} FPS</Badge>
                  {runtimeLimitedDetectionCount > 0 ? (
                    <Badge variant="info">{runtimeLimitedDetectionCount} scheduled</Badge>
                  ) : (
                    <Badge variant="default">Always active</Badge>
                  )}
                  {Object.keys(safetyRuleOverridesPayload).length > 0 && (
                    <Badge variant="info">{Object.keys(safetyRuleOverridesPayload).length} rule override{Object.keys(safetyRuleOverridesPayload).length !== 1 ? "s" : ""}</Badge>
                  )}
                  {customDetectionPolicyCount > 0 && (
                    <Badge variant="info">{customDetectionPolicyCount} custom alert polic{customDetectionPolicyCount === 1 ? "y" : "ies"}</Badge>
                  )}
                  {trainedPpeEnabled && (
                    <Badge variant="info">Trained apron / harness detector</Badge>
                  )}
                </div>
              </div>

              <div className="grid gap-3 md:grid-cols-2">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">Expected Load</p>
                  <p className="mt-1 text-sm font-medium text-[var(--color-text-primary)]">
                    {previewLoading ? "Refreshing…" : planPreview?.execution_plan?.runtime_load || "Low"}
                  </p>
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">Zones</p>
                  <p className="mt-1 text-sm font-medium text-[var(--color-text-primary)]">
                    {requiresZone
                      ? isEdit
                        ? zones.length > 0
                          ? `${zones.length} configured`
                          : "Setup required"
                        : "Required after save"
                      : "Not required"}
                  </p>
                </div>
              </div>

              {planPreview?.missing_model_keys?.length ? (
                <div className="rounded-[var(--radius-lg)] border border-[var(--color-warning)] bg-[var(--color-warning-bg)] p-4">
                  <p className="text-sm font-semibold text-[var(--color-text-primary)]">Detection setup needed</p>
                  <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                    Saving will pause while {planPreview.missing_model_keys.length} required detector asset{planPreview.missing_model_keys.length !== 1 ? "s are" : " is"} prepared.
                  </p>
                </div>
              ) : (
                <div className="rounded-[var(--radius-lg)] border border-[var(--color-success)] bg-[var(--color-success-bg)] p-4">
                  <p className="text-sm font-semibold text-[var(--color-text-primary)]">Required models are ready</p>
                  <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                    The current execution plan can run immediately after save.
                  </p>
                </div>
              )}

              {preservedRuleIds.length > 0 && (
                <p className="text-xs text-[var(--color-text-secondary)]">
                  Additional custom rules already attached to this camera will be preserved.
                </p>
              )}
            </div>

            <div className="flex flex-col gap-2 pt-2">
              <Button onClick={() => void submitCamera()} disabled={saving}>
                {saving
                  ? isEdit ? "Saving…" : "Creating…"
                  : hasUnsavedZoneDraft
                    ? "Save Zone Draft First"
                    : needsZoneSetup
                      ? "Zone Still Needs Saving"
                      : isEdit ? "Save Camera Changes" : requiresZone ? "Save and Continue to Zones" : "Create Camera"}
              </Button>
              <Button variant="secondary" onClick={() => navigate(isEdit && cameraId ? `/configure/cameras/${cameraId}` : "/configure/cameras")}>
                Cancel
              </Button>
            </div>
          </Card>

          {import.meta.env.DEV && (
            <details className="rounded-[var(--radius-lg)] border bg-white p-4">
              <summary className="cursor-pointer text-sm font-semibold text-[var(--color-text-primary)]">Diagnostics</summary>
              <div className="mt-3 space-y-2 text-sm text-[var(--color-text-secondary)]">
                <p><span className="font-medium text-[var(--color-text-primary)]">Camera ID:</span> {camera?.id || "Not created yet"}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Source Type:</span> {streamType === "rtsp" ? "RTSP" : "Video File"}</p>
                <p className="break-all"><span className="font-medium text-[var(--color-text-primary)]">Source Value:</span> {streamType === "rtsp" ? rtspUrl || "Not set" : video || "Not set"}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Credentials Update:</span> {streamType === "rtsp" ? (replaceStoredCredentials || rtspUsername || rtspPassword ? "Provided in request" : camera?.credentials_configured ? "Preserve stored" : "None") : "N/A"}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Profile:</span> {profile}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Stream Processing FPS:</span> {streamFps}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Primary Inference FPS:</span> {inferenceFps}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Capabilities:</span> {selectedDetections.join(", ") || "None"}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Custom Terms:</span> {customDetectionTerms.join(", ") || "None"}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Rule IDs:</span> {selectedRuleIds.join(", ") || "None"}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Model Overrides:</span> {JSON.stringify(capabilityModelOverridesPayload)}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Model Stack:</span> {planPreview?.execution_plan?.model_stack?.join(", ") || "None"}</p>
                <p><span className="font-medium text-[var(--color-text-primary)]">Missing Model Keys:</span> {planPreview?.missing_model_keys?.join(", ") || "None"}</p>
              </div>
            </details>
          )}
        </div>
      </div>
    </div>
  )
}
