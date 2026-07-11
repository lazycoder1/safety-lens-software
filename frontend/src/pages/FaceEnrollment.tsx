import { useEffect, useMemo, useRef, useState } from "react"
import { Camera, HelpCircle, Pencil, Plus, Search, Shield, Trash2, Upload, Video } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import {
  API_BASE,
  ApiError,
  deleteFace,
  enrollFace,
  enrollFaceLive,
  getCameras,
  getFaceLogs,
  getFaces,
  getToken,
  type EnrolledFaceApi,
  type FaceLogApi,
} from "@/lib/api"
import { cn } from "@/lib/utils"

type FaceGroup = EnrolledFaceApi["group"]
type EnrollMode = "upload" | "live"

const AVATAR_COLORS = ["#2563eb", "#7c3aed", "#059669", "#dc2626", "#0891b2", "#be185d", "#475569", "#0d9488", "#6366f1", "#f59e0b"]

function isExpired(dateStr: string | null): boolean {
  if (!dateStr) return false
  return new Date(dateStr) < new Date()
}

function formatTime(timestamp: string): string {
  const d = new Date(timestamp)
  return d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })
}

function initialsFor(name: string): string {
  const parts = name.trim().split(/\s+/)
  return parts.length >= 2
    ? `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase()
    : name.trim().substring(0, 2).toUpperCase()
}

function colorFor(id: string): string {
  const total = Array.from(id).reduce((sum, char) => sum + char.charCodeAt(0), 0)
  return AVATAR_COLORS[total % AVATAR_COLORS.length]
}

function avatarDataUrl(face: EnrolledFaceApi): string {
  const initials = initialsFor(face.name)
  const color = colorFor(face.id)
  return `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80"><rect width="80" height="80" rx="40" fill="${color}"/><text x="50%" y="50%" text-anchor="middle" dy=".35em" fill="white" font-family="system-ui" font-size="28" font-weight="600">${initials}</text></svg>`)}`
}

function groupBadgeVariant(group: FaceGroup): "info" | "warning" | "critical" | "default" {
  switch (group) {
    case "employees": return "info"
    case "visitors": return "warning"
    case "watchlist": return "critical"
    case "contractors": return "default"
  }
}

function groupLabel(group: FaceGroup): string {
  return group.charAt(0).toUpperCase() + group.slice(1)
}

export function FaceEnrollment() {
  const [faces, setFaces] = useState<EnrolledFaceApi[]>([])
  const [matches, setMatches] = useState<FaceLogApi[]>([])
  const [cameras, setCameras] = useState<Array<{ id: string; name: string; runtime_status?: string; status?: string }>>([])
  const [groupFilter, setGroupFilter] = useState<"all" | FaceGroup>("all")
  const [search, setSearch] = useState("")
  const [showEnrollModal, setShowEnrollModal] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<EnrolledFaceApi | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const [photoUrls, setPhotoUrls] = useState<Record<string, string>>({})
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const cameraInputRef = useRef<HTMLInputElement | null>(null)

  const [enrollMode, setEnrollMode] = useState<EnrollMode>("upload")
  const [photoFile, setPhotoFile] = useState<File | null>(null)
  const [selectedCameraId, setSelectedCameraId] = useState("")
  const [enrollName, setEnrollName] = useState("")
  const [enrollGroup, setEnrollGroup] = useState<FaceGroup>("employees")
  const [enrollValidUntil, setEnrollValidUntil] = useState("")
  const [enrollConsent, setEnrollConsent] = useState(false)
  const [enrollConsentMethod, setEnrollConsentMethod] = useState("")

  useEffect(() => {
    refreshData()
  }, [])

  useEffect(() => {
    const objectUrls: string[] = []
    let cancelled = false

    async function loadPhotos() {
      const token = getToken()
      const entries = await Promise.all(
        faces
          .filter((face) => face.photoUrl)
          .map(async (face) => {
            try {
              const res = await fetch(`${API_BASE}${face.photoUrl}`, {
                headers: token ? { Authorization: `Bearer ${token}` } : {},
              })
              if (!res.ok) return null
              const url = URL.createObjectURL(await res.blob())
              objectUrls.push(url)
              return [face.id, url] as const
            } catch {
              return null
            }
          })
      )
      if (!cancelled) {
        setPhotoUrls(Object.fromEntries(entries.filter(Boolean) as Array<readonly [string, string]>))
      }
    }

    loadPhotos()
    return () => {
      cancelled = true
      objectUrls.forEach((url) => URL.revokeObjectURL(url))
    }
  }, [faces])

  async function refreshData() {
    setError("")
    setLoading(true)
    try {
      const [faceRows, logRows, cameraRows] = await Promise.all([
        getFaces(),
        getFaceLogs({ limit: 50 }),
        getCameras(),
      ])
      setFaces(faceRows)
      setMatches(logRows)
      setCameras(cameraRows)
      if (!selectedCameraId && cameraRows.length > 0) {
        setSelectedCameraId(cameraRows[0].id)
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load face recognition data")
    } finally {
      setLoading(false)
    }
  }

  const filteredFaces = useMemo(() => {
    let list = faces
    if (groupFilter !== "all") {
      list = list.filter((f) => f.group === groupFilter)
    }
    if (search) {
      const q = search.toLowerCase()
      list = list.filter((f) => f.name.toLowerCase().includes(q))
    }
    return list
  }, [faces, groupFilter, search])

  function resetEnrollModal() {
    setShowEnrollModal(false)
    setEnrollMode("upload")
    setPhotoFile(null)
    setEnrollName("")
    setEnrollGroup("employees")
    setEnrollValidUntil("")
    setEnrollConsent(false)
    setEnrollConsentMethod("")
    setError("")
  }

  async function handleEnroll() {
    if (!enrollCanSubmit || submitting) return
    setSubmitting(true)
    setError("")
    try {
      const payload = {
        name: enrollName.trim(),
        group: enrollGroup,
        validUntil: enrollValidUntil || null,
        consentMethod: enrollConsentMethod,
        consentConfirmed: enrollConsent,
      }
      if (enrollMode === "upload") {
        if (!photoFile) return
        await enrollFace({ ...payload, photo: photoFile })
      } else {
        await enrollFaceLive({ ...payload, cameraId: selectedCameraId })
      }
      resetEnrollModal()
      await refreshData()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Enrollment failed")
    } finally {
      setSubmitting(false)
    }
  }

  function handlePhotoSelected(file: File | undefined | null) {
    if (!file) return
    setPhotoFile(file)
  }

  async function handleDelete() {
    if (!deleteTarget) return
    setSubmitting(true)
    setError("")
    try {
      await deleteFace(deleteTarget.id)
      setDeleteTarget(null)
      await refreshData()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not delete enrolled face")
    } finally {
      setSubmitting(false)
    }
  }

  const enrollCanSubmit = Boolean(
    enrollName.trim() &&
    enrollConsent &&
    enrollConsentMethod &&
    (enrollMode === "upload" ? photoFile : selectedCameraId)
  )

  const filterPills: { label: string; value: "all" | FaceGroup }[] = [
    { label: "All", value: "all" },
    { label: "Employees", value: "employees" },
    { label: "Visitors", value: "visitors" },
    { label: "Contractors", value: "contractors" },
    { label: "Watchlist", value: "watchlist" },
  ]

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold text-[var(--color-text-primary)]">Face Enrollment</h1>
          <Badge variant="info">{faces.length} enrolled</Badge>
        </div>
        <Button variant="primary" size="md" onClick={() => setShowEnrollModal(true)}>
          <Plus className="w-4 h-4" /> Enroll Face
        </Button>
      </div>

      {error && (
        <div className="rounded-[var(--radius-lg)] border border-[var(--color-critical)] bg-[var(--color-critical-bg)] px-4 py-3 text-sm text-[var(--color-critical)]">
          {error}
        </div>
      )}

      <div className="flex gap-1 flex-wrap">
        {filterPills.map((pill) => (
          <button
            key={pill.value}
            onClick={() => setGroupFilter(pill.value)}
            className={cn(
              "px-3 py-1.5 text-xs font-medium rounded-[var(--radius-md)] transition-colors cursor-pointer",
              groupFilter === pill.value
                ? "bg-[var(--color-text-primary)] text-white"
                : "bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] hover:bg-[var(--color-border-default)]"
            )}
          >
            {pill.label}
          </button>
        ))}
      </div>

      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--color-text-tertiary)]" />
        <input
          type="text"
          placeholder="Search by name..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full pl-9 pr-4 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent"
        />
      </div>

      {loading ? (
        <Card className="p-6 text-sm text-[var(--color-text-secondary)]">Loading face recognition data...</Card>
      ) : filteredFaces.length === 0 ? (
        <Card className="p-6 text-sm text-[var(--color-text-secondary)]">No enrolled faces found.</Card>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {filteredFaces.map((face) => (
            <Card key={face.id} className="flex flex-col items-center text-center p-4">
              <img
                src={photoUrls[face.id] || avatarDataUrl(face)}
                alt={face.name}
                className="w-16 h-16 rounded-full mb-3 object-cover"
              />
              <span className="font-medium text-sm text-[var(--color-text-primary)]">{face.name}</span>
              <div className="mt-1">
                <Badge variant={groupBadgeVariant(face.group)}>{groupLabel(face.group)}</Badge>
              </div>
              {face.validUntil && isExpired(face.validUntil) && (
                <Badge variant="warning" className="mt-1">Expired</Badge>
              )}
              <span className="text-xs text-[var(--color-text-secondary)] mt-1">
                Enrolled {new Date(face.enrolledAt).toLocaleDateString("en-IN")}
              </span>
              <div className="flex gap-1 mt-3">
                <Button variant="ghost" size="sm" title="Edit" disabled>
                  <Pencil className="w-3.5 h-3.5" />
                </Button>
                <Button variant="ghost" size="sm" title="Delete" onClick={() => setDeleteTarget(face)}>
                  <Trash2 className="w-3.5 h-3.5" />
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}

      <div className="space-y-3">
        <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Recent Matches</h2>
        <div className="space-y-2">
          {matches.length === 0 ? (
            <Card className="p-4 text-sm text-[var(--color-text-secondary)]">No face events logged yet.</Card>
          ) : matches.map((match) => (
            <div
              key={match.id}
              className={cn(
                "flex items-center gap-3 px-4 py-3 bg-white border rounded-[var(--radius-lg)] transition-colors",
                match.eventType !== "face_match" && "border-l-4 border-l-[var(--color-warning)]"
              )}
            >
              {match.eventType === "face_match" ? (
                <div className="w-10 h-10 rounded-full bg-[var(--color-info-bg)] flex items-center justify-center flex-shrink-0 text-sm font-semibold text-[var(--color-info)]">
                  {initialsFor(match.personName || "OK")}
                </div>
              ) : match.eventType === "face_low_quality" ? (
                <div className="w-10 h-10 rounded-full bg-[var(--color-bg-tertiary)] flex items-center justify-center flex-shrink-0">
                  <Video className="w-5 h-5 text-[var(--color-text-tertiary)]" />
                </div>
              ) : (
                <div className="w-10 h-10 rounded-full bg-[var(--color-bg-tertiary)] flex items-center justify-center flex-shrink-0">
                  <HelpCircle className="w-5 h-5 text-[var(--color-text-tertiary)]" />
                </div>
              )}
              <div className="flex-1 min-w-0">
                <span className="text-sm font-medium text-[var(--color-text-primary)]">
                  {match.eventType === "face_match" ? match.personName : match.eventType === "face_low_quality" ? "Low-quality face" : "Unknown Person"}
                </span>
                <div className="text-xs text-[var(--color-text-secondary)]">
                  {match.cameraName}{match.qualityReason ? ` - ${match.qualityReason}` : ""}
                </div>
              </div>
              <span className="text-xs font-mono text-[var(--color-text-secondary)]">{formatTime(match.timestamp)}</span>
              <span className="text-xs text-[var(--color-text-secondary)] w-12 text-right">
                {match.confidence != null ? `${match.confidence.toFixed(1)}%` : "--"}
              </span>
            </div>
          ))}
        </div>
      </div>

      {showEnrollModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/40" onClick={resetEnrollModal} />
          <div className="relative bg-white rounded-[var(--radius-xl)] shadow-xl border w-full max-w-lg p-6 space-y-4 max-h-[90vh] overflow-y-auto">
            <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">Enroll Face</h2>

            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setEnrollMode("upload")}
                className={cn("rounded-[var(--radius-md)] border px-3 py-2 text-sm", enrollMode === "upload" && "border-[var(--color-info)] bg-[var(--color-info-bg)]")}
              >
                Upload Photo
              </button>
              <button
                type="button"
                onClick={() => setEnrollMode("live")}
                className={cn("rounded-[var(--radius-md)] border px-3 py-2 text-sm", enrollMode === "live" && "border-[var(--color-info)] bg-[var(--color-info-bg)]")}
              >
                Live Capture
              </button>
            </div>

            {enrollMode === "upload" ? (
              <div className="border rounded-[var(--radius-lg)] p-4 space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <Button variant="secondary" size="md" onClick={() => cameraInputRef.current?.click()}>
                    <Camera className="w-4 h-4" /> Take Photo
                  </Button>
                  <Button variant="secondary" size="md" onClick={() => fileInputRef.current?.click()}>
                    <Upload className="w-4 h-4" /> Choose File
                  </Button>
                </div>
                <div
                  onClick={() => fileInputRef.current?.click()}
                  onDrop={(event) => {
                    event.preventDefault()
                    handlePhotoSelected(event.dataTransfer.files?.[0])
                  }}
                  onDragOver={(event) => event.preventDefault()}
                  className="border-2 border-dashed rounded-[var(--radius-lg)] p-6 flex flex-col items-center gap-2 cursor-pointer hover:bg-[var(--color-bg-secondary)] transition-colors"
                >
                  <Upload className="w-7 h-7 text-[var(--color-text-tertiary)]" />
                  <span className="text-sm text-[var(--color-text-secondary)]">{photoFile ? photoFile.name : "Drop image here or choose a file"}</span>
                  <span className="text-xs text-[var(--color-text-tertiary)] text-center">Use one clear, front-facing face. Phone camera photos are supported.</span>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(event) => handlePhotoSelected(event.target.files?.[0])}
                />
                <input
                  ref={cameraInputRef}
                  type="file"
                  accept="image/*"
                  capture="environment"
                  className="hidden"
                  onChange={(event) => handlePhotoSelected(event.target.files?.[0])}
                />
              </div>
            ) : (
              <div>
                <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Capture from camera</label>
                <select
                  value={selectedCameraId}
                  onChange={(e) => setSelectedCameraId(e.target.value)}
                  className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] cursor-pointer"
                >
                  {cameras.map((camera) => (
                    <option key={camera.id} value={camera.id}>{camera.name}</option>
                  ))}
                </select>
              </div>
            )}

            <div>
              <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Name *</label>
              <input
                type="text"
                value={enrollName}
                onChange={(e) => setEnrollName(e.target.value)}
                placeholder="Full name"
                className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)]"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Group</label>
              <select
                value={enrollGroup}
                onChange={(e) => setEnrollGroup(e.target.value as FaceGroup)}
                className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] cursor-pointer"
              >
                <option value="employees">Employees</option>
                <option value="visitors">Visitors</option>
                <option value="contractors">Contractors</option>
                <option value="watchlist">Watchlist</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Valid Until</label>
              <input
                type="date"
                value={enrollValidUntil}
                onChange={(e) => setEnrollValidUntil(e.target.value)}
                className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)]"
              />
              <p className="text-xs text-[var(--color-text-tertiary)] mt-1">Leave empty for permanent enrollment</p>
            </div>

            <div className="border rounded-[var(--radius-lg)] p-4 space-y-3">
              <div className="flex items-center gap-2 text-sm font-medium text-[var(--color-text-primary)]">
                <Shield className="w-4 h-4 text-[var(--color-info)]" />
                DPDPA Consent
              </div>
              <label className="flex items-start gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enrollConsent}
                  onChange={(e) => setEnrollConsent(e.target.checked)}
                  className="mt-0.5 cursor-pointer"
                />
                <span className="text-xs text-[var(--color-text-secondary)] leading-relaxed">
                  I confirm that {enrollName.trim() || "this person"} has been informed that their facial image will be processed by Rakshak Lens for access control and safety monitoring, stored on-premise, and they have provided consent.
                </span>
              </label>
              <div>
                <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Consent Method</label>
                <select
                  value={enrollConsentMethod}
                  onChange={(e) => setEnrollConsentMethod(e.target.value)}
                  className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] cursor-pointer"
                >
                  <option value="">Select method...</option>
                  <option value="Written form">Written form on file</option>
                  <option value="Verbal consent">Verbal consent recorded</option>
                  <option value="Email consent">Email consent received</option>
                </select>
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <Button variant="secondary" size="md" onClick={resetEnrollModal}>Cancel</Button>
              <Button variant="primary" size="md" onClick={handleEnroll} disabled={!enrollCanSubmit || submitting}>
                {submitting ? "Enrolling..." : "Enroll"}
              </Button>
            </div>
          </div>
        </div>
      )}

      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/40" onClick={() => setDeleteTarget(null)} />
          <div className="relative bg-white rounded-[var(--radius-xl)] shadow-xl border w-full max-w-sm p-6 space-y-4">
            <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">Delete Enrolled Face</h2>
            <p className="text-sm text-[var(--color-text-secondary)]">
              This deactivates {deleteTarget.name}&apos;s biometric match record. Historical logs remain for audit review.
            </p>
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="secondary" size="md" onClick={() => setDeleteTarget(null)}>Cancel</Button>
              <Button variant="danger" size="md" onClick={handleDelete} disabled={submitting}>Delete</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
