import { useState, useMemo } from "react"
import { Plus, Search, Trash2, Pencil, Upload, Shield, HelpCircle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { mockFaces, mockFaceMatches, getAvatarUrl, type EnrolledFace, type FaceMatch } from "@/data/mockFaces"

function isExpired(dateStr: string | null): boolean {
  if (!dateStr) return false
  return new Date(dateStr) < new Date()
}

function formatTime(timestamp: string): string {
  const d = new Date(timestamp)
  return d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })
}

const AVATAR_COLORS = ["#2563eb", "#7c3aed", "#059669", "#dc2626", "#0891b2", "#be185d", "#475569", "#0d9488", "#6366f1", "#f59e0b"]

function groupBadgeVariant(group: EnrolledFace["group"]): "info" | "warning" | "default" {
  switch (group) {
    case "employees": return "info"
    case "visitors": return "warning"
    case "contractors": return "default"
  }
}

function groupLabel(group: EnrolledFace["group"]): string {
  return group.charAt(0).toUpperCase() + group.slice(1)
}

export function FaceEnrollment() {
  const [faces, setFaces] = useState<EnrolledFace[]>(mockFaces)
  const [matches] = useState<FaceMatch[]>(mockFaceMatches)
  const [groupFilter, setGroupFilter] = useState<"all" | "employees" | "visitors" | "contractors">("all")
  const [search, setSearch] = useState("")
  const [showEnrollModal, setShowEnrollModal] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<EnrolledFace | null>(null)

  // Enroll modal state
  const [enrollName, setEnrollName] = useState("")
  const [enrollGroup, setEnrollGroup] = useState<EnrolledFace["group"]>("employees")
  const [enrollValidUntil, setEnrollValidUntil] = useState("")
  const [enrollConsent, setEnrollConsent] = useState(false)
  const [enrollConsentMethod, setEnrollConsentMethod] = useState("")

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
    setEnrollName("")
    setEnrollGroup("employees")
    setEnrollValidUntil("")
    setEnrollConsent(false)
    setEnrollConsentMethod("")
  }

  function handleEnroll() {
    if (!enrollName.trim() || !enrollConsent || !enrollConsentMethod) return
    const nameParts = enrollName.trim().split(" ")
    const initials = nameParts.length >= 2
      ? (nameParts[0][0] + nameParts[nameParts.length - 1][0]).toUpperCase()
      : enrollName.trim().substring(0, 2).toUpperCase()
    const color = AVATAR_COLORS[Math.floor(Math.random() * AVATAR_COLORS.length)]
    const face: EnrolledFace = {
      id: `fc-${Date.now()}`,
      name: enrollName.trim(),
      group: enrollGroup,
      initials,
      color,
      validUntil: enrollValidUntil || null,
      enrolledAt: new Date().toISOString().split("T")[0],
      consentMethod: enrollConsentMethod,
    }
    setFaces((prev) => [face, ...prev])
    resetEnrollModal()
  }

  function handleDelete() {
    if (!deleteTarget) return
    setFaces((prev) => prev.filter((f) => f.id !== deleteTarget.id))
    setDeleteTarget(null)
  }

  const enrollCanSubmit = enrollName.trim() && enrollConsent && enrollConsentMethod

  const filterPills: { label: string; value: typeof groupFilter }[] = [
    { label: "All", value: "all" },
    { label: "Employees", value: "employees" },
    { label: "Visitors", value: "visitors" },
    { label: "Contractors", value: "contractors" },
  ]

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold text-[var(--color-text-primary)]">Face Enrollment</h1>
          <Badge variant="info">{faces.length} enrolled</Badge>
        </div>
        <Button variant="primary" size="md" onClick={() => setShowEnrollModal(true)}>
          <Plus className="w-4 h-4" /> Enroll Face
        </Button>
      </div>

      {/* Group filter pills */}
      <div className="flex gap-1">
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

      {/* Search */}
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

      {/* Face card grid */}
      <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
        {filteredFaces.map((face) => (
          <Card key={face.id} className="flex flex-col items-center text-center p-4">
            <img
              src={getAvatarUrl(face.initials, face.color)}
              alt={face.name}
              className="w-16 h-16 rounded-full mb-3"
            />
            <span className="font-medium text-sm text-[var(--color-text-primary)]">{face.name}</span>
            <div className="mt-1">
              <Badge variant={groupBadgeVariant(face.group)}>{groupLabel(face.group)}</Badge>
            </div>
            {face.group === "visitors" && face.validUntil && isExpired(face.validUntil) && (
              <Badge variant="warning" className="mt-1">Expired</Badge>
            )}
            <span className="text-xs text-[var(--color-text-secondary)] mt-1">Enrolled {face.enrolledAt}</span>
            <div className="flex gap-1 mt-3">
              <Button variant="ghost" size="sm" title="Edit">
                <Pencil className="w-3.5 h-3.5" />
              </Button>
              <Button variant="ghost" size="sm" title="Delete" onClick={() => setDeleteTarget(face)}>
                <Trash2 className="w-3.5 h-3.5" />
              </Button>
            </div>
          </Card>
        ))}
      </div>

      {/* Recent Matches */}
      <div className="space-y-3">
        <h2 className="text-base font-semibold text-[var(--color-text-primary)]">Recent Matches</h2>
        <div className="space-y-2">
          {matches.map((match) => (
            <div
              key={match.id}
              className={cn(
                "flex items-center gap-3 px-4 py-3 bg-white border rounded-[var(--radius-lg)] transition-colors",
                match.isUnknown && "border-l-4 border-l-[var(--color-warning)]"
              )}
            >
              {match.isUnknown ? (
                <div className="w-10 h-10 rounded-full bg-[var(--color-bg-tertiary)] flex items-center justify-center flex-shrink-0">
                  <HelpCircle className="w-5 h-5 text-[var(--color-text-tertiary)]" />
                </div>
              ) : (
                <img
                  src={getAvatarUrl(
                    match.personName ? match.personName.split(" ").map((n) => n[0]).join("").substring(0, 2).toUpperCase() : "??",
                    faces.find((f) => f.id === match.personId)?.color || "#a3a3a3"
                  )}
                  alt={match.personName || "Unknown"}
                  className="w-10 h-10 rounded-full flex-shrink-0"
                />
              )}
              <div className="flex-1 min-w-0">
                <span className="text-sm font-medium text-[var(--color-text-primary)]">
                  {match.isUnknown ? "Unknown Person" : match.personName}
                </span>
                <div className="text-xs text-[var(--color-text-secondary)]">{match.cameraName}</div>
              </div>
              <span className="text-xs font-mono text-[var(--color-text-secondary)]">{formatTime(match.timestamp)}</span>
              <span className="text-xs text-[var(--color-text-secondary)] w-12 text-right">
                {match.confidence != null ? `${match.confidence.toFixed(1)}%` : "—"}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Enroll Modal */}
      {showEnrollModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/40" onClick={resetEnrollModal} />
          <div className="relative bg-white rounded-[var(--radius-xl)] shadow-xl border w-full max-w-lg p-6 space-y-4 max-h-[90vh] overflow-y-auto">
            <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">Enroll Face</h2>

            <div className="space-y-3">
              {/* Photo upload area */}
              <div
                onClick={() => alert("Photo upload available in production")}
                className="border-2 border-dashed rounded-[var(--radius-lg)] p-8 flex flex-col items-center gap-2 cursor-pointer hover:bg-[var(--color-bg-secondary)] transition-colors"
              >
                <Upload className="w-8 h-8 text-[var(--color-text-tertiary)]" />
                <span className="text-sm text-[var(--color-text-secondary)]">Drop image or click to upload</span>
              </div>

              {/* Name */}
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

              {/* Group */}
              <div>
                <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Group</label>
                <select
                  value={enrollGroup}
                  onChange={(e) => setEnrollGroup(e.target.value as EnrolledFace["group"])}
                  className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] cursor-pointer"
                >
                  <option value="employees">Employees</option>
                  <option value="visitors">Visitors</option>
                  <option value="contractors">Contractors</option>
                </select>
              </div>

              {/* Valid Until */}
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

              {/* DPDPA Consent */}
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
                    I confirm that {enrollName.trim() || "this person"} has been informed that their facial image will be processed by SafetyLens for access control and safety monitoring, stored on-premise, and they have provided consent.
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
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <Button variant="secondary" size="md" onClick={resetEnrollModal}>Cancel</Button>
              <Button variant="primary" size="md" onClick={handleEnroll} disabled={!enrollCanSubmit}>Enroll</Button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/40" onClick={() => setDeleteTarget(null)} />
          <div className="relative bg-white rounded-[var(--radius-xl)] shadow-xl border w-full max-w-sm p-6 space-y-4">
            <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">Delete Enrolled Face</h2>
            <p className="text-sm text-[var(--color-text-secondary)]">
              This will permanently delete {deleteTarget.name}&apos;s facial data from the system. This action cannot be undone.
            </p>
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="secondary" size="md" onClick={() => setDeleteTarget(null)}>Cancel</Button>
              <Button variant="danger" size="md" onClick={handleDelete}>Delete</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
