import { useEffect, useMemo, useRef, useState } from "react"
import * as Tabs from "@radix-ui/react-tabs"
import { AlertTriangle, CheckCircle, Eye, Pencil, Plus, Search, Trash2, Upload, User, XCircle } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  API_BASE,
  createPlateListEntry,
  deletePlateListEntry,
  getPlateListEntries,
  getPlateReads,
  getToken,
  importPlateListCsv,
  updatePlateListEntry,
  type PlateListEntryApi,
  type PlateListType,
  type PlateMatchStatus,
  type PlateReadApi,
} from "@/lib/api"
import { cn } from "@/lib/utils"

function normalizePlate(value: string) {
  return value.toUpperCase().replace(/[^A-Z0-9]/g, "")
}

function isIndianPlate(value: string) {
  return /^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$/.test(normalizePlate(value))
}

function formatTime(timestamp: string): string {
  const d = new Date(timestamp)
  return d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })
}

function isExpired(dateStr: string | null): boolean {
  if (!dateStr) return false
  return new Date(dateStr) < new Date()
}

function StatusBadge({ status }: { status: PlateMatchStatus }) {
  switch (status) {
    case "whitelist":
      return <Badge variant="success"><CheckCircle className="w-3 h-3" /> Whitelist</Badge>
    case "blocked":
      return <Badge variant="critical"><XCircle className="w-3 h-3" /> Blocked</Badge>
    case "visitor":
      return <Badge variant="info"><User className="w-3 h-3" /> Visitor</Badge>
    case "low_confidence":
      return <Badge variant="warning"><AlertTriangle className="w-3 h-3" /> Low confidence</Badge>
    default:
      return <Badge variant="warning"><AlertTriangle className="w-3 h-3" /> Unknown</Badge>
  }
}

function ListBadge({ list }: { list: PlateListType }) {
  switch (list) {
    case "whitelist":
      return <Badge variant="success">Whitelist</Badge>
    case "blocked":
      return <Badge variant="critical">Blocked</Badge>
    case "visitors":
      return <Badge variant="info">Visitors</Badge>
  }
}

function ProtectedThumb({ url, alt, className = "w-14 h-10" }: { url: string | null; alt: string; className?: string }) {
  const [src, setSrc] = useState<string | null>(null)

  useEffect(() => {
    let objectUrl: string | null = null
    let cancelled = false
    setSrc(null)
    if (!url) return
    fetch(`${API_BASE}${url}`, { headers: getToken() ? { Authorization: `Bearer ${getToken()}` } : {} })
      .then((res) => (res.ok ? res.blob() : null))
      .then((blob) => {
        if (!blob || cancelled) return
        objectUrl = URL.createObjectURL(blob)
        setSrc(objectUrl)
      })
      .catch(() => {})
    return () => {
      cancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [url])

  if (!src) return <div className={cn(className, "rounded bg-[var(--color-bg-tertiary)] border")} />
  return <img src={src} alt={alt} className={cn(className, "rounded object-cover border")} />
}

const emptyForm = {
  id: "",
  plateNumber: "",
  owner: "",
  vehicle: "",
  list: "whitelist" as PlateListType,
  validFrom: "",
  validUntil: "",
}

export function PlateManagement() {
  const [reads, setReads] = useState<PlateReadApi[]>([])
  const [plates, setPlates] = useState<PlateListEntryApi[]>([])
  const [readsSearch, setReadsSearch] = useState("")
  const [listFilter, setListFilter] = useState<"all" | PlateListType>("all")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showModal, setShowModal] = useState(false)
  const [selectedRead, setSelectedRead] = useState<PlateReadApi | null>(null)
  const [form, setForm] = useState(emptyForm)
  const fileInputRef = useRef<HTMLInputElement>(null)

  async function loadData() {
    setLoading(true)
    setError(null)
    try {
      const [nextReads, nextPlates] = await Promise.all([
        getPlateReads({ limit: 100 }),
        getPlateListEntries(),
      ])
      setReads(nextReads)
      setPlates(nextPlates)
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load ANPR data"
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const filteredReads = useMemo(() => {
    if (!readsSearch.trim()) return reads
    const q = readsSearch.toLowerCase()
    return reads.filter(
      (r) => r.plateNumber.toLowerCase().includes(q) || r.normalizedPlate.toLowerCase().includes(q) || r.cameraName.toLowerCase().includes(q)
    )
  }, [reads, readsSearch])

  const filteredPlates = useMemo(() => {
    if (listFilter === "all") return plates
    return plates.filter((p) => p.list === listFilter)
  }, [plates, listFilter])

  function openCreateModal(plateNumber = "", list: PlateListType = "whitelist") {
    setForm({ ...emptyForm, plateNumber, list })
    setShowModal(true)
  }

  function openEditModal(entry: PlateListEntryApi) {
    setForm({
      id: entry.id,
      plateNumber: entry.plateNumber,
      owner: entry.owner,
      vehicle: entry.vehicle,
      list: entry.list,
      validFrom: entry.validFrom || "",
      validUntil: entry.validUntil || "",
    })
    setShowModal(true)
  }

  async function handleSave() {
    if (!form.plateNumber.trim()) return
    const payload = {
      plateNumber: form.plateNumber,
      owner: form.owner,
      vehicle: form.vehicle,
      list: form.list,
      validFrom: form.validFrom || null,
      validUntil: form.validUntil || null,
    }
    try {
      if (form.id) {
        await updatePlateListEntry(form.id, payload)
        toast.success("Plate updated")
      } else {
        await createPlateListEntry(payload)
        toast.success("Plate added")
      }
      setShowModal(false)
      await loadData()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to save plate")
    }
  }

  async function handleDeletePlate(id: string) {
    try {
      await deletePlateListEntry(id)
      toast.success("Plate removed")
      await loadData()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to remove plate")
    }
  }

  async function handleImport(file: File | undefined) {
    if (!file) return
    try {
      const result = await importPlateListCsv(file)
      toast.success(`Imported ${result.created.length} plate${result.created.length === 1 ? "" : "s"}${result.failed.length ? `, ${result.failed.length} failed` : ""}`)
      await loadData()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "CSV import failed")
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = ""
    }
  }

  const filterPills: { label: string; value: typeof listFilter }[] = [
    { label: "All", value: "all" },
    { label: "Whitelist", value: "whitelist" },
    { label: "Blocked", value: "blocked" },
    { label: "Visitors", value: "visitors" },
  ]

  const normalizedPreview = normalizePlate(form.plateNumber)
  const formatWarning = normalizedPreview && !isIndianPlate(normalizedPreview)

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-[var(--color-text-primary)]">ANPR Plate Management</h1>
          <p className="text-sm text-[var(--color-text-secondary)] mt-1">
            Monitor plate reads and manage vehicle access lists
          </p>
        </div>
        <Button variant="primary" size="sm" onClick={() => openCreateModal()}>
          <Plus className="w-3.5 h-3.5" /> Add Plate
        </Button>
      </div>

      {error && (
        <div className="border border-[var(--color-critical)]/30 rounded-[var(--radius-md)] bg-red-50 px-4 py-3 text-sm text-[var(--color-critical)]">
          {error}
        </div>
      )}

      <Tabs.Root defaultValue="recent">
        <Tabs.List className="flex gap-1 border-b mb-4">
          <Tabs.Trigger value="recent" className="px-4 py-2 text-sm font-medium border-b-2 border-transparent data-[state=active]:border-[var(--color-text-primary)] data-[state=active]:text-[var(--color-text-primary)] text-[var(--color-text-secondary)] transition-colors cursor-pointer">
            Recent Reads
          </Tabs.Trigger>
          <Tabs.Trigger value="manage" className="px-4 py-2 text-sm font-medium border-b-2 border-transparent data-[state=active]:border-[var(--color-text-primary)] data-[state=active]:text-[var(--color-text-primary)] text-[var(--color-text-secondary)] transition-colors cursor-pointer">
            Manage Lists
          </Tabs.Trigger>
        </Tabs.List>

        <Tabs.Content value="recent" className="space-y-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--color-text-tertiary)]" />
            <input type="text" placeholder="Search by plate number or camera..." value={readsSearch} onChange={(e) => setReadsSearch(e.target.value)} className="w-full pl-9 pr-4 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent" />
          </div>

          <div className="border rounded-[var(--radius-lg)] overflow-hidden bg-white">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-[var(--color-bg-tertiary)]">
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Plate</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Camera</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Time</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Status</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Crop</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Confidence</th>
                  <th className="px-4 py-2.5"></th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan={7} className="px-4 py-8 text-center text-[var(--color-text-secondary)]">Loading ANPR data...</td></tr>
                ) : filteredReads.length === 0 ? (
                  <tr><td colSpan={7} className="px-4 py-8 text-center text-[var(--color-text-secondary)]">No plate reads logged yet.</td></tr>
                ) : filteredReads.map((read) => (
                  <tr key={read.id} className="border-b last:border-b-0 hover:bg-[var(--color-bg-secondary)] transition-colors">
                    <td className="px-4 py-2.5 font-mono font-medium text-[var(--color-text-primary)]">{read.plateNumber || "Unread"}</td>
                    <td className="px-4 py-2.5 text-[var(--color-text-secondary)]">{read.cameraName}</td>
                    <td className="px-4 py-2.5 font-mono text-[var(--color-text-secondary)]">{formatTime(read.timestamp)}</td>
                    <td className="px-4 py-2.5"><StatusBadge status={read.matchStatus} /></td>
                    <td className="px-4 py-2.5"><ProtectedThumb url={read.cropUrl} alt={read.plateNumber || "Plate crop"} /></td>
                    <td className="px-4 py-2.5 text-[var(--color-text-secondary)]">{read.confidence == null ? "--" : `${read.confidence.toFixed(1)}%`}</td>
                    <td className="px-4 py-2.5">
                      <div className="flex justify-end gap-1">
                        <Button variant="ghost" size="sm" title="View" onClick={() => setSelectedRead(read)}><Eye className="w-3.5 h-3.5" /></Button>
                        {(read.matchStatus === "unknown" || read.matchStatus === "low_confidence") && read.normalizedPlate && (
                          <Button variant="ghost" size="sm" onClick={() => openCreateModal(read.normalizedPlate)}><Plus className="w-3 h-3" /> Add</Button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Tabs.Content>

        <Tabs.Content value="manage" className="space-y-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex gap-1">
              {filterPills.map((pill) => (
                <button key={pill.value} onClick={() => setListFilter(pill.value)} className={cn("px-3 py-1.5 text-xs font-medium rounded-[var(--radius-md)] transition-colors cursor-pointer", listFilter === pill.value ? "bg-[var(--color-text-primary)] text-white" : "bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] hover:bg-[var(--color-border-default)]")}>{pill.label}</button>
              ))}
            </div>
            <div className="flex gap-2">
              <input ref={fileInputRef} type="file" accept=".csv,text/csv" className="hidden" onChange={(e) => handleImport(e.target.files?.[0])} />
              <Button variant="secondary" size="sm" onClick={() => fileInputRef.current?.click()}><Upload className="w-3.5 h-3.5" /> Import CSV</Button>
              <Button variant="primary" size="sm" onClick={() => openCreateModal()}><Plus className="w-3.5 h-3.5" /> Add Plate</Button>
            </div>
          </div>

          <div className="border rounded-[var(--radius-lg)] overflow-hidden bg-white">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-[var(--color-bg-tertiary)]">
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Plate</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Owner</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Vehicle</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">List</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Valid Until</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan={6} className="px-4 py-8 text-center text-[var(--color-text-secondary)]">Loading plate lists...</td></tr>
                ) : filteredPlates.length === 0 ? (
                  <tr><td colSpan={6} className="px-4 py-8 text-center text-[var(--color-text-secondary)]">No plates in this list.</td></tr>
                ) : filteredPlates.map((plate) => (
                  <tr key={plate.id} className="border-b last:border-b-0 hover:bg-[var(--color-bg-secondary)] transition-colors">
                    <td className="px-4 py-2.5 font-mono font-medium text-[var(--color-text-primary)]">{plate.plateNumber}</td>
                    <td className="px-4 py-2.5 text-[var(--color-text-secondary)]">{plate.owner || "Unknown"}</td>
                    <td className="px-4 py-2.5 text-[var(--color-text-secondary)]">{plate.vehicle || "--"}</td>
                    <td className="px-4 py-2.5"><ListBadge list={plate.list} /></td>
                    <td className="px-4 py-2.5">{plate.validUntil ? <span className={cn(isExpired(plate.validUntil) && "text-[#92400e] font-medium")}>{plate.validUntil}{isExpired(plate.validUntil) && " (expired)"}</span> : <span className="text-[var(--color-text-tertiary)]">--</span>}</td>
                    <td className="px-4 py-2.5"><div className="flex gap-1"><Button variant="ghost" size="sm" title="Edit" onClick={() => openEditModal(plate)}><Pencil className="w-3.5 h-3.5" /></Button><Button variant="ghost" size="sm" title="Delete" onClick={() => handleDeletePlate(plate.id)}><Trash2 className="w-3.5 h-3.5" /></Button></div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Tabs.Content>
      </Tabs.Root>

      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/40" onClick={() => setShowModal(false)} />
          <div className="relative bg-white rounded-[var(--radius-xl)] shadow-xl border w-full max-w-md p-6 space-y-4">
            <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">{form.id ? "Edit Plate" : "Add Plate"}</h2>
            <div className="space-y-3">
              <div><label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Plate Number</label><input type="text" value={form.plateNumber} onChange={(e) => setForm((prev) => ({ ...prev, plateNumber: e.target.value }))} placeholder="e.g. KA05AB1234" className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)]" />{normalizedPreview && <p className={cn("mt-1 text-xs", formatWarning ? "text-[#92400e]" : "text-[var(--color-text-secondary)]")}>Normalized: {normalizedPreview}{formatWarning ? " · non-standard Indian format, override allowed" : ""}</p>}</div>
              <div><label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Owner Name</label><input type="text" value={form.owner} onChange={(e) => setForm((prev) => ({ ...prev, owner: e.target.value }))} placeholder="Vehicle owner name" className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)]" /></div>
              <div><label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Vehicle Description</label><input type="text" value={form.vehicle} onChange={(e) => setForm((prev) => ({ ...prev, vehicle: e.target.value }))} placeholder="e.g. White Toyota Innova" className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)]" /></div>
              <div><label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">List</label><select value={form.list} onChange={(e) => setForm((prev) => ({ ...prev, list: e.target.value as PlateListType }))} className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] cursor-pointer"><option value="whitelist">Whitelist</option><option value="blocked">Blocked</option><option value="visitors">Visitors</option></select></div>
              <div><label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Valid Until (optional)</label><input type="date" value={form.validUntil} onChange={(e) => setForm((prev) => ({ ...prev, validUntil: e.target.value }))} className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)]" /></div>
            </div>
            <div className="flex justify-end gap-2 pt-2"><Button variant="secondary" size="md" onClick={() => setShowModal(false)}>Cancel</Button><Button variant="primary" size="md" onClick={handleSave} disabled={!form.plateNumber.trim()}>{form.id ? "Save" : "Add"}</Button></div>
          </div>
        </div>
      )}

      {selectedRead && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/40" onClick={() => setSelectedRead(null)} />
          <div className="relative bg-white rounded-[var(--radius-xl)] shadow-xl border w-full max-w-2xl p-6 space-y-4">
            <div className="flex items-start justify-between gap-4"><div><h2 className="text-lg font-semibold text-[var(--color-text-primary)] font-mono">{selectedRead.plateNumber || "Unread plate"}</h2><p className="text-sm text-[var(--color-text-secondary)]">{selectedRead.cameraName} · {formatTime(selectedRead.timestamp)}</p></div><StatusBadge status={selectedRead.matchStatus} /></div>
            <div className="grid grid-cols-2 gap-4"><div><p className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">Snapshot</p><ProtectedThumb url={selectedRead.snapshotUrl} alt="Plate snapshot" className="w-full h-48" /></div><div><p className="text-xs font-medium text-[var(--color-text-secondary)] mb-2">Plate Crop</p><ProtectedThumb url={selectedRead.cropUrl} alt="Plate crop" className="w-full h-48" /></div></div>
            <div className="grid grid-cols-3 gap-3 text-sm"><div><p className="text-xs text-[var(--color-text-secondary)]">Overall</p><p>{selectedRead.confidence == null ? "--" : `${selectedRead.confidence.toFixed(1)}%`}</p></div><div><p className="text-xs text-[var(--color-text-secondary)]">Detector</p><p>{selectedRead.detectionConfidence == null ? "--" : `${selectedRead.detectionConfidence.toFixed(1)}%`}</p></div><div><p className="text-xs text-[var(--color-text-secondary)]">OCR</p><p>{selectedRead.ocrConfidence == null ? "--" : `${selectedRead.ocrConfidence.toFixed(1)}%`}</p></div></div>
            {selectedRead.qualityReason && <p className="text-sm text-[#92400e]">{selectedRead.qualityReason}</p>}
            <div className="flex justify-end gap-2 pt-2"><Button variant="secondary" size="md" onClick={() => setSelectedRead(null)}>Close</Button>{selectedRead.normalizedPlate && <><Button variant="secondary" size="md" onClick={() => { setSelectedRead(null); openCreateModal(selectedRead.normalizedPlate, "blocked") }}>Block</Button><Button variant="primary" size="md" onClick={() => { setSelectedRead(null); openCreateModal(selectedRead.normalizedPlate, "whitelist") }}>Add to Whitelist</Button></>}</div>
          </div>
        </div>
      )}
    </div>
  )
}
