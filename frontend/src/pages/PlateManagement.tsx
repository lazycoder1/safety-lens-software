import { useState, useMemo } from "react"
import * as Tabs from "@radix-ui/react-tabs"
import { CheckCircle, XCircle, User, AlertTriangle, Plus, Search, Pencil, Trash2, Upload } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { mockPlates, mockPlateReads, type Plate, type PlateRead } from "@/data/mockPlates"

function formatTime(timestamp: string): string {
  const d = new Date(timestamp)
  return d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })
}

function isExpired(dateStr: string | null): boolean {
  if (!dateStr) return false
  return new Date(dateStr) < new Date()
}

function StatusBadge({ status }: { status: PlateRead["matchStatus"] }) {
  switch (status) {
    case "whitelist":
      return <Badge variant="success"><CheckCircle className="w-3 h-3" /> Whitelist</Badge>
    case "blocked":
      return <Badge variant="critical"><XCircle className="w-3 h-3" /> Blocked</Badge>
    case "visitor":
      return <Badge variant="info"><User className="w-3 h-3" /> Visitor</Badge>
    case "unknown":
      return <Badge variant="warning"><AlertTriangle className="w-3 h-3" /> Unknown</Badge>
  }
}

function ListBadge({ list }: { list: Plate["list"] }) {
  switch (list) {
    case "whitelist":
      return <Badge variant="success">Whitelist</Badge>
    case "blocked":
      return <Badge variant="critical">Blocked</Badge>
    case "visitors":
      return <Badge variant="info">Visitors</Badge>
  }
}

export function PlateManagement() {
  const [plates, setPlates] = useState<Plate[]>(mockPlates)
  const [reads] = useState<PlateRead[]>(mockPlateReads)
  const [readsSearch, setReadsSearch] = useState("")
  const [listFilter, setListFilter] = useState<"all" | "whitelist" | "blocked" | "visitors">("all")
  const [showAddModal, setShowAddModal] = useState(false)

  // Add plate modal state
  const [newPlateNumber, setNewPlateNumber] = useState("")
  const [newOwner, setNewOwner] = useState("")
  const [newVehicle, setNewVehicle] = useState("")
  const [newList, setNewList] = useState<"whitelist" | "blocked" | "visitors">("whitelist")
  const [newValidUntil, setNewValidUntil] = useState("")

  const filteredReads = useMemo(() => {
    if (!readsSearch) return reads
    const q = readsSearch.toLowerCase()
    return reads.filter(
      (r) => r.plateNumber.toLowerCase().includes(q) || r.cameraName.toLowerCase().includes(q)
    )
  }, [reads, readsSearch])

  const filteredPlates = useMemo(() => {
    if (listFilter === "all") return plates
    return plates.filter((p) => p.list === listFilter)
  }, [plates, listFilter])

  function handleAddPlate() {
    if (!newPlateNumber.trim()) return
    const plate: Plate = {
      id: `pl-${Date.now()}`,
      plateNumber: newPlateNumber.trim().toUpperCase(),
      owner: newOwner.trim() || "Unknown",
      vehicle: newVehicle.trim() || "—",
      list: newList,
      validUntil: newValidUntil || null,
      createdAt: new Date().toISOString().split("T")[0],
    }
    setPlates((prev) => [plate, ...prev])
    resetAddModal()
  }

  function handleDeletePlate(id: string) {
    setPlates((prev) => prev.filter((p) => p.id !== id))
  }

  function resetAddModal() {
    setShowAddModal(false)
    setNewPlateNumber("")
    setNewOwner("")
    setNewVehicle("")
    setNewList("whitelist")
    setNewValidUntil("")
  }

  const filterPills: { label: string; value: typeof listFilter }[] = [
    { label: "All", value: "all" },
    { label: "Whitelist", value: "whitelist" },
    { label: "Blocked", value: "blocked" },
    { label: "Visitors", value: "visitors" },
  ]

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-xl font-semibold text-[var(--color-text-primary)]">ANPR Plate Management</h1>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Monitor plate reads and manage vehicle access lists
        </p>
      </div>

      <Tabs.Root defaultValue="recent">
        <Tabs.List className="flex gap-1 border-b mb-4">
          <Tabs.Trigger
            value="recent"
            className="px-4 py-2 text-sm font-medium border-b-2 border-transparent data-[state=active]:border-[var(--color-text-primary)] data-[state=active]:text-[var(--color-text-primary)] text-[var(--color-text-secondary)] transition-colors cursor-pointer"
          >
            Recent Reads
          </Tabs.Trigger>
          <Tabs.Trigger
            value="manage"
            className="px-4 py-2 text-sm font-medium border-b-2 border-transparent data-[state=active]:border-[var(--color-text-primary)] data-[state=active]:text-[var(--color-text-primary)] text-[var(--color-text-secondary)] transition-colors cursor-pointer"
          >
            Manage Lists
          </Tabs.Trigger>
        </Tabs.List>

        {/* Recent Reads Tab */}
        <Tabs.Content value="recent" className="space-y-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--color-text-tertiary)]" />
            <input
              type="text"
              placeholder="Search by plate number or camera..."
              value={readsSearch}
              onChange={(e) => setReadsSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] focus:border-transparent"
            />
          </div>

          <div className="border rounded-[var(--radius-lg)] overflow-hidden bg-white">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-[var(--color-bg-tertiary)]">
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Plate</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Camera</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Time</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Status</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Confidence</th>
                  <th className="px-4 py-2.5"></th>
                </tr>
              </thead>
              <tbody>
                {filteredReads.map((read) => (
                  <tr key={read.id} className="border-b last:border-b-0 hover:bg-[var(--color-bg-secondary)] transition-colors">
                    <td className="px-4 py-2.5 font-mono font-medium text-[var(--color-text-primary)]">{read.plateNumber}</td>
                    <td className="px-4 py-2.5 text-[var(--color-text-secondary)]">{read.cameraName}</td>
                    <td className="px-4 py-2.5 font-mono text-[var(--color-text-secondary)]">{formatTime(read.timestamp)}</td>
                    <td className="px-4 py-2.5"><StatusBadge status={read.matchStatus} /></td>
                    <td className="px-4 py-2.5 text-[var(--color-text-secondary)]">{read.confidence.toFixed(1)}%</td>
                    <td className="px-4 py-2.5">
                      {read.matchStatus === "unknown" && (
                        <Button variant="ghost" size="sm" onClick={() => { setShowAddModal(true); setNewPlateNumber(read.plateNumber) }}>
                          <Plus className="w-3 h-3" /> Add to list
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Tabs.Content>

        {/* Manage Lists Tab */}
        <Tabs.Content value="manage" className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex gap-1">
              {filterPills.map((pill) => (
                <button
                  key={pill.value}
                  onClick={() => setListFilter(pill.value)}
                  className={cn(
                    "px-3 py-1.5 text-xs font-medium rounded-[var(--radius-md)] transition-colors cursor-pointer",
                    listFilter === pill.value
                      ? "bg-[var(--color-text-primary)] text-white"
                      : "bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] hover:bg-[var(--color-border-default)]"
                  )}
                >
                  {pill.label}
                </button>
              ))}
            </div>
            <div className="flex gap-2">
              <Button variant="secondary" size="sm" disabled title="Coming soon">
                <Upload className="w-3.5 h-3.5" /> Import CSV
              </Button>
              <Button variant="primary" size="sm" onClick={() => setShowAddModal(true)}>
                <Plus className="w-3.5 h-3.5" /> Add Plate
              </Button>
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
                {filteredPlates.map((plate) => (
                  <tr key={plate.id} className="border-b last:border-b-0 hover:bg-[var(--color-bg-secondary)] transition-colors">
                    <td className="px-4 py-2.5 font-mono font-medium text-[var(--color-text-primary)]">{plate.plateNumber}</td>
                    <td className="px-4 py-2.5 text-[var(--color-text-secondary)]">{plate.owner}</td>
                    <td className="px-4 py-2.5 text-[var(--color-text-secondary)]">{plate.vehicle}</td>
                    <td className="px-4 py-2.5"><ListBadge list={plate.list} /></td>
                    <td className="px-4 py-2.5">
                      {plate.validUntil ? (
                        <span className={cn(isExpired(plate.validUntil) && "text-[#92400e] font-medium")}>
                          {plate.validUntil}
                          {isExpired(plate.validUntil) && " (expired)"}
                        </span>
                      ) : (
                        <span className="text-[var(--color-text-tertiary)]">—</span>
                      )}
                    </td>
                    <td className="px-4 py-2.5">
                      <div className="flex gap-1">
                        <Button variant="ghost" size="sm" title="Edit">
                          <Pencil className="w-3.5 h-3.5" />
                        </Button>
                        <Button variant="ghost" size="sm" title="Delete" onClick={() => handleDeletePlate(plate.id)}>
                          <Trash2 className="w-3.5 h-3.5" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Tabs.Content>
      </Tabs.Root>

      {/* Add Plate Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/40" onClick={resetAddModal} />
          <div className="relative bg-white rounded-[var(--radius-xl)] shadow-xl border w-full max-w-md p-6 space-y-4">
            <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">Add Plate</h2>

            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Plate Number</label>
                <input
                  type="text"
                  value={newPlateNumber}
                  onChange={(e) => setNewPlateNumber(e.target.value)}
                  placeholder="e.g. KA05AB1234"
                  className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)]"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Owner Name</label>
                <input
                  type="text"
                  value={newOwner}
                  onChange={(e) => setNewOwner(e.target.value)}
                  placeholder="Vehicle owner name"
                  className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)]"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Vehicle Description</label>
                <input
                  type="text"
                  value={newVehicle}
                  onChange={(e) => setNewVehicle(e.target.value)}
                  placeholder="e.g. White Toyota Innova"
                  className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] placeholder:text-[var(--color-text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)]"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">List</label>
                <select
                  value={newList}
                  onChange={(e) => setNewList(e.target.value as "whitelist" | "blocked" | "visitors")}
                  className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)] cursor-pointer"
                >
                  <option value="whitelist">Whitelist</option>
                  <option value="blocked">Blocked</option>
                  <option value="visitors">Visitors</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-[var(--color-text-secondary)] mb-1">Valid Until (optional)</label>
                <input
                  type="date"
                  value={newValidUntil}
                  onChange={(e) => setNewValidUntil(e.target.value)}
                  className="w-full px-3 py-2 text-sm border rounded-[var(--radius-md)] bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-info)]"
                />
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <Button variant="secondary" size="md" onClick={resetAddModal}>Cancel</Button>
              <Button variant="primary" size="md" onClick={handleAddPlate} disabled={!newPlateNumber.trim()}>Add</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
