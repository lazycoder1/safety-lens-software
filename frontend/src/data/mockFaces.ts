export interface EnrolledFace {
  id: string
  name: string
  group: "employees" | "visitors" | "contractors"
  initials: string
  color: string
  validUntil: string | null
  enrolledAt: string
  consentMethod: string
}

export interface FaceMatch {
  id: string
  personName: string | null
  personId: string | null
  cameraName: string
  timestamp: string
  confidence: number | null
  isUnknown: boolean
}

export function getAvatarUrl(initials: string, color: string): string {
  return `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80"><rect width="80" height="80" rx="40" fill="${color}"/><text x="50%" y="50%" text-anchor="middle" dy=".35em" fill="white" font-family="system-ui" font-size="28" font-weight="600">${initials}</text></svg>`)}`
}

export const mockFaces: EnrolledFace[] = [
  // Employees (6)
  {
    id: "fc-001",
    name: "Rajesh Kumar",
    group: "employees",
    initials: "RK",
    color: "#2563eb",
    validUntil: null,
    enrolledAt: "2025-06-10",
    consentMethod: "Written form",
  },
  {
    id: "fc-002",
    name: "Priya Sharma",
    group: "employees",
    initials: "PS",
    color: "#7c3aed",
    validUntil: null,
    enrolledAt: "2025-06-15",
    consentMethod: "Email consent",
  },
  {
    id: "fc-003",
    name: "Vikram Singh",
    group: "employees",
    initials: "VS",
    color: "#059669",
    validUntil: null,
    enrolledAt: "2025-07-01",
    consentMethod: "Written form",
  },
  {
    id: "fc-004",
    name: "Anitha Rao",
    group: "employees",
    initials: "AR",
    color: "#dc2626",
    validUntil: null,
    enrolledAt: "2025-08-20",
    consentMethod: "Verbal consent",
  },
  {
    id: "fc-005",
    name: "Suresh Patel",
    group: "employees",
    initials: "SP",
    color: "#0891b2",
    validUntil: null,
    enrolledAt: "2025-09-05",
    consentMethod: "Written form",
  },
  {
    id: "fc-006",
    name: "Kavitha Nair",
    group: "employees",
    initials: "KN",
    color: "#be185d",
    validUntil: null,
    enrolledAt: "2025-10-12",
    consentMethod: "Email consent",
  },
  // Visitors (2) — 1 expired, 1 valid
  {
    id: "fc-007",
    name: "Deepak Joshi",
    group: "visitors",
    initials: "DJ",
    color: "#f59e0b",
    validUntil: "2026-03-10",
    enrolledAt: "2026-02-15",
    consentMethod: "Written form",
  },
  {
    id: "fc-008",
    name: "Meena Krishnan",
    group: "visitors",
    initials: "MK",
    color: "#6366f1",
    validUntil: "2026-05-01",
    enrolledAt: "2026-03-01",
    consentMethod: "Verbal consent",
  },
  // Contractors (2)
  {
    id: "fc-009",
    name: "Arjun Reddy",
    group: "contractors",
    initials: "AR",
    color: "#475569",
    validUntil: null,
    enrolledAt: "2026-01-10",
    consentMethod: "Written form",
  },
  {
    id: "fc-010",
    name: "Fatima Begum",
    group: "contractors",
    initials: "FB",
    color: "#0d9488",
    validUntil: null,
    enrolledAt: "2026-02-20",
    consentMethod: "Email consent",
  },
]

export const mockFaceMatches: FaceMatch[] = [
  {
    id: "fm-001",
    personName: "Rajesh Kumar",
    personId: "fc-001",
    cameraName: "Gate 1 Entry",
    timestamp: "2026-03-28T08:10:15",
    confidence: 96.5,
    isUnknown: false,
  },
  {
    id: "fm-002",
    personName: "Priya Sharma",
    personId: "fc-002",
    cameraName: "Lobby Camera",
    timestamp: "2026-03-28T08:32:40",
    confidence: 94.2,
    isUnknown: false,
  },
  {
    id: "fm-003",
    personName: null,
    personId: null,
    cameraName: "Gate 2 Entry",
    timestamp: "2026-03-28T09:15:22",
    confidence: null,
    isUnknown: true,
  },
  {
    id: "fm-004",
    personName: "Vikram Singh",
    personId: "fc-003",
    cameraName: "Gate 1 Entry",
    timestamp: "2026-03-28T09:45:08",
    confidence: 97.8,
    isUnknown: false,
  },
  {
    id: "fm-005",
    personName: "Arjun Reddy",
    personId: "fc-009",
    cameraName: "Gate 3 Entry",
    timestamp: "2026-03-28T10:20:33",
    confidence: 91.3,
    isUnknown: false,
  },
  {
    id: "fm-006",
    personName: null,
    personId: null,
    cameraName: "Parking Camera",
    timestamp: "2026-03-28T10:55:17",
    confidence: null,
    isUnknown: true,
  },
  {
    id: "fm-007",
    personName: "Kavitha Nair",
    personId: "fc-006",
    cameraName: "Lobby Camera",
    timestamp: "2026-03-28T11:10:45",
    confidence: 95.0,
    isUnknown: false,
  },
  {
    id: "fm-008",
    personName: "Deepak Joshi",
    personId: "fc-007",
    cameraName: "Gate 1 Entry",
    timestamp: "2026-03-28T11:40:22",
    confidence: 89.6,
    isUnknown: false,
  },
]
