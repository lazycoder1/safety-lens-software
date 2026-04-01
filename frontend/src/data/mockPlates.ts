export interface Plate {
  id: string
  plateNumber: string
  owner: string
  vehicle: string
  list: "whitelist" | "blocked" | "visitors"
  validUntil: string | null
  createdAt: string
}

export interface PlateRead {
  id: string
  plateNumber: string
  cameraName: string
  timestamp: string
  matchStatus: "whitelist" | "blocked" | "visitor" | "unknown"
  confidence: number
}

export const mockPlates: Plate[] = [
  // Whitelist (5)
  {
    id: "pl-001",
    plateNumber: "KA05AB1234",
    owner: "Rajesh Kumar",
    vehicle: "White Toyota Innova",
    list: "whitelist",
    validUntil: null,
    createdAt: "2025-06-15",
  },
  {
    id: "pl-002",
    plateNumber: "KA01MN5678",
    owner: "Priya Sharma",
    vehicle: "Silver Honda City",
    list: "whitelist",
    validUntil: null,
    createdAt: "2025-07-20",
  },
  {
    id: "pl-003",
    plateNumber: "KA03CD9012",
    owner: "Vikram Singh",
    vehicle: "Black Hyundai Creta",
    list: "whitelist",
    validUntil: null,
    createdAt: "2025-08-10",
  },
  {
    id: "pl-004",
    plateNumber: "KA09EF3456",
    owner: "Anitha Rao",
    vehicle: "Red Maruti Swift",
    list: "whitelist",
    validUntil: null,
    createdAt: "2025-09-01",
  },
  {
    id: "pl-005",
    plateNumber: "KA12GH7890",
    owner: "Suresh Patel",
    vehicle: "Blue Tata Nexon",
    list: "whitelist",
    validUntil: null,
    createdAt: "2025-10-05",
  },
  // Blocked (3)
  {
    id: "pl-006",
    plateNumber: "MH04JK2345",
    owner: "Unknown",
    vehicle: "Grey Sedan",
    list: "blocked",
    validUntil: null,
    createdAt: "2025-11-12",
  },
  {
    id: "pl-007",
    plateNumber: "TN07LM6789",
    owner: "Flagged – Repeated Violations",
    vehicle: "White Truck",
    list: "blocked",
    validUntil: null,
    createdAt: "2025-12-01",
  },
  {
    id: "pl-008",
    plateNumber: "AP09NP1122",
    owner: "Former Employee",
    vehicle: "Black SUV",
    list: "blocked",
    validUntil: null,
    createdAt: "2026-01-15",
  },
  // Visitors (4) — 2 expired, 2 valid
  {
    id: "pl-009",
    plateNumber: "KA02QR3344",
    owner: "Deepak Joshi",
    vehicle: "White Hyundai i20",
    list: "visitors",
    validUntil: "2026-02-28",
    createdAt: "2026-02-01",
  },
  {
    id: "pl-010",
    plateNumber: "KA05ST5566",
    owner: "Meena Krishnan",
    vehicle: "Silver Kia Seltos",
    list: "visitors",
    validUntil: "2026-03-15",
    createdAt: "2026-03-01",
  },
  {
    id: "pl-011",
    plateNumber: "KA08UV7788",
    owner: "Arjun Reddy",
    vehicle: "Blue Mahindra XUV700",
    list: "visitors",
    validUntil: "2026-04-30",
    createdAt: "2026-03-20",
  },
  {
    id: "pl-012",
    plateNumber: "KA11WX9900",
    owner: "Fatima Begum",
    vehicle: "Grey Honda Amaze",
    list: "visitors",
    validUntil: "2026-05-15",
    createdAt: "2026-03-25",
  },
]

export const mockPlateReads: PlateRead[] = [
  {
    id: "pr-001",
    plateNumber: "KA05AB1234",
    cameraName: "Gate 1 Entry",
    timestamp: "2026-03-28T08:15:32",
    matchStatus: "whitelist",
    confidence: 97.2,
  },
  {
    id: "pr-002",
    plateNumber: "MH04JK2345",
    cameraName: "Gate 1 Entry",
    timestamp: "2026-03-28T08:22:10",
    matchStatus: "blocked",
    confidence: 94.8,
  },
  {
    id: "pr-003",
    plateNumber: "KA02QR3344",
    cameraName: "Gate 2 Exit",
    timestamp: "2026-03-28T09:05:45",
    matchStatus: "visitor",
    confidence: 91.5,
  },
  {
    id: "pr-004",
    plateNumber: "DL08XY4455",
    cameraName: "Gate 1 Entry",
    timestamp: "2026-03-28T09:30:18",
    matchStatus: "unknown",
    confidence: 88.3,
  },
  {
    id: "pr-005",
    plateNumber: "KA01MN5678",
    cameraName: "Gate 2 Exit",
    timestamp: "2026-03-28T10:12:07",
    matchStatus: "whitelist",
    confidence: 96.1,
  },
  {
    id: "pr-006",
    plateNumber: "TN07LM6789",
    cameraName: "Gate 1 Entry",
    timestamp: "2026-03-28T10:45:33",
    matchStatus: "blocked",
    confidence: 93.4,
  },
  {
    id: "pr-007",
    plateNumber: "KA19ZZ1122",
    cameraName: "Gate 3 Entry",
    timestamp: "2026-03-28T11:02:55",
    matchStatus: "unknown",
    confidence: 85.7,
  },
  {
    id: "pr-008",
    plateNumber: "KA09EF3456",
    cameraName: "Gate 2 Exit",
    timestamp: "2026-03-28T11:30:20",
    matchStatus: "whitelist",
    confidence: 98.0,
  },
  {
    id: "pr-009",
    plateNumber: "KA11WX9900",
    cameraName: "Gate 1 Entry",
    timestamp: "2026-03-28T12:15:42",
    matchStatus: "visitor",
    confidence: 90.2,
  },
  {
    id: "pr-010",
    plateNumber: "KA03CD9012",
    cameraName: "Gate 3 Entry",
    timestamp: "2026-03-28T12:45:10",
    matchStatus: "whitelist",
    confidence: 95.6,
  },
]
