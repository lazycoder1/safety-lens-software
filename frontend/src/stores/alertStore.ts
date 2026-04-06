import { create } from "zustand"
import {
  getAlerts,
  acknowledgeAlert as apiAcknowledge,
  resolveAlert as apiResolve,
  snoozeAlert as apiSnooze,
  markFalsePositive as apiFalsePositive,
} from "@/lib/api"

export type Severity = "P1" | "P2" | "P3" | "P4"
export type AlertStatus = "active" | "acknowledged" | "resolved" | "snoozed"

export interface Alert {
  id: string
  severity: Severity
  status: AlertStatus
  rule: string
  cameraId: string
  cameraName: string
  zone: string
  confidence: number
  timestamp: string
  source: string
  description: string
  snapshotUrl: string | null
  cleanSnapshotUrl: string | null
  bboxes: Array<{ label: string; bbox: [number, number, number, number]; confidence: number }>
  acknowledgedBy?: string | null
  acknowledgedAt?: string | null
  resolvedAt?: string | null
  snoozedUntil?: string | null
  falsePositive?: boolean
}

interface AlertStore {
  alerts: Alert[]
  loading: boolean
  error: string | null

  fetchAlerts: () => Promise<void>
  addOrUpdateAlert: (alert: Alert) => void

  acknowledge: (id: string) => Promise<void>
  snooze: (id: string, minutes?: number) => Promise<void>
  resolve: (id: string) => Promise<void>
  markFalsePositive: (id: string) => Promise<void>
}

export const useAlertStore = create<AlertStore>((set, get) => ({
  alerts: [],
  loading: true,
  error: null,

  fetchAlerts: async () => {
    set({ loading: true, error: null })
    try {
      const data = await getAlerts({ limit: 500 })
      set({ alerts: data, loading: false })
    } catch (e: any) {
      set({ error: e.message, loading: false })
    }
  },

  addOrUpdateAlert: (alert: Alert) => {
    set((state) => {
      const idx = state.alerts.findIndex((a) => a.id === alert.id)
      if (idx >= 0) {
        const updated = [...state.alerts]
        updated[idx] = alert
        return { alerts: updated }
      }
      return { alerts: [alert, ...state.alerts] }
    })
  },

  acknowledge: async (id) => {
    const result = await apiAcknowledge(id)
    get().addOrUpdateAlert(result)
  },

  snooze: async (id, minutes = 15) => {
    const result = await apiSnooze(id, minutes)
    get().addOrUpdateAlert(result)
  },

  resolve: async (id) => {
    const result = await apiResolve(id)
    get().addOrUpdateAlert(result)
  },

  markFalsePositive: async (id) => {
    const result = await apiFalsePositive(id)
    get().addOrUpdateAlert(result)
  },
}))
