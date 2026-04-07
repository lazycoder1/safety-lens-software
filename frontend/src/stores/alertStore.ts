import { create } from "zustand"
import {
  getAlerts,
  acknowledgeAlert as apiAcknowledge,
  resolveAlert as apiResolve,
  snoozeAlert as apiSnooze,
  markFalsePositive as apiFalsePositive,
} from "@/lib/api"
import type { Alert } from "@/types"

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
