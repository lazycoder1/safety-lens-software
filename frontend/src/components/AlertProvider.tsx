import { useEffect, useRef } from "react"
import { create } from "zustand"
import { useAlertStore } from "@/stores/alertStore"
import { useAuthStore } from "@/stores/authStore"
import type { Alert } from "@/types"
import { WS_BASE } from "@/lib/api"

interface AlertConnectionStore {
  connected: boolean
  setConnected: (v: boolean) => void
}

export const useAlertConnection = create<AlertConnectionStore>((set) => ({
  connected: false,
  setConnected: (v) => set({ connected: v }),
}))

export function AlertProvider() {
  const wsRef = useRef<WebSocket | null>(null)
  const addOrUpdateAlert = useAlertStore((s) => s.addOrUpdateAlert)
  const fetchAlerts = useAlertStore((s) => s.fetchAlerts)
  const setConnected = useAlertConnection((s) => s.setConnected)
  const token = useAuthStore((s) => s.token)

  useEffect(() => {
    if (!token) return

    let retryDelay = 2000
    let cancelled = false

    // Backfill the alert store on (re)connect so alerts that fired while
    // this client was offline still show up in the Live Alerts panel.
    // Without this, the panel only shows alerts received via WebSocket
    // push since the user's current session started.
    fetchAlerts().catch(() => {})

    function connect() {
      if (cancelled) return
      const ws = new WebSocket(`${WS_BASE}/ws/alerts?token=${token}`)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        retryDelay = 2000
        // Re-backfill on every successful reconnect, not just initial mount.
        // If the WS dropped and we missed a push, this catches us up.
        fetchAlerts().catch(() => {})
      }

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)
        if (msg.type === "alert") {
          const alert = msg.data as Alert
          addOrUpdateAlert(alert)
        } else if (msg.type === "updated") {
          const alert = msg.data as Alert
          addOrUpdateAlert(alert)
        }
      }

      ws.onclose = () => {
        setConnected(false)
        if (!cancelled) {
          setTimeout(connect, retryDelay)
          retryDelay = Math.min(retryDelay * 1.5, 15000)
        }
      }

      ws.onerror = () => {
        ws.close()
      }
    }

    connect()
    return () => {
      cancelled = true
      wsRef.current?.close()
    }
  }, [addOrUpdateAlert, fetchAlerts, setConnected, token])

  return null
}
