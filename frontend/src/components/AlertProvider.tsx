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
  const setConnected = useAlertConnection((s) => s.setConnected)
  const token = useAuthStore((s) => s.token)

  useEffect(() => {
    if (!token) return

    let retryDelay = 2000
    let cancelled = false

    function connect() {
      if (cancelled) return
      const ws = new WebSocket(`${WS_BASE}/ws/alerts?token=${token}`)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        retryDelay = 2000
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
  }, [addOrUpdateAlert, setConnected, token])

  return null
}
