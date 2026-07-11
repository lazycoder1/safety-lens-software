/**
 * Fire-and-forget error reporter — POSTs errors to /api/errors.
 * Never throws, never blocks. Silently drops if the server is unreachable.
 */

const DEV_API_HOST = window.location.hostname || "localhost"
const API_BASE = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? `http://${DEV_API_HOST}:8000` : "")

export function reportError(
  error: unknown,
  context?: {
    component?: string
    url?: string
    requestId?: string
    extra?: Record<string, unknown>
  },
) {
  try {
    const message =
      error instanceof Error ? error.message : typeof error === "string" ? error : "Unknown error"
    const stack = error instanceof Error ? error.stack : undefined

    const body = {
      message,
      stack,
      url: context?.url || window.location.href,
      component: context?.component,
      request_id: context?.requestId,
      context: context?.extra,
    }

    fetch(`${API_BASE}/api/errors`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).catch(() => {
      // Server unreachable — silently drop
    })
  } catch {
    // Never let error reporting itself cause errors
  }
}
