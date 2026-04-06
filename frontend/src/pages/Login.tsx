import { useState, FormEvent } from "react"
import { useNavigate } from "react-router-dom"
import { Eye, EyeOff, ShieldCheck, Loader2 } from "lucide-react"
import { useAuthStore } from "@/stores/authStore"

export function Login() {
  const navigate = useNavigate()
  const { login, register, clearError } = useAuthStore()

  const [mode, setMode] = useState<"login" | "register">("login")
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [successMsg, setSuccessMsg] = useState<string | null>(null)

  function resetForm() {
    setUsername("")
    setPassword("")
    setConfirmPassword("")
    setError(null)
    setSuccessMsg(null)
    clearError()
  }

  function switchMode() {
    resetForm()
    setMode(mode === "login" ? "register" : "login")
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setSuccessMsg(null)

    if (!username.trim() || !password.trim()) {
      setError("Username and password are required")
      return
    }

    if (mode === "register") {
      if (password !== confirmPassword) {
        setError("Passwords do not match")
        return
      }
      if (password.length < 6) {
        setError("Password must be at least 6 characters")
        return
      }
    }

    setSubmitting(true)
    try {
      if (mode === "login") {
        await login(username, password)
        const user = useAuthStore.getState().user
        if (user?.mustChangePassword) {
          navigate("/change-password")
        } else {
          navigate("/live")
        }
      } else {
        const msg = await register(username, password)
        setSuccessMsg(msg || "Account pending admin approval")
        setUsername("")
        setPassword("")
        setConfirmPassword("")
      }
    } catch (err: any) {
      setError(err.message || "Something went wrong")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-neutral-100">
      <div className="w-full max-w-sm">
        <div className="bg-white rounded-xl shadow-lg p-8 relative">
          {/* Close button — only in register mode */}
          {mode === "register" && (
            <button
              onClick={() => { setMode("login"); setError(null); setSuccessMsg(null) }}
              className="absolute top-4 right-4 p-1.5 rounded-lg hover:bg-neutral-100 text-neutral-400 hover:text-neutral-600 transition-colors cursor-pointer"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          )}
          {/* Branding */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-neutral-900 text-white mb-3">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <h1 className="text-2xl font-bold text-neutral-900">SafetyLens</h1>
            <p className="text-sm text-neutral-500 mt-1">Industrial Safety Monitoring</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Username */}
            <div>
              <label className="block text-sm font-medium text-neutral-700 mb-1">Username</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full rounded-lg border border-neutral-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
                placeholder="Enter username"
                autoComplete="username"
              />
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-neutral-700 mb-1">Password</label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full rounded-lg border border-neutral-300 px-3 py-2 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
                  placeholder="Enter password"
                  autoComplete={mode === "login" ? "current-password" : "new-password"}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-neutral-400 hover:text-neutral-600"
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Confirm Password (register only) */}
            {mode === "register" && (
              <div>
                <label className="block text-sm font-medium text-neutral-700 mb-1">Confirm Password</label>
                <div className="relative">
                  <input
                    type={showConfirm ? "text" : "password"}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full rounded-lg border border-neutral-300 px-3 py-2 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
                    placeholder="Confirm password"
                    autoComplete="new-password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirm(!showConfirm)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-neutral-400 hover:text-neutral-600"
                    tabIndex={-1}
                  >
                    {showConfirm ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            )}

            {/* Error */}
            {error && <p className="text-sm text-red-600">{error}</p>}

            {/* Success */}
            {successMsg && <p className="text-sm text-green-600">{successMsg}</p>}

            {/* Submit */}
            <button
              type="submit"
              disabled={submitting}
              className="w-full flex items-center justify-center gap-2 rounded-lg bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-800 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {submitting && <Loader2 className="w-4 h-4 animate-spin" />}
              {mode === "login" ? "Sign In" : "Request Access"}
            </button>
          </form>

          {/* Toggle mode */}
          <p className="mt-6 text-center text-sm text-neutral-500">
            {mode === "login" ? (
              <>
                Don&apos;t have an account?{" "}
                <button onClick={switchMode} className="font-medium text-neutral-900 hover:underline">
                  Register
                </button>
              </>
            ) : (
              <>
                Already have an account?{" "}
                <button onClick={switchMode} className="font-medium text-neutral-900 hover:underline">
                  Sign In
                </button>
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  )
}
