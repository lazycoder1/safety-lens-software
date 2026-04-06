import { useState, FormEvent } from "react"
import { useNavigate } from "react-router-dom"
import { ShieldCheck, Loader2, Eye, EyeOff, X } from "lucide-react"
import { useAuthStore } from "@/stores/authStore"

export function ChangePassword() {
  const navigate = useNavigate()
  const { changePassword, user } = useAuthStore()
  const isForced = user?.mustChangePassword

  const [currentPassword, setCurrentPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showNew, setShowNew] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)

  function handleClose() {
    navigate(-1)
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)

    if (!currentPassword.trim() || !newPassword.trim()) {
      setError("All fields are required")
      return
    }

    if (newPassword.length < 6) {
      setError("New password must be at least 6 characters")
      return
    }

    if (newPassword !== confirmPassword) {
      setError("New passwords do not match")
      return
    }

    setSubmitting(true)
    try {
      await changePassword(currentPassword, newPassword)
      navigate("/live")
    } catch (err: any) {
      setError(err.message || "Failed to change password")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-neutral-100">
      <div className="w-full max-w-sm">
        <div className="bg-white rounded-xl shadow-lg p-8 relative">
          {/* Close button — only if not forced */}
          {!isForced && (
            <button
              onClick={handleClose}
              className="absolute top-4 right-4 p-1.5 rounded-lg hover:bg-neutral-100 text-neutral-400 hover:text-neutral-600 transition-colors cursor-pointer"
            >
              <X size={18} />
            </button>
          )}

          {/* Branding */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-neutral-900 text-white mb-3">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <h1 className="text-xl font-bold text-neutral-900">Change Your Password</h1>
            <p className="text-sm text-neutral-500 mt-1">
              {isForced
                ? "You must change your password before continuing"
                : "Enter your current password and choose a new one"}
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Current Password */}
            <div>
              <label className="block text-sm font-medium text-neutral-700 mb-1">Current Password</label>
              <input
                type="password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                className="w-full rounded-lg border border-neutral-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
                placeholder="Enter current password"
                autoComplete="current-password"
              />
            </div>

            {/* New Password */}
            <div>
              <label className="block text-sm font-medium text-neutral-700 mb-1">New Password</label>
              <div className="relative">
                <input
                  type={showNew ? "text" : "password"}
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="w-full rounded-lg border border-neutral-300 px-3 py-2 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
                  placeholder="At least 6 characters"
                  autoComplete="new-password"
                />
                <button
                  type="button"
                  onClick={() => setShowNew(!showNew)}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-neutral-400 hover:text-neutral-600 cursor-pointer"
                >
                  {showNew ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            {/* Confirm New Password */}
            <div>
              <label className="block text-sm font-medium text-neutral-700 mb-1">Confirm New Password</label>
              <div className="relative">
                <input
                  type={showConfirm ? "text" : "password"}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full rounded-lg border border-neutral-300 px-3 py-2 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
                  placeholder="Re-enter new password"
                  autoComplete="new-password"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirm(!showConfirm)}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-neutral-400 hover:text-neutral-600 cursor-pointer"
                >
                  {showConfirm ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            {/* Error */}
            {error && <p className="text-sm text-red-600">{error}</p>}

            {/* Buttons */}
            <div className="flex gap-2">
              {!isForced && (
                <button
                  type="button"
                  onClick={handleClose}
                  className="flex-1 rounded-lg border border-neutral-300 px-4 py-2 text-sm font-medium text-neutral-700 hover:bg-neutral-50 cursor-pointer"
                >
                  Cancel
                </button>
              )}
              <button
                type="submit"
                disabled={submitting}
                className="flex-1 flex items-center justify-center gap-2 rounded-lg bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-800 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {submitting && <Loader2 className="w-4 h-4 animate-spin" />}
                Update Password
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
