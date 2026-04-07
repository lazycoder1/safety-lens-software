import { useState, useEffect, useCallback, useRef } from "react"
import { CheckCircle2, XCircle, Trash2, KeyRound, Copy, X, Loader2, RefreshCw } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getUsers, approveUser, rejectUser, updateUserRole, deleteUser, resetUserPassword } from "@/lib/api"
import { useAuthStore } from "@/stores/authStore"
import { usePendingCount } from "@/components/Sidebar"
import { toast } from "sonner"
import { PasswordInput } from "@/components/ui/PasswordInput"
import { isPasswordValid, generateStrongPassword } from "@/lib/passwordValidation"

interface ManagedUser {
  id: string
  username: string
  role: "admin" | "operator" | "viewer"
  status: string
  createdAt: string
  lastLogin: string | null
}

const statusVariant: Record<string, "success" | "warning" | "critical" | "default"> = {
  active: "success",
  pending: "warning",
  rejected: "critical",
  disabled: "default",
}

const roleVariant: Record<string, "info" | "default"> = {
  admin: "info",
  operator: "info",
  viewer: "default",
}

const roleColors: Record<string, string> = {
  admin: "bg-purple-100 text-purple-800",
  operator: "bg-blue-100 text-blue-800",
  viewer: "",
}

function formatDate(iso: string | null): string {
  if (!iso) return "—"
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}

export function UserManagement() {
  const [users, setUsers] = useState<ManagedUser[]>([])
  const [loading, setLoading] = useState(true)
  const currentUser = useAuthStore((s) => s.user)
  const refreshPending = usePendingCount((s) => s.refresh)

  const fetchUsers = useCallback(async () => {
    try {
      const data = await getUsers()
      setUsers(data.users ?? data)
    } catch {
      toast.error("Failed to load users")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchUsers()
  }, [fetchUsers])

  const pendingUsers = users.filter((u) => u.status === "pending")
  const otherUsers = users.filter((u) => u.status !== "pending")

  async function handleApprove(id: string) {
    try {
      await approveUser(id)
      toast.success("User approved")
      fetchUsers()
      refreshPending()
    } catch {
      toast.error("Failed to approve user")
    }
  }

  async function handleReject(id: string) {
    try {
      await rejectUser(id)
      toast.success("User rejected")
      fetchUsers()
      refreshPending()
    } catch {
      toast.error("Failed to reject user")
    }
  }

  async function handleRoleChange(id: string, role: string) {
    try {
      await updateUserRole(id, role)
      toast.success("Role updated")
      fetchUsers()
    } catch {
      toast.error("Failed to update role")
    }
  }

  // Delete confirmation modal state
  const [deleteModal, setDeleteModal] = useState<{ userId: string; username: string } | null>(null)
  const [deleteSubmitting, setDeleteSubmitting] = useState(false)

  async function confirmDelete() {
    if (!deleteModal) return
    setDeleteSubmitting(true)
    try {
      await deleteUser(deleteModal.userId)
      toast.success("User deleted")
      setDeleteModal(null)
      fetchUsers()
      refreshPending()
    } catch {
      toast.error("Failed to delete user")
    } finally {
      setDeleteSubmitting(false)
    }
  }

  // Reset password modal state
  const [resetModal, setResetModal] = useState<{ userId: string; username: string } | null>(null)
  const [resetPasswordInput, setResetPasswordInput] = useState("")
  const [resetResult, setResetResult] = useState<string | null>(null)
  const [resetSubmitting, setResetSubmitting] = useState(false)
  const [resetError, setResetError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  function openResetModal(userId: string, username: string) {
    setResetModal({ userId, username })
    setResetPasswordInput("")
    setResetResult(null)
    setResetError(null)
    setCopied(false)
  }

  async function handleResetSubmit() {
    if (!resetModal) return
    if (resetPasswordInput && !isPasswordValid(resetPasswordInput)) {
      setResetError("Password does not meet strength requirements")
      return
    }
    setResetSubmitting(true)
    setResetError(null)
    try {
      const result = await resetUserPassword(resetModal.userId, resetPasswordInput || undefined)
      setResetResult(result.newPassword)
    } catch {
      setResetError("Failed to reset password")
    } finally {
      setResetSubmitting(false)
    }
  }

  function handleCopyPassword() {
    if (resetResult) {
      navigator.clipboard.writeText(resetResult)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  function handleDelete(id: string, username: string) {
    if (currentUser?.id === id) {
      toast.error("You cannot delete your own account")
      return
    }
    setDeleteModal({ userId: id, username })
  }

  function UserRow({ user }: { user: ManagedUser }) {
    const isSelf = currentUser?.id === user.id
    return (
      <tr className="border-b border-[var(--color-border-default)] last:border-0">
        <td className="px-4 py-3 text-sm font-medium text-[var(--color-text-primary)]">
          {user.username}
          {isSelf && (
            <span className="ml-2 text-xs text-[var(--color-text-tertiary)]">(you)</span>
          )}
        </td>
        <td className="px-4 py-3">
          <select
            value={user.role}
            onChange={(e) => handleRoleChange(user.id, e.target.value)}
            disabled={isSelf}
            className="rounded-md border border-[var(--color-border-default)] bg-[var(--color-bg-primary)] px-2 py-1 text-xs font-medium text-[var(--color-text-primary)] focus:outline-none focus:ring-1 focus:ring-[var(--color-info)] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <option value="admin">Admin</option>
            <option value="operator">Operator</option>
            <option value="viewer">Viewer</option>
          </select>
        </td>
        <td className="px-4 py-3">
          <Badge
            variant={statusVariant[user.status] ?? "default"}
            className={roleColors[user.role]}
          >
            {user.status}
          </Badge>
        </td>
        <td className="px-4 py-3 text-sm text-[var(--color-text-secondary)]">
          {formatDate(user.createdAt)}
        </td>
        <td className="px-4 py-3 text-sm text-[var(--color-text-secondary)]">
          {formatDate(user.lastLogin)}
        </td>
        <td className="px-4 py-3">
          <div className="flex items-center gap-1">
            {user.status === "pending" && (
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleApprove(user.id)}
                  title="Approve"
                >
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleReject(user.id)}
                  title="Reject"
                >
                  <XCircle className="h-4 w-4 text-red-500" />
                </Button>
              </>
            )}
            {!isSelf && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => openResetModal(user.id, user.username)}
                title="Reset password"
              >
                <KeyRound className="h-4 w-4 text-[var(--color-text-tertiary)]" />
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleDelete(user.id, user.username)}
              disabled={isSelf}
              title={isSelf ? "Cannot delete yourself" : "Delete user"}
            >
              <Trash2 className="h-4 w-4 text-[var(--color-text-tertiary)]" />
            </Button>
          </div>
        </td>
      </tr>
    )
  }

  const tableHead = (
    <thead>
      <tr className="border-b border-[var(--color-border-default)] text-left text-xs font-medium uppercase tracking-wider text-[var(--color-text-tertiary)]">
        <th className="px-4 py-3">Username</th>
        <th className="px-4 py-3">Role</th>
        <th className="px-4 py-3">Status</th>
        <th className="px-4 py-3">Created</th>
        <th className="px-4 py-3">Last Login</th>
        <th className="px-4 py-3">Actions</th>
      </tr>
    </thead>
  )

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center text-[var(--color-text-tertiary)]">
        Loading users...
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-[var(--color-text-primary)]">
          User Management
        </h1>
        <span className="text-sm text-[var(--color-text-tertiary)]">
          {users.length} user{users.length !== 1 ? "s" : ""}
        </span>
      </div>

      {pendingUsers.length > 0 && (
        <Card className="border-amber-300 bg-amber-50/60">
          <div className="px-4 py-3 border-b border-amber-200">
            <h2 className="text-sm font-semibold text-amber-800">
              Pending Approval ({pendingUsers.length})
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              {tableHead}
              <tbody>
                {pendingUsers.map((u) => (
                  <UserRow key={u.id} user={u} />
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card>
        <div className="overflow-x-auto">
          <table className="w-full">
            {tableHead}
            <tbody>
              {otherUsers.length === 0 ? (
                <tr>
                  <td
                    colSpan={6}
                    className="px-4 py-8 text-center text-sm text-[var(--color-text-tertiary)]"
                  >
                    No users found
                  </td>
                </tr>
              ) : (
                otherUsers.map((u) => (
                  <UserRow key={u.id} user={u} />
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Delete Confirmation Modal */}
      {deleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => !deleteSubmitting && setDeleteModal(null)}>
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-sm mx-4" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-5 py-4 border-b">
              <h3 className="font-semibold text-[var(--color-text-primary)]">Delete User</h3>
              <button onClick={() => setDeleteModal(null)} disabled={deleteSubmitting} className="p-1 rounded-lg hover:bg-neutral-100 text-neutral-400 hover:text-neutral-600 cursor-pointer disabled:opacity-50">
                <X size={18} />
              </button>
            </div>
            <div className="px-5 py-4 space-y-4">
              <p className="text-sm text-[var(--color-text-secondary)]">
                Are you sure you want to delete <strong>{deleteModal.username}</strong>? This action cannot be undone.
              </p>
              <div className="flex gap-3 justify-end">
                <Button variant="secondary" onClick={() => setDeleteModal(null)} disabled={deleteSubmitting}>
                  Cancel
                </Button>
                <Button variant="danger" onClick={confirmDelete} disabled={deleteSubmitting}>
                  {deleteSubmitting && <Loader2 className="w-4 h-4 animate-spin mr-2" />}
                  Delete
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Reset Password Modal */}
      {resetModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => setResetModal(null)}>
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-sm mx-4" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-5 py-4 border-b">
              <h3 className="font-semibold text-[var(--color-text-primary)]">Reset Password</h3>
              <button onClick={() => setResetModal(null)} className="p-1 rounded-lg hover:bg-neutral-100 text-neutral-400 hover:text-neutral-600 cursor-pointer">
                <X size={18} />
              </button>
            </div>

            <div className="px-5 py-4 space-y-4">
              <p className="text-sm text-[var(--color-text-secondary)]">
                Reset password for <strong>{resetModal.username}</strong>. Enter a new password or leave empty to auto-generate one.
              </p>

              {!resetResult ? (
                <>
                  <PasswordInput
                    value={resetPasswordInput}
                    onChange={setResetPasswordInput}
                    label="New Password"
                    placeholder="Type a password or generate one"
                    showStrengthMeter={resetPasswordInput.length > 0}
                    showRequirements={resetPasswordInput.length > 0}
                  />
                  <Button
                    type="button"
                    variant="secondary"
                    className="w-full"
                    onClick={() => setResetPasswordInput(generateStrongPassword())}
                  >
                    <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
                    Generate Strong Password
                  </Button>
                  {resetError && <p className="text-sm text-red-600">{resetError}</p>}
                  <Button
                    variant="primary"
                    className="w-full"
                    onClick={handleResetSubmit}
                    disabled={resetSubmitting || (resetPasswordInput.length > 0 && !isPasswordValid(resetPasswordInput))}
                  >
                    {resetSubmitting && <Loader2 className="w-4 h-4 animate-spin mr-2" />}
                    Reset Password
                  </Button>
                </>
              ) : (
                <div className="space-y-3">
                  <p className="text-sm text-emerald-700 font-medium">Password has been reset. Share this with the user:</p>
                  <div className="flex items-center gap-2 bg-neutral-100 rounded-lg px-3 py-2.5">
                    <code className="flex-1 text-sm font-mono font-semibold text-[var(--color-text-primary)]">{resetResult}</code>
                    <button
                      onClick={handleCopyPassword}
                      className="p-1.5 rounded-md hover:bg-neutral-200 text-neutral-500 hover:text-neutral-700 cursor-pointer transition-colors"
                      title="Copy to clipboard"
                    >
                      <Copy size={16} />
                    </button>
                  </div>
                  {copied && <p className="text-xs text-emerald-600">Copied to clipboard</p>}
                  <p className="text-xs text-[var(--color-text-tertiary)]">The user will be asked to change this password on their next login.</p>
                  <Button variant="secondary" className="w-full" onClick={() => setResetModal(null)}>
                    Done
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
