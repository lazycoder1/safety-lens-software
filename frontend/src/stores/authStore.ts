import { create } from "zustand"
import { apiLogin, apiRegister, apiChangePassword, getMe, setToken, clearToken, getToken } from "@/lib/api"

export interface User {
  id: string
  username: string
  role: "admin" | "operator" | "viewer"
  status: string
  mustChangePassword: boolean
  createdAt: string
  lastLogin: string | null
}

interface AuthStore {
  token: string | null
  user: User | null
  loading: boolean
  error: string | null

  login: (username: string, password: string) => Promise<void>
  register: (username: string, password: string) => Promise<string>
  logout: () => void
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>
  checkAuth: () => Promise<void>
  clearError: () => void
}

export const useAuthStore = create<AuthStore>((set) => ({
  token: getToken(),
  user: null,
  loading: true,
  error: null,

  login: async (username, password) => {
    set({ error: null, loading: true })
    try {
      const data = await apiLogin(username, password)
      setToken(data.token)
      set({ token: data.token, user: data.user, loading: false })
    } catch (e: any) {
      set({ error: e.message, loading: false })
      throw e
    }
  },

  register: async (username, password) => {
    set({ error: null })
    try {
      const data = await apiRegister(username, password)
      return data.message
    } catch (e: any) {
      set({ error: e.message })
      throw e
    }
  },

  logout: () => {
    clearToken()
    set({ token: null, user: null, loading: false, error: null })
  },

  changePassword: async (currentPassword, newPassword) => {
    set({ error: null })
    try {
      const data = await apiChangePassword(currentPassword, newPassword)
      setToken(data.token)
      set({ token: data.token, user: data.user })
    } catch (e: any) {
      set({ error: e.message })
      throw e
    }
  },

  checkAuth: async () => {
    const token = getToken()
    if (!token) {
      set({ token: null, user: null, loading: false })
      return
    }
    try {
      const data = await getMe()
      set({ token, user: data.user, loading: false })
    } catch {
      clearToken()
      set({ token: null, user: null, loading: false })
    }
  },

  clearError: () => set({ error: null }),
}))
