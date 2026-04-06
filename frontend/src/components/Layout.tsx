import { Outlet } from "react-router-dom"
import { Sidebar } from "./Sidebar"
import { Topbar } from "./Topbar"
import { Toaster } from "sonner"
import { AlertProvider } from "./AlertProvider"
import { ViolationModal } from "./ViolationModal"
import { LiveAlertsPanel } from "./LiveAlertsPanel"

export function Layout() {
  return (
    <div className="flex h-screen overflow-hidden">
      <AlertProvider />
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <Topbar />
        <div className="flex flex-1 min-h-0 overflow-hidden">
          <main className="flex-1 overflow-y-auto bg-[var(--color-bg-secondary)]">
            <Outlet />
          </main>
          <LiveAlertsPanel />
        </div>
      </div>
      <Toaster position="top-right" richColors expand={false} visibleToasts={3} />
      <ViolationModal />
    </div>
  )
}
