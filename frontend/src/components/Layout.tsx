import { Outlet } from "react-router-dom"
import { Sidebar } from "./Sidebar"
import { Topbar } from "./Topbar"
import { Toaster } from "sonner"

export function Layout() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <Topbar />
        <main className="flex-1 overflow-y-auto bg-[var(--color-bg-secondary)]">
          <Outlet />
        </main>
      </div>
      <Toaster position="top-right" richColors expand={false} visibleToasts={3} />
    </div>
  )
}
