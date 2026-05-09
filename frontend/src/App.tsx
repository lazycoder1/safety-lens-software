import { lazy, Suspense } from "react"
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { Layout } from "@/components/Layout"
import { ProtectedRoute } from "@/components/ProtectedRoute"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { Login } from "@/pages/Login"
import { Loader2 } from "lucide-react"

const ChangePassword = lazy(() => import("@/pages/ChangePassword").then(m => ({ default: m.ChangePassword })))
const LiveView = lazy(() => import("@/pages/LiveView").then(m => ({ default: m.LiveView })))
const AlertCenter = lazy(() => import("@/pages/AlertCenter").then(m => ({ default: m.AlertCenter })))
const Dashboard = lazy(() => import("@/pages/Dashboard").then(m => ({ default: m.Dashboard })))
const CameraConfig = lazy(() => import("@/pages/CameraConfig").then(m => ({ default: m.CameraConfig })))
const CameraSetupPage = lazy(() => import("@/pages/CameraSetupPage").then(m => ({ default: m.CameraSetupPage })))
const CameraDiscoverPage = lazy(() => import("@/pages/CameraDiscoverPage").then(m => ({ default: m.CameraDiscoverPage })))
const CameraDetailsPage = lazy(() => import("@/pages/CameraDetailsPage").then(m => ({ default: m.CameraDetailsPage })))
const CameraEditPage = lazy(() => import("@/pages/CameraEditPage").then(m => ({ default: m.CameraEditPage })))
const RulesEngine = lazy(() => import("@/pages/RulesEngine").then(m => ({ default: m.RulesEngine })))
const RuleEditor = lazy(() => import("@/pages/RuleEditor").then(m => ({ default: m.RuleEditor })))
const PlateManagement = lazy(() => import("@/pages/PlateManagement").then(m => ({ default: m.PlateManagement })))
const FaceEnrollment = lazy(() => import("@/pages/FaceEnrollment").then(m => ({ default: m.FaceEnrollment })))
const AlertRouting = lazy(() => import("@/pages/AlertRouting").then(m => ({ default: m.AlertRouting })))
const Placeholder = lazy(() => import("@/pages/Placeholder").then(m => ({ default: m.Placeholder })))
const Reports = lazy(() => import("@/pages/Reports").then(m => ({ default: m.Reports })))
const AlertDetail = lazy(() => import("@/pages/AlertDetail").then(m => ({ default: m.AlertDetail })))
const SystemSettings = lazy(() => import("@/pages/SystemSettings").then(m => ({ default: m.SystemSettings })))
const SystemModels = lazy(() => import("@/pages/SystemModels").then(m => ({ default: m.SystemModels })))
const ZoneManagement = lazy(() => import("@/pages/ZoneManagement").then(m => ({ default: m.ZoneManagement })))
const AISearch = lazy(() => import("@/pages/AISearch").then(m => ({ default: m.AISearch })))
const LicenseStatus = lazy(() => import("@/pages/LicenseStatus").then(m => ({ default: m.LicenseStatus })))
const UserManagement = lazy(() => import("@/pages/UserManagement").then(m => ({ default: m.UserManagement })))

function PageLoader() {
  return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="w-6 h-6 animate-spin text-neutral-400" />
    </div>
  )
}

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/change-password" element={<ChangePassword />} />
            <Route
              element={
                <ProtectedRoute>
                  <Layout />
                </ProtectedRoute>
              }
            >
              <Route path="/" element={<Navigate to="/live" replace />} />
              <Route path="/live" element={<ErrorBoundary><LiveView /></ErrorBoundary>} />
              <Route path="/alerts" element={<ErrorBoundary><AlertCenter /></ErrorBoundary>} />
              <Route path="/dashboard" element={<ErrorBoundary><Dashboard /></ErrorBoundary>} />
              <Route path="/reports" element={<ErrorBoundary><Reports /></ErrorBoundary>} />
              <Route path="/alerts/:alertId" element={<ErrorBoundary><AlertDetail /></ErrorBoundary>} />
              <Route path="/search" element={<ErrorBoundary><AISearch /></ErrorBoundary>} />
              <Route path="/configure/cameras" element={<ErrorBoundary><CameraConfig /></ErrorBoundary>} />
              <Route
                path="/configure/cameras/discover"
                element={
                  <ProtectedRoute requiredRole={["admin"]}>
                    <ErrorBoundary><CameraDiscoverPage /></ErrorBoundary>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/configure/cameras/new"
                element={
                  <ProtectedRoute requiredRole={["admin"]}>
                    <ErrorBoundary><CameraSetupPage /></ErrorBoundary>
                  </ProtectedRoute>
                }
              />
              <Route path="/configure/cameras/:cameraId" element={<ErrorBoundary><CameraDetailsPage /></ErrorBoundary>} />
              <Route
                path="/configure/cameras/:cameraId/edit"
                element={
                  <ProtectedRoute requiredRole={["admin"]}>
                    <ErrorBoundary><CameraEditPage /></ErrorBoundary>
                  </ProtectedRoute>
                }
              />
              <Route path="/configure/rules" element={<ErrorBoundary><RulesEngine /></ErrorBoundary>} />
              <Route path="/configure/rules/new" element={<ErrorBoundary><RuleEditor /></ErrorBoundary>} />
              <Route path="/configure/rules/:ruleId" element={<ErrorBoundary><RuleEditor /></ErrorBoundary>} />
              <Route path="/configure/zones" element={<ErrorBoundary><ZoneManagement /></ErrorBoundary>} />
              <Route path="/configure/alerts" element={<ErrorBoundary><AlertRouting /></ErrorBoundary>} />
              <Route path="/configure/plates" element={<PlateManagement />} />
              <Route path="/configure/faces" element={<FaceEnrollment />} />
              <Route path="/configure/integrations" element={<Placeholder title="Integrations" />} />
              <Route path="/system/settings" element={<ErrorBoundary><SystemSettings /></ErrorBoundary>} />
              <Route
                path="/system/models"
                element={
                  <ProtectedRoute requiredRole={["admin"]}>
                    <ErrorBoundary><SystemModels /></ErrorBoundary>
                  </ProtectedRoute>
                }
              />
              <Route path="/system/license" element={<LicenseStatus />} />
              <Route
                path="/system/users"
                element={
                  <ProtectedRoute requiredRole={["admin"]}>
                    <ErrorBoundary><UserManagement /></ErrorBoundary>
                  </ProtectedRoute>
                }
              />
            </Route>
          </Routes>
        </Suspense>
      </BrowserRouter>
    </ErrorBoundary>
  )
}
