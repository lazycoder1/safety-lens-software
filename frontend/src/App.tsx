import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { Layout } from "@/components/Layout"
import { ProtectedRoute } from "@/components/ProtectedRoute"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { Login } from "@/pages/Login"
import { ChangePassword } from "@/pages/ChangePassword"
import { LiveView } from "@/pages/LiveView"
import { AlertCenter } from "@/pages/AlertCenter"
import { Dashboard } from "@/pages/Dashboard"
import { CameraConfig } from "@/pages/CameraConfig"
import { RulesEngine } from "@/pages/RulesEngine"
import { RuleEditor } from "@/pages/RuleEditor"
import { PlateManagement } from "@/pages/PlateManagement"
import { FaceEnrollment } from "@/pages/FaceEnrollment"
import { AlertRouting } from "@/pages/AlertRouting"
import { Placeholder } from "@/pages/Placeholder"
import { SystemSettings } from "@/pages/SystemSettings"
import { ZoneManagement } from "@/pages/ZoneManagement"
import { AISearch } from "@/pages/AISearch"
import { LicenseStatus } from "@/pages/LicenseStatus"
import { UserManagement } from "@/pages/UserManagement"

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
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
            <Route path="/reports" element={<Placeholder title="Reports" />} />
            <Route path="/search" element={<ErrorBoundary><AISearch /></ErrorBoundary>} />
            <Route path="/configure/cameras" element={<ErrorBoundary><CameraConfig /></ErrorBoundary>} />
            <Route path="/configure/rules" element={<ErrorBoundary><RulesEngine /></ErrorBoundary>} />
            <Route path="/configure/rules/new" element={<ErrorBoundary><RuleEditor /></ErrorBoundary>} />
            <Route path="/configure/rules/:ruleId" element={<ErrorBoundary><RuleEditor /></ErrorBoundary>} />
            <Route path="/configure/zones" element={<ErrorBoundary><ZoneManagement /></ErrorBoundary>} />
            <Route path="/configure/alerts" element={<ErrorBoundary><AlertRouting /></ErrorBoundary>} />
            <Route path="/configure/plates" element={<PlateManagement />} />
            <Route path="/configure/faces" element={<FaceEnrollment />} />
            <Route path="/configure/integrations" element={<Placeholder title="Integrations" />} />
            <Route path="/system/settings" element={<ErrorBoundary><SystemSettings /></ErrorBoundary>} />
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
      </BrowserRouter>
    </ErrorBoundary>
  )
}
