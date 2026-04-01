import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { Layout } from "@/components/Layout"
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

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/live" replace />} />
          <Route path="/live" element={<LiveView />} />
          <Route path="/alerts" element={<AlertCenter />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/reports" element={<Placeholder title="Reports" />} />
          <Route path="/search" element={<AISearch />} />
          <Route path="/configure/cameras" element={<CameraConfig />} />
          <Route path="/configure/rules" element={<RulesEngine />} />
          <Route path="/configure/rules/new" element={<RuleEditor />} />
          <Route path="/configure/rules/:ruleId" element={<RuleEditor />} />
          <Route path="/configure/zones" element={<ZoneManagement />} />
          <Route path="/configure/alerts" element={<AlertRouting />} />
          <Route path="/configure/plates" element={<PlateManagement />} />
          <Route path="/configure/faces" element={<FaceEnrollment />} />
          <Route path="/configure/integrations" element={<Placeholder title="Integrations" />} />
          <Route path="/system/settings" element={<SystemSettings />} />
          <Route path="/system/license" element={<LicenseStatus />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
