import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { reportError } from '@/lib/errorReporter'
import './index.css'
import App from './App.tsx'

// Catch unhandled promise rejections globally
window.addEventListener('unhandledrejection', (event) => {
  reportError(event.reason, { component: 'global', extra: { type: 'unhandledrejection' } })
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
