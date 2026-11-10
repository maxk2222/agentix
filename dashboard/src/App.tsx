import { useEffect, Suspense, lazy } from 'react'
import type { ComponentType } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './api/AuthContext'
import { DataProvider } from './api/DataContext'
import { BrandingProvider } from './ui/BrandingContext'
import Login from './components/Login'
import Layout from './components/Layout'
import { useAppPreferences } from './ui/AppPreferences'
import { useT } from './i18n'

const CHUNK_RELOAD_FLAG = 'chunk-reload-at'

/**
 * Wraps React.lazy so a failed dynamic import (stale chunk hash after a
 * redeploy while index.html is cached) triggers one hard reload instead
 * of leaving a white page. A timestamp window prevents reload loops.
 */
function lazyWithReload<T extends ComponentType<unknown>>(
  factory: () => Promise<{ default: T }>
) {
  return lazy(async () => {
    try {
      const mod = await factory()
      sessionStorage.removeItem(CHUNK_RELOAD_FLAG)
      return mod
    } catch (err) {
      const last = Number(sessionStorage.getItem(CHUNK_RELOAD_FLAG) || 0)
      if (Date.now() - last > 10_000) {
        sessionStorage.setItem(CHUNK_RELOAD_FLAG, String(Date.now()))
        window.location.reload()
        return new Promise<{ default: T }>(() => {})
      }
      throw err
    }
  })
}

const Overview = lazyWithReload(() => import('./components/Overview'))
const Models = lazyWithReload(() => import('./components/Models'))
const History = lazyWithReload(() => import('./components/History'))
const Blocked = lazyWithReload(() => import('./components/Blocked'))
const ConfigEditor = lazyWithReload(() => import('./components/ConfigEditor'))
const KnowledgeUpload = lazyWithReload(() => import('./components/KnowledgeUpload'))
const WidgetAndChatPage = lazyWithReload(() => import('./components/WidgetAndChatPage'))
const ChatPage = lazyWithReload(() => import('./components/ChatPage'))
const Conversations = lazyWithReload(() => import('./components/Conversations'))
const RuntimeConfig = lazyWithReload(() => import('./components/RuntimeConfig'))
const UserManagement = lazyWithReload(() => import('./components/UserManagement'))
const ToolsPage = lazyWithReload(() => import('./components/ToolsPage'))
const TriggersPage = lazyWithReload(() => import('./components/TriggersPage'))
const AgentChainsPage = lazyWithReload(() => import('./components/AgentChainsPage'))
const AudioPage = lazyWithReload(() => import('./components/AudioPage'))
const EmotionModelsPage = lazyWithReload(() => import('./components/EmotionModelsPage'))
const MetricsPage = lazyWithReload(() => import('./components/MetricsPage'))
const BackupPage = lazyWithReload(() => import('./components/BackupPage'))
const BrandingPage = lazyWithReload(() => import('./components/BrandingPage'))

function PageLoader() {
  return (
    <div className="login-page">
      <div className="loading">Loading...</div>
    </div>
  )
}

function AppRoutes() {
  const { authenticated, loading, refresh } = useAuth()
  const { language } = useAppPreferences()
  const $t = useT(language)

  useEffect(() => {
    refresh()
  }, [refresh])

  if (loading) {
    return (
      <div className="login-page">
        <div className="loading">{$t('common.loading')}</div>
      </div>
    )
  }

  if (!authenticated) {
    return <Login />
  }

  return (
    <Suspense fallback={<PageLoader />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/overview" replace />} />
          <Route path="overview" element={<Overview />} />
          <Route path="models" element={<Models />} />
          <Route path="history" element={<History />} />
          <Route path="metrics" element={<MetricsPage />} />
          <Route path="conversations" element={<Conversations />} />
          <Route path="blocked" element={<Blocked />} />
          <Route path="config-assistants" element={<ConfigEditor />} />
          <Route path="config-knowledge" element={<KnowledgeUpload />} />
          <Route path="config-runtime" element={<RuntimeConfig />} />
          <Route path="config-tools" element={<ToolsPage />} />
          <Route path="voice-management" element={<AudioPage />} />
          <Route path="emotion-models" element={<EmotionModelsPage />} />
          <Route path="agent-chains" element={<AgentChainsPage />} />
          <Route path="triggers" element={<TriggersPage />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="widget" element={<WidgetAndChatPage />} />
          <Route path="config-branding" element={<BrandingPage />} />
          <Route path="users" element={<UserManagement />} />
          <Route path="backups" element={<BackupPage />} />
          <Route path="*" element={<Navigate to="/overview" replace />} />
        </Route>
      </Routes>
    </Suspense>
  )
}

function getDashboardBasename(): string {
  const path = window.location.pathname
  const idx = path.lastIndexOf('/dashboard')
  if (idx === -1) return '/dashboard'
  return path.slice(0, idx) + '/dashboard'
}

export default function App() {
  return (
    <AuthProvider>
      <BrandingProvider>
        <DataProvider>
          <BrowserRouter basename={getDashboardBasename()}>
            <AppRoutes />
          </BrowserRouter>
        </DataProvider>
      </BrandingProvider>
    </AuthProvider>
  )
}
