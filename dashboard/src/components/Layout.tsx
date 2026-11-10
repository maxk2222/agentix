import { useState, useMemo } from 'react'
import { Link, useLocation, useNavigate, Outlet } from 'react-router-dom'
import {
  LayoutDashboard,
  BarChart3,
  LogOut,
  Settings,
  MessageSquare,
  MessageCircle,
  Users,
  Database,
  ChevronDown,
  ChevronUp,
  Globe,
  Moon,
  Sun,
  Languages,
  Wrench,
  Clock,
  Workflow,
  Volume2,
  BrainCircuit,
  KeyRound,
  DatabaseBackup,
  Palette,
  X,
} from 'lucide-react'
import logo from '../assets/logo.png'
import { useAuth } from '../api/AuthContext'
import { auth } from '../api/client'
import { useAppPreferences } from '../ui/AppPreferences'
import { useBranding } from '../ui/BrandingContext'
import { useT } from '../i18n'

type PageItem = {
  id: string
  i18nKey: string
  icon: React.ComponentType<any>
  children?: PageItem[]
  permission?: string
}

export default function Layout() {
  const location = useLocation()
  const navigate = useNavigate()
  const active = location.pathname.slice(1) || 'overview'
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set(['config']))
  const [showPwdModal, setShowPwdModal] = useState(false)
  const [pwdForm, setPwdForm] = useState({ current_password: '', new_password: '', confirm_password: '' })
  const [pwdError, setPwdError] = useState('')
  const [pwdSuccess, setPwdSuccess] = useState(false)
  const [pwdLoading, setPwdLoading] = useState(false)
  const { theme, setTheme, language, setLanguage } = useAppPreferences()
  const $t = useT(language)
  const { logout, user, hasPermission } = useAuth()
  const { branding } = useBranding()

  const pages: PageItem[] = useMemo(() => {
    const role = user?.role
    const all: PageItem[] = [
      { id: 'overview', i18nKey: 'nav.overview', icon: LayoutDashboard, permission: 'metrics:read' },
      { id: 'chat', i18nKey: 'nav.chat', icon: MessageCircle },
      { id: 'conversations', i18nKey: 'nav.conversations', icon: MessageSquare, permission: 'conversations:read' },
      { id: 'config-assistants', i18nKey: 'nav.configAssistants', icon: Users, permission: 'instructions:read' },
      { id: 'config-knowledge', i18nKey: 'nav.configKnowledge', icon: Database, permission: 'knowledge:read' },
      { id: 'config-tools', i18nKey: 'nav.configTools', icon: Wrench, permission: 'tools:read' },
      { id: 'agent-chains', i18nKey: 'nav.agentChains', icon: Workflow, permission: 'agent_chains:read' },
      { id: 'triggers', i18nKey: 'nav.triggers', icon: Clock, permission: 'triggers:read' },
      { id: 'metrics', i18nKey: 'nav.metrics', icon: BarChart3, permission: 'metrics:read' },
      { id: 'backups', i18nKey: 'nav.backups', icon: DatabaseBackup, permission: 'backup:read' },
      {
        id: 'config',
        i18nKey: 'nav.config',
        icon: Settings,
        children: [
          { id: 'widget', i18nKey: 'nav.widgetAndChat', icon: MessageSquare },
          { id: 'config-branding', i18nKey: 'nav.branding', icon: Palette, permission: 'branding:write' },
          { id: 'voice-management', i18nKey: 'nav.voiceManagement', icon: Volume2, permission: 'instructions:read' },
          { id: 'emotion-models', i18nKey: 'nav.emotionModels', icon: BrainCircuit, permission: 'instructions:read' },
          { id: 'config-runtime', i18nKey: 'nav.configRuntime', icon: Globe, permission: 'runtime_config:read' },
          { id: 'users', i18nKey: 'nav.users', icon: Users, permission: 'users:read' },
        ],
      },
    ]
    return all.filter((p) => {
      // Role-based page visibility
      if (p.id === 'users' && role !== 'admin' && role !== 'owner') return false
      if (role === 'viewer') {
        const viewerPages = ['overview', 'models', 'history', 'chat', 'widget']
        if (!viewerPages.includes(p.id)) return false
      }
      if (p.permission && !hasPermission(p.permission)) return false
      if (p.children) {
        p.children = p.children.filter((c) => !c.permission || hasPermission(c.permission))
        return p.children.length > 0
      }
      return true
    })
  }, [hasPermission, user])

  async function handleLogout() {
    await logout()
  }

  function openPwdModal() {
    setPwdForm({ current_password: '', new_password: '', confirm_password: '' })
    setPwdError('')
    setPwdSuccess(false)
    setShowPwdModal(true)
  }

  async function handleChangePassword(e: React.FormEvent) {
    e.preventDefault()
    setPwdError('')
    setPwdSuccess(false)
    if (pwdForm.new_password !== pwdForm.confirm_password) {
      setPwdError(language === 'de' ? 'Passwörter stimmen nicht überein' : 'Passwords do not match')
      return
    }
    if (pwdForm.new_password.length < 6) {
      setPwdError(language === 'de' ? 'Neues Passwort muss mindestens 6 Zeichen lang sein' : 'New password must be at least 6 characters')
      return
    }
    setPwdLoading(true)
    try {
      await auth.changePassword(pwdForm.current_password, pwdForm.new_password)
      setPwdSuccess(true)
      setPwdForm({ current_password: '', new_password: '', confirm_password: '' })
      setTimeout(() => setShowPwdModal(false), 1500)
    } catch (err: any) {
      setPwdError(err?.message || (language === 'de' ? 'Passwort konnte nicht geändert werden' : 'Failed to change password'))
    } finally {
      setPwdLoading(false)
    }
  }

  const toggleGroup = (groupId: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(groupId)) {
        next.delete(groupId)
      } else {
        next.add(groupId)
      }
      return next
    })
  }

  const isInGroup = (page: PageItem, activeId: string): boolean => {
    if (!page.children) return false
    return page.children.some((c) => c.id === activeId)
  }

  const handleGroupClick = (page: PageItem) => {
    if (page.children && page.children.length > 0) {
      toggleGroup(page.id)
      if (!isInGroup(page, active)) {
        navigate(`/${page.children[0].id}`)
      }
    }
  }

  const renderNavItem = (page: PageItem, depth: number = 0) => {
    const hasChildren = page.children && page.children.length > 0
    const isExpanded = expandedGroups.has(page.id)
    const activeInGroup = hasChildren && isInGroup(page, active)
    const isActive = active === page.id || activeInGroup

    return (
      <li key={page.id} className={depth > 0 ? 'sub-nav-item' : 'nav-item-wrap'}>
        {hasChildren ? (
          <div
            className={`nav-item ${isActive ? 'active' : ''}`}
            onClick={() => handleGroupClick(page)}
            style={{ paddingLeft: depth > 0 ? `${20 + depth * 16}px` : undefined }}
          >
            <page.icon size={18} />
            <span className="nav-label">{$t(page.i18nKey)}</span>
            <span className="nav-chevron">
              {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </span>
          </div>
        ) : (
          <Link
            to={`/${page.id}`}
            className={`nav-item ${isActive ? 'active' : ''}`}
            style={{ paddingLeft: depth > 0 ? `${20 + depth * 16}px` : undefined }}
          >
            <page.icon size={18} />
            <span className="nav-label">{$t(page.i18nKey)}</span>
          </Link>
        )}
        {hasChildren && isExpanded && (
          <ul className="sub-nav">
            {page.children!.map((child) => renderNavItem(child, depth + 1))}
          </ul>
        )}
      </li>
    )
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-header">
          <img src={branding.logo_url || logo} alt={branding.app_name || 'Agentix'} className="sidebar-logo" />
        </div>
        <ul className="sidebar-nav">
          {pages.map((p) => renderNavItem(p))}
        </ul>
        <div className="sidebar-footer">
          <div className="sidebar-controls">
            <button
              className="sidebar-control-btn"
              onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
              title={theme === 'light' ? $t('preferences.dark') : $t('preferences.light')}
            >
              {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
            </button>
            <button
              className="sidebar-control-btn"
              onClick={() => setLanguage(language === 'de' ? 'en' : 'de')}
              title={$t('preferences.language')}
            >
              <Languages size={16} />
              <span className="sidebar-lang-label">{language.toUpperCase()}</span>
            </button>
          </div>
          <div className="sidebar-user">
            <span>{user?.display_name || user?.username || $t('common.admin')}</span>
            <div style={{ display: 'flex', gap: 4 }}>
              <button onClick={openPwdModal} title={language === 'de' ? 'Passwort ändern' : 'Change password'}>
                <KeyRound size={14} />
              </button>
              <button onClick={handleLogout} title={$t('common.logout')}>
                <LogOut size={14} />
              </button>
            </div>
          </div>
        </div>
      </aside>
      <main className="content">
        <Outlet />
      </main>

      {showPwdModal && (
        <div className="modal-overlay" onClick={() => setShowPwdModal(false)}>
          <div className="modal-box" style={{ maxWidth: 420 }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{language === 'de' ? 'Passwort ändern' : 'Change Password'}</h3>
              <button className="btn-icon" onClick={() => setShowPwdModal(false)} title={$t('common.close')}>
                <X size={18} />
              </button>
            </div>
            {pwdError && <div className="alert-error" style={{ marginBottom: 12 }}>{pwdError}</div>}
            {pwdSuccess && (
              <div className="alert-success" style={{ marginBottom: 12 }}>
                {language === 'de' ? 'Passwort erfolgreich geändert' : 'Password changed successfully'}
              </div>
            )}
            <form onSubmit={handleChangePassword}>
              <div className="rc-field">
                <label>{language === 'de' ? 'Aktuelles Passwort' : 'Current Password'}</label>
                <input
                  type="password"
                  value={pwdForm.current_password}
                  onChange={(e) => setPwdForm((f) => ({ ...f, current_password: e.target.value }))}
                  required
                  autoComplete="current-password"
                />
              </div>
              <div className="rc-field">
                <label>{language === 'de' ? 'Neues Passwort' : 'New Password'}</label>
                <input
                  type="password"
                  value={pwdForm.new_password}
                  onChange={(e) => setPwdForm((f) => ({ ...f, new_password: e.target.value }))}
                  required
                  minLength={6}
                  maxLength={100}
                  autoComplete="new-password"
                />
              </div>
              <div className="rc-field">
                <label>{language === 'de' ? 'Neues Passwort bestätigen' : 'Confirm New Password'}</label>
                <input
                  type="password"
                  value={pwdForm.confirm_password}
                  onChange={(e) => setPwdForm((f) => ({ ...f, confirm_password: e.target.value }))}
                  required
                  minLength={6}
                  maxLength={100}
                  autoComplete="new-password"
                />
              </div>
              <div style={{ display: 'flex', gap: 12, marginTop: 16 }}>
                <button type="submit" className="btn-primary" disabled={pwdLoading}>
                  {pwdLoading ? (language === 'de' ? 'Speichern...' : 'Saving...') : (language === 'de' ? 'Speichern' : 'Save')}
                </button>
                <button type="button" className="btn-secondary" onClick={() => setShowPwdModal(false)}>
                  {$t('common.cancel')}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
