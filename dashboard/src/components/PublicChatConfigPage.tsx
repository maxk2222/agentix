import { useEffect, useRef, useState } from 'react'
import { Copy, Check, MessageSquare, Shield } from 'lucide-react'
import UnsavedChangesBar, { useUnsavedGuard } from './UnsavedChangesBar'
import LogoField from './LogoField'
import { useData } from '../api/DataContext'
import { useBranding } from '../ui/BrandingContext'
import { publicChatConfig, chatRoles, agentChains, type ChatRoleMappings } from '../api/client'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

interface ChatWidgetColors {
  primary?: string
  headerBackground?: string
  headerText?: string
  buttonBackground?: string
  buttonText?: string
  background?: string
  text?: string
  assistantMessageBackground?: string
  userMessageBackground?: string
  inputBackground?: string
  inputText?: string
}

interface PublicChatConfigState {
  enabled: boolean
  title: string
  subtitle: string
  logo_url: string
  favicon_url: string
  theme: 'light' | 'dark'
  language: 'auto' | 'de' | 'en'
  colors: ChatWidgetColors
  privacy_text: string
  initial_message: boolean
  initial_message_content: string
}

const DEFAULT_COLORS: ChatWidgetColors = {
  primary: '#0d3d66',
  headerBackground: '#0d3d66',
  headerText: '#ffffff',
  buttonBackground: '#0d3d66',
  buttonText: '#ffffff',
  background: '#ffffff',
  text: '#333333',
  assistantMessageBackground: '#f5f5f5',
  userMessageBackground: '#0d3d66',
  inputBackground: '#ffffff',
  inputText: '#333333',
}

function getDefaultConfig(lang: 'de' | 'en'): PublicChatConfigState {
  const isDe = lang === 'de'
  return {
    enabled: false,
    title: 'Agentix Public Chat',
    subtitle: isDe ? 'Powered by Agentix' : 'Powered by Agentix',
    logo_url: '',
    favicon_url: '',
    theme: 'light',
    language: 'auto',
    colors: { ...DEFAULT_COLORS },
    privacy_text: isDe
      ? 'Ihre Konversationen werden ausschließlich auf den Servern des Betreibers verarbeitet. Es werden keine personenbezogenen Daten an Dritte weitergegeben.'
      : 'Your conversations are processed exclusively on the operator\'s servers. No personal data is shared with third parties.',
    initial_message: true,
    initial_message_content: isDe ? 'Hallo! Wie kann ich dir helfen?' : 'Hello! How can I help you?',
  }
}

function getPublicBase(): string {
  const path = window.location.pathname
  const idx = path.lastIndexOf('/dashboard/')
  if (idx === -1) return '/dashboard'
  return path.slice(0, idx) + '/dashboard'
}

export default function PublicChatConfigPage() {
  const { language } = useAppPreferences()
  const $t = useT(language)

  const [cfg, setCfg] = useState<PublicChatConfigState>(getDefaultConfig(language))
  const [assistants, setAssistants] = useState<string[]>([])
  const [chains, setChains] = useState<string[]>([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [saveSuccess, setSaveSuccess] = useState(false)
  const [saveError, setSaveError] = useState('')
  const [saving, setSaving] = useState(false)
  const [copied, setCopied] = useState(false)
  const { getInstructions } = useData()
  const { branding } = useBranding()
  const cfgBaselineRef = useRef<string>(JSON.stringify(cfg))

  // Role mapping state
  const [roleMappings, setRoleMappings] = useState<ChatRoleMappings>({})
  const [rolesLoading, setRolesLoading] = useState(false)
  const [rolesSaveSuccess, setRolesSaveSuccess] = useState(false)
  const [rolesSaveError, setRolesSaveError] = useState('')
  const rolesBaselineRef = useRef<string>(JSON.stringify({}))

  const ROLE_LABELS: Record<string, string> = {
    anonymous: 'Anonym',
    standard: 'Standard',
    pro: 'Pro',
    admin: 'Admin',
  }

  // Load available assistants
  useEffect(() => {
    setLoadingModels(true)
    getInstructions()
      .then((data) => {
        const names = data.assistants.map((a: { name: string }) => a.name)
        setAssistants(names)
      })
      .catch(() => {
        setAssistants([])
      })
      .finally(() => setLoadingModels(false))
    agentChains
      .list()
      .then((data) => {
        setChains(data.filter((c) => c.enabled).map((c) => c.name))
      })
      .catch(() => {
        setChains([])
      })
  }, [getInstructions])

  // Load saved public chat config from backend
  useEffect(() => {
    publicChatConfig
      .get()
      .then((defaults) => {
        if (defaults && Object.keys(defaults).length > 0) {
          setCfg((prev) => {
            const merged = {
              ...prev,
              ...(defaults as Partial<PublicChatConfigState>),
              colors: {
                ...prev.colors,
                ...(defaults.colors as ChatWidgetColors),
              },
            }
            cfgBaselineRef.current = JSON.stringify(merged)
            return merged
          })
        }
      })
      .catch(() => {
        // ignore
      })
  }, [])

  // Load role mappings
  useEffect(() => {
    setRolesLoading(true)
    chatRoles
      .list()
      .then((data) => {
        const mappings = data.mappings || {}
        setRoleMappings(mappings)
        rolesBaselineRef.current = JSON.stringify(mappings)
      })
      .catch(() => {
        setRoleMappings({})
      })
      .finally(() => setRolesLoading(false))
  }, [])

  const cfgDirty = JSON.stringify(cfg) !== cfgBaselineRef.current
  const rolesDirty = JSON.stringify(roleMappings) !== rolesBaselineRef.current
  const isDirty = cfgDirty || rolesDirty

  useUnsavedGuard(isDirty, $t('common.leaveWarning'))

  async function handleSave() {
    setSaveSuccess(false)
    setSaveError('')
    try {
      await publicChatConfig.save(cfg as unknown as Record<string, unknown>)
      cfgBaselineRef.current = JSON.stringify(cfg)
      setSaveSuccess(true)
      setTimeout(() => setSaveSuccess(false), 3000)
    } catch (err: any) {
      setSaveError(err.message || $t('common.saveError'))
      setTimeout(() => setSaveError(''), 5000)
      throw err
    }
  }

  function toggleRoleAssistant(role: string, assistant: string) {
    setRoleMappings((prev) => {
      const current = new Set(prev[role] || [])
      if (current.has(assistant)) {
        current.delete(assistant)
      } else {
        current.add(assistant)
      }
      return { ...prev, [role]: Array.from(current) }
    })
  }

  async function handleSaveRoles() {
    setRolesSaveSuccess(false)
    setRolesSaveError('')
    try {
      await chatRoles.save(roleMappings)
      rolesBaselineRef.current = JSON.stringify(roleMappings)
      setRolesSaveSuccess(true)
      setTimeout(() => setRolesSaveSuccess(false), 3000)
    } catch (err: any) {
      setRolesSaveError(err.message || $t('common.saveError'))
      setTimeout(() => setRolesSaveError(''), 5000)
      throw err
    }
  }

  async function handleSaveAll() {
    setSaving(true)
    try {
      if (cfgDirty) await handleSave()
      if (rolesDirty) await handleSaveRoles()
    } finally {
      setSaving(false)
    }
  }

  function handleDiscard() {
    if (cfgDirty) setCfg(JSON.parse(cfgBaselineRef.current))
    if (rolesDirty) setRoleMappings(JSON.parse(rolesBaselineRef.current))
  }

  function copyEmbedUrl() {
    const publicBase = getPublicBase().replace(/\/dashboard$/, '')
    const url = window.location.origin + publicBase + '/chat'
    navigator.clipboard.writeText(url)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="widget-container">
      <section className="widget-section">
        <h2>{$t('publicChat.configuration')}</h2>

        <div className="widget-configurator">
          {/* Enabled toggle */}
          <div className="widget-config-row widget-config-checkbox">
            <input
              type="checkbox"
              id="publicChatEnabled"
              checked={cfg.enabled}
              onChange={(e) => setCfg({ ...cfg, enabled: e.target.checked })}
            />
            <label htmlFor="publicChatEnabled">{$t('publicChat.enabled')}</label>
          </div>

          {/* Title */}
          <div className="widget-config-row">
            <label>{$t('publicChat.title')}</label>
            <input
              type="text"
              value={cfg.title}
              onChange={(e) => setCfg({ ...cfg, title: e.target.value })}
            />
          </div>

          {/* Subtitle */}
          <div className="widget-config-row">
            <label>{$t('publicChat.subtitle')}</label>
            <input
              type="text"
              value={cfg.subtitle}
              onChange={(e) => setCfg({ ...cfg, subtitle: e.target.value })}
            />
          </div>

          {/* Logo */}
          <div className="widget-config-row">
            <LogoField
              label={$t('publicChat.logoUrl')}
              value={cfg.logo_url}
              onChange={(v) => setCfg({ ...cfg, logo_url: v })}
              hint={$t('publicChat.logoInheritHint')}
              previewHeight={branding.logo_url ? 40 : 24}
            />
          </div>

          {/* Favicon */}
          <div className="widget-config-row">
            <LogoField
              label={$t('publicChat.faviconUrl')}
              value={cfg.favicon_url}
              onChange={(v) => setCfg({ ...cfg, favicon_url: v })}
              hint={$t('publicChat.logoInheritHint')}
              previewHeight={24}
            />
          </div>

          {/* Theme */}
          <div className="widget-config-row">
            <label>{$t('publicChat.theme')}</label>
            <select
              value={cfg.theme}
              onChange={(e) => setCfg({ ...cfg, theme: e.target.value as 'light' | 'dark' })}
            >
              <option value="light">{$t('widget.light')}</option>
              <option value="dark">{$t('widget.dark')}</option>
            </select>
          </div>

          {/* Language */}
          <div className="widget-config-row">
            <label>{$t('publicChat.language')}</label>
            <select
              value={cfg.language}
              onChange={(e) => setCfg({ ...cfg, language: e.target.value as 'auto' | 'de' | 'en' })}
            >
              <option value="auto">{$t('publicChat.languageAuto')}</option>
              <option value="de">Deutsch</option>
              <option value="en">English</option>
            </select>
          </div>

          {/* Privacy text */}
          <div className="widget-config-row">
            <label>{$t('publicChat.privacyText')}</label>
            <textarea
              rows={4}
              value={cfg.privacy_text}
              onChange={(e) => setCfg({ ...cfg, privacy_text: e.target.value })}
            />
          </div>

          {/* Initial message */}
          <div className="widget-config-row widget-config-checkbox">
            <input
              type="checkbox"
              id="publicChatInitialMessage"
              checked={cfg.initial_message}
              onChange={(e) => setCfg({ ...cfg, initial_message: e.target.checked })}
            />
            <label htmlFor="publicChatInitialMessage">{$t('publicChat.initialMessage')}</label>
          </div>

          {cfg.initial_message && (
            <div className="widget-config-row">
              <label>{$t('publicChat.initialMessageContent')}</label>
              <input
                type="text"
                value={cfg.initial_message_content}
                onChange={(e) => setCfg({ ...cfg, initial_message_content: e.target.value })}
              />
            </div>
          )}

          {/* Colors */}
          <div className="widget-config-divider">{$t('publicChat.colors')}</div>
          {Object.entries(cfg.colors).map(([key, value]) => (
            <div key={key} className="widget-config-row widget-config-color">
              <label>{key}</label>
              <input
                type="color"
                value={value || '#000000'}
                onChange={(e) =>
                  setCfg({
                    ...cfg,
                    colors: { ...cfg.colors, [key]: e.target.value },
                  })
                }
              />
            </div>
          ))}

          {/* Status */}
          {(saveSuccess || saveError) && (
            <div className="widget-config-row" style={{ marginTop: 16 }}>
              {saveSuccess && <span style={{ color: 'green' }}>{$t('common.saved')}</span>}
              {saveError && <span style={{ color: 'red' }}>{saveError}</span>}
            </div>
          )}
        </div>
      </section>

      {/* Role-Assistant Mapping */}
      <section className="widget-section">
        <h2><Shield size={20} style={{ marginRight: 8, verticalAlign: 'text-bottom' }} />Rollen & Assistenten</h2>
        <div className="widget-configurator">
          <p style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 12 }}>
            Konfiguriere, welche Assistenten für jede Nutzerrolle sichtbar sind.
            Ohne Zuordnung sieht die Rolle alle freigeschalteten Assistenten.
          </p>
          {rolesLoading && <div style={{ fontSize: 13, color: 'var(--muted)' }}>Laden...</div>}
          {!rolesLoading && Object.keys(ROLE_LABELS).map((role) => (
            <div key={role} className="widget-config-row">
              <label>{ROLE_LABELS[role]}</label>
              <div
                style={{
                  maxHeight: 160,
                  overflowY: 'auto',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  padding: '8px 12px',
                }}
              >
                {assistants.length === 0 && !loadingModels && (
                  <div style={{ color: 'var(--muted)', fontSize: 14 }}>Keine Assistenten verfügbar.</div>
                )}
                {assistants.map((name) => (
                  <div key={name} className="widget-config-row widget-config-checkbox" style={{ marginBottom: 4 }}>
                    <input
                      type="checkbox"
                      id={`role-${role}-assistant-${name}`}
                      checked={(roleMappings[role] || []).includes(name)}
                      onChange={() => toggleRoleAssistant(role, name)}
                    />
                    <label htmlFor={`role-${role}-assistant-${name}`}>{name}</label>
                  </div>
                ))}
                {chains.map((name) => (
                  <div key={`chain:${name}`} className="widget-config-row widget-config-checkbox" style={{ marginBottom: 4 }}>
                    <input
                      type="checkbox"
                      id={`role-${role}-chain-${name}`}
                      checked={(roleMappings[role] || []).includes(`chain:${name}`)}
                      onChange={() => toggleRoleAssistant(role, `chain:${name}`)}
                    />
                    <label htmlFor={`role-${role}-chain-${name}`}>{name} (Chain)</label>
                  </div>
                ))}
              </div>
            </div>
          ))}
          {(rolesSaveSuccess || rolesSaveError) && (
            <div className="widget-config-row" style={{ marginTop: 16 }}>
              {rolesSaveSuccess && <span style={{ color: 'green' }}>{$t('common.saved')}</span>}
              {rolesSaveError && <span style={{ color: 'red' }}>{rolesSaveError}</span>}
            </div>
          )}
        </div>
      </section>

      {/* Embed URL */}
      <section className="widget-section">
        <h2>{$t('publicChat.embedUrl')}</h2>
        <div className="widget-embed-preview">
          <div className="widget-embed-code">
            <pre>{window.location.origin + getPublicBase().replace(/\/dashboard$/, '') + '/chat'}</pre>
            <button className="widget-copy-btn" onClick={copyEmbedUrl} title="Copy">
              {copied ? <Check size={16} /> : <Copy size={16} />}
            </button>
          </div>
          <a
            href={window.location.origin + getPublicBase().replace(/\/dashboard$/, '') + '/chat'}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-primary"
            style={{ marginTop: 12, display: 'inline-flex', alignItems: 'center', gap: 8 }}
          >
            <MessageSquare size={16} /> {$t('publicChat.openPublicChat')}
          </a>
        </div>
      </section>

      <UnsavedChangesBar
        dirty={isDirty}
        saving={saving}
        onSave={handleSaveAll}
        onDiscard={handleDiscard}
      />
    </div>
  )
}
