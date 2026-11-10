import { useEffect, useRef, useState } from 'react'
import UnsavedChangesBar, { useUnsavedGuard } from './UnsavedChangesBar'
import LogoField from './LogoField'
import { branding as brandingApi } from '../api/client'
import { useBranding, type BrandingThemeVars } from '../ui/BrandingContext'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

type ThemeMode = 'light' | 'dark'

interface BrandingState {
  app_name: string
  powered_by: string
  logo_url: string
  favicon_url: string
  theme: { light: BrandingThemeVars; dark: BrandingThemeVars }
}

// Editable theme variables — the same curated set the backend sanitizer
// accepts; derived accent/-light/-border vars are computed automatically.
const THEME_VAR_KEYS = [
  'bg',
  'card',
  'surface',
  'surface_hover',
  'text',
  'text_secondary',
  'text_muted',
  'border',
  'primary',
  'danger',
  'success',
  'warning',
  'info',
] as const

// Baseline values from index.css — used as color-input fallbacks and for
// "reset to default" semantics.
const LIGHT_DEFAULTS: BrandingThemeVars = {
  bg: '#f3f5f8',
  card: '#ffffff',
  surface: '#f0f4f8',
  surface_hover: '#e3eaf2',
  text: '#18222e',
  text_secondary: '#4a5b6e',
  text_muted: '#7b8a99',
  border: '#d8e0e8',
  primary: '#1d4e89',
  danger: '#dc2626',
  success: '#059669',
  warning: '#d97706',
  info: '#3d6ea8',
}

const DARK_DEFAULTS: BrandingThemeVars = {
  bg: '#0c1420',
  card: '#131d2b',
  surface: '#15202f',
  surface_hover: '#1d2c3f',
  text: '#e2e9f0',
  text_secondary: '#9bacc0',
  text_muted: '#647689',
  border: '#24344a',
  primary: '#6b9bd4',
  danger: '#f87171',
  success: '#34d399',
  warning: '#fbbf24',
  info: '#7fa8d9',
}

const EMPTY_STATE: BrandingState = {
  app_name: '',
  powered_by: '',
  logo_url: '',
  favicon_url: '',
  theme: { light: {}, dark: {} },
}

export default function BrandingPage() {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const { refresh } = useBranding()

  const [cfg, setCfg] = useState<BrandingState>(EMPTY_STATE)
  const [themeTab, setThemeTab] = useState<ThemeMode>('light')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saveSuccess, setSaveSuccess] = useState(false)
  const [saveError, setSaveError] = useState('')
  const baselineRef = useRef<string>(JSON.stringify(EMPTY_STATE))

  useEffect(() => {
    setLoading(true)
    brandingApi
      .get()
      .then((data) => {
        if (!data) return
        const merged: BrandingState = {
          app_name: (data.app_name as string) || '',
          powered_by: (data.powered_by as string) || '',
          logo_url: (data.logo_url as string) || '',
          favicon_url: (data.favicon_url as string) || '',
          theme: {
            light: { ...((data.theme as any)?.light ?? {}) },
            dark: { ...((data.theme as any)?.dark ?? {}) },
          },
        }
        setCfg(merged)
        baselineRef.current = JSON.stringify(merged)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const isDirty = JSON.stringify(cfg) !== baselineRef.current
  useUnsavedGuard(isDirty, $t('common.leaveWarning'))

  function setThemeVar(mode: ThemeMode, key: string, value: string) {
    setCfg((prev) => ({
      ...prev,
      theme: { ...prev.theme, [mode]: { ...prev.theme[mode], [key]: value } },
    }))
  }

  async function handleSave() {
    setSaving(true)
    setSaveSuccess(false)
    setSaveError('')
    try {
      // Empty theme maps carry no information — drop them from the payload.
      const theme: Record<string, BrandingThemeVars> = {}
      if (Object.keys(cfg.theme.light).length) theme.light = cfg.theme.light
      if (Object.keys(cfg.theme.dark).length) theme.dark = cfg.theme.dark
      const payload: Record<string, unknown> = {
        app_name: cfg.app_name,
        powered_by: cfg.powered_by,
        logo_url: cfg.logo_url,
        favicon_url: cfg.favicon_url,
      }
      if (Object.keys(theme).length) payload.theme = theme
      await brandingApi.save(payload)
      baselineRef.current = JSON.stringify(cfg)
      setSaveSuccess(true)
      await refresh()
      setTimeout(() => setSaveSuccess(false), 3000)
    } catch (err: any) {
      setSaveError(err.message || $t('common.error'))
      setTimeout(() => setSaveError(''), 5000)
      throw err
    } finally {
      setSaving(false)
    }
  }

  function handleDiscard() {
    setCfg(JSON.parse(baselineRef.current))
  }

  const themeDefaults = themeTab === 'light' ? LIGHT_DEFAULTS : DARK_DEFAULTS

  return (
    <div className="widget-container">
      <section className="widget-section">
        <h2>{$t('branding.title')}</h2>
        {loading && <div style={{ fontSize: 13, color: 'var(--muted)' }}>{$t('common.loading')}</div>}
        {!loading && (
          <div className="widget-configurator">
            <div className="widget-config-divider">{$t('branding.appSection')}</div>

            <div className="widget-config-row">
              <label>{$t('branding.appName')}</label>
              <input
                type="text"
                value={cfg.app_name}
                onChange={(e) => setCfg({ ...cfg, app_name: e.target.value })}
                placeholder="Agentix"
              />
            </div>

            <div className="widget-config-row">
              <label>{$t('branding.poweredBy')}</label>
              <input
                type="text"
                value={cfg.powered_by}
                onChange={(e) => setCfg({ ...cfg, powered_by: e.target.value })}
                placeholder="Powered by Agentix"
              />
            </div>

            <div className="widget-config-divider">{$t('branding.logoSection')}</div>

            <div className="widget-config-row">
              <LogoField
                label={$t('branding.logo')}
                value={cfg.logo_url}
                onChange={(v) => setCfg({ ...cfg, logo_url: v })}
                hint={$t('branding.logoHint')}
              />
            </div>

            <div className="widget-config-row">
              <LogoField
                label={$t('branding.favicon')}
                value={cfg.favicon_url}
                onChange={(v) => setCfg({ ...cfg, favicon_url: v })}
                hint={$t('branding.faviconHint')}
                previewHeight={24}
              />
            </div>

            <div className="widget-config-divider">{$t('branding.themeSection')}</div>

            <div className="widget-embed-toggle" style={{ marginBottom: 12 }}>
              <button
                className={themeTab === 'light' ? 'active' : ''}
                onClick={() => setThemeTab('light')}
              >
                {$t('branding.lightTheme')}
              </button>
              <button
                className={themeTab === 'dark' ? 'active' : ''}
                onClick={() => setThemeTab('dark')}
              >
                {$t('branding.darkTheme')}
              </button>
            </div>

            {THEME_VAR_KEYS.map((key) => {
              const value = cfg.theme[themeTab][key] ?? ''
              const fallback = themeDefaults[key] ?? '#000000'
              return (
                <div key={key} className="widget-config-row widget-config-color">
                  <label>{$t(`branding.vars.${key}`)}</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <input
                      type="color"
                      value={value || fallback}
                      onChange={(e) => setThemeVar(themeTab, key, e.target.value)}
                    />
                    <input
                      type="text"
                      value={value}
                      placeholder={fallback}
                      style={{ width: 110, fontFamily: 'monospace', fontSize: 12 }}
                      onChange={(e) => setThemeVar(themeTab, key, e.target.value)}
                    />
                    {value && (
                      <button
                        type="button"
                        className="btn btn-secondary"
                        style={{ padding: '2px 8px', fontSize: 12 }}
                        onClick={() => {
                          setCfg((prev) => {
                            const next = { ...prev.theme[themeTab] }
                            delete next[key]
                            return { ...prev, theme: { ...prev.theme, [themeTab]: next } }
                          })
                        }}
                        title={$t('branding.resetVar')}
                      >
                        ↺
                      </button>
                    )}
                  </div>
                </div>
              )
            })}
            <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 8 }}>
              {$t('branding.themeHint')}
            </p>

            {(saveSuccess || saveError) && (
              <div className="widget-config-row" style={{ marginTop: 16 }}>
                {saveSuccess && <span style={{ color: 'green' }}>{$t('common.saved')}</span>}
                {saveError && <span style={{ color: 'red' }}>{saveError}</span>}
              </div>
            )}
          </div>
        )}
      </section>

      <UnsavedChangesBar
        dirty={isDirty}
        saving={saving}
        onSave={handleSave}
        onDiscard={handleDiscard}
      />
    </div>
  )
}
