import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import { branding as brandingApi } from '../api/client'

export interface BrandingThemeVars {
  [key: string]: string
}

export interface Branding {
  app_name?: string
  powered_by?: string
  logo_url?: string
  favicon_url?: string
  theme?: {
    light?: BrandingThemeVars
    dark?: BrandingThemeVars
  }
}

interface BrandingContextValue {
  branding: Branding
  refresh: () => Promise<void>
}

const BrandingContext = createContext<BrandingContextValue>({
  branding: {},
  refresh: async () => {},
})

const THEME_STYLE_ID = 'agentix-brand-theme'
const DEFAULT_TITLE = 'Agentix Dashboard'

// Editable theme var names (snake_case) -> CSS custom property names.
const VAR_KEYS: Record<string, string> = {
  bg: '--bg',
  card: '--card',
  surface: '--surface',
  surface_hover: '--surface-hover',
  text: '--text',
  text_secondary: '--text-secondary',
  text_muted: '--text-muted',
  border: '--border',
  primary: '--primary',
  danger: '--danger',
  success: '--success',
  warning: '--warning',
  info: '--info',
}

// Alpha values differ per theme in index.css — keep them theme-specific.
const ACCENT_ALPHAS = {
  light: { light: 0.09, border: 0.28 },
  dark: { light: 0.14, border: 0.35 },
} as const

function hexToRgba(color: string, alpha: number): string {
  let clean = color.trim().replace(/^#/, '')
  if (clean.length === 3) {
    clean = clean.split('').map((c) => c + c).join('')
  }
  if (clean.length === 8) clean = clean.substring(0, 6)
  const num = parseInt(clean, 16)
  if (Number.isNaN(num)) return color
  return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`
}

function mixColors(color: string, target: string, amount: number): string {
  const parse = (c: string) => {
    let clean = c.trim().replace(/^#/, '')
    if (clean.length === 3) clean = clean.split('').map((ch) => ch + ch).join('')
    const num = parseInt(clean, 16)
    return Number.isNaN(num) ? [0, 0, 0] : [(num >> 16) & 255, (num >> 8) & 255, num & 255]
  }
  const [r1, g1, b1] = parse(color)
  const [r2, g2, b2] = parse(target)
  const m = (a: number, b: number) => Math.round(a + (b - a) * amount)
  const hex = (n: number) => n.toString(16).padStart(2, '0')
  return `#${hex(m(r1, r2))}${hex(m(g1, g2))}${hex(m(b1, b2))}`
}

function isDarkColor(color: string): boolean {
  let clean = color.trim().replace(/^#/, '')
  if (clean.length === 3) clean = clean.split('').map((c) => c + c).join('')
  const num = parseInt(clean, 16)
  if (Number.isNaN(num)) return false
  const r = (num >> 16) & 255
  const g = (num >> 8) & 255
  const b = num & 255
  return 0.299 * r + 0.587 * g + 0.114 * b < 150
}

/** Derive the accent/-hover/-light/-border/-secondary/on-primary set from `primary`. */
function deriveAccentVars(primary: string, mode: 'light' | 'dark'): Record<string, string> {
  const alphas = ACCENT_ALPHAS[mode]
  const hoverTarget = mode === 'light' ? '#000000' : '#ffffff'
  const secondaryTarget = mode === 'light' ? '#ffffff' : '#ffffff'
  return {
    '--primary': primary,
    '--primary-light': hexToRgba(primary, alphas.light),
    '--accent': primary,
    '--accent-hover': mixColors(primary, hoverTarget, mode === 'light' ? 0.15 : 0.12),
    '--accent-light': hexToRgba(primary, alphas.light),
    '--accent-border': hexToRgba(primary, alphas.border),
    '--accent-secondary': mixColors(primary, secondaryTarget, mode === 'light' ? 0.2 : 0.12),
    '--on-primary': isDarkColor(primary) ? '#ffffff' : '#0c1826',
  }
}

function buildThemeCss(branding: Branding): string {
  const theme = branding.theme
  if (!theme) return ''
  const blocks: string[] = []
  const render = (vars: BrandingThemeVars | undefined, mode: 'light' | 'dark', selector: string) => {
    if (!vars) return
    const declarations: Record<string, string> = {}
    for (const [key, cssVar] of Object.entries(VAR_KEYS)) {
      const value = vars[key]
      if (value && key !== 'primary') declarations[cssVar] = value
    }
    if (vars.primary) {
      Object.assign(declarations, deriveAccentVars(vars.primary, mode))
    }
    const body = Object.entries(declarations)
      .map(([k, v]) => `${k}: ${v};`)
      .join(' ')
    if (body) blocks.push(`${selector} { ${body} }`)
  }
  render(theme.light, 'light', ':root')
  render(theme.dark, 'dark', '[data-theme="dark"]')
  return blocks.join('\n')
}

function applyBranding(branding: Branding): void {
  document.title = branding.app_name ? `${branding.app_name} Dashboard` : DEFAULT_TITLE

  const favicon = document.querySelector<HTMLLinkElement>('link[rel~="icon"]')
  if (favicon && branding.favicon_url) {
    favicon.href = branding.favicon_url
  }

  let styleEl = document.getElementById(THEME_STYLE_ID) as HTMLStyleElement | null
  const css = buildThemeCss(branding)
  if (css) {
    if (!styleEl) {
      styleEl = document.createElement('style')
      styleEl.id = THEME_STYLE_ID
      document.head.appendChild(styleEl)
    }
    styleEl.textContent = css
  } else if (styleEl) {
    styleEl.remove()
  }
}

export function BrandingProvider({ children }: { children: ReactNode }) {
  const [branding, setBranding] = useState<Branding>({})

  const refresh = useCallback(async () => {
    const cfg = await brandingApi.get()
    const next = (cfg ?? {}) as Branding
    setBranding(next)
    applyBranding(next)
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  return (
    <BrandingContext.Provider value={{ branding, refresh }}>
      {children}
    </BrandingContext.Provider>
  )
}

export function useBranding() {
  return useContext(BrandingContext)
}
