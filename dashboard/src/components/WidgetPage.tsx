import { useEffect, useRef, useState } from 'react'
import { Copy, Check, RefreshCw, Trash2 } from 'lucide-react'
import { useData } from '../api/DataContext'
import { useAuth } from '../api/AuthContext'
import { widgetDefaults } from '../api/client'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'
import ConfirmModal from './ConfirmModal'
import LogoField from './LogoField'
import { useBranding } from '../ui/BrandingContext'

declare global {
  interface Window {
    AgentixChatWidget?: any
    initAgentixChat?: (config: any) => any
  }
}

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

interface WidgetConfig {
  apiUrl: string
  apiKey: string
  model: string
  position: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left'
  theme: 'light' | 'dark'
  language: 'auto' | 'de' | 'en'
  width: number
  height: number
  showOnLoad: boolean
  title: string
  subtitle: string
  logoUrlButton: string
  logoUrlHeader: string
  showLogoInButton: boolean
  showLogoInHeader: boolean
  logoBorder: { enabled: boolean; color: string; opacity: number }
  logoGlow: { enabled: boolean; color: string; opacity: number }
  logoBackground: { enabled: boolean; color: string; opacity: number }
  privacyText: string
  privacyTextDe: string
  privacyTextEn: string
  privacyEnabled: boolean
  initialMessage: boolean
  initialMessageContent: string
  initialMessageContentDe: string
  initialMessageContentEn: string
  initialMessageSilent: boolean
  bounceInterval: number
  bounceText: string
  bounceTextDe: string
  bounceTextEn: string
  inputPlaceholder: string
  inputPlaceholderDe: string
  inputPlaceholderEn: string
  colors: ChatWidgetColors
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

function getPublicBase(): string {
  const path = window.location.pathname
  const idx = path.lastIndexOf('/dashboard/')
  if (idx === -1) return '/dashboard'
  return path.slice(0, idx) + '/dashboard'
}

// The live preview always talks to the same backend that serves this
// dashboard — same-origin requests carry the dashboard session cookie, which
// authenticates /v1/* calls without an API key (APIKeyMiddleware bypass) and
// avoids CORS entirely. cfg.apiUrl may point at a deployed public origin
// (e.g. https://example.org/api/chat/v1); that value is only meaningful for
// external embeds and the generated embed code, never for the preview.
function previewApiUrl(): string {
  const apiBase = getPublicBase().replace(/\/dashboard$/, '')
  return window.location.origin + apiBase + '/v1'
}

function getDefaultConfig(lang: 'auto' | 'de' | 'en'): WidgetConfig {
  const isDe = lang !== 'en'
  const publicBase = getPublicBase()
  const apiBase = publicBase.replace(/\/dashboard$/, '')
  return {
    apiUrl: window.location.origin + apiBase + '/v1',
    apiKey: '',
    model: 'CyberSecurityChatbot',
    position: 'bottom-right',
    theme: 'light',
    language: lang,
    width: 650,
    height: 550,
    showOnLoad: false,
    title: 'CyberSecurityChatbot',
    subtitle: 'Powered by Agentix',
    logoUrlButton: '',
    logoUrlHeader: '',
    showLogoInButton: true,
    showLogoInHeader: true,
    logoBorder: { enabled: false, color: '#ffffff', opacity: 0.35 },
    logoGlow: { enabled: true, color: '#ffffff', opacity: 0.4 },
    logoBackground: { enabled: true, color: '#ffffff', opacity: 0.35 },
    privacyText: isDe
      ? 'Willkommen beim Chatbot! Ihre Konversationen werden ausschließlich auf den Servern des Betreibers verarbeitet. Es werden keine personenbezogenen Daten an Dritte weitergegeben.'
      : 'Welcome to the chatbot! Your conversations are processed exclusively on the operator\'s servers. No personal data is shared with third parties.',
    privacyTextDe: 'Willkommen beim Chatbot! Ihre Konversationen werden ausschließlich auf den Servern des Betreibers verarbeitet. Es werden keine personenbezogenen Daten an Dritte weitergegeben.',
    privacyTextEn: 'Welcome to the chatbot! Your conversations are processed exclusively on the operator\'s servers. No personal data is shared with third parties.',
    privacyEnabled: true,
    initialMessage: true,
    initialMessageContent: isDe ? 'Hallo!' : 'Hello! Language: English',
    initialMessageContentDe: 'Hallo!',
    initialMessageContentEn: 'Hello! Language: English',
    initialMessageSilent: true,
    bounceInterval: 15000,
    bounceText: isDe ? 'Hier chatten!' : 'Chat here!',
    bounceTextDe: 'Hier chatten!',
    bounceTextEn: 'Chat here!',
    inputPlaceholder: isDe ? 'Nachricht eingeben...' : 'Type your message here...',
    inputPlaceholderDe: 'Nachricht eingeben...',
    inputPlaceholderEn: 'Type your message here...',
    colors: DEFAULT_COLORS,
  }
}

function jsLiteral(value: unknown): string {
  if (typeof value === 'string') {
    return `'${value.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, '\\n')}'`
  }
  if (typeof value === 'boolean' || typeof value === 'number') {
    return String(value)
  }
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const entries = Object.entries(value)
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => `${k}: ${jsLiteral(v)}`)
    return `{ ${entries.join(', ')} }`
  }
  return 'undefined'
}

function isDeepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (typeof a !== typeof b) return false
  if (typeof a !== 'object' || a === null || b === null) return false
  const keysA = Object.keys(a as object)
  const keysB = Object.keys(b as object)
  if (keysA.length !== keysB.length) return false
  return keysA.every((k) => keysB.includes(k) && isDeepEqual((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]))
}

function generateEmbedCode(cfg: WidgetConfig): string {
  const colorEntries = Object.entries(cfg.colors).filter(([, v]) => v)
  const colorsJs = colorEntries.length
    ? ',\n      colors: {\n' +
      colorEntries.map(([k, v]) => `        ${k}: ${jsLiteral(v)}`).join(',\n') +
      '\n      }'
    : ''
  const languageJs = cfg.language !== 'auto'
    ? `,\n      language: ${jsLiteral(cfg.language)}`
    : ''
  const onOpenJs = cfg.initialMessage
    ? `,\n      onOpen: function() {\n        setTimeout(() => {\n          if (chat.getMessageCount && chat.getMessageCount() === 0) {\n            var detected = document.documentElement.lang || navigator.language || 'en';\n            var isDe = detected.toLowerCase().indexOf('de') === 0;\n            var msg = isDe ? ${jsLiteral(cfg.initialMessageContentDe)} : ${jsLiteral(cfg.initialMessageContentEn)};\n            chat.sendUserMessage(msg, { silent: ${cfg.initialMessageSilent} });\n          }\n        }, 500);\n      }`
    : ''
  const privacyTextDeJs = cfg.privacyTextDe
    ? `,\n      privacyTextDe: ${jsLiteral(cfg.privacyTextDe)}`
    : ''
  const privacyTextEnJs = cfg.privacyTextEn
    ? `,\n      privacyTextEn: ${jsLiteral(cfg.privacyTextEn)}`
    : ''
  const apiKeyJs = cfg.apiKey ? `,\n      apiKey: ${jsLiteral(cfg.apiKey)}` : ''
  const bounceTextDeJs = cfg.bounceTextDe
    ? `,\n      bounceTextDe: ${jsLiteral(cfg.bounceTextDe)}`
    : ''
  const bounceTextEnJs = cfg.bounceTextEn
    ? `,\n      bounceTextEn: ${jsLiteral(cfg.bounceTextEn)}`
    : ''
  const inputPlaceholderDeJs = cfg.inputPlaceholderDe
    ? `,\n      inputPlaceholderDe: ${jsLiteral(cfg.inputPlaceholderDe)}`
    : ''
  const inputPlaceholderEnJs = cfg.inputPlaceholderEn
    ? `,\n      inputPlaceholderEn: ${jsLiteral(cfg.inputPlaceholderEn)}`
    : ''
  const logoUrlButtonJs = cfg.logoUrlButton
    ? `,\n      logoUrlButton: ${jsLiteral(cfg.logoUrlButton)}`
    : ''
  const logoUrlHeaderJs = cfg.logoUrlHeader
    ? `,\n      logoUrlHeader: ${jsLiteral(cfg.logoUrlHeader)}`
    : ''
  const widgetPrefix = getPublicBase().replace(/\/dashboard$/, '')
  const widgetUrl = window.location.origin + widgetPrefix + '/widget/widget.umd.js'
  return `<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js"></script>
<script src="${widgetUrl}"></script>
<script>
  document.addEventListener('DOMContentLoaded', function() {
    const chat = new window.AgentixChatWidget({
      apiUrl: ${jsLiteral(cfg.apiUrl)}${apiKeyJs},
      model: ${jsLiteral(cfg.model)},
      position: ${jsLiteral(cfg.position)},
      theme: ${jsLiteral(cfg.theme)}${languageJs},
      width: ${cfg.width},
      height: ${cfg.height},
      showOnLoad: ${cfg.showOnLoad},
      title: ${jsLiteral(cfg.title)},
      subtitle: ${jsLiteral(cfg.subtitle)},
      showLogoInButton: ${cfg.showLogoInButton},
      showLogoInHeader: ${cfg.showLogoInHeader},
      logoBorder: ${jsLiteral(cfg.logoBorder)},
      logoGlow: ${jsLiteral(cfg.logoGlow)},
      logoBackground: ${jsLiteral(cfg.logoBackground)}${logoUrlButtonJs}${logoUrlHeaderJs}${privacyTextDeJs}${privacyTextEnJs},
      privacyEnabled: ${cfg.privacyEnabled},
      bounceInterval: ${cfg.bounceInterval}${bounceTextDeJs}${bounceTextEnJs},
      initialMessage: ${cfg.initialMessage},
      initialMessageSilent: ${cfg.initialMessageSilent}${colorsJs}${onOpenJs}${inputPlaceholderDeJs}${inputPlaceholderEnJs}
    });
  });
</script>`
}

function generateMinimalEmbedCode(cfg: WidgetConfig, baseDefaults?: Record<string, unknown> | null): string {
  const entries: string[] = []
  const langDefaults = baseDefaults
    ? ({ ...getDefaultConfig(cfg.language), ...baseDefaults } as WidgetConfig)
    : getDefaultConfig(cfg.language)

  // apiUrl immer aufnehmen (wichtig für externe Einbettungen)
  entries.push(`apiUrl: ${jsLiteral(cfg.apiUrl)}`)

  // Vergleiche einfache Werte mit sprachabhängigem Default
  if (cfg.model !== langDefaults.model) entries.push(`model: ${jsLiteral(cfg.model)}`)
  if (cfg.position !== langDefaults.position) entries.push(`position: ${jsLiteral(cfg.position)}`)
  if (cfg.theme !== langDefaults.theme) entries.push(`theme: ${jsLiteral(cfg.theme)}`)
  if (cfg.language !== langDefaults.language) entries.push(`language: ${jsLiteral(cfg.language)}`)
  if (cfg.width !== langDefaults.width) entries.push(`width: ${cfg.width}`)
  if (cfg.height !== langDefaults.height) entries.push(`height: ${cfg.height}`)
  if (cfg.showOnLoad !== langDefaults.showOnLoad) entries.push(`showOnLoad: ${cfg.showOnLoad}`)
  if (cfg.title !== langDefaults.title) entries.push(`title: ${jsLiteral(cfg.title)}`)
  if (cfg.subtitle !== langDefaults.subtitle) entries.push(`subtitle: ${jsLiteral(cfg.subtitle)}`)
  if (cfg.logoUrlButton !== langDefaults.logoUrlButton) entries.push(`logoUrlButton: ${jsLiteral(cfg.logoUrlButton)}`)
  if (cfg.logoUrlHeader !== langDefaults.logoUrlHeader) entries.push(`logoUrlHeader: ${jsLiteral(cfg.logoUrlHeader)}`)
  if (cfg.showLogoInButton !== langDefaults.showLogoInButton) entries.push(`showLogoInButton: ${cfg.showLogoInButton}`)
  if (cfg.showLogoInHeader !== langDefaults.showLogoInHeader) entries.push(`showLogoInHeader: ${cfg.showLogoInHeader}`)

  // Verschachtelte Objekte
  if (!isDeepEqual(cfg.logoBorder, langDefaults.logoBorder)) {
    entries.push(`logoBorder: ${jsLiteral(cfg.logoBorder)}`)
  }
  if (!isDeepEqual(cfg.logoGlow, langDefaults.logoGlow)) {
    entries.push(`logoGlow: ${jsLiteral(cfg.logoGlow)}`)
  }
  if (!isDeepEqual(cfg.logoBackground, langDefaults.logoBackground)) {
    entries.push(`logoBackground: ${jsLiteral(cfg.logoBackground)}`)
  }

  // Sprachspezifische Text-Overrides (keine generischen privacyText/bounceText/... mehr)
  if (cfg.privacyTextDe !== langDefaults.privacyTextDe) entries.push(`privacyTextDe: ${jsLiteral(cfg.privacyTextDe)}`)
  if (cfg.privacyTextEn !== langDefaults.privacyTextEn) entries.push(`privacyTextEn: ${jsLiteral(cfg.privacyTextEn)}`)
  if (cfg.privacyEnabled !== langDefaults.privacyEnabled) entries.push(`privacyEnabled: ${cfg.privacyEnabled}`)

  if (cfg.initialMessage !== langDefaults.initialMessage) entries.push(`initialMessage: ${cfg.initialMessage}`)
  if (cfg.initialMessageContentDe !== langDefaults.initialMessageContentDe) entries.push(`initialMessageContentDe: ${jsLiteral(cfg.initialMessageContentDe)}`)
  if (cfg.initialMessageContentEn !== langDefaults.initialMessageContentEn) entries.push(`initialMessageContentEn: ${jsLiteral(cfg.initialMessageContentEn)}`)
  if (cfg.initialMessageSilent !== langDefaults.initialMessageSilent) entries.push(`initialMessageSilent: ${cfg.initialMessageSilent}`)

  if (cfg.bounceInterval !== langDefaults.bounceInterval) entries.push(`bounceInterval: ${cfg.bounceInterval}`)
  if (cfg.bounceTextDe !== langDefaults.bounceTextDe) entries.push(`bounceTextDe: ${jsLiteral(cfg.bounceTextDe)}`)
  if (cfg.bounceTextEn !== langDefaults.bounceTextEn) entries.push(`bounceTextEn: ${jsLiteral(cfg.bounceTextEn)}`)
  if (cfg.inputPlaceholderDe !== langDefaults.inputPlaceholderDe) entries.push(`inputPlaceholderDe: ${jsLiteral(cfg.inputPlaceholderDe)}`)
  if (cfg.inputPlaceholderEn !== langDefaults.inputPlaceholderEn) entries.push(`inputPlaceholderEn: ${jsLiteral(cfg.inputPlaceholderEn)}`)
  if (cfg.apiKey) entries.push(`apiKey: ${jsLiteral(cfg.apiKey)}`)

  // Colors
  const colorEntries = Object.entries(cfg.colors).filter(
    ([k, v]) => v && v !== DEFAULT_COLORS[k as keyof ChatWidgetColors]
  )
  if (colorEntries.length > 0) {
    entries.push(
      `colors: {\n        ${colorEntries.map(([k, v]) => `${k}: ${jsLiteral(v)}`).join(',\n        ')}\n      }`
    )
  }

  // onOpen mit Laufzeit-Spracherkennung
  const initialMessageChanged =
    cfg.initialMessage !== langDefaults.initialMessage ||
    cfg.initialMessageContentDe !== langDefaults.initialMessageContentDe ||
    cfg.initialMessageContentEn !== langDefaults.initialMessageContentEn ||
    cfg.initialMessageSilent !== langDefaults.initialMessageSilent

  if (cfg.initialMessage && initialMessageChanged) {
    entries.push(
      `onOpen: function() {\n        setTimeout(() => {\n          if (chat.getMessageCount && chat.getMessageCount() === 0) {\n            var detected = document.documentElement.lang || navigator.language || 'en';\n            var isDe = detected.toLowerCase().indexOf('de') === 0;\n            var msg = isDe ? ${jsLiteral(cfg.initialMessageContentDe)} : ${jsLiteral(cfg.initialMessageContentEn)};\n            chat.sendUserMessage(msg, { silent: ${cfg.initialMessageSilent} });\n          }\n        }, 500);\n      }`
    )
  }

  const widgetPrefix2 = getPublicBase().replace(/\/dashboard$/, '')
  const widgetUrl2 = window.location.origin + widgetPrefix2 + '/widget/widget.umd.js'
  return `<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js"></script>
<script src="${widgetUrl2}"></script>
<script>
  document.addEventListener('DOMContentLoaded', function() {
    const chat = new window.AgentixChatWidget({
      ${entries.join(',\n      ')}
    });
  });
</script>`
}

export default function WidgetPage() {
  const { language } = useAppPreferences()
  const { branding } = useBranding()
  const $t = useT(language)

  const [cfg, setCfg] = useState<WidgetConfig>({ ...getDefaultConfig(language), language: 'auto' })
  const [assistants, setAssistants] = useState<string[]>([])
  const [loadingModels, setLoadingModels] = useState(true)
  const [copied, setCopied] = useState(false)
  const [resetDone, setResetDone] = useState(false)
  const [embedMode, setEmbedMode] = useState<'full' | 'minimal'>('minimal')
  const [loadingDefaults, setLoadingDefaults] = useState(false)
  const [defaultsLoaded, setDefaultsLoaded] = useState(false)
  const [serverDefaults, setServerDefaults] = useState<Record<string, unknown> | null>(null)
  const [deploySuccess, setDeploySuccess] = useState(false)
  const [deployError, setDeployError] = useState('')
  const [deployConfirmOpen, setDeployConfirmOpen] = useState(false)
  const widgetRef = useRef<any>(null)
  const cfgRef = useRef(cfg)
  cfgRef.current = cfg
  const scriptLoadedRef = useRef(false)
  const { getInstructions } = useData()
  const { hasPermission } = useAuth()
  const canDeploy = hasPermission('widget_defaults:write')

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
  }, [])

  // Load saved widget defaults from backend
  useEffect(() => {
    setLoadingDefaults(true)
    widgetDefaults
      .get()
      .then((defaults) => {
        setServerDefaults(defaults)
        if (defaults && Object.keys(defaults).length > 0) {
          setCfg((prev) => ({
            ...prev,
            ...(defaults as Partial<WidgetConfig>),
            colors: {
              ...prev.colors,
              ...(defaults.colors as ChatWidgetColors),
            },
            logoBorder: {
              ...prev.logoBorder,
              ...(defaults.logoBorder as WidgetConfig['logoBorder']),
            },
            logoGlow: {
              ...prev.logoGlow,
              ...(defaults.logoGlow as WidgetConfig['logoGlow']),
            },
            logoBackground: {
              ...prev.logoBackground,
              ...(defaults.logoBackground as WidgetConfig['logoBackground']),
            },
          }))
        }
      })
      .catch(() => {
        // ignore
      })
      .finally(() => {
        setLoadingDefaults(false)
        setDefaultsLoaded(true)
      })
  }, [])

  // Saved widget defaults can refer to an assistant that has since been
  // renamed or deleted. Once both sources are loaded, keep the preview on a
  // currently configured assistant instead of sending requests for a stale
  // model (which the chat API correctly rejects).
  useEffect(() => {
    if (!defaultsLoaded || loadingModels || assistants.length === 0) return
    if (assistants.includes(cfg.model)) return

    const fallbackModel = assistants[0]
    cfgRef.current = { ...cfgRef.current, model: fallbackModel }
    setCfg((current) =>
      current.model === fallbackModel
        ? current
        : { ...current, model: fallbackModel },
    )
    widgetRef.current?.updateConfig?.({ model: fallbackModel })
  }, [assistants, cfg.model, defaultsLoaded, loadingModels])

  // Load widget script dynamically (wait for defaults first)
  useEffect(() => {
    if (!defaultsLoaded) return

    if (!scriptLoadedRef.current && window.AgentixChatWidget) {
      scriptLoadedRef.current = true
    }

    if (!scriptLoadedRef.current) {
      const ortScript = document.createElement('script')
      ortScript.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js'
      ortScript.async = true
      document.body.appendChild(ortScript)
      const script = document.createElement('script')
      const widgetPrefix3 = getPublicBase().replace(/\/dashboard$/, '')
      script.src = widgetPrefix3 + '/widget/widget.umd.js'
      script.async = true
      script.onload = () => {
        scriptLoadedRef.current = true
        initWidget(cfgRef.current)
      }
      document.body.appendChild(script)
    } else {
      initWidget(cfgRef.current)
    }

    return () => {
      if (widgetRef.current) {
        widgetRef.current.destroy()
        widgetRef.current = null
      }
    }
  }, [defaultsLoaded])

  function initWidget(currentCfg = cfg) {
    if (!window.AgentixChatWidget) return
    if (widgetRef.current) {
      widgetRef.current.destroy()
    }
    const chat = new window.AgentixChatWidget({
      ...currentCfg,
      apiUrl: previewApiUrl(),
      apiKey: '',
      // Empty strings count as "explicitly set" in mergeServerDefaults and
      // would block widget-defaults/branding inheritance — treat as unset.
      logoUrlButton: currentCfg.logoUrlButton || undefined,
      logoUrlHeader: currentCfg.logoUrlHeader || undefined,
      language: currentCfg.language === 'auto' ? undefined : currentCfg.language,
      onOpen: function () {
        setTimeout(() => {
          if (chat.getMessageCount && chat.getMessageCount() === 0) {
            var detected = document.documentElement.lang || navigator.language || 'en';
            var isDe = detected.toLowerCase().indexOf('de') === 0;
            var msg = isDe ? currentCfg.initialMessageContentDe : currentCfg.initialMessageContentEn;
            chat.sendUserMessage(msg, { silent: true })
          }
        }, 500)
      },
    })
    widgetRef.current = chat
  }

  function applyConfig() {
    initWidget(cfg)
    if (widgetRef.current && widgetRef.current.updateConfig) {
      // Omit empty logo keys entirely — updateConfig's spread merge would
      // otherwise overwrite an inherited branding/default logo with the
      // builtin fallback.
      widgetRef.current.updateConfig({
        model: cfg.model,
        apiUrl: previewApiUrl(),
        language: cfg.language === 'auto' ? undefined : cfg.language,
        title: cfg.title,
        subtitle: cfg.subtitle,
        ...(cfg.logoUrlButton ? { logoUrlButton: cfg.logoUrlButton } : {}),
        ...(cfg.logoUrlHeader ? { logoUrlHeader: cfg.logoUrlHeader } : {}),
        showLogoInButton: cfg.showLogoInButton,
        showLogoInHeader: cfg.showLogoInHeader,
        logoBorder: cfg.logoBorder,
        logoGlow: cfg.logoGlow,
        logoBackground: cfg.logoBackground,
        privacyText: cfg.privacyText,
        privacyTextDe: cfg.privacyTextDe,
        privacyTextEn: cfg.privacyTextEn,
        privacyEnabled: cfg.privacyEnabled,
        initialMessage: cfg.initialMessage,
        initialMessageContent: cfg.initialMessageContent,
        initialMessageContentDe: cfg.initialMessageContentDe,
        initialMessageContentEn: cfg.initialMessageContentEn,
        initialMessageSilent: cfg.initialMessageSilent,
        bounceInterval: cfg.bounceInterval,
        bounceText: cfg.bounceText,
        bounceTextDe: cfg.bounceTextDe,
        bounceTextEn: cfg.bounceTextEn,
        inputPlaceholder: cfg.inputPlaceholder,
        inputPlaceholderDe: cfg.inputPlaceholderDe,
        inputPlaceholderEn: cfg.inputPlaceholderEn,
        colors: cfg.colors,
      })
    }
  }

  // Update config defaults when dashboard language changes
  useEffect(() => {
    setCfg((prev) => {
      const defaults = getDefaultConfig(language)
      const prevLang = prev.language === 'auto' ? language : prev.language
      const prevDefaults = getDefaultConfig(prevLang)
      return {
        ...prev,
        privacyText: prev.privacyText === prevDefaults.privacyText ? defaults.privacyText : prev.privacyText,
        privacyTextDe: prev.privacyTextDe === prevDefaults.privacyTextDe ? defaults.privacyTextDe : prev.privacyTextDe,
        privacyTextEn: prev.privacyTextEn === prevDefaults.privacyTextEn ? defaults.privacyTextEn : prev.privacyTextEn,
        initialMessageContent: prev.initialMessageContent === prevDefaults.initialMessageContent ? defaults.initialMessageContent : prev.initialMessageContent,
        initialMessageContentDe: prev.initialMessageContentDe === prevDefaults.initialMessageContentDe ? defaults.initialMessageContentDe : prev.initialMessageContentDe,
        initialMessageContentEn: prev.initialMessageContentEn === prevDefaults.initialMessageContentEn ? defaults.initialMessageContentEn : prev.initialMessageContentEn,
        bounceText: prev.bounceText === prevDefaults.bounceText ? defaults.bounceText : prev.bounceText,
        bounceTextDe: prev.bounceTextDe === prevDefaults.bounceTextDe ? defaults.bounceTextDe : prev.bounceTextDe,
        bounceTextEn: prev.bounceTextEn === prevDefaults.bounceTextEn ? defaults.bounceTextEn : prev.bounceTextEn,
        inputPlaceholder: prev.inputPlaceholder === prevDefaults.inputPlaceholder ? defaults.inputPlaceholder : prev.inputPlaceholder,
        inputPlaceholderDe: prev.inputPlaceholderDe === prevDefaults.inputPlaceholderDe ? defaults.inputPlaceholderDe : prev.inputPlaceholderDe,
        inputPlaceholderEn: prev.inputPlaceholderEn === prevDefaults.inputPlaceholderEn ? defaults.inputPlaceholderEn : prev.inputPlaceholderEn,
      }
    })
  }, [language])

  // Re-init widget preview when language changes
  useEffect(() => {
    initWidget(cfg)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [language])

  function resetWidgetData() {
    localStorage.removeItem('agentix-chat-messages')
    localStorage.removeItem('agentix-chat-privacy-accepted')
    localStorage.removeItem('agentix-chat-open')
    sessionStorage.removeItem('agentix-conversation-id')
    const newId = `widget-${(crypto as any).randomUUID ? (crypto as any).randomUUID() : String(Date.now())}`
    sessionStorage.setItem('agentix-conversation-id', newId)
    initWidget(cfg)
    setResetDone(true)
    setTimeout(() => setResetDone(false), 2500)
  }

  function handleModelChange(name: string) {
    const newCfg = { ...cfg, model: name }
    setCfg(newCfg)
    if (widgetRef.current && widgetRef.current.updateConfig) {
      widgetRef.current.updateConfig({ model: name })
    }
  }

  function copyCode() {
    navigator.clipboard.writeText(generateEmbedCode(cfg))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  async function handleDeployDefaults() {
    setDeployConfirmOpen(false)
    setDeploySuccess(false)
    setDeployError('')
    try {
      // Never persist the API key – deployed defaults are served publicly.
      const { apiKey: _apiKey, ...deployable } = cfg
      const payload = deployable as unknown as Record<string, unknown>
      await widgetDefaults.save(payload)
      setServerDefaults(payload)
      setDeploySuccess(true)
      // Reload preview with latest deployed config
      initWidget(cfg)
      setTimeout(() => setDeploySuccess(false), 3000)
    } catch (err: any) {
      setDeployError(err.message || $t('widget.deployDefaultsError'))
      setTimeout(() => setDeployError(''), 5000)
    }
  }

  const embedCode = embedMode === 'full' ? generateEmbedCode(cfg) : generateMinimalEmbedCode(cfg, serverDefaults)

  return (
    <div>
      <div className="widget-container">
        {/* Configurator */}
        <section className="widget-section">
          <h2>{$t('widget.configurator')}</h2>
          <div className="widget-configurator">
            <div className="widget-config-row">
              <label>Modell / Assistent</label>
              <select
                value={cfg.model}
                onChange={(e) => handleModelChange(e.target.value)}
                disabled={loadingModels}
              >
                {assistants.length === 0 && !loadingModels && (
                  <option value={cfg.model}>{cfg.model}</option>
                )}
                {assistants.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>

            <div className="widget-config-row">
              <label>{$t('widget.apiUrl')}</label>
              <input
                type="text"
                value={cfg.apiUrl}
                onChange={(e) => setCfg({ ...cfg, apiUrl: e.target.value })}
              />
              <span className="widget-config-hint">{$t('widget.apiUrlEmbedHint')}</span>
            </div>

            <div className="widget-config-row">
              <label>{$t('widget.apiKey')}</label>
              <input
                type="text"
                value={cfg.apiKey}
                onChange={(e) => setCfg({ ...cfg, apiKey: e.target.value })}
                placeholder="sk-..."
              />
            </div>

            <div className="widget-config-row">
              <label>{$t('widget.widgetTitle')}</label>
              <input
                type="text"
                value={cfg.title}
                onChange={(e) => setCfg({ ...cfg, title: e.target.value })}
              />
            </div>

            <div className="widget-config-row">
              <LogoField
                label={$t('widget.logoUrlButton')}
                value={cfg.logoUrlButton}
                onChange={(v) => setCfg({ ...cfg, logoUrlButton: v })}
                hint={$t('publicChat.logoInheritHint')}
              />
            </div>

            <div className="widget-config-row">
              <LogoField
                label={$t('widget.logoUrlHeader')}
                value={cfg.logoUrlHeader}
                onChange={(v) => setCfg({ ...cfg, logoUrlHeader: v })}
                hint={$t('publicChat.logoInheritHint')}
                previewHeight={36}
              />
            </div>

            <div className="widget-config-row">
              <label>{$t('widget.subtitle')}</label>
              <input
                type="text"
                value={cfg.subtitle}
                onChange={(e) => setCfg({ ...cfg, subtitle: e.target.value })}
              />
            </div>

            <div className="widget-config-row">
              <label>{$t('widget.position')}</label>
              <select
                value={cfg.position}
                onChange={(e) =>
                  setCfg({ ...cfg, position: e.target.value as WidgetConfig['position'] })
                }
              >
                <option value="bottom-right">{$t('widget.bottomRight')}</option>
                <option value="bottom-left">{$t('widget.bottomLeft')}</option>
                <option value="top-right">{$t('widget.topRight')}</option>
                <option value="top-left">{$t('widget.topLeft')}</option>
              </select>
            </div>

            <div className="widget-config-row">
              <label>{$t('widget.theme')}</label>
              <select
                value={cfg.theme}
                onChange={(e) => setCfg({ ...cfg, theme: e.target.value as WidgetConfig['theme'] })}
              >
                <option value="light">{$t('widget.light')}</option>
                <option value="dark">{$t('widget.dark')}</option>
              </select>
            </div>

            <div className="widget-config-row">
              <label>{$t('widget.language')}</label>
              <select
                value={cfg.language}
                onChange={(e) => setCfg({ ...cfg, language: e.target.value as WidgetConfig['language'] })}
              >
                <option value="auto">{$t('widget.languageAuto')}</option>
                <option value="de">Deutsch</option>
                <option value="en">English</option>
              </select>
            </div>

            <div className="widget-config-row">
              <label>{$t('widget.width')}</label>
              <input
                type="number"
                value={cfg.width}
                onChange={(e) => setCfg({ ...cfg, width: Number(e.target.value) })}
              />
            </div>

            <div className="widget-config-row">
              <label>{$t('widget.height')}</label>
              <input
                type="number"
                value={cfg.height}
                onChange={(e) => setCfg({ ...cfg, height: Number(e.target.value) })}
              />
            </div>

            <div className="widget-config-row widget-config-checkbox">
              <input
                type="checkbox"
                id="showOnLoad"
                checked={cfg.showOnLoad}
                onChange={(e) => setCfg({ ...cfg, showOnLoad: e.target.checked })}
              />
              <label htmlFor="showOnLoad">{$t('widget.showOnLoad')}</label>
            </div>

            <div className="widget-config-row widget-config-checkbox">
              <input
                type="checkbox"
                id="showLogoInButton"
                checked={cfg.showLogoInButton}
                onChange={(e) => setCfg({ ...cfg, showLogoInButton: e.target.checked })}
              />
              <label htmlFor="showLogoInButton">{$t('widget.showLogoInButton')}</label>
            </div>

            <div className="widget-config-row widget-config-checkbox">
              <input
                type="checkbox"
                id="showLogoInHeader"
                checked={cfg.showLogoInHeader}
                onChange={(e) => setCfg({ ...cfg, showLogoInHeader: e.target.checked })}
              />
              <label htmlFor="showLogoInHeader">{$t('widget.showLogoInHeader')}</label>
            </div>

            <div className="widget-config-divider">{$t('widget.logoBorder')}</div>

            <div className="widget-config-row widget-config-checkbox">
              <input
                type="checkbox"
                id="logoBorderEnabled"
                checked={cfg.logoBorder.enabled}
                onChange={(e) =>
                  setCfg({ ...cfg, logoBorder: { ...cfg.logoBorder, enabled: e.target.checked } })
                }
              />
              <label htmlFor="logoBorderEnabled">{$t('widget.logoBorderEnabled')}</label>
            </div>

            {cfg.logoBorder.enabled && (
              <>
                <div className="widget-config-row widget-config-color">
                  <label>{$t('widget.borderColor')}</label>
                  <input
                    type="color"
                    value={cfg.logoBorder.color}
                    onChange={(e) =>
                      setCfg({ ...cfg, logoBorder: { ...cfg.logoBorder, color: e.target.value } })
                    }
                  />
                </div>
                <div className="widget-config-row">
                  <label>{$t('widget.transparency')}</label>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.1}
                    value={cfg.logoBorder.opacity}
                    onChange={(e) =>
                      setCfg({
                        ...cfg,
                        logoBorder: { ...cfg.logoBorder, opacity: Number(e.target.value) },
                      })
                    }
                  />
                </div>
              </>
            )}

            <div className="widget-config-divider">{$t('widget.logoGlow')}</div>

            <div className="widget-config-row widget-config-checkbox">
              <input
                type="checkbox"
                id="logoGlowEnabled"
                checked={cfg.logoGlow.enabled}
                onChange={(e) =>
                  setCfg({ ...cfg, logoGlow: { ...cfg.logoGlow, enabled: e.target.checked } })
                }
              />
              <label htmlFor="logoGlowEnabled">{$t('widget.logoGlowEnabled')}</label>
            </div>

            {cfg.logoGlow.enabled && (
              <>
                <div className="widget-config-row widget-config-color">
                  <label>{$t('widget.glowColor')}</label>
                  <input
                    type="color"
                    value={cfg.logoGlow.color}
                    onChange={(e) =>
                      setCfg({ ...cfg, logoGlow: { ...cfg.logoGlow, color: e.target.value } })
                    }
                  />
                </div>
                <div className="widget-config-row">
                  <label>{$t('widget.transparency')}</label>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.1}
                    value={cfg.logoGlow.opacity}
                    onChange={(e) =>
                      setCfg({
                        ...cfg,
                        logoGlow: { ...cfg.logoGlow, opacity: Number(e.target.value) },
                      })
                    }
                  />
                </div>
              </>
            )}

            <div className="widget-config-divider">{$t('widget.logoBackground')}</div>

            <div className="widget-config-row widget-config-checkbox">
              <input
                type="checkbox"
                id="logoBackgroundEnabled"
                checked={cfg.logoBackground.enabled}
                onChange={(e) =>
                  setCfg({ ...cfg, logoBackground: { ...cfg.logoBackground, enabled: e.target.checked } })
                }
              />
              <label htmlFor="logoBackgroundEnabled">{$t('widget.logoBackgroundEnabled')}</label>
            </div>

            {cfg.logoBackground.enabled && (
              <>
                <div className="widget-config-row widget-config-color">
                  <label>{$t('widget.backgroundColor')}</label>
                  <input
                    type="color"
                    value={cfg.logoBackground.color === 'transparent' ? '#000000' : cfg.logoBackground.color}
                    onChange={(e) =>
                      setCfg({ ...cfg, logoBackground: { ...cfg.logoBackground, color: e.target.value } })
                    }
                  />
                </div>
                <div className="widget-config-row">
                  <label>{$t('widget.transparency')}</label>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.1}
                    value={cfg.logoBackground.opacity}
                    onChange={(e) =>
                      setCfg({
                        ...cfg,
                        logoBackground: { ...cfg.logoBackground, opacity: Number(e.target.value) },
                      })
                    }
                  />
                </div>
              </>
            )}

            <div className="widget-config-divider">{$t('widget.privacyText')}</div>

            <div className="widget-config-row widget-config-checkbox">
              <input
                type="checkbox"
                id="privacyEnabled"
                checked={cfg.privacyEnabled}
                onChange={(e) => setCfg({ ...cfg, privacyEnabled: e.target.checked })}
              />
              <label htmlFor="privacyEnabled">{$t('widget.privacyEnabled')}</label>
            </div>

            {cfg.privacyEnabled && (
              <>
                <div className="widget-config-row widget-config-fullwidth">
                  <label>{$t('widget.privacyTextDe')}</label>
                  <textarea
                    rows={3}
                    value={cfg.privacyTextDe}
                    onChange={(e) =>
                      setCfg((prev) => ({
                        ...prev,
                        privacyTextDe: e.target.value,
                        privacyText: prev.language === 'de' ? e.target.value : prev.privacyText,
                      }))
                    }
                  />
                </div>
                <div className="widget-config-row widget-config-fullwidth">
                  <label>{$t('widget.privacyTextEn')}</label>
                  <textarea
                    rows={3}
                    value={cfg.privacyTextEn}
                    onChange={(e) =>
                      setCfg((prev) => ({
                        ...prev,
                        privacyTextEn: e.target.value,
                        privacyText: prev.language === 'en' ? e.target.value : prev.privacyText,
                      }))
                    }
                  />
                </div>
              </>
            )}

            <div className="widget-config-divider">{$t('widget.bounceText')}</div>

            <div className="widget-config-row">
              <label>{$t('widget.bounceInterval')}</label>
              <input
                type="number"
                min={0}
                step={1000}
                value={cfg.bounceInterval}
                onChange={(e) => setCfg({ ...cfg, bounceInterval: Number(e.target.value) })}
              />
            </div>
            <div className="widget-config-row">
              <label>{$t('widget.bounceTextDe')}</label>
              <input
                type="text"
                value={cfg.bounceTextDe}
                onChange={(e) =>
                  setCfg((prev) => ({
                    ...prev,
                    bounceTextDe: e.target.value,
                    bounceText: prev.language === 'de' ? e.target.value : prev.bounceText,
                  }))
                }
              />
            </div>
            <div className="widget-config-row">
              <label>{$t('widget.bounceTextEn')}</label>
              <input
                type="text"
                value={cfg.bounceTextEn}
                onChange={(e) =>
                  setCfg((prev) => ({
                    ...prev,
                    bounceTextEn: e.target.value,
                    bounceText: prev.language === 'en' ? e.target.value : prev.bounceText,
                  }))
                }
              />
            </div>

            <div className="widget-config-divider">{$t('widget.inputPlaceholder')}</div>

            <div className="widget-config-row">
              <label>{$t('widget.inputPlaceholderDe')}</label>
              <input
                type="text"
                value={cfg.inputPlaceholderDe}
                onChange={(e) =>
                  setCfg((prev) => ({
                    ...prev,
                    inputPlaceholderDe: e.target.value,
                    inputPlaceholder: prev.language === 'de' ? e.target.value : prev.inputPlaceholder,
                  }))
                }
              />
            </div>
            <div className="widget-config-row">
              <label>{$t('widget.inputPlaceholderEn')}</label>
              <input
                type="text"
                value={cfg.inputPlaceholderEn}
                onChange={(e) =>
                  setCfg((prev) => ({
                    ...prev,
                    inputPlaceholderEn: e.target.value,
                    inputPlaceholder: prev.language === 'en' ? e.target.value : prev.inputPlaceholder,
                  }))
                }
              />
            </div>

            <div className="widget-config-divider">{$t('widget.initialMessage')}</div>

            <div className="widget-config-row widget-config-checkbox">
              <input
                type="checkbox"
                id="initialMessage"
                checked={cfg.initialMessage}
                onChange={(e) => setCfg({ ...cfg, initialMessage: e.target.checked })}
              />
              <label htmlFor="initialMessage">{$t('widget.initialMessage')}</label>
            </div>

            {cfg.initialMessage && (
              <>
                <div className="widget-config-row">
                  <label>{$t('widget.initialMessageContentDe')}</label>
                  <input
                    type="text"
                    value={cfg.initialMessageContentDe}
                    onChange={(e) =>
                      setCfg((prev) => ({
                        ...prev,
                        initialMessageContentDe: e.target.value,
                        initialMessageContent: prev.language === 'de' ? e.target.value : prev.initialMessageContent,
                      }))
                    }
                  />
                </div>
                <div className="widget-config-row">
                  <label>{$t('widget.initialMessageContentEn')}</label>
                  <input
                    type="text"
                    value={cfg.initialMessageContentEn}
                    onChange={(e) =>
                      setCfg((prev) => ({
                        ...prev,
                        initialMessageContentEn: e.target.value,
                        initialMessageContent: prev.language === 'en' ? e.target.value : prev.initialMessageContent,
                      }))
                    }
                  />
                </div>
                <div className="widget-config-row widget-config-checkbox">
                  <input
                    type="checkbox"
                    id="initialMessageSilent"
                    checked={cfg.initialMessageSilent}
                    onChange={(e) => setCfg({ ...cfg, initialMessageSilent: e.target.checked })}
                  />
                  <label htmlFor="initialMessageSilent">{$t('widget.initialMessageSilent')}</label>
                </div>
              </>
            )}

            <div className="widget-config-divider">{$t('widget.colors')}</div>

            <div className="widget-config-row widget-config-color">
              <label>{$t('widget.primaryColor')}</label>
              <input
                type="color"
                value={cfg.colors.primary || DEFAULT_COLORS.primary}
                onChange={(e) =>
                  setCfg({ ...cfg, colors: { ...cfg.colors, primary: e.target.value } })
                }
              />
            </div>

            <div className="widget-config-row widget-config-color">
              <label>{$t('widget.headerBackground')}</label>
              <input
                type="color"
                value={cfg.colors.headerBackground || DEFAULT_COLORS.headerBackground}
                onChange={(e) =>
                  setCfg({ ...cfg, colors: { ...cfg.colors, headerBackground: e.target.value } })
                }
              />
            </div>

            <div className="widget-config-row widget-config-color">
              <label>{$t('widget.headerText')}</label>
              <input
                type="color"
                value={cfg.colors.headerText || DEFAULT_COLORS.headerText}
                onChange={(e) =>
                  setCfg({ ...cfg, colors: { ...cfg.colors, headerText: e.target.value } })
                }
              />
            </div>

            <div className="widget-config-row widget-config-color">
              <label>{$t('widget.buttonBackground')}</label>
              <input
                type="color"
                value={cfg.colors.buttonBackground || DEFAULT_COLORS.buttonBackground}
                onChange={(e) =>
                  setCfg({ ...cfg, colors: { ...cfg.colors, buttonBackground: e.target.value } })
                }
              />
            </div>

            <div className="widget-config-row widget-config-color">
              <label>{$t('widget.buttonText')}</label>
              <input
                type="color"
                value={cfg.colors.buttonText || DEFAULT_COLORS.buttonText}
                onChange={(e) =>
                  setCfg({ ...cfg, colors: { ...cfg.colors, buttonText: e.target.value } })
                }
              />
            </div>

            <div className="widget-config-row widget-config-color">
              <label>{$t('widget.chatBackground')}</label>
              <input
                type="color"
                value={cfg.colors.background || DEFAULT_COLORS.background}
                onChange={(e) =>
                  setCfg({ ...cfg, colors: { ...cfg.colors, background: e.target.value } })
                }
              />
            </div>

            <div className="widget-config-row widget-config-color">
              <label>{$t('widget.chatText')}</label>
              <input
                type="color"
                value={cfg.colors.text || DEFAULT_COLORS.text}
                onChange={(e) =>
                  setCfg({ ...cfg, colors: { ...cfg.colors, text: e.target.value } })
                }
              />
            </div>

            <div className="widget-config-actions">
              <button className="widget-apply-btn" onClick={applyConfig}>
                <RefreshCw size={16} />
                {$t('widget.apply')}
              </button>
              {canDeploy && (
                <button className="widget-deploy-btn" onClick={() => setDeployConfirmOpen(true)}>
                  {$t('widget.deployDefaults')}
                </button>
              )}
              <button className="widget-reset-btn" onClick={resetWidgetData}>
                <Trash2 size={16} />
                {$t('widget.resetHistory')}
              </button>
            </div>
            {resetDone && (
              <div className="widget-reset-status">
                {$t('widget.resetHistoryDone')}
              </div>
            )}
            {deploySuccess && (
              <div className="widget-deploy-status widget-deploy-success">
                {$t('widget.deployDefaultsSuccess')}
              </div>
            )}
            {deployError && (
              <div className="widget-deploy-status widget-deploy-error">
                {deployError}
              </div>
            )}
            {loadingDefaults && (
              <div className="widget-loading-status">
                {$t('widget.loadingDefaults')}
              </div>
            )}
          </div>
        </section>

        {/* Embed Code */}
        <section className="widget-section">
          <h2>{$t('widget.embedCode')}</h2>
          <div className="widget-embed-toggle">
            <button
              className={embedMode === 'full' ? 'active' : ''}
              onClick={() => setEmbedMode('full')}
            >
              {$t('common.full')}
            </button>
            <button
              className={embedMode === 'minimal' ? 'active' : ''}
              onClick={() => setEmbedMode('minimal')}
            >
              {$t('common.minimal')}
            </button>
          </div>
          <div className="widget-code-block">
            <pre>
              <code>{embedCode}</code>
            </pre>
            <button className="widget-copy-btn" onClick={copyCode}>
              {copied ? <Check size={16} /> : <Copy size={16} />}
              {copied ? $t('common.copied') : $t('common.copy')}
            </button>
          </div>
        </section>
      </div>

      <footer className="widget-footer">
        <p>{branding.powered_by || 'Powered by Agentix'}</p>
      </footer>

      <ConfirmModal
        open={deployConfirmOpen}
        title={$t('widget.deployConfirmTitle')}
        message={$t('widget.deployConfirmMessage')}
        confirmText={$t('widget.deployConfirmButton')}
        onConfirm={handleDeployDefaults}
        onCancel={() => setDeployConfirmOpen(false)}
      />
    </div>
  )
}
