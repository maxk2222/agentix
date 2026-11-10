import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

type Theme = 'light' | 'dark'
type Language = 'de' | 'en'

interface AppPreferences {
  theme: Theme
  language: Language
  setTheme: (t: Theme) => void
  setLanguage: (l: Language) => void
}

const STORAGE_KEY = 'agentix-dashboard-prefs'

const defaultPrefs: AppPreferences = {
  theme: 'light',
  language: 'de',
  setTheme: () => {},
  setLanguage: () => {},
}

const AppPreferencesContext = createContext<AppPreferences>(defaultPrefs)

function detectBrowserLanguage(): Language {
  const navLang = navigator.language
  if (navLang) {
    const normalized = navLang.toLowerCase().split('-')[0]
    if (normalized === 'en') return 'en'
    if (normalized === 'de') return 'de'
  }
  const navLangs = navigator.languages
  if (navLangs && navLangs.length > 0) {
    const normalized = navLangs.map((l) => l.toLowerCase().split('-')[0])
    if (normalized.includes('de')) return 'de'
    if (normalized.includes('en')) return 'en'
  }
  return 'de'
}

function loadPrefs(): { theme: Theme; language: Language } {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      return {
        theme: parsed.theme === 'dark' ? 'dark' : 'light',
        language: parsed.language === 'en' ? 'en' : 'de',
      }
    }
  } catch {}
  return { theme: 'light', language: detectBrowserLanguage() }
}

export function AppPreferencesProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(loadPrefs().theme)
  const [language, setLanguageState] = useState<Language>(loadPrefs().language)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    // MDXEditor (and other vendored libs) switch to their dark palettes via
    // a .dark/.dark-theme class — our data-theme attribute alone leaves them
    // in light mode with unreadable grays and dark-blue links.
    document.documentElement.classList.toggle('dark-theme', theme === 'dark')
  }, [theme])

  useEffect(() => {
    document.documentElement.lang = language
  }, [language])

  const setTheme = (t: Theme) => {
    setThemeState(t)
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ theme: t, language }))
    } catch {}
  }

  const setLanguage = (l: Language) => {
    setLanguageState(l)
    document.documentElement.lang = l
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ theme, language: l }))
    } catch {}
  }

  return (
    <AppPreferencesContext.Provider value={{ theme, language, setTheme, setLanguage }}>
      {children}
    </AppPreferencesContext.Provider>
  )
}

export function useAppPreferences() {
  return useContext(AppPreferencesContext)
}
