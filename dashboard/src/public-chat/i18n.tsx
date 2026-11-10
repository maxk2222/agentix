import { createContext, useContext, useState, useEffect, type ReactNode } from 'react'

export type Language = 'de' | 'en'

export const PUBLIC_CHAT_I18N: Record<Language, Record<string, string>> = {
  de: {
    emptyStateText: 'Starte eine Konversation, indem du unten eine Nachricht eingibst.',
    inputPlaceholder: 'Nachricht eingeben...',
    send: 'Senden',
    stop: 'Stop',
    speak: 'Vorlesen',
    copy: 'Kopieren',
    copied: 'Kopiert',
    close: 'Schließen',
    save: 'Speichern',
    cancel: 'Abbrechen',
    code: 'Code',
    web: 'Web',
    htmlPreview: 'HTML Vorschau',
    fullscreen: 'Vollbild',
    thinking: 'Denkt nach',
    reasoning: 'Reasoning',
    arguments: 'Argumente',
    duration: 'Dauer',
    output: 'Ausgabe',
    codeSandbox: 'Code Sandbox',
    pythonSandbox: 'python sandbox',
    pipInstall: 'pip install',
    configError: 'Public-Chat-Konfiguration konnte nicht geladen werden.',
    login: 'Anmelden',
    register: 'Registrieren',
    username: 'Benutzername',
    displayName: 'Anzeigename',
    emailOptional: 'E-Mail (optional)',
    password: 'Passwort',
    loading: 'Laden...',
    conversations: 'Konversationen',
    noConversations: 'Noch keine gespeicherten Konversationen.',
    messages: 'Nachrichten',
    rename: 'Umbenennen',
    load: 'Laden',
    delete: 'Löschen',
    deleteConfirm: 'Konversation löschen?',
    preview: 'Vorschau',
    loadIntoChat: 'In Chat laden',
    history: 'Verlauf',
    logout: 'Abmelden',
    loginRegister: 'Anmelden / Registrieren',
    reset: 'Zurücksetzen',
    emotionActive: 'Emotion Detection aktiv',
    emotionStart: 'Emotion Detection starten',
    phoneModeOn: 'Telefonmodus beenden',
    phoneModeOff: 'Telefonmodus',
    sttStart: 'Spracheingabe',
    sttStop: 'Aufnahme stoppen',
    phaseWaiting: 'Warten auf Antwort',
    phaseReading: 'Beim Lesen',
    phaseTyping: 'Beim Tippen',
    phaseSpeaking: 'Beim Sprechen',
    emotionHistory: 'Emotionshistorie',
    text: 'Text',
    webcam: 'Webcam',
    audioText: 'Audio Text',
    audioProsody: 'Audio Prosody',
    phases: 'Phasen',
    fused: 'Fused',
    queueTitle: 'Warteschlange',
    queueSendNow: 'Jetzt senden (laufenden Lauf stoppen)',
    queueInject: 'In laufenden Agent einfügen',
    queueInjectFailed: 'Einfügen fehlgeschlagen',
    queueEdit: 'Bearbeiten',
    queueRemove: 'Entfernen',
    queueResume: 'Fortsetzen',
    queueImageMessage: 'Bildnachricht',
    artifactDownload: 'Herunterladen',
    artifactPreview: 'Vorschau',
    artifactHidePreview: 'Vorschau schließen',
    artifactDownloadFailed: 'Download fehlgeschlagen',
    artifactPreviewFailed: 'Vorschau fehlgeschlagen',
    artifactExportAs: 'Exportieren als',
    artifactPageOf: 'Seite {current} von {total}',
    queueEnqueue: 'Einreihen',
    agentMemory: 'Agent-Gedächtnis',
    memoryLoading: 'Gedächtnis wird geladen…',
    memoryEmpty: 'Der Agent hat für diese Konversation noch nichts gespeichert.',
    uploadFile: 'Datei hochladen',
    uploading: 'Wird hochgeladen…',
    uploadFailed: 'Upload fehlgeschlagen',
    dropFiles: 'Dateien hier ablegen',
    dropImages: 'Bilder hier ablegen',
  },
  en: {
    emptyStateText: 'Start a conversation by typing a message below.',
    inputPlaceholder: 'Type your message...',
    send: 'Send',
    stop: 'Stop',
    speak: 'Speak',
    copy: 'Copy',
    copied: 'Copied',
    close: 'Close',
    save: 'Save',
    cancel: 'Cancel',
    code: 'Code',
    web: 'Web',
    htmlPreview: 'HTML Preview',
    fullscreen: 'Fullscreen',
    thinking: 'Thinking',
    reasoning: 'Reasoning',
    arguments: 'Arguments',
    duration: 'Duration',
    output: 'Output',
    codeSandbox: 'Code Sandbox',
    pythonSandbox: 'python sandbox',
    pipInstall: 'pip install',
    configError: 'Public chat configuration could not be loaded.',
    login: 'Login',
    register: 'Register',
    username: 'Username',
    displayName: 'Display Name',
    emailOptional: 'Email (optional)',
    password: 'Password',
    loading: 'Loading...',
    conversations: 'Conversations',
    noConversations: 'No saved conversations yet.',
    messages: 'messages',
    rename: 'Rename',
    load: 'Load',
    delete: 'Delete',
    deleteConfirm: 'Delete conversation?',
    preview: 'Preview',
    loadIntoChat: 'Load into Chat',
    history: 'History',
    logout: 'Logout',
    loginRegister: 'Login / Register',
    reset: 'Reset',
    emotionActive: 'Emotion Detection active',
    emotionStart: 'Start Emotion Detection',
    phoneModeOn: 'End phone mode',
    phoneModeOff: 'Phone mode',
    sttStart: 'Voice input',
    sttStop: 'Stop recording',
    phaseWaiting: 'Waiting for response',
    phaseReading: 'Reading',
    phaseTyping: 'Typing',
    phaseSpeaking: 'Speaking',
    emotionHistory: 'Emotion History',
    text: 'Text',
    webcam: 'Webcam',
    audioText: 'Audio Text',
    audioProsody: 'Audio Prosody',
    phases: 'Phases',
    fused: 'Fused',
    queueTitle: 'Queue',
    queueSendNow: 'Send now (stops the running answer)',
    queueInject: 'Inject into running agent',
    queueInjectFailed: 'Injection failed',
    queueEdit: 'Edit',
    queueRemove: 'Remove',
    queueResume: 'Resume',
    queueImageMessage: 'Image message',
    artifactDownload: 'Download',
    artifactPreview: 'Preview',
    artifactHidePreview: 'Hide preview',
    artifactDownloadFailed: 'Download failed',
    artifactPreviewFailed: 'Preview failed',
    artifactExportAs: 'Export as',
    artifactPageOf: 'Page {current} of {total}',
    queueEnqueue: 'Queue message',
    agentMemory: 'Agent Memory',
    memoryLoading: 'Loading memory…',
    memoryEmpty: 'The agent has not stored anything for this conversation yet.',
    uploadFile: 'Upload file',
    uploading: 'Uploading…',
    uploadFailed: 'Upload failed',
    dropFiles: 'Drop files here',
    dropImages: 'Drop images here',
  },
}

export function detectLanguage(): Language {
  const browserLang = navigator.language
  if (browserLang) {
    const normalized = browserLang.toLowerCase().split('-')[0]
    if (normalized === 'de') return 'de'
    if (normalized === 'en') return 'en'
  }
  const browserLangs = navigator.languages
  if (browserLangs && browserLangs.length > 0) {
    const normalized = browserLangs.map((l) => l.toLowerCase().split('-')[0])
    if (normalized.includes('de')) return 'de'
    if (normalized.includes('en')) return 'en'
  }
  const pageLang = document.documentElement.lang
  if (pageLang) {
    const normalized = pageLang.toLowerCase().split('-')[0]
    if (normalized === 'de') return 'de'
    if (normalized === 'en') return 'en'
  }
  return 'de'
}

const STORAGE_KEY = 'agentix-public-chat-lang'

export function getStoredLanguage(): Language {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw === 'en') return 'en'
  } catch {}
  return detectLanguage()
}

export function storeLanguage(lang: Language): void {
  try {
    localStorage.setItem(STORAGE_KEY, lang)
  } catch {}
}

/* Simple context so child components can access translations */
interface I18nCtx {
  language: Language
  t: Record<string, string>
  setLanguage: (l: Language) => void
}

const I18nContext = createContext<I18nCtx>({
  language: 'de',
  t: PUBLIC_CHAT_I18N.de,
  setLanguage: () => {},
})

export function I18nProvider({ children, forcedLanguage }: { children: ReactNode; forcedLanguage?: 'auto' | Language }) {
  const [language, setLanguageState] = useState<Language>(() => {
    if (forcedLanguage && forcedLanguage !== 'auto') return forcedLanguage
    return getStoredLanguage()
  })
  const t = PUBLIC_CHAT_I18N[language]

  useEffect(() => {
    if (forcedLanguage && forcedLanguage !== 'auto') {
      setLanguageState(forcedLanguage)
    }
  }, [forcedLanguage])

  useEffect(() => {
    document.documentElement.lang = language
  }, [language])

  const setLanguage = (l: Language) => {
    if (forcedLanguage && forcedLanguage !== 'auto') return
    setLanguageState(l)
    storeLanguage(l)
  }

  return (
    <I18nContext.Provider value={{ language, t, setLanguage }}>
      {children}
    </I18nContext.Provider>
  )
}

export function useI18n() {
  return useContext(I18nContext)
}
