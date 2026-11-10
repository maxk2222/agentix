import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { Send, Square, Bot, User, Copy, Check, Volume2, RotateCcw, Mic, Phone, PhoneOff, X, Sparkles, Smile, History, LogOut, LogIn, AlertCircle, BookOpen, KeyRound, Sun, Moon, ListPlus, Paperclip, Brain, Trash2, Pencil, Folder, EllipsisVertical } from 'lucide-react'
import { publicAuth, publicConversations, publicKnowledge, type ChatUserInfo, type ConversationItem, type OKFConcept } from './api'
import type { AgentTodoSnapshot, ObserverEvent, SubagentEvent, SubagentMeta } from '../api/client'
import { SubagentModal, SubagentTagsRow, upsertSubagentEvent } from '../chat-shared/SubagentTags'
import { AgentTodoPanel } from '../chat-shared/AgentTodoPanel'
import { ObserverTag } from '../chat-shared/ObserverTag'
import {
  ChatLanguageContext,
  HtmlPreviewContext,
  INTERNAL_TOOL_NAMES,
  MarkdownContent,
  ReasoningPanel,
  TerminalBox,
  ToolCallBadge,
  ToolUsageBadges,
  TypingIndicator,
  addToolProgress,
  collectArtifactFiles,
  resolveToolProgress,
  settleToolProgress,
  type ToolCallLive,
  type ToolExecutionEvent,
  type ToolProgressItem,
} from '../chat-shared/ChatElements'
import { StreamBatcher } from '../chat-shared/streamBatcher'
import { metricsFromRetrievalInfo } from '../chat-shared/messageMetrics'
import { loadChatMessages, persistChatMessagesDebounced, flushPendingChatWrites, slimMessageForStorage } from '../chat-shared/messageStorage'
import { QueuedMessagesBar, type QueuedChatMessage } from '../chat-shared/QueuedMessagesBar'
import { ArtifactCard } from '../chat-shared/ArtifactCard'
import { WorkspaceBrowser } from '../chat-shared/WorkspaceBrowser'
import { UploadTray, AttachmentChips } from '../chat-shared/UploadTray'
import { uploadChatAttachment, deleteChatAttachment, isImageFile, isDocFile, ACCEPT_STRING, type PendingFile, type AttachmentMeta } from '../chat-shared/uploads'
import { blobToWav } from '../utils/audio'
import { TTSSegmentPlayer, extractCompleteSentences } from '../utils/tts'
import { createVideoSnippetBuffer } from '../emotion/videoSnippetBuffer'
import ConfirmModal from '../components/ConfirmModal'
import AlertModal from '../components/AlertModal'
import { createAudioSnippetBuffer } from '../emotion/audioSnippetBuffer'
import { fuseEmotions } from '../emotion/emotionAggregator'
import type { EmotionContextPayload, EmotionRuntimeStatus, SnippetConfig, EmotionResult, PhaseEmotion, EmotionPhase } from '../emotion/types'
import { applyHighpassFilter } from '../emotion/audioFilter'
import { useI18n } from './i18n'
import { EMBEDDED_FAVICON, EMBEDDED_LOGO } from './embeddedAssets'
import { OKFGraphView } from '../components/OKFGraphView'
import { ResponseMetricsBadge, LiveTokenChip, type ResponseMetrics, type LiveUsage } from '../components/ResponseMetrics'

interface Message {
  role: 'user' | 'assistant'
  content: string
  // Injected system/task notes (e.g. background task finished) render as a
  // centered notice, not a user bubble — role stays 'user' for API compat.
  systemNote?: boolean
  reasoning?: string
  toolCalls?: ToolCallLive[]
  subagents?: SubagentEvent[]
  todoList?: AgentTodoSnapshot
  observerEvents?: ObserverEvent[]
  terminalEvents?: ToolExecutionEvent[]
  emotion?: {
    fused_emotion?: EmotionResult | null
    text_emotion?: EmotionResult | null
    visual_pre_send?: EmotionResult | null
    audio_text_emotion?: EmotionResult | null
    audio_prosody?: EmotionResult | null
    phases?: PhaseEmotion[] | null
    consent?: {
      webcam_opt_in: boolean
      frames_sent_to_backend: boolean
      audio_opt_in: boolean
    }
  } | null
  retrievalInfo?: {
    status: string
    chunks_retrieved: number
    rag_context_tokens: number
    rag_context: Array<{ content: string; metadata?: Record<string, unknown> }>
    todo_list?: AgentTodoSnapshot
    attachments?: AttachmentMeta[]
  }
  attachments?: AttachmentMeta[]
  metrics?: ResponseMetrics
  // Transient while streaming — cumulative live_usage snapshot from the run.
  liveUsage?: LiveUsage
}

function getApiBase(): string {
  const path = window.location.pathname
  const idx = path.lastIndexOf('/chat')
  if (idx === -1) return ''
  return path.slice(0, idx)
}

function getApiUrl(): string {
  return window.location.origin + getApiBase() + '/v1'
}

function getPublicChatConfigUrl(): string {
  return window.location.origin + getApiBase() + '/public-chat/config'
}

const CONFIG_CACHE_KEY = 'agentix-public-chat-config-cache'

/* Scoped token issued by /public-chat/config — sent on every /chat/* request
   so public-chat availability rules apply ONLY to this page's calls. */
let publicAccessToken: string | null = null

function pcHeaders(): Record<string, string> {
  return publicAccessToken ? { 'X-Public-Chat-Token': publicAccessToken } : {}
}

async function fetchPublicChatConfig(): Promise<any | null> {
  try {
    // Clear stale entries written by older builds. Assistant availability is
    // mutable (for example, an assistant may just have been deleted), so it
    // must be re-resolved by the server rather than served from sessionStorage.
    sessionStorage.removeItem(CONFIG_CACHE_KEY)
  } catch {
    // ignore storage errors; the network request below is authoritative
  }
  try {
    // A group-scoped link carries ?pct=<signed token> — forward it so the
    // config endpoint resolves that scope (and re-mints the token with the
    // same groups) instead of minting a plain anonymous/session token.
    const incomingPct = new URLSearchParams(window.location.search).get('pct')
    const res = await fetch(getPublicChatConfigUrl(), {
      credentials: 'include',
      cache: 'no-store',
      headers: incomingPct ? { 'X-Public-Chat-Token': incomingPct } : {},
    })
    if (!res.ok) return null
    const data = await res.json()
    publicAccessToken = data?.access_token ?? null
    return data
  } catch {
    return null
  }
}

function invalidatePublicChatConfigCache(): void {
  try {
    sessionStorage.removeItem(CONFIG_CACHE_KEY)
  } catch {
    // ignore
  }
}

const PUBLIC_THEME_KEY = 'agentix-public-chat-theme'

function getOrCreateConversationId(model: string): string {
  const key = `agentix-public-chat-conversation-id:${model}`
  const existing = sessionStorage.getItem(key)
  if (existing && existing.startsWith('public-chat-')) {
    return existing
  }
  const hash = (crypto as any).randomUUID ? (crypto as any).randomUUID() : String(Date.now())
  const id = `public-chat-${hash}`
  sessionStorage.setItem(key, id)
  return id
}

function chatMessagesKey(conversationId: string): string {
  return `agentix-public-chat-messages:${conversationId}`
}

function normalizeEmotion(emotion: any): Message['emotion'] {
  if (!emotion) return undefined
  // Convert dict-shaped phases (backend format) to array (frontend format)
  if (emotion.phases && !Array.isArray(emotion.phases)) {
    emotion = {
      ...emotion,
      phases: Object.entries(emotion.phases).map(([phase, data]: [string, any]) => ({
        phase,
        primary_emotion: data.primary_emotion,
        confidence: data.confidence,
        intensity: data.intensity,
        source: data.source,
        timestamp_ms: data.timestamp_ms,
        sampled_frames: data.sampled_frames,
        valid_face_frames: data.valid_face_frames,
        face_detection_rate: data.face_detection_rate,
        window_ms: data.window_ms,
      })),
    }
  }
  if (emotion.fused_emotion !== undefined || emotion.phases !== undefined) {
    return emotion as Message['emotion']
  }
  if (emotion.primary_emotion) {
    return {
      fused_emotion: {
        primary_emotion: emotion.primary_emotion,
        confidence: emotion.confidence,
        intensity: emotion.intensity,
        source: emotion.source,
        timestamp_ms: emotion.timestamp_ms,
        signals: emotion.signals,
      },
    }
  }
  return emotion as Message['emotion']
}

/* The server persists each turn's user text merged ("A\n\ninjectedB") while
   the live stream shows injected inputs as separate bubbles — so dedupe must
   match a text in ANY '\n\n'-separated segment of a user message, not just on
   full-message equality. Otherwise the same injected message reappears after
   a reload/attach that reconstructed the thread from server turns. */
function userTextMatches(content: string, text: string): boolean {
  return (
    content === text ||
    content.startsWith(text + '\n\n') ||
    content.endsWith('\n\n' + text) ||
    content.includes('\n\n' + text + '\n\n')
  )
}

function hasUserText(msgs: Message[], text: string): boolean {
  return msgs.some((m) => m.role === 'user' && userTextMatches(m.content, text))
}

/* Insert a user message bubble right before the trailing (streaming) assistant
   placeholder — keeps user→assistant order when a mid-run injection arrives. */
function filterUnconsumed(
  items: Array<string | { text: string; images?: string[] }>,
  queuedTexts: ReadonlySet<string>,
  displayed: Message[],
): Array<{ text: string; images?: string[] }> {
  // Unconsumed inputs race two other paths: a message already shown in the
  // thread was actually consumed (must not re-send) and one already back in
  // the queue must not be requeued a second time.
  const seen = new Set<string>()
  const out: Array<{ text: string; images?: string[] }> = []
  for (const it of items) {
    const text = typeof it === 'string' ? it : it.text
    const images = typeof it === 'string' ? undefined : it.images
    if (!text || hasUserText(displayed, text) || queuedTexts.has(text) || seen.has(text)) continue
    seen.add(text)
    out.push({ text, images })
  }
  return out
}

function insertUserBeforeLastAssistant(prev: Message[], text: string, source?: string): Message[] {
  // Idempotent: a replayed delta or a second attached stream must not splice
  // the same user text in twice.
  if (hasUserText(prev, text)) return prev
  const msg: Message = { role: 'user', content: text, systemNote: !!source && source !== 'user' }
  const idx = [...prev.entries()].reverse().find(([, m]) => m.role === 'assistant')?.[0]
  if (idx === undefined) return [...prev, msg]
  const next = [...prev]
  next.splice(idx, 0, msg)
  return next
}

/* metricsFromRetrievalInfo lives in chat-shared/messageMetrics so dashboard
   and public chat rebuild identical badges from persisted turns. */

function loadMessagesForConversation(conversationId: string): Message[] {
  return loadChatMessages(chatMessagesKey(conversationId)).map((msg: any) => ({
    ...msg,
    emotion: normalizeEmotion(msg.emotion),
  }))
}

function persistMessages(conversationId: string, messages: Message[]) {
  persistChatMessagesDebounced(chatMessagesKey(conversationId), messages)
}

function cleanTextForTTS(text: string): string {
  let clean = text
  clean = clean.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
  clean = clean.replace(/https?:\/\/[^\s]+/g, 'Link')
  clean = clean.replace(/^#{1,6}\s+/gm, '')
  clean = clean.replace(/[*_`#]/g, '')
  clean = clean.replace(/^[-*]\s+/gm, '')
  clean = clean.replace(/^\d+\.\s+/gm, '')
  clean = clean.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{FE00}-\u{FE0F}\u{1F900}-\u{1F9FF}\u{1F018}-\u{1F270}\u{238C}\u{2B06}\u{2B07}\u{2B05}\u{27A1}\u{2194}-\u{21AA}]/gu, '')
  clean = clean.replace(/<[^>]+>/g, '')
  clean = clean.replace(/[ \t]+/g, ' ')
  clean = clean.split('\n').map((l) => l.trim()).join('\n')
  clean = clean.replace(/\n{3,}/g, '\n\n')
  return clean.trim()
}

function ToolCallModal({
  tool,
  onClose,
}: {
  tool: ToolCallLive | null
  onClose: () => void
}) {
  const { t } = useI18n()
  const [toolViewMode, setToolViewMode] = useState<'list' | 'graph'>('list')
  const [toolOkfConcepts, setToolOkfConcepts] = useState<OKFConcept[] | null>(null)
  const [toolOkfLoading, setToolOkfLoading] = useState(false)
  const [toolOkfError, setToolOkfError] = useState<string | null>(null)

  const retrievedDocIds = useMemo(() => {
    if (!tool || !tool.output) return new Set<string>()
    if (tool.tool_name !== 'rag_search') return new Set<string>()
    const sources = new Set<string>()
    for (const line of tool.output.split('\n')) {
      const m = line.match(/Source:\s*(.+?)(?:,\s*|$)/)
      if (m && m[1]) {
        const src = m[1].trim()
        sources.add(src)
        const filename = src.split('/').pop() || ''
        if (filename && filename !== src) sources.add(filename)
      }
    }
    return sources
  }, [tool])

  if (!tool) return null

  const isRagSearch = tool.tool_name === 'rag_search'
  const ragCollection = isRagSearch ? String(tool.args?.collection || '') : ''
  const canShowGraph = isRagSearch && !!ragCollection

  async function loadToolGraph() {
    if (!ragCollection || toolOkfConcepts !== null) return
    setToolOkfLoading(true)
    setToolOkfError(null)
    try {
      const res = await publicKnowledge.getOKFGraph(ragCollection)
      setToolOkfConcepts(res.concepts)
    } catch (err) {
      setToolOkfError(String(err instanceof Error ? err.message : err))
    } finally {
      setToolOkfLoading(false)
    }
  }

  function handleToolToggleGraph() {
    if (toolViewMode === 'list') {
      setToolViewMode('graph')
      loadToolGraph()
    } else {
      setToolViewMode('list')
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={(e) => e.stopPropagation()} style={{ maxWidth: toolViewMode === 'graph' ? '96vw' : '640px', width: '90vw', height: toolViewMode === 'graph' ? '90vh' : 'auto', display: toolViewMode === 'graph' ? 'flex' : 'block', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <h4 style={{ margin: 0 }}>{tool.tool_name}</h4>
            {canShowGraph && (
              <button
                className="btn-secondary"
                onClick={handleToolToggleGraph}
                style={{ fontSize: '11px', padding: '2px 8px' }}
              >
                {toolViewMode === 'list' ? 'Graph View' : 'List View'}
              </button>
            )}
          </div>
          <button className="btn-icon" onClick={onClose} title={t.close}>
            <X size={18} />
          </button>
        </div>
        {toolViewMode === 'graph' ? (
          <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <OKFGraphView
              concepts={toolOkfConcepts || []}
              retrievedDocIds={retrievedDocIds}
              loading={toolOkfLoading}
              error={toolOkfError}
            />
          </div>
        ) : (
          <>
            <div style={{ marginBottom: '12px' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                {t.arguments}
              </div>
              <pre className="chat-toolcall-overlay-args" style={{ margin: 0, borderRadius: '8px' }}>
                {JSON.stringify(tool.args, null, 2)}
              </pre>
            </div>
            {typeof tool.duration_ms === 'number' && (
              <div className="chat-toolcall-overlay-meta" style={{ marginBottom: '12px' }}>
                {t.duration}: {tool.duration_ms}ms
              </div>
            )}
            {tool.error && (
              <div className="chat-toolcall-overlay-err" style={{ marginBottom: '12px' }}>
                {tool.error}
              </div>
            )}
            {tool.output && (
              <div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  {t.output}
                </div>
                <pre className="chat-toolcall-overlay-args" style={{ margin: 0, borderRadius: '8px' }}>
                  {tool.output}
                </pre>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  RAG badge + modal                                                    */
/* ------------------------------------------------------------------ */
function RAGBadge({ info, onClick }: { info: NonNullable<Message['retrievalInfo']>; onClick: () => void }) {
  const isOk = info.status === 'ok'
  const label = isOk
    ? `RAG \u00b7 ${info.chunks_retrieved} Docs \u00b7 ${(info.rag_context_tokens || 0).toLocaleString()} Tokens`
    : `RAG \u00b7 ${info.status}`
  return (
    <span
      className={`chat-rag-badge ${isOk ? 'ok' : 'err'}`}
      onClick={onClick}
      style={{ cursor: 'pointer' }}
    >
      <BookOpen size={12} /> {label}
    </span>
  )
}

function RAGModal({
  info,
  onClose,
}: {
  info: Message['retrievalInfo']
  onClose: () => void
}) {
  const { t } = useI18n()
  const [viewMode, setViewMode] = useState<'list' | 'graph'>('list')
  const [okfConcepts, setOkfConcepts] = useState<OKFConcept[] | null>(null)
  const [okfLoading, setOkfLoading] = useState(false)
  const [okfError, setOkfError] = useState<string | null>(null)

  if (!info) return null

  const okfDocIds = new Set<string>()
  let okfCollection = ''
  if (info.rag_context) {
    for (const doc of info.rag_context) {
      const meta = doc.metadata as Record<string, unknown> | undefined
      const conceptId = meta?.okf_concept_id as string | undefined
      if (conceptId) {
        okfDocIds.add(conceptId)
        if (!okfCollection) {
          okfCollection = (meta?.collection as string) || ''
        }
      }
    }
  }
  const hasOkf = okfDocIds.size > 0 && !!okfCollection

  async function loadGraph() {
    if (!okfCollection || okfConcepts !== null) return
    setOkfLoading(true)
    setOkfError(null)
    try {
      const res = await publicKnowledge.getOKFGraph(okfCollection)
      setOkfConcepts(res.concepts)
    } catch (err) {
      setOkfError(String(err instanceof Error ? err.message : err))
    } finally {
      setOkfLoading(false)
    }
  }

  function handleToggleGraph() {
    if (viewMode === 'list') {
      setViewMode('graph')
      loadGraph()
    } else {
      setViewMode('list')
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={(e) => e.stopPropagation()} style={{ maxWidth: viewMode === 'graph' ? '96vw' : '640px', width: '90vw', height: viewMode === 'graph' ? '90vh' : 'auto', display: viewMode === 'graph' ? 'flex' : 'block', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <h4 style={{ margin: 0 }}>RAG Context</h4>
            {hasOkf && (
              <button
                className="btn-secondary"
                onClick={handleToggleGraph}
                style={{ fontSize: '11px', padding: '2px 8px' }}
              >
                {viewMode === 'list' ? 'Graph View' : 'List View'}
              </button>
            )}
          </div>
          <button className="btn-icon" onClick={onClose} title={t.close}>
            <X size={18} />
          </button>
        </div>
        <div className="chat-rag-overlay-meta" style={{ marginBottom: '12px' }}>
          Status: {info.status} \u00b7 Documents: {info.chunks_retrieved} \u00b7 Tokens: {(info.rag_context_tokens || 0).toLocaleString()}
          {hasOkf && <span> \u00b7 OKF Collection: {okfCollection}</span>}
        </div>
        {viewMode === 'graph' ? (
          <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <OKFGraphView
              concepts={okfConcepts || []}
              retrievedDocIds={okfDocIds}
              loading={okfLoading}
              error={okfError}
            />
          </div>
        ) : (
          info.rag_context && info.rag_context.length > 0 && (
            <div style={{ maxHeight: '60vh', overflowY: 'auto' }}>
              {info.rag_context.map((doc, idx) => (
                <div key={idx} className="chat-rag-overlay-doc">
                  <div className="chat-rag-overlay-doc-title">Document {idx + 1}</div>
                  <pre className="chat-rag-overlay-doc-content">{doc.content}</pre>
                  {doc.metadata && Object.keys(doc.metadata).length > 0 && (
                    <pre className="chat-rag-overlay-doc-meta">{JSON.stringify(doc.metadata, null, 2)}</pre>
                  )}
                </div>
              ))}
            </div>
          )
        )}
      </div>
    </div>
  )
}

function HtmlPreviewModal({ html, onClose }: { html: string | null; onClose: () => void }) {
  const { t } = useI18n()
  if (!html) return null
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box preview-modal" onClick={(e) => e.stopPropagation()}>
        <div className="preview-header">
          <div>
            <h4>{t.htmlPreview}</h4>
          </div>
          <div className="preview-header-actions">
            <button className="btn-icon" onClick={onClose} title={t.close}>
              <X size={18} />
            </button>
          </div>
        </div>
        <div className="preview-content">
          <iframe className="preview-frame" sandbox="allow-scripts" srcDoc={html} title={t.htmlPreview} />
        </div>
      </div>
    </div>
  )
}

export default function PublicChatPage() {
  const [assistants, setAssistants] = useState<string[]>([])
  const [assistantDescriptions, setAssistantDescriptions] = useState<Record<string, string>>({})
  const [model, setModel] = useState('')
  const [wsOpen, setWsOpen] = useState(false)
  const [wsEnabledMap, setWsEnabledMap] = useState<Record<string, boolean>>({})
  const wsEnabled = Boolean(wsEnabledMap[model])
  const [loadingModels, setLoadingModels] = useState(false)

  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [messageQueue, setMessageQueue] = useState<QueuedChatMessage[]>([])
  const [queuePaused, setQueuePaused] = useState(false)
  /* Injection is agent-only — unknown until the run reports it, so stay
     optimistic and latch to false on an explicit "not_agent" rejection. */
  const [runIsAgent, setRunIsAgent] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [sttToast, setSttToast] = useState<string | null>(null)
  const [configError, setConfigError] = useState<string | null>(null)

  const [configLoaded, setConfigLoaded] = useState(false)
  const [headerCollapsed, setHeaderCollapsed] = useState(false)
  const lastScrollTopRef = useRef(0)
  const lastHeaderToggleRef = useRef(0)
  const [headerMenuOpen, setHeaderMenuOpen] = useState(false)
  const headerMenuRef = useRef<HTMLDivElement>(null)
  const [title, setTitle] = useState('')
  const [subtitle, setSubtitle] = useState('')
  const [logoUrl, setLogoUrl] = useState('')
  const [theme, setThemeState] = useState<'light' | 'dark'>(() => {
    try {
      const stored = localStorage.getItem(PUBLIC_THEME_KEY)
      if (stored === 'light' || stored === 'dark') return stored
    } catch {}
    return 'light'
  })
  const [cfgLanguage, setCfgLanguage] = useState<string>('auto')
  const { language, t, setLanguage } = useI18n()

  const setTheme = (next: 'light' | 'dark') => {
    setThemeState(next)
    try {
      localStorage.setItem(PUBLIC_THEME_KEY, next)
    } catch {}
  }

  /* Auth & History state */
  const [chatUser, setChatUser] = useState<ChatUserInfo | null>(null)
  const [authModalOpen, setAuthModalOpen] = useState(false)
  const [authTab, setAuthTab] = useState<'login' | 'register'>('login')
  const [authError, setAuthError] = useState<string | null>(null)
  const [authLoading, setAuthLoading] = useState(false)

  const [pwdModalOpen, setPwdModalOpen] = useState(false)
  const [pwdForm, setPwdForm] = useState({ current_password: '', new_password: '', confirm_password: '' })
  const [pwdError, setPwdError] = useState('')
  const [pwdSuccess, setPwdSuccess] = useState(false)
  const [pwdLoading, setPwdLoading] = useState(false)

  const [historyOpen, setHistoryOpen] = useState(false)
  const [historyItems, setHistoryItems] = useState<ConversationItem[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [previewConversationId, setPreviewConversationId] = useState<string | null>(null)
  const [previewMessages, setPreviewMessages] = useState<Message[] | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [editingTitleId, setEditingTitleId] = useState<string | null>(null)
  const [editingTitleValue, setEditingTitleValue] = useState('')

  /* Upload & memory state — flags come from /public-chat/config `uploads`. */
  const [uploadsCfg, setUploadsCfg] = useState<{ images: boolean; files: boolean }>({ images: false, files: false })
  const [memoryEnabled, setMemoryEnabled] = useState<Record<string, boolean>>({})
  const [pendingImages, setPendingImages] = useState<string[]>([])
  const [pendingFiles, setPendingFiles] = useState<PendingFile[]>([])
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [memoryOpen, setMemoryOpen] = useState(false)
  const [memoryEntries, setMemoryEntries] = useState<any[]>([])
  const [editingMemory, setEditingMemory] = useState<{ assistant_name: string; memory: string } | null>(null)
  const [memoryLoading, setMemoryLoading] = useState(false)
  const [memoryError, setMemoryError] = useState<string | null>(null)

  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  /* Synchronous in-flight guard for message sends. State updates are batched,
     so `chatLoading` alone cannot prevent re-entrant sends while an async send
     is suspended before its first state commit — the queue drain effect would
     otherwise fire every remaining queued message in parallel. */
  const sendInFlightRef = useRef(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const activeModelRef = useRef<string>(model)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [playingMsgIdx, setPlayingMsgIdx] = useState<number | null>(null)
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null)

  const [audioConfig, setAudioConfig] = useState<{
    stt: { button_visible: boolean; language?: string; provider?: string; model?: string }
    tts: { button_visible: boolean; voice_id: string | null; auto_read?: boolean; streaming_enabled?: boolean }
    phone: { button_visible: boolean }
  } | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [phoneMode, setPhoneMode] = useState(false)
  const [awaitingUserSpeech, setAwaitingUserSpeech] = useState(false)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const vadSilenceStartRef = useRef<number>(0)
  const vadRafRef = useRef<number | null>(null)
  const phoneModeRef = useRef(false)
  const vadSpeechDetectedRef = useRef(false)
  const recordingStartTimeRef = useRef(0)
  const phoneEmptyRetriesRef = useRef(0)
  const lastAssistantIdxRef = useRef<number | null>(null)
  const [selectedToolCall, setSelectedToolCall] = useState<ToolCallLive | null>(null)
  const [selectedSubagent, setSelectedSubagent] = useState<SubagentMeta | null>(null)
  const [selectedRAGInfo, setSelectedRAGInfo] = useState<Message['retrievalInfo']>(undefined)
  const [previewHtml, setPreviewHtml] = useState<string | null>(null)
  const [openEmotionIdx, setOpenEmotionIdx] = useState<number | null>(null)
  const ttsPlayerRef = useRef<TTSSegmentPlayer | null>(null)
  const ttsBufferRef = useRef('')
  const userScrolledUpRef = useRef(false)
  const continuousAudioStreamRef = useRef<MediaStream | null>(null)
  const hasTransitionedToTypingRef = useRef(false)
  const audioSnapshotsRef = useRef<EmotionResult[]>([])
  const isStartingAudioRef = useRef(false)
  const isStartingWebcamRef = useRef(false)
  const messagesRef = useRef<Message[]>(messages)

  const [emotionConfig, setEmotionConfig] = useState<{
    enabled: boolean
    text: { enabled: boolean; rules_enabled: boolean }
    webcam: {
      enabled: boolean
      button_visible: boolean
      runtime: string
      execution_providers: string[]
      face_detector: string
      face_model_url: string
      expression_model_url: string
      expression_model_type?: string
      expression_input_size?: number
      snippet_window_ms: number
      sample_fps: number
      max_frames_per_snippet: number
      min_valid_face_frames: number
      frame_upload_enabled: boolean
    }
    audio: { enabled: boolean; model_url: string; min_snr_db?: number; noise_gate_min?: number; noise_gate_max?: number }
    prompt_steering: { enabled: boolean; min_confidence: number }
    analytics: { enabled: boolean }
  } | null>(null)
  const [webcamConsent, setWebcamConsent] = useState(false)
  const [webcamConsentOpen, setWebcamConsentOpen] = useState(false)
  const [notice, setNotice] = useState<{ message: string; danger?: boolean } | null>(null)
  const [convDeleteTarget, setConvDeleteTarget] = useState<string | null>(null)
  const [webcamActive, setWebcamActive] = useState(false)
  const [emotionRuntimeStatus, setEmotionRuntimeStatus] = useState<EmotionRuntimeStatus>('uninitialized')
  const snippetBufferRef = useRef<ReturnType<typeof createVideoSnippetBuffer> | null>(null)
  const audioSnippetBufferRef = useRef<ReturnType<typeof createAudioSnippetBuffer> | null>(null)
  const emotionTimelineRef = useRef<{
    phases: Record<EmotionPhase, { snapshots: EmotionResult[]; startTime: number; endTime?: number }>
    activePhase: EmotionPhase
    startTime: number
  } | null>(null)
  const emotionPollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const lastEmotionModelRef = useRef<string | null>(null)
  const lastAudioModelRef = useRef<string | null>(null)

  /* ------------------------------------------------------------------ */
  /*  Smart auto-scroll                                                 */
  /* ------------------------------------------------------------------ */
  const scrollToBottom = useCallback((instant = false) => {
    if (!userScrolledUpRef.current && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: instant ? 'auto' : 'smooth' })
    }
  }, [])

  // Flush debounced sessionStorage writes on unmount.
  useEffect(() => () => flushPendingChatWrites(), [])

  // Read-only lookup of the active conversation id (getOrCreate* would mint
  // one as a side effect, which we don't want during render). Recompute when
  // messages change — a conversation switch always accompanies one.
  const activeConversationId = useMemo(() => {
    try {
      return sessionStorage.getItem(`agentix-public-chat-conversation-id:${model}`) || undefined
    } catch {
      return undefined
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, messages])

  useEffect(() => {
    // Instant scrolling while streaming — smooth animation cannot keep up with
    // batched updates and piles up expensive scroll interruptions.
    scrollToBottom(chatLoading)
  }, [messages, chatLoading, scrollToBottom])

  useEffect(() => {
    const container = messagesContainerRef.current
    if (!container) return
    const onScroll = () => {
      const st = container.scrollTop
      const nearBottom = container.scrollHeight - st - container.clientHeight < 100
      userScrolledUpRef.current = !nearBottom
      // Auto-collapse the header when scrolling down (mobile gains space
      // back automatically; the class only takes effect under 640px).
      // Large deltas + cooldown prevent flicker from momentum bounce,
      // iOS rubber-banding and streaming auto-scroll pushes.
      const delta = st - lastScrollTopRef.current
      lastScrollTopRef.current = st
      const now = Date.now()
      if (st <= 40) {
        setHeaderCollapsed(false)
      } else if (now - lastHeaderToggleRef.current > 350) {
        if (delta > 12 && st > 140) {
          setHeaderCollapsed(true)
          lastHeaderToggleRef.current = now
        } else if (delta < -12) {
          setHeaderCollapsed(false)
          lastHeaderToggleRef.current = now
        }
      }
    }
    container.addEventListener('scroll', onScroll, { passive: true })
    return () => container.removeEventListener('scroll', onScroll)
  }, [])

  // Close the header overflow menu on outside click / Escape.
  useEffect(() => {
    if (!headerMenuOpen) return
    const onDown = (e: MouseEvent) => {
      if (headerMenuRef.current && !headerMenuRef.current.contains(e.target as Node)) {
        setHeaderMenuOpen(false)
      }
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setHeaderMenuOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [headerMenuOpen])

  const copy = useCallback((id: string) => {
    setCopiedId(id)
    setTimeout(() => setCopiedId((current) => (current === id ? null : current)), 2000)
  }, [])

  // Sync phoneModeRef with state
  useEffect(() => {
    phoneModeRef.current = phoneMode
  }, [phoneMode])

  /* Sync messagesRef with state */
  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  /* Cleanup emotion on unmount */
  useEffect(() => {
    return () => {
      stopWebcamEmotion()
      stopEmotionPolling()
      attachAbortRef.current?.abort()
    }
  }, [])

  /* Auto-focus textarea on mount */
  useEffect(() => {
    textareaRef.current?.focus()
  }, [])

  /* Close emotion tooltip on outside click */
  useEffect(() => {
    const handleDocClick = () => setOpenEmotionIdx(null)
    document.addEventListener('click', handleDocClick)
    return () => document.removeEventListener('click', handleDocClick)
  }, [])

  async function fetchEmotionConfig(modelName: string) {
    if (lastEmotionModelRef.current === modelName) return
    try {
      const res = await fetch(window.location.origin + getApiBase() + `/emotion/config?model=${encodeURIComponent(modelName)}`, { credentials: 'omit' })
      if (!res.ok) {
        setEmotionConfig(null)
        return
      }
      const data = await res.json()
      setEmotionConfig(data)
      lastEmotionModelRef.current = modelName
    } catch {
      setEmotionConfig(null)
    }
  }

  async function fetchAudioConfig() {
    const currentModel = activeModelRef.current
    if (!currentModel) return
    if (lastAudioModelRef.current === currentModel) return
    try {
      const res = await fetch(window.location.origin + getApiBase() + `/audio/config?model=${encodeURIComponent(currentModel)}`, { credentials: 'include' })
      if (!res.ok) {
        setAudioConfig(null)
        return
      }
      const data = await res.json()
      setAudioConfig(data)
      lastAudioModelRef.current = currentModel
    } catch {
      setAudioConfig(null)
      lastAudioModelRef.current = currentModel
    }
  }

  // Load public config and assistants (with credentials so backend can resolve session)
  /* Apply a fetched /public-chat/config uniformly. Used by init, login and
     logout so capability flags (uploads, memory) always reflect the current
     caller — the backend issues them per-caller (files require auth). */
  function applyPublicChatConfig(cfg: any, opts?: { resetConversation?: boolean }) {
    // Global branding supplies defaults for unset surface-specific fields.
    const brand = cfg.branding || {}
    const effectiveTitle = cfg.title || brand.app_name || ''
    if (effectiveTitle) setTitle(effectiveTitle)
    if (cfg.subtitle || brand.powered_by) setSubtitle(cfg.subtitle || brand.powered_by)
    document.title = effectiveTitle || 'Agentix Chat'
    setLogoUrl(cfg.logo_url || brand.logo_url || EMBEDDED_LOGO)
    const faviconHref = cfg.favicon_url || brand.favicon_url || EMBEDDED_FAVICON
    let link = document.querySelector("link[rel~='icon']") as HTMLLinkElement | null
    if (!link) {
      link = document.createElement('link')
      link.rel = 'icon'
      document.head.appendChild(link)
    }
    link.type = 'image/png'
    link.href = faviconHref
    // Server theme is only a default — a stored user preference wins.
    if (cfg.theme) {
      try {
        if (!localStorage.getItem(PUBLIC_THEME_KEY)) {
          setThemeState(cfg.theme as 'light' | 'dark')
        }
      } catch {}
    }
    setCfgLanguage(cfg.language || 'auto')
    if (cfg.language && cfg.language !== 'auto') setLanguage(cfg.language as 'de' | 'en')
    if (cfg.uploads) setUploadsCfg({ images: !!cfg.uploads.images, files: !!cfg.uploads.files })
    setMemoryEnabled(cfg.assistant_memory_enabled || {})
    setWsEnabledMap(cfg.assistant_workspace_enabled || {})
    const enabled = cfg.enabled_assistants || []
    setAssistants(enabled)
    setConfigLoaded(true)
    setAssistantDescriptions(cfg.assistant_descriptions || {})
    if (enabled.length === 0) return
    if (opts?.resetConversation) {
      const target = enabled[0]
      setModel(target)
      activeModelRef.current = target
      const convId = getOrCreateConversationId(target)
      setMessages(loadMessagesForConversation(convId))
      attachToRunIfActive(convId)
      fetchAudioConfig()
      fetchEmotionConfig(target)
    } else if (!enabled.includes(model)) {
      handleModelChange(enabled[0])
    }
  }

  useEffect(() => {
    async function init() {
      setLoadingModels(true)
      try {
        // First check auth so we know if a session cookie exists
        try {
          const me = await publicAuth.me()
          if (me.authenticated && me.user) {
            setChatUser(me.user)
          }
        } catch {
          // ignore auth errors
        }
        const cfg = await fetchPublicChatConfig()
        if (cfg) {
          // Master switch: when the feature is disabled the backend issues no
          // access token and returns no assistants — mirror that by blocking
          // the page instead of falling back to anonymous API access.
          if (cfg.enabled === false) {
            setConfigError('This chat is currently disabled.')
            setLoadingModels(false)
            return
          }
          applyPublicChatConfig(cfg, { resetConversation: true })
        } else {
          setConfigError('Public chat configuration could not be loaded.')
        }
      } catch {
        setConfigError('Public chat configuration could not be loaded.')
      } finally {
        setLoadingModels(false)
      }
    }
    init()
  }, [])

  /* Auto-resize textarea */
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    const h = Math.min(ta.scrollHeight, 100)
    ta.style.height = h + 'px'
    ta.style.overflowY = ta.scrollHeight > 100 ? 'auto' : 'hidden'
  }, [input])

  /* Dynamic input bar width */
  useEffect(() => {
    const inner = document.querySelector<HTMLDivElement>('.chat-input-inner.chat-input-dynamic')
    if (!inner) return
    if (phoneMode) {
      inner.style.width = ''
      return
    }
    const computeWidth = () => {
      const ta = textareaRef.current
      if (!ta) return
      const cs = getComputedStyle(ta)
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.font = `${cs.fontSize} ${cs.fontFamily}`
      const textWidth = ctx.measureText(ta.value || ta.placeholder).width
      let buttonWidth = 0
      inner.querySelectorAll('button').forEach(btn => { buttonWidth += btn.offsetWidth })
      const gap = parseFloat(getComputedStyle(inner).gap) || 10
      const buttonCount = inner.querySelectorAll('button').length
      const padding = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight) + parseFloat(getComputedStyle(inner).paddingLeft) + parseFloat(getComputedStyle(inner).paddingRight)
      const target = Math.min(Math.max(textWidth + buttonWidth + gap * (buttonCount + 1) + padding + 24, Math.max(400, window.innerWidth * 0.5)), window.innerWidth * 0.9)
      inner.style.width = target + 'px'
    }
    computeWidth()
    window.addEventListener('resize', computeWidth)
    return () => window.removeEventListener('resize', computeWidth)
  }, [input, phoneMode])

  // Persist messages
  useEffect(() => {
    const convId = getOrCreateConversationId(activeModelRef.current)
    persistMessages(convId, messages)
  }, [messages])

  // Theme
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    document.documentElement.classList.toggle('dark-theme', theme === 'dark')
  }, [theme])

  function handleModelChange(name: string) {
    setModel(name)
    activeModelRef.current = name
    setMessageQueue([])
    setQueuePaused(false)
    setPendingImages([])
    setPendingFiles([])
    const convId = getOrCreateConversationId(name)
    setMessages(loadMessagesForConversation(convId))
    attachToRunIfActive(convId)
    setError(null)
    fetchAudioConfig()
    fetchEmotionConfig(name)
  }

  /* ------------------------------------------------------------------ */
  /*  Resume a run that survived a page reload                            */
  /* ------------------------------------------------------------------ */
  const attachAbortRef = useRef<AbortController | null>(null)

  async function attachToRunIfActive(conversationId: string) {
    if (!conversationId || abortRef.current) return
    try {
      const st = await fetch(`${getApiUrl()}/chat/runs/${encodeURIComponent(conversationId)}`, {
        credentials: 'include',
        headers: pcHeaders(),
      })
      const info = await st.json().catch(() => null)
      if (info?.running) {
        setRunIsAgent(info?.agent === true)
        await streamRunResumed(conversationId)
      } else {
        await syncConversationFromServer(conversationId)
      }
      // Inputs the run never consumed — recover them after missed broadcast,
      // deduped against items a live delta already restored.
      const unconsumed = info?.unconsumed_inputs as Array<{ text: string; images?: string[] }> | undefined
      if (unconsumed?.length) requeueUnconsumed(unconsumed)
    } catch {
      // best effort — older backend or network issue just skips resume
    }
  }

  async function syncConversationFromServer(conversationId: string) {
    if (!chatUser) return // server history is only available to logged-in users
    try {
      const data = await publicConversations.messages(conversationId)
      const loadedMessages = normalizeServerMessages(data)
      if (loadedMessages.length === 0) return
      const local = messagesRef.current
      const lastLocalAssistant = [...local].reverse().find((m) => m.role === 'assistant')
      const lastServerAssistant = [...loadedMessages].reverse().find((m) => m.role === 'assistant')
      const localLen = lastLocalAssistant?.content?.length ?? -1
      const serverLen = lastServerAssistant?.content?.length ?? -1
      if (loadedMessages.length > local.length || serverLen > localLen) {
        setMessages(loadedMessages)
      }
    } catch {
      // conversation not persisted server-side — keep local state
    }
  }

  async function streamRunResumed(conversationId: string) {
    attachAbortRef.current?.abort()
    const ctrl = new AbortController()
    attachAbortRef.current = ctrl
    setChatLoading(true)
    setError(null)
    const batcher = new StreamBatcher()
    try {
      const res = await fetch(`${getApiUrl()}/chat/runs/${encodeURIComponent(conversationId)}/stream`, {
        credentials: 'include',
        headers: pcHeaders(),
        signal: ctrl.signal,
      })
      if (!res.ok || !res.body) {
        await syncConversationFromServer(conversationId)
        return
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let assistantContent = ''
      let assistantReasoning = ''
      let hasReasoning = false
      let assistantToolCalls: ToolCallLive[] = []
      let assistantSubagents: SubagentEvent[] = []
      let assistantTodoList: AgentTodoSnapshot | undefined
      let observerEventsAcc: ObserverEvent[] = []
      let assistantRetrievalInfo: Message['retrievalInfo'] = undefined
      let assistantMetrics: ResponseMetrics | undefined
      let assistantLiveUsage: LiveUsage | undefined
      let terminalEventsAcc: ToolExecutionEvent[] = []

      const applyAccumulated = () => {
        setMessages((prev) => {
          const next = [...prev]
          const lastIdx = next.length - 1
          if (lastIdx < 0 || next[lastIdx].role !== 'assistant') return prev
          const existing = next[lastIdx]
          next[lastIdx] = {
            role: 'assistant',
            content: assistantContent,
            reasoning: hasReasoning ? assistantReasoning : undefined,
            toolCalls: assistantToolCalls.length > 0 ? assistantToolCalls : undefined,
            subagents: assistantSubagents.length > 0 ? assistantSubagents : undefined,
            todoList: assistantTodoList,
            observerEvents: observerEventsAcc.length > 0 ? observerEventsAcc : existing?.observerEvents,
            terminalEvents: terminalEventsAcc.length > 0 ? terminalEventsAcc : existing?.terminalEvents,
            retrievalInfo: assistantRetrievalInfo !== undefined ? assistantRetrievalInfo : existing?.retrievalInfo,
            metrics: assistantMetrics ?? existing?.metrics,
            liveUsage: assistantLiveUsage ?? existing?.liveUsage,
          }
          return next
        })
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6)
          if (data === '[DONE]') break
          try {
            const chunk = JSON.parse(data)
            const delta = chunk.choices?.[0]?.delta || {}
            if (delta.resumed) {
              assistantContent = delta.content || ''
              assistantReasoning = delta.reasoning_content || ''
              hasReasoning = Boolean(assistantReasoning)
              assistantToolCalls = delta.tool_calls || []
              for (const tp of (delta.tool_progress as ToolProgressItem[] | undefined) || []) {
                assistantToolCalls = addToolProgress(assistantToolCalls, tp)
              }
              for (const call of [...assistantToolCalls]) {
                if (!call.running) assistantToolCalls = resolveToolProgress(assistantToolCalls, call)
              }
              assistantSubagents = delta.subagent_events || []
              assistantTodoList = delta.todo_list ?? undefined
              observerEventsAcc = (delta.observer_events as ObserverEvent[] | undefined) || []
              if (delta.live_usage) assistantLiveUsage = delta.live_usage as LiveUsage
              const runUserMessage = typeof delta.user_message === 'string' ? delta.user_message : ''
              const resumedInjected = Array.isArray(delta.injected_messages) ? delta.injected_messages : []
              setMessages((prev) => {
                const next = [...prev]
                if (runUserMessage && !hasUserText(next, runUserMessage)) {
                  next.push({ role: 'user', content: runUserMessage })
                }
                for (const injected of resumedInjected) {
                  const injectedText = typeof injected === 'string' ? injected : (injected as any)?.text
                  const injectedSrc = typeof injected === 'object' && injected ? (injected as any).source as string | undefined : undefined
                  if (typeof injectedText === 'string' && injectedText && !hasUserText(next, injectedText)) {
                    next.push({ role: 'user', content: injectedText, systemNote: !!injectedSrc && injectedSrc !== 'user' })
                  }
                }
                if (next.length === 0 || next[next.length - 1].role !== 'assistant') {
                  next.push({ role: 'assistant', content: '' })
                }
                return next
              })
            } else {
              const content = delta.content
              const reasoning = delta.reasoning_content
              const toolExecution = delta.tool_execution
              const toolCalls = delta.tool_calls as ToolCallLive[] | undefined
              const toolProgress = delta.tool_progress as ToolProgressItem | undefined
              const subagent = delta.subagent as SubagentEvent | undefined
              const todoListDelta = delta.todo_list as AgentTodoSnapshot | undefined
              const retrievalInfo = delta.retrieval_info as Message['retrievalInfo'] | undefined
              const metricsDelta = delta.metrics as ResponseMetrics | undefined
              const liveDelta = delta.live_usage as LiveUsage | undefined
              const injectedUserMessage = delta.injected_user_message as string | undefined
              const injectedSource = delta.injected_source as string | undefined
              const unconsumedInputs = delta.unconsumed_inputs as Array<string | { text: string; images?: string[] }> | undefined
              const observerDelta = delta.observer as ObserverEvent | undefined
              if (injectedUserMessage) {
                setMessages((prev) => insertUserBeforeLastAssistant(prev, injectedUserMessage, injectedSource))
              }
              if (unconsumedInputs?.length) requeueUnconsumed(unconsumedInputs)
              if (observerDelta) observerEventsAcc = [...observerEventsAcc, observerDelta]
              if (liveDelta) assistantLiveUsage = liveDelta
              if (retrievalInfo) {
                assistantRetrievalInfo = retrievalInfo
                if (retrievalInfo.todo_list) assistantTodoList = retrievalInfo.todo_list
              }
              if (todoListDelta) assistantTodoList = todoListDelta
              if (metricsDelta) assistantMetrics = metricsDelta
              if (content) assistantContent += content
              if (reasoning) {
                assistantReasoning += reasoning
                hasReasoning = true
              }
              if (toolExecution) terminalEventsAcc = [...terminalEventsAcc, toolExecution]
              if (toolProgress) assistantToolCalls = addToolProgress(assistantToolCalls, toolProgress)
              if (toolCalls && toolCalls.length > 0) {
                for (const call of toolCalls) {
                  assistantToolCalls = resolveToolProgress(assistantToolCalls, call)
                  const idx = call.id ? assistantToolCalls.findIndex((item) => item.id === call.id) : -1
                  if (idx < 0) assistantToolCalls.push(call)
                  else assistantToolCalls[idx] = { ...assistantToolCalls[idx], ...call }
                }
              }
              if (subagent) assistantSubagents = upsertSubagentEvent(assistantSubagents, subagent)
            }
            batcher.schedule(applyAccumulated)
          } catch {
            // ignore malformed chunk
          }
        }
      }
      batcher.flushNow()
      persistMessages(conversationId, messagesRef.current)
      await syncConversationFromServer(conversationId)
    } catch (e: any) {
      batcher.flushNow()
      settleLastAssistantTools()
      if (e?.name !== 'AbortError') {
        // stream broke mid-run — the run keeps going server-side; pull the state
        await syncConversationFromServer(conversationId).catch(() => {})
      }
    } finally {
      setChatLoading(false)
      if (attachAbortRef.current === ctrl) attachAbortRef.current = null
    }
  }

  /* ------------------------------------------------------------------ */
  /*  Webcam Emotion                                                      */
  /* ------------------------------------------------------------------ */
  async function startWebcamEmotion() {
    if (!emotionConfig?.webcam?.enabled) return
    if (webcamActive) return
    if (isStartingWebcamRef.current) return
    if (!webcamConsent) {
      setWebcamConsentOpen(true)
      return
    }
    await startWebcamEmotionInner()
  }

  async function startWebcamEmotionInner() {
    const cfg = emotionConfig?.webcam
    if (!cfg) return
    isStartingWebcamRef.current = true
    setEmotionRuntimeStatus('loading')
    try {
      const snippetConfig: SnippetConfig = {
        windowMs: cfg.snippet_window_ms,
        sampleFps: cfg.sample_fps,
        maxFrames: cfg.max_frames_per_snippet,
        minValidFaceFrames: cfg.min_valid_face_frames,
        faceModelUrl: cfg.face_model_url,
        faceDetectorType: (cfg.face_detector?.includes('mediapipe') ? 'mediapipe' : 'onnx') as 'mediapipe' | 'onnx',
        expressionModelUrl: cfg.expression_model_url,
        expressionModelType: (cfg.expression_model_type as 'fer' | 'affectnet') ?? 'affectnet',
        expressionInputSize: cfg.expression_input_size ?? 224,
        executionProviders: cfg.execution_providers,
      }
      snippetBufferRef.current = createVideoSnippetBuffer(snippetConfig)
      await snippetBufferRef.current.start()
      if (isStartingWebcamRef.current) {
        // Start audio independently of webcam
        if (emotionConfig?.audio?.enabled) {
          await startContinuousAudioEmotion().catch(() => {})
        }
        setWebcamActive(true)
        setEmotionRuntimeStatus('ready')
      }
    } catch {
      setEmotionRuntimeStatus('error')
      setWebcamActive(false)
    } finally {
      isStartingWebcamRef.current = false
    }
  }

  function stopWebcamEmotion() {
    isStartingWebcamRef.current = false
    if (snippetBufferRef.current) {
      snippetBufferRef.current.stop()
      snippetBufferRef.current = null
    }
    stopContinuousAudioEmotion()
    setWebcamActive(false)
    setEmotionRuntimeStatus('uninitialized')
  }

  async function startContinuousAudioEmotion(): Promise<void> {
    if (audioSnippetBufferRef.current?.isActive()) return
    if (isStartingAudioRef.current) return
    if (!emotionConfig?.audio?.enabled) {
      console.info('[PublicChat] Audio emotion disabled in config')
      return
    }
    isStartingAudioRef.current = true
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      if (!isStartingAudioRef.current) {
        stream.getTracks().forEach((t) => t.stop())
        return
      }
      continuousAudioStreamRef.current = stream
      const cfg = emotionConfig.webcam
      const audioCfg: SnippetConfig = {
        windowMs: cfg?.snippet_window_ms || 3000,
        sampleFps: cfg?.sample_fps || 3,
        maxFrames: cfg?.max_frames_per_snippet || 12,
        minValidFaceFrames: cfg?.min_valid_face_frames || 3,
        faceModelUrl: cfg?.face_model_url || '/models/emotion/face-detector.onnx',
        faceDetectorType: 'mediapipe',
        expressionModelUrl: emotionConfig.audio.model_url || '/models/emotion/audio-emotion.onnx',
        expressionModelType: 'affectnet',
        expressionInputSize: 224,
        executionProviders: cfg?.execution_providers || ['wasm'],
        minSnrDb: emotionConfig.audio.min_snr_db ?? 10,
        noiseGateMin: emotionConfig.audio.noise_gate_min ?? 0.001,
        noiseGateMax: emotionConfig.audio.noise_gate_max ?? 0.5,
      }
      console.info('[PublicChat] Starting continuous audio emotion with modelUrl:', audioCfg.expressionModelUrl)
      audioSnippetBufferRef.current = createAudioSnippetBuffer(audioCfg)
      audioSnippetBufferRef.current.start(stream)
      console.info('[PublicChat] Continuous audio emotion started')
    } catch (err) {
      console.warn('[PublicChat] Audio emotion start failed:', err)
    } finally {
      isStartingAudioRef.current = false
    }
  }

  function stopContinuousAudioEmotion(): void {
    isStartingAudioRef.current = false
    if (audioSnippetBufferRef.current) {
      audioSnippetBufferRef.current.stop()
      audioSnippetBufferRef.current = null
    }
    if (continuousAudioStreamRef.current) {
      continuousAudioStreamRef.current.getTracks().forEach((t) => t.stop())
      continuousAudioStreamRef.current = null
    }
  }

  async function capturePreSendEmotion(): Promise<EmotionContextPayload['visual_pre_send'] | null> {
    if (!snippetBufferRef.current || !snippetBufferRef.current.isActive()) return null
    return snippetBufferRef.current.captureSnapshot()
  }

  function buildEmotionPayload(): EmotionContextPayload | undefined {
    if (!emotionConfig?.enabled) return undefined
    const payload: EmotionContextPayload = {}
    if (emotionConfig.webcam.enabled && webcamActive && webcamConsent) {
      // visual_pre_send will be captured right before sending
    }
    payload.consent = {
      webcam_opt_in: webcamConsent,
      frames_sent_to_backend: false,
      audio_opt_in: audioSnippetBufferRef.current?.isActive() ?? false,
    }
    return payload
  }

  /* ------------------------------------------------------------------ */
  /*  Continuous Emotion Batch Collector (phase-based)                  */
  /* ------------------------------------------------------------------ */
  function createEmptyPhases(): Record<EmotionPhase, { snapshots: EmotionResult[]; startTime: number; endTime?: number }> {
    return {
      waiting_for_response: { snapshots: [], startTime: Date.now() },
      reading: { snapshots: [], startTime: 0 },
      typing: { snapshots: [], startTime: 0 },
      speaking: { snapshots: [], startTime: 0 },
    }
  }

  function startEmotionPolling(initialPhase: EmotionPhase = 'waiting_for_response'): void {
    if (emotionPollIntervalRef.current) return
    hasTransitionedToTypingRef.current = false
    audioSnapshotsRef.current = []
    const phases = createEmptyPhases()
    phases[initialPhase].startTime = Date.now()
    emotionTimelineRef.current = {
      phases,
      activePhase: initialPhase,
      startTime: Date.now(),
    }
    emotionPollIntervalRef.current = setInterval(() => {
      processEmotionBatch()
    }, 2000)
  }

  function stopEmotionPolling(): void {
    if (emotionPollIntervalRef.current) {
      clearInterval(emotionPollIntervalRef.current)
      emotionPollIntervalRef.current = null
    }
    processEmotionBatch()
  }

  function switchEmotionPhase(nextPhase: EmotionPhase): void {
    const tl = emotionTimelineRef.current
    if (!tl || tl.activePhase === nextPhase) return
    const current = tl.phases[tl.activePhase]
    if (current.snapshots.length > 0) {
      const fused = fuseEmotions(current.snapshots)
      if (fused) {
        tl.phases[tl.activePhase] = {
          snapshots: [fused],
          startTime: current.startTime,
          endTime: Date.now(),
        }
      } else {
        current.endTime = Date.now()
      }
    } else {
      current.endTime = Date.now()
    }
    tl.phases[nextPhase].startTime = Date.now()
    tl.activePhase = nextPhase
  }

  function processEmotionBatch(): void {
    const tl = emotionTimelineRef.current
    if (!tl) return
    if (snippetBufferRef.current?.isActive()) {
      snippetBufferRef.current.captureSnapshot().then((res) => {
        if (res) tl.phases[tl.activePhase].snapshots.push(res)
      }).catch(() => {})
    }
    if (audioSnippetBufferRef.current?.isActive()) {
      audioSnippetBufferRef.current.captureSnapshot().then((res) => {
        if (res) {
          tl.phases[tl.activePhase].snapshots.push(res)
          // Bounded window — long runs must not grow this buffer forever.
          if (audioSnapshotsRef.current.length >= 300) audioSnapshotsRef.current.shift()
          audioSnapshotsRef.current.push(res)
        }
      }).catch(() => {})
    }
  }

  function buildEmotionPhases(): PhaseEmotion[] {
    const tl = emotionTimelineRef.current
    if (!tl) return []
    const result: PhaseEmotion[] = []
    const phaseOrder: EmotionPhase[] = ['waiting_for_response', 'reading', 'typing', 'speaking']
    for (const phase of phaseOrder) {
      const data = tl.phases[phase]
      if (data.snapshots.length === 0 && !data.endTime && phase !== tl.activePhase) {
        continue
      }
      if (data.snapshots.length > 0) {
        const fused = fuseEmotions(data.snapshots)
        if (fused) {
          result.push({
            phase,
            primary_emotion: fused.primary_emotion,
            confidence: fused.confidence,
            intensity: fused.intensity,
            source: `phase__${phase}`,
            timestamp_ms: data.startTime || Date.now(),
            sampled_frames: fused.sampled_frames,
            valid_face_frames: fused.valid_face_frames,
            face_detection_rate: fused.face_detection_rate,
            window_ms: fused.window_ms,
          })
          continue
        }
      }
      if (data.startTime > 0 || phase === tl.activePhase) {
        result.push({
          phase,
          primary_emotion: 'unknown',
          confidence: 0,
          intensity: 'low',
          source: `phase__${phase}`,
          timestamp_ms: data.startTime || Date.now(),
        })
      }
    }
    return result
  }

  function computeOverallFused(phases: PhaseEmotion[]): EmotionResult | null {
    if (phases.length === 0) return null
    const asResults: EmotionResult[] = phases.map((p) => ({
      primary_emotion: p.primary_emotion,
      confidence: p.confidence,
      intensity: p.intensity,
      source: p.source,
      timestamp_ms: p.timestamp_ms,
      sampled_frames: p.sampled_frames,
      valid_face_frames: p.valid_face_frames,
      face_detection_rate: p.face_detection_rate,
      window_ms: p.window_ms,
    }))
    return fuseEmotions(asResults)
  }

  function handleReset() {
    const convId = getOrCreateConversationId(activeModelRef.current)
    attachAbortRef.current?.abort()
    /* "New conversation" must mint a NEW conversation id — otherwise the next
       request reuses the old one and the backend keeps injecting that
       conversation's stored attachments and agent memory. Best-effort
       server-side delete first (401 for anonymous callers is fine — they
       cannot upload files anyway), then purge every local binding so
       getOrCreateConversationId generates a fresh id. */
    fetch(`${getApiUrl()}/chat/runs/${encodeURIComponent(convId)}/cancel`, {
      method: 'POST',
      credentials: 'include',
      headers: pcHeaders(),
    }).catch(() => {})
    publicConversations.delete(convId).catch(() => {})
    purgeConversationState(convId)
    setHistoryItems((prev) => prev.filter((i) => i.conversation_id !== convId))
    if (previewConversationId === convId) {
      setPreviewConversationId(null)
    }
    setMessages([])
    setError(null)
    stopAllTTS()
    stopRecording()
    setPhoneMode(false)
    setAwaitingUserSpeech(false)
    stopWebcamEmotion()
    stopEmotionPolling()
    setMessageQueue([])
    setQueuePaused(false)
  }

  function shouldStreamTTS(): boolean {
    if (!audioConfig?.tts?.button_visible || !audioConfig?.tts?.voice_id) return false
    if (phoneModeRef.current) return true
    if (audioConfig.tts.auto_read && audioConfig.tts.streaming_enabled) return true
    return false
  }

  async function fetchTTSSegment(text: string, voiceId: string): Promise<Blob> {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 10000)
    try {
      const res = await fetch(window.location.origin + getApiBase() + '/audio/tts/segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, voice_id: voiceId, segment_index: 0 }),
        signal: controller.signal,
      })
      if (!res.ok) throw new Error(`TTS segment failed: ${res.status}`)
      return res.blob()
    } finally {
      clearTimeout(timeout)
    }
  }

  function stopAllTTS() {
    if (ttsAudioRef.current) {
      ttsAudioRef.current.pause()
      ttsAudioRef.current = null
    }
    if (ttsPlayerRef.current) {
      ttsPlayerRef.current.abort()
      ttsPlayerRef.current = null
    }
    ttsBufferRef.current = ''
    setPlayingMsgIdx(null)
  }

  /* ------------------------------------------------------------------ */
  /*  Message queue (in-memory; drains automatically after each run)     */
  /* ------------------------------------------------------------------ */

  /* Clear `running` flags on the last assistant message's tool calls — after
     an abort/cancel mid-run, spinner badges would otherwise stick forever. */
  function settleLastAssistantTools() {
    setMessages((prev) => {
      const idx = [...prev.entries()].reverse().find(
        ([, m]) => m.role === 'assistant' && m.toolCalls?.some((t) => t.running),
      )?.[0]
      if (idx === undefined) return prev
      const next = [...prev]
      next[idx] = { ...next[idx], toolCalls: settleToolProgress(next[idx].toolCalls!) }
      return next
    })
  }

  function requestRunCancel() {
    const convId = sessionStorage.getItem(`agentix-public-chat-conversation-id:${activeModelRef.current}`)
    if (!convId) return
    fetch(`${getApiUrl()}/chat/runs/${encodeURIComponent(convId)}/cancel`, {
      method: 'POST',
      credentials: 'include',
      headers: pcHeaders(),
    }).catch(() => {})
    recoverUnconsumedInputs(convId)
  }

  /* After a cancel the run may still hold injected inputs that never reached
     an iteration — the live broadcast raced the aborted reader, so poll the
     run status briefly and requeue whatever the server stashed. */
  async function recoverUnconsumedInputs(convId: string) {
    for (let attempt = 0; attempt < 8; attempt++) {
      await new Promise((r) => setTimeout(r, 500))
      try {
        const st = await fetch(`${getApiUrl()}/chat/runs/${encodeURIComponent(convId)}`, {
          credentials: 'include',
          headers: pcHeaders(),
        })
        const info = await st.json().catch(() => null)
        if (!info) return
        const unconsumed = info.unconsumed_inputs as Array<{ text: string; images?: string[] }> | undefined
        if (unconsumed?.length) {
          requeueUnconsumed(unconsumed)
          return
        }
        if (info.done || !info.running) return
      } catch {
        return
      }
    }
  }

  function requeueUnconsumed(items: Array<string | { text: string; images?: string[] }>) {
    setMessageQueue((q) => {
      const queued = new Set(q.map((m) => m.text))
      const restored = filterUnconsumed(items, queued, messagesRef.current)
      if (!restored.length) return q
      return [
        ...restored.map((it) => ({ id: newQueueId(), text: it.text, images: it.images })),
        ...q,
      ]
    })
  }

  function newQueueId(): string {
    return (crypto as any).randomUUID ? (crypto as any).randomUUID() : `q-${Date.now()}-${Math.random()}`
  }

  function enqueueMessage(
    text: string,
    audioEmotions?: { audio_prosody?: EmotionContextPayload['audio_prosody'] | null },
    clearInput = true,
    images?: string[],
  ) {
    if (!text && !(images && images.length)) return
    setMessageQueue((q) => {
      if (q.length >= 20) return q
      return [...q, { id: newQueueId(), text, audioEmotions, images: images && images.length ? images : undefined }]
    })
    if (clearInput) {
      setInput('')
      if (textareaRef.current) textareaRef.current.style.height = 'auto'
    }
    setTimeout(() => textareaRef.current?.focus(), 0)
  }

  function sendQueuedNow(item: QueuedChatMessage) {
    setMessageQueue((q) => [item, ...q.filter((m) => m.id !== item.id)])
    setQueuePaused(false)
    abortRef.current?.abort()
    attachAbortRef.current?.abort()
    requestRunCancel()
    // The aborted send's finally clears chatLoading; the drain effect then
    // sends the item (now at the queue head) through the normal send path.
  }

  /* Inject a queued message into the currently running agent loop — it lands
     in the agent's context at the next iteration instead of waiting for the
     run to finish. Agent-only; the backend rejects other runs cleanly. */
  async function injectQueued(item: QueuedChatMessage) {
    const convId = sessionStorage.getItem(`agentix-public-chat-conversation-id:${activeModelRef.current}`)
    if (!convId) return
    try {
      const res = await fetch(`${getApiUrl()}/chat/runs/${encodeURIComponent(convId)}/input`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json', ...pcHeaders() },
        body: JSON.stringify({ message: item.text, images: item.images }),
      })
      const data = await res.json().catch(() => null)
      if (res.ok && data?.accepted) {
        setMessageQueue((q) => q.filter((m) => m.id !== item.id))
      } else {
        if (data?.reason === 'not_agent') setRunIsAgent(false)
        setError(t.queueInjectFailed + (data?.reason ? `: ${data.reason}` : ''))
      }
    } catch (e: any) {
      setError(`${t.queueInjectFailed}: ${String(e?.message || e)}`)
    }
  }

  function editQueued(item: QueuedChatMessage) {
    setMessageQueue((q) => q.filter((m) => m.id !== item.id))
    setInput(item.text)
    if (item.images && item.images.length) setPendingImages(item.images)
    setTimeout(() => textareaRef.current?.focus(), 0)
  }

  function deleteQueued(item: QueuedChatMessage) {
    setMessageQueue((q) => q.filter((m) => m.id !== item.id))
  }

  function reorderQueued(fromIndex: number, toIndex: number) {
    setMessageQueue((q) => {
      const next = [...q]
      const [moved] = next.splice(fromIndex, 1)
      next.splice(toIndex, 0, moved)
      return next
    })
  }

  const uploadsEnabled = uploadsCfg.images || uploadsCfg.files

  /* Upload a document attachment to the conversation */
  const uploadDoc = useCallback((file: File) => {
    const convId = getOrCreateConversationId(activeModelRef.current)
    const localId = `up-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    setPendingFiles(prev => [...prev, {
      localId, name: file.name, size: file.size,
      ext: `.${(file.name.split('.').pop() || '').toLowerCase()}`,
      status: 'uploading',
    }])
    uploadChatAttachment(getApiUrl(), convId, file, pcHeaders(), activeModelRef.current)
      .then((meta) => {
        setPendingFiles(prev => prev.map(f => f.localId === localId
          ? { ...f, attachmentId: meta.id, status: 'ready' } : f))
      })
      .catch((e: any) => {
        setPendingFiles(prev => prev.map(f => f.localId === localId
          ? { ...f, status: 'error', error: String(e?.message || e) } : f))
      })
  }, [])

  /* Handle file selection for image upload */
  const handleImageFiles = useCallback((files: FileList | File[]) => {
    const fileArr = Array.from(files).filter(isImageFile)
    if (fileArr.length === 0) return
    const readers = fileArr.map(file => {
      return new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result as string)
        reader.onerror = reject
        reader.readAsDataURL(file)
      })
    })
    Promise.all(readers).then((dataUris) => {
      setPendingImages(prev => [...prev, ...dataUris])
    }).catch(() => {})
  }, [])

  /* Handle file selection: images prefer the multimodal vision path when
     enabled; otherwise they go through document upload (OCR/vision text
     extraction) rather than being silently dropped. */
  const handleFiles = useCallback((files: FileList | File[]) => {
    const arr = Array.from(files)
    const imgs = arr.filter(isImageFile)
    if (uploadsCfg.images) {
      if (imgs.length > 0) handleImageFiles(imgs)
    } else if (uploadsCfg.files) {
      imgs.forEach(uploadDoc)
    }
    const docs = uploadsCfg.files ? arr.filter(isDocFile) : []
    docs.forEach(uploadDoc)
  }, [uploadsCfg, handleImageFiles, uploadDoc])

  const removePendingImage = useCallback((index: number) => {
    setPendingImages(prev => prev.filter((_, i) => i !== index))
  }, [])

  const removePendingFile = useCallback((localId: string) => {
    setPendingFiles(prev => {
      const item = prev.find(f => f.localId === localId)
      if (item?.attachmentId) {
        const convId = sessionStorage.getItem(`agentix-public-chat-conversation-id:${activeModelRef.current}`)
        if (convId) deleteChatAttachment(getApiUrl(), convId, item.attachmentId, pcHeaders())
      }
      return prev.filter(f => f.localId !== localId)
    })
  }, [])

  /* Drag & drop handlers */
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
    if (e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files)
    }
  }, [handleFiles])

  /* Open the agent-memory viewer for the current conversation */
  async function openMemory() {
    const convId = sessionStorage.getItem(`agentix-public-chat-conversation-id:${activeModelRef.current}`)
    if (!convId) return
    setMemoryOpen(true)
    setMemoryLoading(true)
    setMemoryError(null)
    try {
      const data = await publicConversations.memory(convId)
      setMemoryEntries(data?.memories || [])
    } catch (e: any) {
      setMemoryError(String(e?.message || e))
      setMemoryEntries([])
    } finally {
      setMemoryLoading(false)
    }
  }

  async function saveMemory() {
    const convId = activeConversationId
    if (!convId || !editingMemory) return
    try {
      await publicConversations.setMemory(convId, editingMemory.assistant_name, editingMemory.memory)
      setEditingMemory(null)
      await openMemory()
    } catch (e: any) {
      setMemoryError(String(e?.message || e))
    }
  }

  async function deleteMemoryEntry(assistantName: string) {
    const convId = activeConversationId
    if (!convId) return
    try {
      await publicConversations.deleteMemory(convId, assistantName)
      if (editingMemory?.assistant_name === assistantName) setEditingMemory(null)
      await openMemory()
    } catch (e: any) {
      setMemoryError(String(e?.message || e))
      setMemoryEntries([])
    } finally {
      setMemoryLoading(false)
    }
  }

  useEffect(() => {
    if (chatLoading || queuePaused || sendInFlightRef.current || messageQueue.length === 0) return
    const head = messageQueue[0]
    setMessageQueue((q) => q.slice(1))
    sendFromText(
      head.text,
      head.audioEmotions as { audio_prosody?: EmotionContextPayload['audio_prosody'] | null } | undefined,
      head.images,
    ).catch((err) => {
      if (err?.name === 'AbortError') return
      console.error('[PublicChat] queued message send failed:', err)
      setError(String(err?.message || err))
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatLoading, messageQueue, queuePaused])

  async function handleSend() {
    const hasReadyFiles = pendingFiles.some(f => f.status === 'ready')
    if (!input.trim() && pendingImages.length === 0 && !hasReadyFiles) return
    if (chatLoading) {
      enqueueMessage(input.trim(), undefined, true, [...pendingImages])
      setPendingImages([])
      setPendingFiles([])
      return
    }
    await sendFromText(input.trim())
  }

  async function sendFromText(
    userMsg: string,
    audioEmotions?: { audio_prosody?: EmotionContextPayload['audio_prosody'] | null },
    images?: string[],
  ) {
    if (chatLoading || sendInFlightRef.current) {
      // Enqueue instead of dropping/parallel-sending — covers both the UI
      // send path and STT transcripts arriving mid-run.
      enqueueMessage(userMsg, audioEmotions, false, images)
      return
    }
    setQueuePaused(false)
    sendInFlightRef.current = true
    setChatLoading(true)
    setRunIsAgent(true)
    const batcher = new StreamBatcher()
    setInput('')
    const imagesToSend = images ?? [...pendingImages]
    setPendingImages([])
    const attachmentsToSend: AttachmentMeta[] = pendingFiles
      .filter(f => f.status === 'ready')
      .map(f => ({ id: f.attachmentId, name: f.name, ext: f.ext }))
    setPendingFiles([])
    setError(null)
    if (textareaRef.current) textareaRef.current.style.height = 'auto'

    try {
      // Stop continuous emotion polling and flush the last batch
      stopEmotionPolling()
      const phases = buildEmotionPhases()
      const fused = computeOverallFused(phases)

      const newMessages: Message[] = [...messagesRef.current, { role: 'user', content: userMsg, attachments: attachmentsToSend.length ? attachmentsToSend : undefined, emotion: fused ? { fused_emotion: fused, phases } : undefined }]
      setMessages(newMessages)
      if (webcamActive) {
        startEmotionPolling()
      }

      // Capture pre-send emotion snapshot
      const preSendEmotion = await capturePreSendEmotion()
      const emotionPayload = buildEmotionPayload()
      if (emotionPayload && preSendEmotion) {
        emotionPayload.visual_pre_send = preSendEmotion
      }
      if (emotionPayload && audioEmotions?.audio_prosody) {
        emotionPayload.audio_prosody = audioEmotions.audio_prosody
        if (emotionPayload.consent) {
          emotionPayload.consent.audio_opt_in = true
        }
      }
      if (emotionPayload) {
        if (fused) emotionPayload.fused = fused
        if (phases.length > 0) {
          emotionPayload.phases = phases
        }
        // Extract best audio prosody from continuous audio snapshots
        console.info('[PublicChat] audioSnapshotsRef.length before send:', audioSnapshotsRef.current.length)
        if (audioSnapshotsRef.current.length > 0) {
          const bestAudio = audioSnapshotsRef.current.reduce((best, curr) =>
            curr.confidence > best.confidence ? curr : best
          )
          emotionPayload.audio_prosody = bestAudio
          console.info('[PublicChat] Setting audio_prosody from continuous snapshots:', bestAudio)
          audioSnapshotsRef.current = []
        }
      }

      const payload: any = {
        model: activeModelRef.current,
        messages: newMessages.map((m) => ({ role: m.role, content: m.content })),
        temperature: 0.7,
        stream: true,
        conversation_id: getOrCreateConversationId(activeModelRef.current),
      }
      if (imagesToSend.length > 0) {
        payload.images = imagesToSend
      }
      if (emotionPayload) {
        payload.emotion_context = emotionPayload
      }

      attachAbortRef.current?.abort()
      abortRef.current = new AbortController()
      const res = await fetch(`${getApiUrl()}/chat/completions`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json', ...pcHeaders() },
        body: JSON.stringify(payload),
        signal: abortRef.current.signal,
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      if (!res.body) throw new Error('No response body')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let assistantContent = ''
      let assistantReasoning = ''
      let hasReasoning = false
      let assistantToolCalls: ToolCallLive[] = []
      let assistantSubagents: SubagentEvent[] = []
      let assistantTodoList: AgentTodoSnapshot | undefined
      let observerEventsAcc: ObserverEvent[] = []
      let assistantRetrievalInfo: Message['retrievalInfo'] = undefined
      let assistantMetrics: ResponseMetrics | undefined
      let assistantLiveUsage: LiveUsage | undefined
      let backendEmotion: Message['emotion'] = null
      let terminalEventsAcc: ToolExecutionEvent[] = []

      const applyAccumulated = () => {
        setMessages((prev) => {
          const next = [...prev]
          const existing = next[next.length - 1]
          next[next.length - 1] = {
            role: 'assistant',
            content: assistantContent,
            reasoning: hasReasoning ? assistantReasoning : undefined,
            toolCalls: assistantToolCalls.length > 0 ? assistantToolCalls : undefined,
            subagents: assistantSubagents.length > 0 ? assistantSubagents : undefined,
            todoList: assistantTodoList,
            observerEvents: observerEventsAcc.length > 0 ? observerEventsAcc : existing?.observerEvents,
            terminalEvents: terminalEventsAcc.length > 0 ? terminalEventsAcc : existing?.terminalEvents,
            retrievalInfo: assistantRetrievalInfo !== undefined ? assistantRetrievalInfo : existing?.retrievalInfo,
            metrics: assistantMetrics ?? existing?.metrics,
            liveUsage: assistantLiveUsage ?? existing?.liveUsage,
          }
          return next
        })
      }

      const streamTTS = shouldStreamTTS()
      if (streamTTS) {
        ttsPlayerRef.current = new TTSSegmentPlayer()
      }
      const voiceId = audioConfig?.tts?.voice_id || null

      const pending: Message[] = [...newMessages, { role: 'assistant', content: '', reasoning: '' }]
      setMessages(pending)

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') break
            try {
              const chunk = JSON.parse(data)
              const content = chunk.choices?.[0]?.delta?.content
              const reasoning = chunk.choices?.[0]?.delta?.reasoning_content
              const toolExecution = chunk.choices?.[0]?.delta?.tool_execution
              const toolCalls = chunk.choices?.[0]?.delta?.tool_calls as ToolCallLive[] | undefined
              const toolProgress = chunk.choices?.[0]?.delta?.tool_progress as ToolProgressItem | undefined
              const subagent = chunk.choices?.[0]?.delta?.subagent as SubagentEvent | undefined
              const todoListDelta = chunk.choices?.[0]?.delta?.todo_list as AgentTodoSnapshot | undefined
              const emotion = chunk.choices?.[0]?.delta?.emotion
              const retrievalInfo = chunk.choices?.[0]?.delta?.retrieval_info as Message['retrievalInfo'] | undefined
              const metricsDelta = chunk.choices?.[0]?.delta?.metrics as ResponseMetrics | undefined
              const liveDelta = chunk.choices?.[0]?.delta?.live_usage as LiveUsage | undefined
              const injectedUserMessage = chunk.choices?.[0]?.delta?.injected_user_message as string | undefined
              const injectedSource = chunk.choices?.[0]?.delta?.injected_source as string | undefined
              const unconsumedInputs = chunk.choices?.[0]?.delta?.unconsumed_inputs as Array<string | { text: string; images?: string[] }> | undefined
              const observerDelta = chunk.choices?.[0]?.delta?.observer as ObserverEvent | undefined
              if (injectedUserMessage) {
                setMessages((prev) => insertUserBeforeLastAssistant(prev, injectedUserMessage, injectedSource))
              }
              if (unconsumedInputs?.length) requeueUnconsumed(unconsumedInputs)
              if (observerDelta) observerEventsAcc = [...observerEventsAcc, observerDelta]
              if (liveDelta) {
                assistantLiveUsage = liveDelta
                // live_usage-only chunks match no other branch — schedule
                // explicitly or the chip wouldn't re-render until the next
                // content/tool event arrives.
                batcher.schedule(applyAccumulated)
              }
              if (emotion) {
                backendEmotion = emotion
              }
              if (retrievalInfo) {
                assistantRetrievalInfo = retrievalInfo
                if (retrievalInfo.todo_list) assistantTodoList = retrievalInfo.todo_list
              }
              if (todoListDelta) assistantTodoList = todoListDelta
              if (metricsDelta) {
                assistantMetrics = metricsDelta
              }
              if (subagent) {
                assistantSubagents = upsertSubagentEvent(assistantSubagents, subagent)
                batcher.schedule(applyAccumulated)
              }
              if (content) {
                assistantContent += content
                // Phase transition: content arriving means assistant is typing
                if (!hasTransitionedToTypingRef.current) {
                  switchEmotionPhase('typing')
                  hasTransitionedToTypingRef.current = true
                }
                if (streamTTS && voiceId) {
                  ttsBufferRef.current += content
                  const { chunks, remainder } = extractCompleteSentences(ttsBufferRef.current)
                  ttsBufferRef.current = remainder
                  for (const chunk of chunks) {
                    const clean = cleanTextForTTS(chunk.text)
                    if (clean) {
                      ttsPlayerRef.current!.add(clean, voiceId, fetchTTSSegment, chunk.pauseAfterMs)
                    }
                  }
                }
                batcher.schedule(applyAccumulated)
              }
              if (reasoning) {
                assistantReasoning += reasoning
                hasReasoning = true
                batcher.schedule(applyAccumulated)
              }
              if (toolExecution) {
                terminalEventsAcc = [...terminalEventsAcc, toolExecution]
                batcher.schedule(applyAccumulated)
              }
              if (toolProgress) {
                assistantToolCalls = addToolProgress(assistantToolCalls, toolProgress)
                batcher.schedule(applyAccumulated)
              }
              if (toolCalls && toolCalls.length > 0) {
                for (const call of toolCalls) {
                  assistantToolCalls = resolveToolProgress(assistantToolCalls, call)
                  const idx = call.id ? assistantToolCalls.findIndex((item) => item.id === call.id) : -1
                  if (idx < 0) assistantToolCalls.push(call)
                  else assistantToolCalls[idx] = { ...assistantToolCalls[idx], ...call }
                }
                batcher.schedule(applyAccumulated)
              }
            } catch {
              // ignore malformed chunk
            }
          }
        }
      }

      // Flush remaining buffered text as a final segment
      if (streamTTS && voiceId && ttsBufferRef.current.trim()) {
        const clean = cleanTextForTTS(ttsBufferRef.current.trim())
        if (clean) {
          ttsPlayerRef.current!.add(clean, voiceId, fetchTTSSegment)
        }
        ttsBufferRef.current = ''
      }

      batcher.flushNow()
      const prevEmotion = newMessages[newMessages.length - 1].emotion
      const backendNormalized = normalizeEmotion(backendEmotion)
      const mergedEmotion: Message['emotion'] = backendNormalized
        ? { ...prevEmotion, ...backendNormalized, phases: backendNormalized.phases ?? prevEmotion?.phases }
        : prevEmotion
      const userMsgWithEmotion = { ...newMessages[newMessages.length - 1], emotion: mergedEmotion }
      // messagesRef carries any user messages injected mid-run (spliced before
      // the streaming placeholder) — rebuilding from it keeps them in place.
      // Fall back to newMessages if the ref hasn't synced the placeholder yet.
      const finalizedHistory =
        messagesRef.current.length > newMessages.length
          ? messagesRef.current.slice(0, -1)
          : newMessages.slice(0, -1)
      if (finalizedHistory.length > newMessages.length - 1) {
        finalizedHistory[newMessages.length - 1] = userMsgWithEmotion
      } else {
        finalizedHistory.push(userMsgWithEmotion)
      }
      const finalMessages: Message[] = [
        ...finalizedHistory,
        {
          role: 'assistant',
          content: assistantContent,
          reasoning: hasReasoning ? assistantReasoning : undefined,
          toolCalls: assistantToolCalls.length > 0 ? settleToolProgress(assistantToolCalls) : undefined,
          subagents: assistantSubagents.length > 0 ? assistantSubagents : undefined,
          todoList: assistantTodoList,
          observerEvents: observerEventsAcc.length > 0 ? observerEventsAcc : undefined,
          retrievalInfo: assistantRetrievalInfo,
          metrics: assistantMetrics,
        },
      ]
      setMessages(finalMessages)
      switchEmotionPhase('reading')

      // In phone mode: speak last assistant message, then start listening
      if (phoneModeRef.current && assistantContent.trim()) {
        const lastIdx = finalMessages.length - 1
        lastAssistantIdxRef.current = lastIdx
        if (!ttsPlayerRef.current?.hasPlayedAny()) {
          setTimeout(() => handleSpeak(assistantContent.trim(), lastIdx), 200)
        } else {
          const waitForPlayer = () => {
            if (ttsPlayerRef.current && ttsPlayerRef.current.queueLength > 0) {
              setTimeout(waitForPlayer, 200)
            } else {
              setAwaitingUserSpeech(true)
              setTimeout(() => startRecording(), 300)
            }
          }
          setTimeout(waitForPlayer, 200)
        }
      }
    } catch (err: any) {
      batcher.flushNow()
      stopAllTTS()
      settleLastAssistantTools()
      if (err.name !== 'AbortError') {
        setError(err.message || 'An error occurred')
      }
    } finally {
      sendInFlightRef.current = false
      setChatLoading(false)
      abortRef.current = null
      publicConversations.invalidate()
      if (!phoneModeRef.current) {
        setTimeout(() => textareaRef.current?.focus(), 0)
      }
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function handleStop() {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
    attachAbortRef.current?.abort()
    requestRunCancel()
    setQueuePaused(true)
  }

  function handleCopyMessage(text: string, idx: number) {
    navigator.clipboard.writeText(text)
    copy(`msg-${idx}`)
  }

  async function handleSpeak(text: string, idx: number) {
    if (!audioConfig?.tts?.button_visible || !audioConfig?.tts?.voice_id) return
    if (playingMsgIdx === idx) {
      stopAllTTS()
      return
    }
    stopAllTTS()
    const cleanText = cleanTextForTTS(text)
    if (!cleanText) return
    setPlayingMsgIdx(idx)
    try {
      // Use sentence-level streaming for faster playback start
      ttsPlayerRef.current = new TTSSegmentPlayer()
      const { chunks } = extractCompleteSentences(cleanText)
      for (const chunk of chunks) {
        ttsPlayerRef.current!.add(chunk.text, audioConfig.tts.voice_id!, fetchTTSSegment, chunk.pauseAfterMs)
      }

      // Watch for playback completion
      const watchPlayback = () => {
        if (ttsPlayerRef.current && ttsPlayerRef.current.queueLength > 0) {
          setTimeout(watchPlayback, 200)
        } else {
          setPlayingMsgIdx((current) => (current === idx ? null : current))
          ttsPlayerRef.current = null
          if (phoneModeRef.current && lastAssistantIdxRef.current === idx) {
            setAwaitingUserSpeech(true)
            setTimeout(() => startRecording(), 300)
          }
        }
      }
      setTimeout(watchPlayback, 200)
    } catch {
      setPlayingMsgIdx((current) => (current === idx ? null : current))
      ttsPlayerRef.current = null
    }
  }

  async function startRecording() {
    if (!audioConfig?.stt?.button_visible || isRecording) return
    // If emotion button is not active but audio emotion is enabled,
    // start a temporary continuous audio buffer for this STT session only
    if (!webcamActive && emotionConfig?.audio?.enabled && !audioSnippetBufferRef.current?.isActive()) {
      console.info('[PublicChat] startRecording: auto-starting temporary audio emotion buffer')
      await startContinuousAudioEmotion().catch(() => {})
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mimeType = MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : MediaRecorder.isTypeSupported('audio/ogg')
          ? 'audio/ogg'
          : ''
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      mediaRecorderRef.current = recorder
      audioChunksRef.current = []
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data)
      }
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        setIsRecording(false)
        setAwaitingUserSpeech(false)

        // Capture audio prosody emotion from the continuous buffer (if running)
        let audioEmotion: EmotionContextPayload['audio_prosody'] | null = null
        console.info('[PublicChat] STT stop: audioSnippetBuffer active=', audioSnippetBufferRef.current?.isActive())
        if (audioSnippetBufferRef.current?.isActive()) {
          audioEmotion = await audioSnippetBufferRef.current.captureSnapshot()
          console.info('[PublicChat] STT captureSnapshot result:', audioEmotion)
        }

        // Stop the temporary audio emotion stream if emotion button is not active
        if (!webcamActive) {
          stopContinuousAudioEmotion()
        }

        if (audioChunksRef.current.length === 0) return
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType || 'audio/webm' })
        try {
          const wavBlob = await blobToWav(audioBlob)
          const formData = new FormData()
          formData.append('audio', wavBlob, 'recording.wav')
          if (audioConfig?.stt?.language) {
            formData.append('language', audioConfig.stt.language)
          }
          const res = await fetch(window.location.origin + getApiBase() + '/audio/stt', {
            method: 'POST',
            body: formData,
          })
          if (!res.ok) {
            if (res.status === 400) {
              setSttToast('Ich konnte dich leider nicht verstehen.')
              setTimeout(() => setSttToast((current) => (current ? null : current)), 2800)
            }
            throw new Error(await res.text())
          }
          const data = await res.json()
          const transcript = (data.text || '').trim()

          const hasAudioEmotions = audioEmotion
          if (transcript.length >= 2) {
            setInput(transcript)
          }
          if (transcript.length >= 2 || hasAudioEmotions) {
            phoneEmptyRetriesRef.current = 0
            // Auto-send after transcription (always send if audio emotions exist)
            setTimeout(() => {
              setInput('')
              sendFromText(transcript, { audio_prosody: audioEmotion })
            }, 200)
          } else if (phoneModeRef.current) {
            phoneEmptyRetriesRef.current++
            if (phoneEmptyRetriesRef.current < 3) {
              setAwaitingUserSpeech(true)
              setTimeout(() => startRecording(), 400)
            } else {
              phoneEmptyRetriesRef.current = 0
              setAwaitingUserSpeech(false)
            }
          }
        } catch (err: any) {
          console.error('STT error:', err)
        }
      }

      recorder.start()
      setIsRecording(true)
      setAwaitingUserSpeech(false)

      // Track emotion during voice input as a distinct phase
      if (emotionTimelineRef.current) {
        switchEmotionPhase('speaking')
      } else if (webcamActive || (phoneModeRef.current && audioSnippetBufferRef.current?.isActive())) {
        startEmotionPolling('speaking')
      }

      /* ---- Voice Activity Detection (VAD) for phone mode ---- */
      if (phoneModeRef.current) {
        const audioCtx = new AudioContext()
        const highpass = audioCtx.createBiquadFilter()
        highpass.type = 'highpass'
        highpass.frequency.value = 80
        const analyser = audioCtx.createAnalyser()
        analyser.fftSize = 512
        const source = audioCtx.createMediaStreamSource(stream)
        source.connect(highpass)
        highpass.connect(analyser)
        audioContextRef.current = audioCtx
        analyserRef.current = analyser
        vadSilenceStartRef.current = performance.now()
        vadSpeechDetectedRef.current = false
        recordingStartTimeRef.current = performance.now()

        const floatArray = new Float32Array(analyser.frequencyBinCount)
        function checkSilence() {
          if (!analyserRef.current || !mediaRecorderRef.current) return
          analyser.getFloatTimeDomainData(floatArray)
          const filtered = applyHighpassFilter(floatArray, audioCtx.sampleRate, 80)
          let sum = 0
          for (let i = 0; i < filtered.length; i++) {
            sum += filtered[i] * filtered[i]
          }
          const rms = Math.sqrt(sum / filtered.length)
          const now = performance.now()
          if (rms >= 0.02) {
            vadSpeechDetectedRef.current = true
            vadSilenceStartRef.current = now
          }
          if (vadSpeechDetectedRef.current && now - recordingStartTimeRef.current > 2000 && rms < 0.02) {
            if (now - vadSilenceStartRef.current > 1500) {
              stopRecording()
              return
            }
          }
          vadRafRef.current = requestAnimationFrame(checkSilence)
        }
        vadRafRef.current = requestAnimationFrame(checkSilence)
      }

      const timeoutRecorder = recorder
      setTimeout(() => {
        if (timeoutRecorder.state === 'recording') {
          timeoutRecorder.stop()
        }
      }, 30000)
    } catch (err: any) {
      setNotice({
        danger: true,
        message:
          'Mikrofon-Zugriff verweigert.\n\n' +
          'Bitte prüfe:\n' +
          '1. Browser-Adressleiste → 🔒 → Berechtigungen → Mikrofon → Erlauben\n' +
          '2. macOS → Systemeinstellungen → Datenschutz & Sicherheit → Mikrofon → Dein Browser muss aktiviert sein\n' +
          '3. Lade die Seite danach neu (F5).',
      })
    }
  }

  function stopRecording() {
    if (vadRafRef.current) {
      cancelAnimationFrame(vadRafRef.current)
      vadRafRef.current = null
    }
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }
    analyserRef.current = null
    vadSpeechDetectedRef.current = false
    recordingStartTimeRef.current = 0
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current.stream?.getTracks().forEach((t) => t.stop())
    }
    setIsRecording(false)
  }

  function toggleRecording() {
    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }

  function togglePhoneMode() {
    const next = !phoneMode
    setPhoneMode(next)
    if (next) {
      phoneEmptyRetriesRef.current = 0
      if (!isRecording) {
        setTimeout(() => startRecording(), 100)
      }
    } else {
      stopRecording()
      stopAllTTS()
      if (!webcamActive) {
        stopEmotionPolling()
        stopContinuousAudioEmotion()
      }
      setAwaitingUserSpeech(false)
      phoneEmptyRetriesRef.current = 0
    }
  }

  /* ------------------------------------------------------------------ */
  /*  Auth & History helpers                                            */
  /* ------------------------------------------------------------------ */

  async function handleLogin(username: string, password: string) {
    setAuthLoading(true)
    setAuthError(null)
    try {
      const data = await publicAuth.login(username, password)
      if (data.success && data.user) {
        setChatUser(data.user)
        setAuthModalOpen(false)
        // Refresh config — role-filtered assistants AND per-caller capability
        // flags (uploads.files becomes available once authenticated).
        invalidatePublicChatConfigCache()
        const cfg = await fetchPublicChatConfig()
        if (cfg && cfg.enabled !== false) {
          applyPublicChatConfig(cfg)
        } else if (cfg && cfg.enabled === false) {
          setConfigError('This chat is currently disabled.')
        }
      }
    } catch (e: any) {
      setAuthError(e.message || 'Login failed')
    } finally {
      setAuthLoading(false)
    }
  }

  async function handleRegister(username: string, password: string, display_name: string, email?: string) {
    setAuthLoading(true)
    setAuthError(null)
    try {
      await publicAuth.register(username, password, display_name, email)
      await handleLogin(username, password)
    } catch (e: any) {
      setAuthError(e.message || 'Registration failed')
    } finally {
      setAuthLoading(false)
    }
  }

  function openPwdModal() {
    setPwdForm({ current_password: '', new_password: '', confirm_password: '' })
    setPwdError('')
    setPwdSuccess(false)
    setPwdModalOpen(true)
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
      await publicAuth.changePassword(pwdForm.current_password, pwdForm.new_password)
      setPwdSuccess(true)
      setPwdForm({ current_password: '', new_password: '', confirm_password: '' })
      setTimeout(() => setPwdModalOpen(false), 1500)
    } catch (err: any) {
      setPwdError(err?.message || (language === 'de' ? 'Passwort konnte nicht geändert werden' : 'Failed to change password'))
    } finally {
      setPwdLoading(false)
    }
  }

  async function handleLogout() {
    try {
      await publicAuth.logout()
      setChatUser(null)
      setHistoryOpen(false)
      // Reset to anonymous config — uploads.files drops back to false and
      // role-filtered assistants are re-resolved for the anonymous role.
      invalidatePublicChatConfigCache()
      const cfg = await fetchPublicChatConfig()
      if (cfg && cfg.enabled !== false) {
        applyPublicChatConfig(cfg)
      } else if (cfg && cfg.enabled === false) {
        setConfigError('This chat is currently disabled.')
      }
    } catch {
      setChatUser(null)
    }
  }

  async function fetchHistory() {
    if (!chatUser) return
    setHistoryLoading(true)
    setHistoryError(null)
    try {
      const data = await publicConversations.list()
      setHistoryItems(data.items)
    } catch (e: any) {
      setHistoryError(e.message || 'Failed to load history')
    } finally {
      setHistoryLoading(false)
    }
  }

  function normalizeServerMessages(data: { messages: any[] }): Message[] {
    const loadedMessages: Message[] = []
    for (const msg of data.messages) {
      if (msg.user_content) {
        loadedMessages.push({
          role: 'user',
          content: msg.user_content,
          emotion: normalizeEmotion(msg.emotion),
        })
      }
      if (msg.assistant_content) {
        const toolCalls = msg.tool_calls
          ? msg.tool_calls.map((tc: any) => ({
              id: tc.id,
              tool_name: tc.tool_name,
              args: tc.args,
              ok: tc.ok,
              error: tc.error || undefined,
              duration_ms: tc.duration_ms || undefined,
              output: tc.output || undefined,
              kind: tc.kind,
              metadata: tc.metadata as ToolCallLive['metadata'],
            }))
          : undefined
        const info = msg.retrieval_info as Message['retrievalInfo'] | null
        loadedMessages.push({
          role: 'assistant',
          content: msg.assistant_content,
          reasoning: msg.assistant_reasoning || undefined,
          toolCalls,
          retrievalInfo: info || undefined,
          todoList: info?.todo_list,
          observerEvents: (msg.retrieval_info as any)?._agent_trace?.observer_events,
          metrics: metricsFromRetrievalInfo(msg.retrieval_info),
        })
        // Attachment meta lives on the assistant turn's retrieval_info — map
        // it back onto the preceding user message so chips survive a reload.
        const atts = info?.attachments
        if (atts?.length) {
          for (let i = loadedMessages.length - 2; i >= 0; i--) {
            if (loadedMessages[i].role === 'user') {
              loadedMessages[i] = { ...loadedMessages[i], attachments: atts }
              break
            }
          }
        }
      }
    }
    // Cap bulky fields (tool outputs, subagent payloads, rag_context) before
    // they reach React state — server history can carry MBs of raw JSONB.
    return loadedMessages.map(slimMessageForStorage)
  }

  async function loadPreview(conversationId: string) {
    setPreviewLoading(true)
    setPreviewError(null)
    try {
      const data = await publicConversations.messages(conversationId)
      setPreviewMessages(normalizeServerMessages(data))
      setPreviewConversationId(conversationId)
    } catch (e: any) {
      setPreviewError(e.message || 'Failed to load preview')
    } finally {
      setPreviewLoading(false)
    }
  }

  async function loadConversation(conversationId: string) {
    setHistoryOpen(false)
    setPreviewConversationId(null)
    setMessageQueue([])
    setQueuePaused(false)
    setChatLoading(true)
    try {
      const data = await publicConversations.messages(conversationId)
      const loadedMessages = normalizeServerMessages(data)
      setMessages(loadedMessages)
      // Update model from first message if available
      const firstMsg = data.messages[0]
      const effectiveModel =
        firstMsg?.model_name && assistants.includes(firstMsg.model_name)
          ? firstMsg.model_name
          : activeModelRef.current
      if (firstMsg?.model_name && assistants.includes(firstMsg.model_name)) {
        setModel(firstMsg.model_name)
        activeModelRef.current = firstMsg.model_name
      }
      // Bind this model's conversation slot to the loaded conversation —
      // otherwise the next send would land in a different (or brand new)
      // conversation than the history just displayed.
      try {
        sessionStorage.setItem(`agentix-public-chat-conversation-id:${effectiveModel}`, conversationId)
      } catch {}
      attachToRunIfActive(conversationId)
    } catch (e: any) {
      setError(e.message || 'Failed to load conversation')
    } finally {
      setChatLoading(false)
    }
  }

  /* Remove every client-side trace of a conversation: the persisted message
     cache plus every model→conversation-id binding that still points at it.
     Without this a deleted conversation resurrects on reload — the stored id
     is reused and the cached messages are rendered again (and the next send
     would even recreate the server link via ensure_exists). */
  function purgeConversationState(conversationId: string) {
    try {
      sessionStorage.removeItem(chatMessagesKey(conversationId))
      const prefix = 'agentix-public-chat-conversation-id:'
      const staleKeys: string[] = []
      for (let i = 0; i < sessionStorage.length; i++) {
        const key = sessionStorage.key(i)
        if (key && key.startsWith(prefix) && sessionStorage.getItem(key) === conversationId) {
          staleKeys.push(key)
        }
      }
      staleKeys.forEach((k) => sessionStorage.removeItem(k))
    } catch {}
  }

  async function deleteConversation(conversationId: string) {
    try {
      // Cancel a still-active run first — its finalize step would otherwise
      // recreate the conversation link right after the delete.
      await fetch(`${getApiUrl()}/chat/runs/${encodeURIComponent(conversationId)}/cancel`, {
        method: 'POST',
        credentials: 'include',
        headers: pcHeaders(),
      }).catch(() => {})
      await publicConversations.delete(conversationId)
      purgeConversationState(conversationId)
      setHistoryItems((prev) => prev.filter((i) => i.conversation_id !== conversationId))
      if (previewConversationId === conversationId) {
        setPreviewConversationId(null)
      }
      const activeConvId = sessionStorage.getItem(
        `agentix-public-chat-conversation-id:${activeModelRef.current}`
      )
      if (!activeConvId || activeConvId === conversationId) {
        setMessages([])
      }
    } catch (e: any) {
      setHistoryError(e.message || 'Failed to delete')
    }
  }

  async function renameConversation(conversationId: string, title: string) {
    try {
      await publicConversations.updateTitle(conversationId, title)
      setHistoryItems((prev) =>
        prev.map((i) => (i.conversation_id === conversationId ? { ...i, title } : i))
      )
      setEditingTitleId(null)
    } catch (e: any) {
      setHistoryError(e.message || 'Failed to rename')
    }
  }

  /* ------------------------------------------------------------------ */

  return (
    <div className="chat-page public-chat-page" data-theme={theme}>
      {/* Header */}
      <div className={`chat-page-header${headerCollapsed ? ' header-collapsed' : ''}`}>
        <div className="chat-header-brand" style={{ flex: 1, display: 'flex', alignItems: 'center' }}>
          {configLoaded && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              {logoUrl && <img src={logoUrl} alt={t.title || 'Logo'} style={{ height: 32, borderRadius: 4, display: 'block' }} />}
              <div className="chat-header-title">
                <h2 style={{ margin: 0 }}>{title}</h2>
                {subtitle && <p style={{ margin: 0, fontSize: 13, color: 'var(--text-secondary)' }}>{subtitle}</p>}
              </div>
            </div>
          )}
        </div>
        <select
          className="chat-model-select"
          value={model}
          onChange={(e) => handleModelChange(e.target.value)}
          disabled={loadingModels || chatLoading}
          style={{ margin: '0 auto' }}
        >
          {assistants.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <div className="chat-page-controls" style={{ flex: 1, justifyContent: 'flex-end' }}>
          {wsEnabled && (
            <button
              className="chat-history-header-btn"
              type="button"
              title="Workspace"
              onClick={() => setWsOpen((v) => !v)}
            >
              <Folder size={16} />
            </button>
          )}
          {chatUser && memoryEnabled[model] && (
            <button
              className="chat-icon-btn"
              onClick={openMemory}
              title={t.agentMemory}
            >
              <Brain size={16} />
            </button>
          )}
          {chatUser && (
            <button
              className="chat-history-header-btn"
              onClick={() => { setHistoryOpen(true); fetchHistory() }}
              title={t.history}
            >
              <History size={16} />
            </button>
          )}
          <button className="chat-reset-btn" onClick={handleReset} title={t.reset}>
            <RotateCcw size={16} />
          </button>
          <div className="chat-header-menu" ref={headerMenuRef}>
            <button
              className="chat-icon-btn"
              onClick={() => setHeaderMenuOpen((v) => !v)}
              title={language === 'de' ? 'Menü' : 'Menu'}
              aria-expanded={headerMenuOpen}
            >
              <EllipsisVertical size={16} />
            </button>
            {headerMenuOpen && (
              <div className="chat-header-menu-panel" role="menu">
                {chatUser && (
                  <div className="chat-menu-user">{chatUser.display_name || chatUser.username}</div>
                )}
                {cfgLanguage === 'auto' && (
                  <div className="chat-menu-langrow" role="group" aria-label="Language">
                    <button
                      type="button"
                      className={language === 'de' ? 'active' : ''}
                      onClick={() => setLanguage('de')}
                    >
                      DE
                    </button>
                    <button
                      type="button"
                      className={language === 'en' ? 'active' : ''}
                      onClick={() => setLanguage('en')}
                    >
                      EN
                    </button>
                  </div>
                )}
                <button
                  className="chat-menu-item"
                  onClick={() => { setTheme(theme === 'dark' ? 'light' : 'dark'); setHeaderMenuOpen(false) }}
                >
                  {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
                  <span>{theme === 'dark' ? t.lightMode : t.darkMode}</span>
                </button>
                {chatUser && (
                  <button
                    className="chat-menu-item"
                    onClick={() => { setHeaderMenuOpen(false); openPwdModal() }}
                  >
                    <KeyRound size={14} />
                    <span>{language === 'de' ? 'Passwort ändern' : 'Change password'}</span>
                  </button>
                )}
                {chatUser ? (
                  <button
                    className="chat-menu-item"
                    onClick={() => { setHeaderMenuOpen(false); handleLogout() }}
                  >
                    <LogOut size={14} />
                    <span>{t.logout}</span>
                  </button>
                ) : (
                  <button
                    className="chat-menu-item"
                    onClick={() => { setHeaderMenuOpen(false); setAuthModalOpen(true); setAuthTab('login'); setAuthError(null) }}
                  >
                    <LogIn size={14} />
                    <span>{t.login}</span>
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Config error */}
      {configError && (
        <div className="error-box" style={{ margin: '0 24px 16px', flexShrink: 0 }}>
          {configError}
        </div>
      )}

      {/* Messages */}
      <ChatLanguageContext.Provider value={language}>
      <HtmlPreviewContext.Provider value={{ open: setPreviewHtml, close: () => setPreviewHtml(null) }}>
        <div className="chat-messages-scroll" ref={messagesContainerRef}>
          <div className="chat-messages-col">
            {messages.length === 0 && !chatLoading && (
              <div className="chat-empty-state">
                <div className="chat-empty-icon">
                  <Bot size={28} />
                </div>
                <h3>{title}</h3>
                {model && assistantDescriptions[model] && (
                  <p className="chat-empty-desc">
                    <strong>{model}</strong> — {assistantDescriptions[model]}
                  </p>
                )}
                <p>{t.emptyStateText}</p>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`chat-message-row ${msg.role}${msg.systemNote ? ' system-note' : ''}`}>
                <div className={`chat-avatar ${msg.role === 'assistant' ? 'assistant-avatar' : 'user-avatar'}`}>
                  {msg.role === 'assistant' ? <Bot size={16} /> : <User size={16} />}
                </div>
                <div className="chat-message-content">
                  {/* Tool calls */}
                  {msg.role === 'assistant' && (
                    <SubagentTagsRow
                      events={msg.subagents || []}
                      toolCalls={msg.toolCalls || []}
                      language={language}
                      onSelect={setSelectedSubagent}
                    />
                  )}
                  {msg.role === 'assistant' && (
                    <AgentTodoPanel snapshot={msg.todoList || msg.retrievalInfo?.todo_list} language={language} />
                  )}
                  {msg.role === 'assistant' && (
                    <ObserverTag events={msg.observerEvents} language={language} />
                  )}
                  {/* Tool calls — flat fallback row only without reasoning */}
                  {msg.role === 'assistant' && !msg.reasoning && msg.toolCalls && msg.toolCalls.length > 0 && (
                    <div className="chat-toolcalls-row">
                      {msg.toolCalls.filter((tc) => !INTERNAL_TOOL_NAMES.has(tc.tool_name)).map((tc, tidx) => (
                        <ToolCallBadge key={tidx} tool={tc} onClick={() => setSelectedToolCall(tc)} />
                      ))}
                    </div>
                  )}
                  {/* RAG context */}
                  {msg.role === 'assistant' && msg.retrievalInfo && (msg.retrievalInfo.status === 'error' || (msg.retrievalInfo.status === 'ok' && msg.retrievalInfo.chunks_retrieved > 0)) && (
                    <div className="chat-rag-badge-row">
                      <RAGBadge info={msg.retrievalInfo} onClick={() => setSelectedRAGInfo(msg.retrievalInfo)} />
                    </div>
                  )}

                  {/* Reasoning */}
                  {msg.role === 'assistant' && msg.reasoning && (
                    <ReasoningPanel
                      reasoning={msg.reasoning}
                      toolCalls={msg.toolCalls}
                      isThinking={msg.content === ''}
                      onToolCallClick={setSelectedToolCall}
                    />
                  )}

                  {/* Terminal per message */}
                  {msg.role === 'assistant' && msg.terminalEvents && msg.terminalEvents.length > 0 && (
                    <TerminalBox events={msg.terminalEvents} />
                  )}

                  {/* Aggregated live tool usage (root + subagent tree) */}
                  {msg.role === 'assistant' && (
                    <ToolUsageBadges toolCalls={msg.toolCalls} subagents={msg.subagents} />
                  )}

                  {/* Message bubble */}
                  <div className="chat-bubble">
                    {msg.role === 'assistant' && msg.content === '' && !msg.reasoning ? (
                      <TypingIndicator />
                    ) : (
                      <MarkdownContent text={msg.content} />
                    )}
                    {msg.role === 'user' && (
                      <AttachmentChips
                        attachments={msg.attachments}
                        apiUrl={getApiUrl()}
                        conversationId={activeConversationId}
                        headers={pcHeaders()}
                      />
                    )}
                  </div>

                  {/* Generated document artifacts — below the answer bubble */}
                  {msg.role === 'assistant' && collectArtifactFiles(msg.toolCalls).length > 0 && (
                    <div className="chat-artifact-row">
                      {collectArtifactFiles(msg.toolCalls).map((f) => (
                        <ArtifactCard
                          key={`${f.conversation_id}/${f.name}`}
                          file={f}
                          apiUrl={getApiUrl()}
                          accessToken={publicAccessToken}
                          labels={{
                            download: t.artifactDownload,
                            preview: t.artifactPreview,
                            hidePreview: t.artifactHidePreview,
                            downloadFailed: t.artifactDownloadFailed,
                            previewFailed: t.artifactPreviewFailed,
                            exportAs: t.artifactExportAs,
                            pageOf: (c, n) => t.artifactPageOf.replace('{current}', String(c)).replace('{total}', String(n)),
                          }}
                        />
                      ))}
                    </div>
                  )}
                  <div className="chat-message-meta">
                    {msg.role === 'user' && webcamActive && msg.emotion?.fused_emotion && (
                      <span
                        className={`chat-message-emotion-tooltip-wrapper ${openEmotionIdx === idx ? 'open' : ''}`}
                        onClick={(e) => {
                          e.stopPropagation()
                          setOpenEmotionIdx(openEmotionIdx === idx ? null : idx)
                        }}
                      >
                        <span
                          className="chat-message-emotion"
                          title={`${t.text}: ${msg.emotion.fused_emotion.source}, ${t.confidence}: ${msg.emotion.fused_emotion.confidence}`}
                        >
                          <Smile size={10} />
                          {msg.emotion.fused_emotion.primary_emotion} ({msg.emotion.fused_emotion.intensity})
                        </span>
                        <div className="chat-message-emotion-tooltip" onClick={(e) => e.stopPropagation()}>
                          <div className="emotion-tooltip-header">{t.emotionHistory}</div>
                          {msg.emotion.text_emotion && (
                            <div className="emotion-tooltip-row">
                              <span>{t.text}:</span>
                              <span>{msg.emotion.text_emotion.primary_emotion} ({msg.emotion.text_emotion.confidence})</span>
                            </div>
                          )}
                          {msg.emotion.visual_pre_send && (
                            <div className="emotion-tooltip-row">
                              <span>{t.webcam}:</span>
                              <span>{msg.emotion.visual_pre_send.primary_emotion} ({msg.emotion.visual_pre_send.confidence})</span>
                            </div>
                          )}
                          {msg.emotion.audio_text_emotion && (
                            <div className="emotion-tooltip-row">
                              <span>{t.audioText}:</span>
                              <span>{msg.emotion.audio_text_emotion.primary_emotion} ({msg.emotion.audio_text_emotion.confidence})</span>
                            </div>
                          )}
                          {msg.emotion.audio_prosody && (
                            <div className="emotion-tooltip-row">
                              <span>{t.audioProsody}:</span>
                              <span>{msg.emotion.audio_prosody.primary_emotion} ({msg.emotion.audio_prosody.confidence})</span>
                            </div>
                          )}
                          {msg.emotion.phases && msg.emotion.phases.length > 0 && (
                            <>
                              <div className="emotion-tooltip-header" style={{ marginTop: 6, borderTop: '1px solid var(--border)', paddingTop: 4 }}>
                                {t.phases}
                              </div>
                              {msg.emotion.phases.map((p) => {
                                const phaseLabels: Record<string, string> = {
                                  waiting_for_response: t.phaseWaiting,
                                  reading: t.phaseReading,
                                  typing: t.phaseTyping,
                                  speaking: t.phaseSpeaking,
                                }
                                return (
                                  <div key={p.phase} className="emotion-tooltip-row">
                                    <span>{phaseLabels[p.phase] || p.phase}:</span>
                                    <span>{p.primary_emotion} ({p.confidence}, {p.intensity})</span>
                                  </div>
                                )
                              })}
                            </>
                          )}
                          <div className="emotion-tooltip-row fused">
                            <span>{t.fused}:</span>
                            <span>{msg.emotion.fused_emotion.primary_emotion} ({msg.emotion.fused_emotion.confidence}, {msg.emotion.fused_emotion.intensity})</span>
                          </div>
                        </div>
                      </span>
                    )}
                    {msg.role === 'assistant' && (
                      <button
                        className={`chat-message-copy ${playingMsgIdx === idx ? 'speaking' : ''}`}
                        onClick={() => handleSpeak(msg.content, idx)}
                        title={playingMsgIdx === idx ? t.stop : t.speak}
                      >
                        <Volume2 size={14} />
                      </button>
                    )}
                    <button
                      className="chat-message-copy"
                      onClick={() => handleCopyMessage(msg.content, idx)}
                      title={t.copy}
                    >
                      {copiedId === `msg-${idx}` ? <Check size={14} /> : <Copy size={14} />}
                    </button>
                    {msg.role === 'assistant' && (msg.metrics ? (
                      <ResponseMetricsBadge metrics={msg.metrics} ragContextTokens={msg.retrievalInfo?.rag_context_tokens} lang={language} />
                    ) : (
                      chatLoading && idx === messages.length - 1 && (
                        <LiveTokenChip
                          live={msg.liveUsage}
                          streamedChars={(msg.content?.length ?? 0) + (msg.reasoning?.length ?? 0)}
                          lang={language}
                        />
                      )
                    ))}
                  </div>
                </div>
              </div>
            ))}

            {chatLoading && messages.length > 0 && messages[messages.length - 1].role === 'user' && (
              <div className="chat-message-row assistant">
                <div className="chat-avatar assistant-avatar">
                  <Bot size={16} />
                </div>
                <div className="chat-message-content">
                  <div className="chat-bubble">
                    <TypingIndicator />
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="error-box" style={{ margin: '8px 0' }}>
                {error}
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>
      </HtmlPreviewContext.Provider>
      </ChatLanguageContext.Provider>

      <HtmlPreviewModal html={previewHtml} onClose={() => setPreviewHtml(null)} />
      <ToolCallModal tool={selectedToolCall} onClose={() => setSelectedToolCall(null)} />
      <SubagentModal
        agent={selectedSubagent}
        language={language}
        events={[...messages, ...(previewMessages || [])].flatMap((message) => message.subagents || [])}
        toolCalls={[...messages, ...(previewMessages || [])].flatMap((message) => message.toolCalls || [])}
        conversationId={activeConversationId || sessionStorage.getItem(`agentix-public-chat-conversation-id:${activeModelRef.current}`) || undefined}
        apiUrl={getApiUrl()}
        requestHeaders={pcHeaders()}
        onClose={() => setSelectedSubagent(null)}
      />
      <RAGModal info={selectedRAGInfo} onClose={() => setSelectedRAGInfo(undefined)} />

      {/* Input */}
      <div className="chat-input-bar">
        <div
          className={`chat-input-inner${phoneMode ? ' phone-mode' : ' chat-input-dynamic'}`} style={{ position: 'relative' }}
          {...(uploadsEnabled ? { onDragOver: handleDragOver, onDragLeave: handleDragLeave, onDrop: handleDrop } : {})}
        >
          {uploadsEnabled && dragOver && (
            <div
              style={{ position: 'absolute', inset: 0, background: 'rgba(0,122,255,0.1)', border: '2px dashed rgba(0,122,255,0.5)', borderRadius: 'inherit', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10, pointerEvents: 'none' }}
            >
              <span style={{ color: 'rgba(0,122,255,0.8)', fontSize: '14px' }}>{uploadsCfg.files ? t.dropFiles : t.dropImages}</span>
            </div>
          )}
          <UploadTray
            pendingImages={uploadsCfg.images ? pendingImages : []}
            pendingFiles={pendingFiles}
            onRemoveImage={removePendingImage}
            onRemoveFile={removePendingFile}
            labels={{ uploading: t.uploading, failed: t.uploadFailed }}
          />
          {uploadsEnabled && (
            <input
              ref={fileInputRef}
              type="file"
              accept={uploadsCfg.files ? ACCEPT_STRING : 'image/*'}
              multiple
              style={{ display: 'none' }}
              onChange={(e) => {
                if (e.target.files) handleFiles(e.target.files)
                e.target.value = ''
              }}
            />
          )}
          <QueuedMessagesBar
            items={messageQueue}
            paused={queuePaused}
            labels={{
              title: t.queueTitle,
              sendNow: t.queueSendNow,
              inject: t.queueInject,
              edit: t.queueEdit,
              remove: t.queueRemove,
              resume: t.queueResume,
              imageMessage: t.queueImageMessage,
            }}
            onSendNow={sendQueuedNow}
            onInject={runIsAgent ? injectQueued : undefined}
            onEdit={editQueued}
            onDelete={deleteQueued}
            onReorder={reorderQueued}
            onResume={() => setQueuePaused(false)}
            hidden={phoneMode}
          />
          {sttToast && (
            <>
              <style>{`
                @keyframes stt-slide-in {
                  from { transform: translateX(100%); opacity: 0; }
                  to { transform: translateX(0); opacity: 1; }
                }
                @keyframes stt-fade-out {
                  from { opacity: 1; }
                  to { opacity: 0; }
                }
              `}</style>
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  background: 'rgba(220,53,69,0.6)',
                  color: '#fff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '13px',
                  borderRadius: 'inherit',
                  zIndex: 2,
                  animation: 'stt-slide-in 0.4s ease-out',
                  pointerEvents: 'none',
                  textAlign: 'center',
                  padding: '0 8px',
                }}
                onAnimationEnd={(e) => {
                  if (e.animationName === 'stt-slide-in') {
                    const el = e.currentTarget as HTMLDivElement
                    setTimeout(() => {
                      el.style.animation = 'stt-fade-out 0.3s ease forwards'
                    }, 2100)
                  }
                }}
              >
                {sttToast}
              </div>
            </>
          )}
          <textarea
            ref={textareaRef}
            className={`chat-textarea${phoneMode ? ' phone-mode-hidden' : ''}`}
            placeholder={t.inputPlaceholder}
            rows={1}
            value={input}
            onChange={(e) => {
              setInput(e.target.value)
              if (!hasTransitionedToTypingRef.current && emotionTimelineRef.current?.activePhase === 'reading' && e.target.value.length > 0) {
                hasTransitionedToTypingRef.current = true
                switchEmotionPhase('typing')
              }
            }}
            onKeyDown={handleKeyDown}
            disabled={isRecording}
          />
          {/* Emotion webcam toggle */}
          {emotionConfig?.enabled && (!chatLoading || phoneMode) && (
            <button
              className={`chat-emotion-btn ${webcamActive ? 'active' : ''} ${emotionRuntimeStatus === 'error' ? 'error' : ''}${(isRecording && !phoneMode) || (phoneMode && !emotionConfig?.enabled) ? ' phone-mode-hidden' : ''}`}
              title={webcamActive ? 'Emotion Detection aktiv' : 'Emotion Detection starten'}
              onClick={webcamActive ? stopWebcamEmotion : startWebcamEmotion}
            >
              <Smile size={18} />
              {webcamActive && emotionRuntimeStatus === 'ready' && (
                <span className="chat-emotion-indicator" />
              )}
            </button>
          )}
          {/* Phone mode toggle */}
          {audioConfig?.phone?.button_visible && (!chatLoading || phoneMode) && (
            <button
              className={`chat-phone-btn ${phoneMode ? 'active' : ''}`}
              title={phoneMode ? t.phoneModeOn : t.phoneModeOff}
              onClick={togglePhoneMode}
            >
              {phoneMode ? <PhoneOff size={18} /> : <Phone size={18} />}
            </button>
          )}
          {/* Mic / STT button */}
          {audioConfig?.stt?.button_visible && !chatLoading && (
            <button
              className={`chat-mic-btn ${isRecording ? 'recording' : ''} ${awaitingUserSpeech ? 'awaiting' : ''}${phoneMode ? ' phone-mode-hidden' : ''}`}
              title={isRecording ? t.sttStop : t.sttStart}
              onClick={toggleRecording}
            >
              {isRecording ? (
                <span className="chat-mic-recording">
                  <span className="chat-mic-ring" />
                  <span className="chat-mic-ring" />
                  <span className="chat-mic-ring" />
                  <Mic size={18} />
                </span>
              ) : (
                <Mic size={18} />
              )}
            </button>
          )}
          {chatLoading && (
            <button className={`chat-stop-indicator${phoneMode ? ' phone-mode-hidden' : ''}`} onClick={handleStop} title={t.stop}>
              <span className="chat-stop-pulse" />
              <span className="chat-stop-icon">
                <Square size={14} fill="currentColor" />
              </span>
            </button>
          )}
          {uploadsEnabled && !isRecording && (
            <button
              className={`chat-image-btn${phoneMode ? ' phone-mode-hidden' : ''}`}
              title={t.uploadFile}
              onClick={() => fileInputRef.current?.click()}
            >
              <Paperclip size={18} />
            </button>
          )}
          <button
            className={`chat-send-btn${phoneMode ? ' phone-mode-hidden' : ''}`}
            onClick={handleSend}
            disabled={(!input.trim() && pendingImages.length === 0 && !pendingFiles.some(f => f.status === 'ready')) || isRecording}
            title={chatLoading ? t.queueEnqueue : t.send}
          >
            {chatLoading ? <ListPlus size={18} /> : <Send size={18} />}
          </button>
        </div>
      </div>

      {/* Agent memory modal — same glass look as dashboard chat */}
      {memoryOpen && (
        <div
          className="chat-memory-modal-overlay"
          onClick={() => { setMemoryOpen(false); setEditingMemory(null) }}
        >
          <div
            className="chat-memory-modal-box"
            onClick={(e) => e.stopPropagation()}
            style={{ maxWidth: 640 }}
          >
            <div className="chat-history-modal-header">
              <h4>{t.agentMemory}</h4>
              <button
                className="btn-icon"
                onClick={() => { setMemoryOpen(false); setEditingMemory(null) }}
                title={t.close}
              >
                <X size={18} />
              </button>
            </div>
            <div className="chat-memory-modal-body">
              {memoryLoading && <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{t.memoryLoading}</div>}
              {memoryError && (
                <div className="chat-auth-error">
                  <AlertCircle size={14} />
                  {memoryError}
                </div>
              )}
              {!memoryLoading && !memoryError && memoryEntries.length === 0 && (
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', padding: '16px 0' }}>{t.memoryEmpty}</div>
              )}
              {memoryEntries.map((entry: any, i: number) => {
                const name = entry.assistant_name || model
                const content = entry.content ?? entry.memory ?? ''
                const isEditing = editingMemory?.assistant_name === name
                return (
                  <div key={i} className="chat-memory-modal-item">
                    <div className="chat-memory-modal-item-header">
                      <strong>{name}</strong>
                      {!isEditing && (
                        <div className="chat-memory-modal-item-actions">
                          <button
                            className="btn-icon"
                            title="Edit"
                            onClick={() => setEditingMemory({ assistant_name: name, memory: content })}
                          >
                            <Pencil size={13} />
                          </button>
                          <button
                            className="btn-icon"
                            title="Delete"
                            onClick={() => deleteMemoryEntry(name)}
                          >
                            <Trash2 size={13} />
                          </button>
                        </div>
                      )}
                    </div>
                    {isEditing && editingMemory ? (
                      <div>
                        <textarea
                          value={editingMemory.memory}
                          onChange={(e) => setEditingMemory({ assistant_name: editingMemory.assistant_name, memory: e.target.value })}
                          rows={4}
                          style={{ width: '100%', fontSize: 13 }}
                        />
                        <div style={{ display: 'flex', gap: 8, marginTop: 8, justifyContent: 'flex-end' }}>
                          <button className="btn-secondary btn-small" onClick={() => setEditingMemory(null)}>
                            {t.cancel || 'Abbrechen'}
                          </button>
                          <button className="btn-primary btn-small" onClick={saveMemory}>
                            {t.save || 'Speichern'}
                          </button>
                        </div>
                      </div>
                    ) : (
                      <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12, background: 'var(--bg)', padding: 10, borderRadius: 6, margin: 0, color: 'var(--text)' }}>
                        {content}
                      </pre>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}

      {/* Auth Modal */}
      {authModalOpen && (
        <div className="chat-auth-modal-overlay" onClick={() => setAuthModalOpen(false)}>
          <div className="chat-auth-modal-box" onClick={(e) => e.stopPropagation()}>
            <div className="chat-auth-modal-header">
              <h4>{authTab === 'login' ? 'Anmelden' : 'Registrieren'}</h4>
              <button className="btn-icon" onClick={() => setAuthModalOpen(false)} title={t.close}>
                <X size={18} />
              </button>
            </div>
            <div className="chat-auth-tab-row">
              <button
                className={`chat-auth-tab ${authTab === 'login' ? 'active' : ''}`}
                onClick={() => { setAuthTab('login'); setAuthError(null) }}
              >
                Anmelden
              </button>
              <button
                className={`chat-auth-tab ${authTab === 'register' ? 'active' : ''}`}
                onClick={() => { setAuthTab('register'); setAuthError(null) }}
              >
                Registrieren
              </button>
            </div>
            {authError && (
              <div className="chat-auth-error">
                <AlertCircle size={14} />
                {authError}
              </div>
            )}
            <form
              onSubmit={(e) => {
                e.preventDefault()
                const fd = new FormData(e.currentTarget)
                const username = fd.get('username') as string
                const password = fd.get('password') as string
                if (authTab === 'login') {
                  handleLogin(username, password)
                } else {
                  const display_name = fd.get('display_name') as string
                  const email = (fd.get('email') as string) || undefined
                  handleRegister(username, password, display_name, email)
                }
              }}
            >
              <div className="chat-auth-field">
                <label>{t.username}</label>
                <input name="username" type="text" required minLength={2} maxLength={100} autoComplete="username" />
              </div>
              {authTab === 'register' && (
                <>
                  <div className="chat-auth-field">
                    <label>{t.displayName}</label>
                    <input name="display_name" type="text" required minLength={1} maxLength={200} autoComplete="name" />
                  </div>
                  <div className="chat-auth-field">
                    <label>{t.emailOptional}</label>
                    <input name="email" type="email" maxLength={200} autoComplete="email" />
                  </div>
                </>
              )}
              <div className="chat-auth-field">
                <label>{t.password}</label>
                <input name="password" type="password" required minLength={6} maxLength={100} autoComplete={authTab === 'login' ? 'current-password' : 'new-password'} />
              </div>
              <button
                type="submit"
                disabled={authLoading}
                className="chat-auth-submit"
              >
                {authLoading ? t.loading : (authTab === 'login' ? t.login : t.register)}
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Change Password Modal */}
      {pwdModalOpen && (
        <div className="chat-auth-modal-overlay" onClick={() => setPwdModalOpen(false)}>
          <div className="chat-auth-modal-box" onClick={(e) => e.stopPropagation()}>
            <div className="chat-auth-modal-header">
              <h4>{language === 'de' ? 'Passwort ändern' : 'Change Password'}</h4>
              <button className="btn-icon" onClick={() => setPwdModalOpen(false)} title={t.close}>
                <X size={18} />
              </button>
            </div>
            {pwdError && (
              <div className="chat-auth-error">
                <AlertCircle size={14} />
                {pwdError}
              </div>
            )}
            {pwdSuccess && (
              <div className="chat-auth-error" style={{ color: 'var(--success, #10b981)' }}>
                {language === 'de' ? 'Passwort erfolgreich geändert' : 'Password changed successfully'}
              </div>
            )}
            <form onSubmit={handleChangePassword}>
              <div className="chat-auth-field">
                <label>{language === 'de' ? 'Aktuelles Passwort' : 'Current Password'}</label>
                <input
                  type="password"
                  value={pwdForm.current_password}
                  onChange={(e) => setPwdForm((f) => ({ ...f, current_password: e.target.value }))}
                  required
                  autoComplete="current-password"
                />
              </div>
              <div className="chat-auth-field">
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
              <div className="chat-auth-field">
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
              <button
                type="submit"
                disabled={pwdLoading}
                className="chat-auth-submit"
              >
                {pwdLoading ? t.loading : (language === 'de' ? 'Speichern' : 'Save')}
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Conversation History Modal */}
      {historyOpen && (
        <div className="chat-history-modal-overlay" onClick={() => setHistoryOpen(false)}>
          <div className="chat-history-modal-box" onClick={(e) => e.stopPropagation()}>
            <div className="chat-history-modal-header">
              <h4>{t.conversations}</h4>
              <button className="btn-icon" onClick={() => setHistoryOpen(false)} title={t.close}>
                <X size={18} />
              </button>
            </div>
            <div className="chat-history-modal-body">
              {historyLoading && (
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', padding: '16px 0' }}>{t.loading}</div>
              )}
              {historyError && (
                <div style={{ fontSize: 13, color: 'var(--danger)', padding: '16px 0' }}>{historyError}</div>
              )}
              {!historyLoading && historyItems.length === 0 && (
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', padding: '16px 0' }}>
                  {t.noConversations}
                </div>
              )}
              {!historyLoading && historyItems.map((item) => (
                <div
                  key={item.conversation_id}
                  className="chat-history-modal-item"
                  onClick={() => loadPreview(item.conversation_id)}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="chat-history-modal-item-info">
                    <div className="chat-history-modal-item-title">
                      {editingTitleId === item.conversation_id ? (
                        <input
                          value={editingTitleValue}
                          onChange={(e) => setEditingTitleValue(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              e.stopPropagation()
                              renameConversation(item.conversation_id, editingTitleValue)
                            }
                            if (e.key === 'Escape') {
                              e.stopPropagation()
                              setEditingTitleId(null)
                            }
                          }}
                          onBlur={() => renameConversation(item.conversation_id, editingTitleValue)}
                          autoFocus
                          style={{ fontSize: 12, padding: '2px 4px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--card)', color: 'var(--text)', width: '100%' }}
                          onClick={(e) => e.stopPropagation()}
                        />
                      ) : (
                        item.title || item.model_name || '—'
                      )}
                    </div>
                    <div className="chat-history-modal-item-meta">
                      {item.message_count} {t.messages} · {new Date(item.updated_at).toLocaleString(language === 'de' ? 'de-DE' : 'en-GB', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                  <div className="chat-history-modal-item-actions" style={{ display: 'flex', gap: 6 }}>
                    <button
                      className="btn-icon btn-small"
                      onClick={(e) => { e.stopPropagation(); setEditingTitleId(item.conversation_id); setEditingTitleValue(item.title || item.model_name || '') }}
                      title={t.rename}
                    >
                      <Sparkles size={12} />
                    </button>
                    <button
                      className="btn-primary btn-small"
                      style={{ padding: '2px 8px', fontSize: 11, whiteSpace: 'nowrap' }}
                      onClick={(e) => { e.stopPropagation(); loadConversation(item.conversation_id) }}
                      disabled={chatLoading}
                    >
                      {t.load}
                    </button>
                    <button
                      className="btn-icon btn-small"
                      onClick={(e) => { e.stopPropagation(); setConvDeleteTarget(item.conversation_id) }}
                      title={t.delete}
                    >
                      <X size={12} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Conversation Preview Modal */}
      {previewConversationId && (
        <div className="chat-history-modal-overlay" onClick={() => setPreviewConversationId(null)}>
          <div className="chat-history-preview-box" onClick={(e) => e.stopPropagation()}>
            <div className="chat-history-modal-header">
              <h4>{historyItems.find((i) => i.conversation_id === previewConversationId)?.title || historyItems.find((i) => i.conversation_id === previewConversationId)?.model_name || t.preview}</h4>
              <button className="btn-icon" onClick={() => setPreviewConversationId(null)} title={t.close}>
                <X size={18} />
              </button>
            </div>
            <div className="chat-history-preview-body">
              {previewLoading && (
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', padding: '16px 0' }}>{t.loading}</div>
              )}
              {previewError && (
                <div style={{ fontSize: 13, color: 'var(--danger)', padding: '16px 0' }}>{previewError}</div>
              )}
              {!previewLoading && !previewError && previewMessages && previewMessages.map((msg, idx) => {
                const role = msg.role
                return (
                  <div key={idx} className={`chat-history-preview-message ${role}`}>
                    {role === 'assistant' && (
                      <div className="chat-avatar assistant-avatar" style={{ width: 28, height: 28, minWidth: 28 }}>
                        <Bot size={14} />
                      </div>
                    )}
                    {role === 'user' && (
                      <div className="chat-avatar user-avatar" style={{ width: 28, height: 28, minWidth: 28 }}>
                        <User size={14} />
                      </div>
                    )}
                    <div className="chat-history-preview-bubble">
                      {role === 'assistant' && (
                        <SubagentTagsRow
                          events={msg.subagents || []}
                          toolCalls={msg.toolCalls || []}
                          language={language}
                          onSelect={setSelectedSubagent}
                        />
                      )}
                      {role === 'assistant' && (
                        <AgentTodoPanel snapshot={msg.todoList || msg.retrievalInfo?.todo_list} language={language} />
                      )}
                      {role === 'assistant' && (
                        <ObserverTag events={msg.observerEvents} language={language} />
                      )}
                      <MarkdownContent text={msg.content} style={{ fontSize: 13 }} />
                    </div>
                  </div>
                )
              })}
            </div>
            <div className="chat-history-preview-footer">
              <button
                className="btn-primary"
                onClick={() => {
                  if (previewConversationId) loadConversation(previewConversationId)
                }}
                disabled={chatLoading}
              >
                {t.loadIntoChat}
              </button>
            </div>
          </div>
        </div>
      )}
      {wsEnabled && wsOpen && (
        <WorkspaceBrowser
          apiUrl={getApiUrl()}
          assistant={model}
          conversationId={activeConversationId}
          accessToken={publicAccessToken || undefined}
          onClose={() => setWsOpen(false)}
        />
      )}
      <ConfirmModal
        open={webcamConsentOpen}
        title="Webcam-Emotion Detection"
        message={
          'Die Kamera wird lokal im Browser ausgewertet. ' +
          'Es werden keine Bilder oder Videos aufgezeichnet oder übertragen. ' +
          'Nur aggregierte Emotions-Metadaten werden gesendet.\n\n' +
          'Möchten Sie fortfahren?'
        }
        confirmText="OK"
        cancelText={t.cancel}
        onConfirm={() => {
          setWebcamConsentOpen(false)
          setWebcamConsent(true)
          void startWebcamEmotionInner()
        }}
        onCancel={() => setWebcamConsentOpen(false)}
      />
      <ConfirmModal
        open={convDeleteTarget !== null}
        title={t.delete}
        message={t.deleteConfirm}
        confirmText={t.delete}
        cancelText={t.cancel}
        onConfirm={() => {
          const id = convDeleteTarget
          setConvDeleteTarget(null)
          if (id) deleteConversation(id)
        }}
        onCancel={() => setConvDeleteTarget(null)}
      />
      <AlertModal
        open={notice !== null}
        title={notice?.danger ? 'Error' : 'Info'}
        message={notice?.message || ''}
        danger={notice?.danger}
        onClose={() => setNotice(null)}
      />
    </div>
  )
}
