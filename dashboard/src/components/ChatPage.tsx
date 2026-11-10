import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { Brain, Copy, Check, Save, Send, Square, RotateCcw, Bot, Trash2, User, X, History, Sparkles, Volume2, Mic, Phone, PhoneOff, Smile, BookOpen, ImagePlus, ListPlus, Folder } from 'lucide-react'
import { createVideoSnippetBuffer } from '../emotion/videoSnippetBuffer'
import { createAudioSnippetBuffer } from '../emotion/audioSnippetBuffer'
import { fuseEmotions } from '../emotion/emotionAggregator'
import type { EmotionContextPayload, EmotionRuntimeStatus, SnippetConfig, EmotionResult, PhaseEmotion, EmotionPhase } from '../emotion/types'
import { applyHighpassFilter } from '../emotion/audioFilter'
import { extractCompleteSentences, TTSSegmentPlayer } from '../utils/tts'
import { blobToWav } from '../utils/audio'
import { agentMemory, conversations, knowledge, type AgentTodoSnapshot, type AssistantEntry, type OKFConcept, type ObserverEvent, type SubagentEvent, type SubagentMeta } from '../api/client'
import { useData } from '../api/DataContext'
import { useAuth } from '../api/AuthContext'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'
import { OKFGraphView } from './OKFGraphView'
import { ResponseMetricsBadge, LiveTokenChip, type ResponseMetrics, type LiveUsage } from './ResponseMetrics'
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
import ConfirmModal from './ConfirmModal'
import AlertModal from './AlertModal'
import { ArtifactCard } from '../chat-shared/ArtifactCard'
import { WorkspaceBrowser } from '../chat-shared/WorkspaceBrowser'
import { UploadTray, AttachmentChips } from '../chat-shared/UploadTray'
import { uploadChatAttachment, deleteChatAttachment, deleteChatConversation, isImageFile, isDocFile, ACCEPT_STRING, type PendingFile, type AttachmentMeta } from '../chat-shared/uploads'

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
        source: data.source || `phase__${phase}`,
        timestamp_ms: data.timestamp_ms,
        sampled_frames: data.sampled_frames,
        valid_face_frames: data.valid_face_frames,
        face_detection_rate: data.face_detection_rate,
        window_ms: data.window_ms,
      })),
    }
  }
  // Already new format
  if (emotion.fused_emotion !== undefined || emotion.phases !== undefined) {
    return emotion as Message['emotion']
  }
  // Old flat format: { primary_emotion, confidence, intensity, source, timestamp_ms }
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

/* Rebuild the metrics badge from persisted retrieval_info after a reload —
   _llm_usage_metadata and _final_answer_usage travel with the stored turn. */
/* Insert a user message bubble right before the trailing (streaming) assistant
   placeholder — keeps user→assistant order when a mid-run injection arrives. */
function filterUnconsumed(
  items: Array<string | { text: string; images?: string[] }>,
  queuedTexts: ReadonlySet<string>,
  displayedTexts: ReadonlySet<string>,
): Array<{ text: string; images?: string[] }> {
  // Unconsumed inputs race two other paths: a message already shown in the
  // thread was actually consumed (must not re-send) and one already back in
  // the queue must not be requeued a second time.
  const seen = new Set<string>()
  const out: Array<{ text: string; images?: string[] }> = []
  for (const it of items) {
    const text = typeof it === 'string' ? it : it.text
    const images = typeof it === 'string' ? undefined : it.images
    if (!text || displayedTexts.has(text) || queuedTexts.has(text) || seen.has(text)) continue
    seen.add(text)
    out.push({ text, images })
  }
  return out
}

function insertUserBeforeLastAssistant(prev: Message[], text: string, source?: string): Message[] {
  // Idempotent: a replayed delta or a second attached stream must not splice
  // the same user text in twice.
  if (prev.some((m) => m.role === 'user' && m.content === text)) return prev
  const msg: Message = { role: 'user', content: text, systemNote: !!source && source !== 'user' }
  const idx = [...prev.entries()].reverse().find(([, m]) => m.role === 'assistant')?.[0]
  if (idx === undefined) return [...prev, msg]
  const next = [...prev]
  next.splice(idx, 0, msg)
  return next
}

function getPublicBase(): string {
  const path = window.location.pathname
  const idx = path.lastIndexOf('/dashboard/')
  if (idx === -1) return '/dashboard'
  return path.slice(0, idx) + '/dashboard'
}

function adminChatConversationIdKey(model: string): string {
  return `agentix-admin-chat-conversation-id:${model}`
}

function chatMessagesKey(conversationId: string): string {
  return `agentix-chat-messages:${conversationId}`
}

function getOrCreateChatConversationId(model: string, userId?: string): string {
  const key = adminChatConversationIdKey(model)
  const existing = sessionStorage.getItem(key)
  if (existing) {
    if (userId && existing.startsWith(`dashboard-${userId}-`)) {
      return existing
    }
    if (!userId && existing.startsWith('admin-chat-')) {
      return existing
    }
  }
  const hash = (crypto as any).randomUUID ? (crypto as any).randomUUID() : String(Date.now())
  const id = userId ? `dashboard-${userId}-${hash}` : `admin-chat-${hash}`
  sessionStorage.setItem(key, id)
  return id
}

function getApiUrl(): string {
  const publicBase = getPublicBase()
  const apiBase = publicBase.replace(/\/dashboard$/, '')
  return window.location.origin + apiBase + '/v1'
}

function useCopyTimeout() {
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const copy = useCallback((id: string) => {
    setCopiedId(id)
    setTimeout(() => setCopiedId((current) => (current === id ? null : current)), 2000)
  }, [])
  return { copiedId, copy }
}

/* ------------------------------------------------------------------ */
/*  Main component                                                      */
/* ------------------------------------------------------------------ */
export default function ChatPage() {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const { getInstructions, getChains } = useData()
  const { user } = useAuth()
  const userId = user?.id

  const [model, setModel] = useState('CyberSecurityChatbot')
  const [wsOpen, setWsOpen] = useState(false)
  const [assistants, setAssistants] = useState<string[]>([])
  const [assistantConfigs, setAssistantConfigs] = useState<AssistantEntry[]>([])
  const wsEnabled = assistantConfigs.find((a) => a.name === model)?.workspace?.enabled === true
  const [chains, setChains] = useState<Array<{ id: string; name: string }>>([])
  const [loadingModels, setLoadingModels] = useState(false)

  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sttToast, setSttToast] = useState<string | null>(null)
  const [selectedToolCall, setSelectedToolCall] = useState<ToolCallLive | null>(null)
  const [selectedSubagent, setSelectedSubagent] = useState<SubagentMeta | null>(null)
  const [selectedRAGInfo, setSelectedRAGInfo] = useState<Message['retrievalInfo']>(undefined)
  const [pendingImages, setPendingImages] = useState<string[]>([])
  const [pendingFiles, setPendingFiles] = useState<PendingFile[]>([])
  const [chatFileUploadEnabled, setChatFileUploadEnabled] = useState(false)
  const [messageQueue, setMessageQueue] = useState<QueuedChatMessage[]>([])
  const [queuePaused, setQueuePaused] = useState(false)
  const [chatImageUploadEnabled, setChatImageUploadEnabled] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [previewHtml, setPreviewHtml] = useState<string | null>(null)

  const [agentMemories, setAgentMemories] = useState<Array<{ assistant_name: string; memory: string; updated_at: string | null }>>([])
  const [memoryPanelOpen, setMemoryPanelOpen] = useState(false)
  const [editingMemory, setEditingMemory] = useState<{ assistant_name: string; memory: string } | null>(null)

  const [historyOpen, setHistoryOpen] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyItems, setHistoryItems] = useState<Array<{ conversation_id: string; model_name?: string | null; message_count: number; last_message_at: string }>>([])
  const [historyError, setHistoryError] = useState<string | null>(null)

  const [previewConversationId, setPreviewConversationId] = useState<string | null>(null)
  const [previewMessages, setPreviewMessages] = useState<Message[] | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)

  const [audioConfig, setAudioConfig] = useState<{
    stt: { button_visible: boolean; language?: string; provider?: string; model?: string }
    tts: { button_visible: boolean; voice_id: string | null; auto_read?: boolean; streaming_enabled?: boolean }
    phone: { enabled: boolean; button_visible: boolean }
  } | null>(null)
  const [emotionConfig, setEmotionConfig] = useState<{
    enabled: boolean
    text: { enabled: boolean; rules_enabled: boolean }
    webcam: {
      enabled: boolean
      button_visible: boolean
      runtime: string
      execution_providers: string[]
      face_detector: string
      face_detector_type?: 'mediapipe' | 'onnx'
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
    audio: {
      enabled: boolean
      model_url: string
      min_snr_db?: number
      noise_gate_min?: number
      noise_gate_max?: number
    }
    prompt_steering: { enabled: boolean; min_confidence: number }
    analytics: { enabled: boolean }
  } | null>(null)
  const [playingMsgIdx, setPlayingMsgIdx] = useState<number | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [phoneMode, setPhoneMode] = useState(false)
  const [awaitingUserSpeech, setAwaitingUserSpeech] = useState(false)

  const [webcamConsent, setWebcamConsent] = useState(false)
  const [webcamConsentOpen, setWebcamConsentOpen] = useState(false)
  const [notice, setNotice] = useState<{ message: string; danger?: boolean } | null>(null)
  const [memDeleteTarget, setMemDeleteTarget] = useState<string | null>(null)
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
  const continuousAudioStreamRef = useRef<MediaStream | null>(null)
  const hasTransitionedToTypingRef = useRef(false)
  const audioSnapshotsRef = useRef<EmotionResult[]>([])
  const isStartingAudioRef = useRef(false)
  const isStartingWebcamRef = useRef(false)
  const messagesRef = useRef<Message[]>(messages)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  /* Synchronous in-flight guard for message sends. State updates are batched,
     so `chatLoading` alone cannot prevent re-entrant sends while an async send
     is suspended before its first state commit — the queue drain effect would
     otherwise fire every remaining queued message in parallel. */
  const sendInFlightRef = useRef(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const activeModelRef = useRef<string>(model)
  const userScrolledUpRef = useRef(false)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null)
  const lastAssistantIdxRef = useRef<number | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const vadSilenceStartRef = useRef<number>(0)
  const vadRafRef = useRef<number | null>(null)
  const phoneModeRef = useRef(false)
  const vadSpeechDetectedRef = useRef(false)
  const recordingStartTimeRef = useRef(0)
  const phoneEmptyRetriesRef = useRef(0)
  const ttsBufferRef = useRef('')
  const ttsPlayerRef = useRef<TTSSegmentPlayer | null>(null)
  const lastAudioModelRef = useRef<string | null>(null)
  const lastEmotionModelRef = useRef<string | null>(null)

  const { copiedId, copy } = useCopyTimeout()

  /* Load assistants, chains and model history */
  // Flush debounced sessionStorage writes on unmount.
  useEffect(() => () => flushPendingChatWrites(), [])

  // Read-only lookup of the active conversation id (getOrCreate* would mint
  // one as a side effect, which we don't want during render).
  const activeConversationId = useMemo(() => {
    try {
      return sessionStorage.getItem(adminChatConversationIdKey(model)) || undefined
    } catch {
      return undefined
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, messages])

  useEffect(() => {
    setLoadingModels(true)
    Promise.all([getInstructions(), getChains()])
      .then(([data, chainData]) => {
        const assistantNames = data.assistants.map((a: { name: string }) => a.name)
        setAssistants(assistantNames)
        setAssistantConfigs(data.assistants)
        const chainItems = chainData.map((c: { id: string; name: string }) => ({ id: c.id, name: c.name }))
        setChains(chainItems)
        const chainValues = chainItems.map((c) => `chain:${c.id}`)
        const all = [...assistantNames, ...chainValues]
        const targetModel = all.length > 0 && model === 'CyberSecurityChatbot' ? all[0] : model
        setModel(targetModel)
        activeModelRef.current = targetModel
        const convId = getOrCreateChatConversationId(targetModel, userId)
        loadMessagesForConversation(convId)
        attachToRunIfActive(convId)
        loadMemoryForConversation(convId)
        fetchAudioConfig(targetModel)
        fetchEmotionConfig(targetModel)
      })
      .catch(() => {
        setAssistants([])
        setChains([])
      })
      .finally(() => setLoadingModels(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  /* Refresh assistants/chains when user returns to the tab */
  useEffect(() => {
    function handleVisibilityChange() {
      if (document.visibilityState === 'visible') {
        Promise.all([getInstructions(), getChains()])
          .then(([data, chainData]) => {
            setAssistants(data.assistants.map((a: { name: string }) => a.name))
            setAssistantConfigs(data.assistants)
            setChains(chainData.map((c: { id: string; name: string }) => ({ id: c.id, name: c.name })))
          })
          .catch(() => {
            // ignore
          })
        // Refresh audio / emotion config on return so dashboard edits are picked up
        lastAudioModelRef.current = null
        lastEmotionModelRef.current = null
        fetchAudioConfig(activeModelRef.current)
        fetchEmotionConfig(activeModelRef.current)
      }
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange)
  }, [getInstructions, getChains])

  /* Sync phoneModeRef with state */
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
      stopEmotionPolling().catch(() => {})
      attachAbortRef.current?.abort()
    }
  }, [])

  /* Auto-focus textarea on mount */
  useEffect(() => {
    requestAnimationFrame(() => textareaRef.current?.focus())
  }, [])

  /* Fetch upload capabilities — unprivileged endpoint so non-admin users
     also see the features admins enabled (effective values apply instantly).
     Retries a few times so a transient failure (restart, blip) does not leave
     the upload UI dead until the next manual reload. */
  useEffect(() => {
    let cancelled = false
    let attempt = 0
    const load = () => {
      fetch(`${getApiUrl()}/chat/capabilities`, { credentials: 'include' })
        .then((res) => (res.ok ? res.json() : null))
        .then((c) => {
          if (!c) {
            if (!cancelled && attempt < 3) {
              attempt += 1
              setTimeout(load, 2000 * attempt)
            }
            return
          }
          setChatImageUploadEnabled(!!c.images)
          setChatFileUploadEnabled(!!c.files)
        })
        .catch(() => {
          if (!cancelled && attempt < 3) {
            attempt += 1
            setTimeout(load, 2000 * attempt)
          }
        })
    }
    load()
    return () => { cancelled = true }
  }, [])

  const uploadsEnabled = chatImageUploadEnabled || chatFileUploadEnabled

  /* Upload a document attachment to the conversation */
  const uploadDoc = useCallback((file: File) => {
    const convId = getOrCreateChatConversationId(activeModelRef.current, userId)
    const localId = `up-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    setPendingFiles(prev => [...prev, {
      localId, name: file.name, size: file.size,
      ext: `.${(file.name.split('.').pop() || '').toLowerCase()}`,
      status: 'uploading',
    }])
    uploadChatAttachment(getApiUrl(), convId, file, undefined, activeModelRef.current)
      .then((meta) => {
        setPendingFiles(prev => prev.map(f => f.localId === localId
          ? { ...f, attachmentId: meta.id, status: 'ready' } : f))
      })
      .catch((e: any) => {
        setPendingFiles(prev => prev.map(f => f.localId === localId
          ? { ...f, status: 'error', error: String(e?.message || e) } : f))
      })
  }, [userId])

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
    if (chatImageUploadEnabled) {
      if (imgs.length > 0) handleImageFiles(imgs)
    } else if (chatFileUploadEnabled) {
      imgs.forEach(uploadDoc)
    }
    const docs = chatFileUploadEnabled ? arr.filter(isDocFile) : []
    docs.forEach(uploadDoc)
  }, [chatImageUploadEnabled, chatFileUploadEnabled, handleImageFiles, uploadDoc])

  const removePendingFile = useCallback((localId: string) => {
    setPendingFiles(prev => {
      const item = prev.find(f => f.localId === localId)
      if (item?.attachmentId) {
        const convId = sessionStorage.getItem(adminChatConversationIdKey(activeModelRef.current))
        if (convId) deleteChatAttachment(getApiUrl(), convId, item.attachmentId)
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

  /* Remove a pending image */
  const removePendingImage = useCallback((index: number) => {
    setPendingImages(prev => prev.filter((_, i) => i !== index))
  }, [])

  function loadMessagesForConversation(conversationId: string) {
    const parsed = loadChatMessages(chatMessagesKey(conversationId))
    if (parsed.length > 0) {
      setMessages(parsed.map((msg: any) => ({
        ...msg,
        emotion: normalizeEmotion(msg.emotion),
      })))
      return
    }
    setMessages([])
  }

  function persistMessages(conversationId: string, next: Message[]) {
    persistChatMessagesDebounced(chatMessagesKey(conversationId), next)
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
      })
      const info = await st.json().catch(() => null)
      if (info?.running) {
        await streamRunResumed(conversationId)
      } else {
        await syncConversationFromServer(conversationId)
      }
      // Inputs the run never consumed (live broadcast may have been missed
      // after a disconnect/reload) — hand them back to the queue, deduped
      // against items a live `unconsumed_inputs` delta already restored.
      const unconsumed = info?.unconsumed_inputs as Array<{ text: string; images?: string[] }> | undefined
      if (unconsumed?.length) requeueUnconsumed(unconsumed)
    } catch {
      // best effort — an older backend or network issue just skips resume
    }
  }

  function requeueUnconsumed(items: Array<string | { text: string; images?: string[] }>) {
    const displayed = new Set(
      messagesRef.current.filter((m) => m.role === 'user').map((m) => m.content)
    )
    setMessageQueue((q) => {
      const queued = new Set(q.map((m) => m.text))
      const restored = filterUnconsumed(items, queued, displayed)
      if (!restored.length) return q
      return [
        ...restored.map((it) => ({ id: newQueueId(), text: it.text, images: it.images })),
        ...q,
      ]
    })
  }

  async function syncConversationFromServer(conversationId: string) {
    try {
      const response = await conversations.detail(conversationId)
      const loadedMessages = normalizeConversationMessages(response)
      if (loadedMessages.length === 0) return
      const local = messagesRef.current
      const lastLocalAssistant = [...local].reverse().find((m) => m.role === 'assistant')
      const lastServerAssistant = [...loadedMessages].reverse().find((m) => m.role === 'assistant')
      const localLen = lastLocalAssistant?.content?.length ?? -1
      const serverLen = lastServerAssistant?.content?.length ?? -1
      if (loadedMessages.length > local.length || serverLen > localLen) {
        setMessages(loadedMessages)
        persistMessages(conversationId, loadedMessages)
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
                if (runUserMessage && !next.some((m) => m.role === 'user' && m.content === runUserMessage)) {
                  next.push({ role: 'user', content: runUserMessage })
                }
                for (const injected of resumedInjected) {
                  const injectedText = typeof injected === 'string' ? injected : (injected as any)?.text
                  const injectedSrc = typeof injected === 'object' && injected ? (injected as any).source as string | undefined : undefined
                  if (typeof injectedText === 'string' && injectedText && !next.some((m) => m.role === 'user' && m.content === injectedText)) {
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
            // skip invalid JSON
          }
        }
      }
      batcher.flushNow()
      persistMessages(conversationId, messagesRef.current)
      await syncConversationFromServer(conversationId)
      loadMemoryForConversation(conversationId)
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

  async function fetchAudioConfig(modelName: string) {
    if (lastAudioModelRef.current === modelName) return
    try {
      const res = await fetch(`/audio/config?model=${encodeURIComponent(modelName)}`, { credentials: 'include' })
      if (!res.ok) {
        setAudioConfig(null)
        return
      }
      const data = await res.json()
      setAudioConfig(data)
      lastAudioModelRef.current = modelName
    } catch {
      setAudioConfig(null)
    }
  }

  async function fetchEmotionConfig(modelName: string) {
    if (lastEmotionModelRef.current === modelName) return
    try {
      const res = await fetch(`/emotion/config?model=${encodeURIComponent(modelName)}`, { credentials: 'include' })
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

  /* ------------------------------------------------------------------ */
  /*  Webcam Emotion                                                      */
  /* ------------------------------------------------------------------ */
  async function startWebcamEmotion() {
    if (webcamActive) return
    if (isStartingWebcamRef.current) return
    const webcamEnabled = emotionConfig?.webcam?.enabled ?? false
    const audioEnabled = emotionConfig?.audio?.enabled ?? false
    if (!webcamEnabled && !audioEnabled) return

    if (webcamEnabled && !webcamConsent) {
      setWebcamConsentOpen(true)
      return
    }
    await startWebcamEmotionInner()
  }

  async function startWebcamEmotionInner() {
    const webcamEnabled = emotionConfig?.webcam?.enabled ?? false
    isStartingWebcamRef.current = true
    setEmotionRuntimeStatus('loading')
    try {
      if (webcamEnabled && emotionConfig?.webcam) {
        const cfg = emotionConfig.webcam
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
      }
      if (isStartingWebcamRef.current) {
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

  async function startContinuousAudioEmotion(): Promise<void> {
    if (audioSnippetBufferRef.current?.isActive()) return
    if (isStartingAudioRef.current) return
    if (!emotionConfig?.audio?.enabled) {
      console.info('[Chat] Audio emotion disabled in config')
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
      const audioCfg: SnippetConfig = {
        windowMs: emotionConfig.webcam.snippet_window_ms || 3000,
        sampleFps: emotionConfig.webcam.sample_fps || 3,
        maxFrames: emotionConfig.webcam.max_frames_per_snippet || 12,
        minValidFaceFrames: emotionConfig.webcam.min_valid_face_frames || 3,
        faceModelUrl: emotionConfig.webcam.face_model_url || '/models/emotion/face-detector.onnx',
        faceDetectorType: 'mediapipe',
        expressionModelUrl: emotionConfig.audio.model_url || '/models/emotion/audio-emotion.onnx',
        expressionModelType: 'affectnet',
        expressionInputSize: 224,
        executionProviders: emotionConfig.webcam.execution_providers || ['wasm'],
        minSnrDb: emotionConfig.audio.min_snr_db ?? 10,
        noiseGateMin: emotionConfig.audio.noise_gate_min ?? 0.001,
        noiseGateMax: emotionConfig.audio.noise_gate_max ?? 0.5,
      }
      console.info('[Chat] Starting continuous audio emotion with modelUrl:', audioCfg.expressionModelUrl)
      audioSnippetBufferRef.current = createAudioSnippetBuffer(audioCfg)
      audioSnippetBufferRef.current.start(stream)
      console.info('[Chat] Continuous audio emotion started')
    } catch (err) {
      console.warn('[Chat] Audio emotion start failed:', err)
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

  async function toggleEmotionDetection() {
    if (webcamActive || isStartingWebcamRef.current) {
      stopWebcamEmotion()
    } else {
      await startWebcamEmotion()
      // Start audio independently of webcam
      if (emotionConfig?.audio?.enabled) {
        await startContinuousAudioEmotion().catch(() => {})
      }
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
      processEmotionBatch().catch(() => {})
    }, 2000)
  }

  async function stopEmotionPolling(): Promise<void> {
    if (emotionPollIntervalRef.current) {
      clearInterval(emotionPollIntervalRef.current)
      emotionPollIntervalRef.current = null
    }
    // Flush remaining snapshots into the active phase
    await processEmotionBatch()
  }

  function switchEmotionPhase(nextPhase: EmotionPhase): void {
    const tl = emotionTimelineRef.current
    if (!tl || tl.activePhase === nextPhase) return
    // Flush current phase snapshots
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
    // Init next phase
    tl.phases[nextPhase].startTime = Date.now()
    tl.activePhase = nextPhase
  }

  async function processEmotionBatch(): Promise<void> {
    const tl = emotionTimelineRef.current
    if (!tl) return

    // Collect current snapshots from active buffers (await both)
    const promises: Promise<void>[] = []
    if (snippetBufferRef.current?.isActive()) {
      promises.push(
        snippetBufferRef.current.captureSnapshot().then((res) => {
          if (res) tl.phases[tl.activePhase].snapshots.push(res)
        }).catch(() => {})
      )
    }
    if (audioSnippetBufferRef.current?.isActive()) {
      promises.push(
        audioSnippetBufferRef.current.captureSnapshot().then((res) => {
          if (res) {
            tl.phases[tl.activePhase].snapshots.push(res)
            audioSnapshotsRef.current.push(res)
          }
        }).catch(() => {})
      )
    }
    await Promise.all(promises)
  }

  function buildEmotionPhases(): PhaseEmotion[] {
    const tl = emotionTimelineRef.current
    if (!tl) return []
    const result: PhaseEmotion[] = []
    const phaseOrder: EmotionPhase[] = ['waiting_for_response', 'reading', 'typing', 'speaking']
    for (const phase of phaseOrder) {
      const data = tl.phases[phase]
      if (data.snapshots.length === 0 && !data.endTime && phase !== tl.activePhase) {
        continue // skip empty phases that were never entered
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
      // Phase was entered but no valid snapshots
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

  function cleanTextForTTS(text: string): string {
    let clean = text
    // Replace markdown links [text](url) with just the text
    clean = clean.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    // Replace bare URLs with a placeholder
    clean = clean.replace(/https?:\/\/[^\s]+/g, 'Link')
    // Remove markdown heading markers
    clean = clean.replace(/^#{1,6}\s+/gm, '')
    // Remove markdown formatting characters
    clean = clean.replace(/[*_`#]/g, '')
    // Remove markdown list markers at line start
    clean = clean.replace(/^[\-*]\s+/gm, '')
    clean = clean.replace(/^\d+\.\s+/gm, '')
    // Remove emojis (Unicode ranges for common emojis)
    clean = clean.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{FE00}-\u{FE0F}\u{1F900}-\u{1F9FF}\u{1F018}-\u{1F270}\u{238C}\u{2B06}\u{2B07}\u{2B05}\u{27A1}\u{2194}-\u{21AA}]/gu, '')
    // Remove HTML tags
    clean = clean.replace(/<[^>]+>/g, '')
    // Collapse multiple spaces inside a line, but keep paragraph breaks
    clean = clean.replace(/[ \t]+/g, ' ')
    // Trim each line
    clean = clean.split('\n').map((l) => l.trim()).join('\n')
    // Normalize paragraph breaks to exactly two newlines
    clean = clean.replace(/\n{3,}/g, '\n\n')
    return clean.trim()
  }

  function stopAllTTS() {
    if (ttsAudioRef.current) {
      ttsAudioRef.current.pause()
      ttsAudioRef.current = null
    }
    ttsPlayerRef.current?.abort()
    ttsPlayerRef.current = null
    ttsBufferRef.current = ''
    setPlayingMsgIdx(null)
  }

  async function speakMessage(text: string, idx: number) {
    if (!audioConfig?.tts?.button_visible || !audioConfig?.tts?.voice_id) return
    if (playingMsgIdx === idx) {
      stopAllTTS()
      return
    }
    // Stop any other playing TTS first
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
          // In phone mode: after TTS finishes, start STT
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

  /* ------------------------------------------------------------------ */
  /*  STT Recording                                                       */
  /* ------------------------------------------------------------------ */
  async function startRecording() {
    if (!audioConfig?.stt?.button_visible || isRecording) return
    // If emotion button is not active but audio emotion is enabled,
    // start a temporary continuous audio buffer for this STT session only
    if (!webcamActive && emotionConfig?.audio?.enabled && !audioSnippetBufferRef.current?.isActive()) {
      console.info('[Chat] startRecording: auto-starting temporary audio emotion buffer')
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

        // Capture audio prosody emotion from the continuous buffer (if running)
        let audioEmotion: EmotionContextPayload['audio_prosody'] | null = null
        console.info('[Chat] STT stop: audioSnippetBuffer active=', audioSnippetBufferRef.current?.isActive())
        if (audioSnippetBufferRef.current?.isActive()) {
          audioEmotion = await audioSnippetBufferRef.current.captureSnapshot()
          console.info('[Chat] STT captureSnapshot result:', audioEmotion)
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
          const res = await fetch('/audio/stt', {
            method: 'POST',
            credentials: 'include',
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
              sendChatMessageFromText(transcript, { audio_prosody: audioEmotion }).catch((err) => {
                console.error('Auto-send after transcription error:', err)
              })
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
      hasTransitionedToTypingRef.current = true

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

      // Auto-stop after 30 seconds
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
      const res = await fetch('/audio/tts/segment', {
        method: 'POST',
        credentials: 'include',
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

  function togglePhoneMode() {
    const next = !phoneMode
    setPhoneMode(next)
    if (next) {
      phoneEmptyRetriesRef.current = 0
      // Turn on: immediately start listening if not already
      if (!isRecording) {
        setTimeout(() => startRecording(), 100)
      }
    } else {
      // Turn off: stop any recording and TTS
      // Detach onstop first so the auto-send logic doesn't fire when
      // the user manually disables phone mode.
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.onstop = null
      }
      stopRecording()
      setIsRecording(false)
      stopAllTTS()
      if (!webcamActive) {
        stopEmotionPolling()
        stopContinuousAudioEmotion()
      }
      setAwaitingUserSpeech(false)
      phoneEmptyRetriesRef.current = 0
    }
  }

  /* Helper to send from a text string directly (used by phone mode) */
  async function sendChatMessageFromText(
    userMsg: string,
    audioEmotions?: { audio_prosody?: EmotionContextPayload['audio_prosody'] | null },
    images?: string[],
  ) {
    if (chatLoading || sendInFlightRef.current) {
      // Programmatic callers (STT) enqueue without touching the input —
      // the user may already be typing the next message.
      enqueueMessage(userMsg, images, audioEmotions, false)
      return
    }
    setQueuePaused(false)
    if (!userMsg && !audioEmotions?.audio_prosody && !(images && images.length)) return
    sendInFlightRef.current = true
    setChatLoading(true)
    const batcher = new StreamBatcher()
    setInput('')
    const imagesToSend = images ?? [...pendingImages]
    setPendingImages([])
    const attachmentsToSend: AttachmentMeta[] = pendingFiles
      .filter(f => f.status === 'ready')
      .map(f => ({ id: f.attachmentId, name: f.name, ext: f.ext }))
    setPendingFiles([])
    setError(null)
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
    const conversationId = getOrCreateChatConversationId(activeModelRef.current, userId)

    try {
      // Stop continuous emotion polling and flush the last batch
      await stopEmotionPolling()
      const phases = buildEmotionPhases()
      const fused = computeOverallFused(phases)

      const newMessages: Message[] = [...messagesRef.current, { role: 'user', content: userMsg, attachments: attachmentsToSend.length ? attachmentsToSend : undefined, emotion: fused ? { fused_emotion: fused, phases } : undefined }]
      setMessages(newMessages)
      persistMessages(conversationId, newMessages)
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
        // Extract best audio prosody from continuous audio snapshots (fallback if STT didn't provide one)
        console.info('[Chat] sendChatMessageFromText audioSnapshotsRef.length:', audioSnapshotsRef.current.length, 'audioEmotions?.audio_prosody:', audioEmotions?.audio_prosody)
        if (!audioEmotions?.audio_prosody && audioSnapshotsRef.current.length > 0) {
          const bestAudio = audioSnapshotsRef.current.reduce((best, curr) =>
            curr.confidence > best.confidence ? curr : best
          )
          emotionPayload.audio_prosody = bestAudio
          console.info('[Chat] Fallback audio_prosody from snapshots:', bestAudio)
        }
        audioSnapshotsRef.current = []
      }

      const payload: any = {
        model: activeModelRef.current,
        messages: newMessages.map((m) => ({ role: m.role, content: m.content })),
        temperature: 0.7,
        stream: true,
        conversation_id: conversationId,
      }
      if (emotionPayload) {
        payload.emotion_context = emotionPayload
      }
      if (imagesToSend.length > 0) {
        payload.images = imagesToSend
      }

      // Reset streaming TTS state for new message
      stopAllTTS()

      attachAbortRef.current?.abort()
      abortRef.current = new AbortController()
      const res = await fetch(`${getApiUrl()}/chat/completions`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
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

      const pendingMessages: Message[] = [
        ...newMessages,
        { role: 'assistant', content: '', reasoning: '' },
      ]
      setMessages(pendingMessages)

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
              if (liveDelta) assistantLiveUsage = liveDelta
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
              if (content) {
                assistantContent += content
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
              }
              if (reasoning) {
                assistantReasoning += reasoning
                hasReasoning = true
              }
              if (toolExecution) {
                terminalEventsAcc = [...terminalEventsAcc, toolExecution]
              }
              if (toolProgress) {
                assistantToolCalls = addToolProgress(assistantToolCalls, toolProgress)
              }
              if (toolCalls && toolCalls.length > 0) {
                for (const call of toolCalls) {
                  assistantToolCalls = resolveToolProgress(assistantToolCalls, call)
                  const idx = call.id ? assistantToolCalls.findIndex((item) => item.id === call.id) : -1
                  if (idx < 0) assistantToolCalls.push(call)
                  else assistantToolCalls[idx] = { ...assistantToolCalls[idx], ...call }
                }
              }
              if (subagent) {
                assistantSubagents = upsertSubagentEvent(assistantSubagents, subagent)
              }
              batcher.schedule(applyAccumulated)
            } catch {
              // skip invalid JSON
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
      persistMessages(conversationId, finalMessages)
      loadMemoryForConversation(conversationId)
      switchEmotionPhase('reading')

      // In phone mode: TTS the last assistant message, then start STT
      if (phoneModeRef.current && assistantContent.trim()) {
        const lastIdx = finalMessages.length - 1
        lastAssistantIdxRef.current = lastIdx
        if (!ttsPlayerRef.current?.hasPlayedAny()) {
          setTimeout(() => speakMessage(assistantContent.trim(), lastIdx), 200)
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
        setError(`${$t('common.error')}: ${err.message}`)
        setMessages((prev) => {
          const next: Message[] = [...prev, { role: 'assistant' as const, content: `${$t('common.error')}: ${err.message}` }]
          persistMessages(conversationId, next)
          return next
        })
      }
    } finally {
      sendInFlightRef.current = false
      setChatLoading(false)
      abortRef.current = null
      if (!phoneModeRef.current) {
        setTimeout(() => textareaRef.current?.focus(), 0)
      }
    }
  }

  function handleModelChange(name: string) {
    setModel(name)
    activeModelRef.current = name
    setMessageQueue([])
    setQueuePaused(false)
    setPendingImages([])
    setPendingFiles([])
    const convId = getOrCreateChatConversationId(name, userId)
    loadMessagesForConversation(convId)
    attachToRunIfActive(convId)
    loadMemoryForConversation(convId)
    fetchAudioConfig(name)
    fetchEmotionConfig(name)
  }

  async function loadMemoryForConversation(conversationId: string) {
    try {
      const res = await agentMemory.get(conversationId)
      setAgentMemories(res.memories)
    } catch {
      setAgentMemories([])
    }
  }

  async function handleSaveChatMemory(assistantName: string, memory: string) {
    const convId = getOrCreateChatConversationId(activeModelRef.current, userId)
    try {
      await agentMemory.set(convId, assistantName, memory)
      setAgentMemories((prev) =>
        prev.map((m) => (m.assistant_name === assistantName ? { ...m, memory, updated_at: new Date().toISOString() } : m))
      )
      setEditingMemory(null)
    } catch (e: any) {
      setNotice({ danger: true, message: 'Failed to save memory: ' + String(e.message || e) })
    }
  }

  async function handleDeleteChatMemory(assistantName: string) {
    const convId = getOrCreateChatConversationId(activeModelRef.current, userId)
    try {
      await agentMemory.delete(convId, assistantName)
      setAgentMemories((prev) => prev.filter((m) => m.assistant_name !== assistantName))
    } catch (e: any) {
      setNotice({ danger: true, message: 'Failed to delete memory: ' + String(e.message || e) })
    }
  }

  /* Scroll handling */
  const scrollToBottom = useCallback((instant = false) => {
    if (!userScrolledUpRef.current && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: instant ? 'auto' : 'smooth' })
    }
  }, [])

  useEffect(() => {
    // Instant scrolling while streaming — smooth animation cannot keep up with
    // batched updates and piles up expensive scroll interruptions.
    scrollToBottom(chatLoading)
  }, [messages, chatLoading, scrollToBottom])

  /* ------------------------------------------------------------------ */
  /*  Message queue (in-memory; drains automatically after each run)     */
  /* ------------------------------------------------------------------ */

  useEffect(() => {
    if (chatLoading || queuePaused || sendInFlightRef.current || messageQueue.length === 0) return
    const head = messageQueue[0]
    setMessageQueue((q) => q.slice(1))
    sendChatMessageFromText(
      head.text,
      head.audioEmotions as { audio_prosody?: EmotionContextPayload['audio_prosody'] | null } | undefined,
      head.images,
    ).catch((err) => {
      if (err?.name === 'AbortError') return
      console.error('[Chat] queued message send failed:', err)
      setError(String(err?.message || err))
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatLoading, messageQueue, queuePaused])

  useEffect(() => {
    const container = messagesContainerRef.current
    if (!container) return

    const onScroll = () => {
      const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100
      userScrolledUpRef.current = !nearBottom
    }
    container.addEventListener('scroll', onScroll)
    return () => container.removeEventListener('scroll', onScroll)
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

  /* Send message */
  async function sendChatMessage() {
    const batcher = new StreamBatcher()
    const hasReadyFiles = pendingFiles.some(f => f.status === 'ready')
    if (!input.trim() && pendingImages.length === 0 && !hasReadyFiles) return
    if (chatLoading || sendInFlightRef.current) {
      enqueueMessage(input.trim(), [...pendingImages])
      setPendingImages([])
      setPendingFiles([])
      return
    }
    setQueuePaused(false)
    sendInFlightRef.current = true
    setChatLoading(true)
    const userMsg = input.trim()
    const imagesToSend = [...pendingImages]
    setInput('')
    setPendingImages([])
    const attachmentsToSend: AttachmentMeta[] = pendingFiles
      .filter(f => f.status === 'ready')
      .map(f => ({ id: f.attachmentId, name: f.name, ext: f.ext }))
    setPendingFiles([])
    setError(null)
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
    const conversationId = getOrCreateChatConversationId(activeModelRef.current, userId)

    try {
      // Stop continuous emotion polling and flush the last batch
      await stopEmotionPolling()
      const phases = buildEmotionPhases()
      const fused = computeOverallFused(phases)

      const newMessages: Message[] = [...messagesRef.current, { role: 'user', content: userMsg, attachments: attachmentsToSend.length ? attachmentsToSend : undefined, emotion: fused ? { fused_emotion: fused, phases } : undefined }]
      setMessages(newMessages)
      persistMessages(conversationId, newMessages)
      if (webcamActive) {
        startEmotionPolling()
      }

      // Capture pre-send emotion snapshot
      const preSendEmotion = await capturePreSendEmotion()
      const emotionPayload = buildEmotionPayload()
      if (emotionPayload && preSendEmotion) {
        emotionPayload.visual_pre_send = preSendEmotion
      }
      if (emotionPayload) {
        if (fused) emotionPayload.fused = fused
        if (phases.length > 0) {
          emotionPayload.phases = phases
        }
        // Extract best audio prosody from continuous audio snapshots
        console.info('[Chat] audioSnapshotsRef.length before send:', audioSnapshotsRef.current.length)
        if (audioSnapshotsRef.current.length > 0) {
          const bestAudio = audioSnapshotsRef.current.reduce((best, curr) =>
            curr.confidence > best.confidence ? curr : best
          )
          emotionPayload.audio_prosody = bestAudio
          console.info('[Chat] Setting audio_prosody from continuous snapshots:', bestAudio)
          audioSnapshotsRef.current = []
        }
      }

      const payload: any = {
        model: activeModelRef.current,
        messages: newMessages.map((m) => ({ role: m.role, content: m.content })),
        temperature: 0.7,
        stream: true,
        conversation_id: conversationId,
      }
      if (emotionPayload) {
        payload.emotion_context = emotionPayload
      }
      if (imagesToSend.length > 0) {
        payload.images = imagesToSend
      }

      // Reset streaming TTS state for new message
      stopAllTTS()

      attachAbortRef.current?.abort()
      abortRef.current = new AbortController()
      const res = await fetch(`${getApiUrl()}/chat/completions`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
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

      const pendingMessages: Message[] = [
        ...newMessages,
        { role: 'assistant', content: '', reasoning: '' },
      ]
      setMessages(pendingMessages)

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
              if (liveDelta) assistantLiveUsage = liveDelta
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
              if (content) {
                assistantContent += content
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
              }
              if (reasoning) {
                assistantReasoning += reasoning
                hasReasoning = true
              }
              if (toolExecution) {
                terminalEventsAcc = [...terminalEventsAcc, toolExecution]
              }
              if (toolProgress) {
                assistantToolCalls = addToolProgress(assistantToolCalls, toolProgress)
              }
              if (toolCalls && toolCalls.length > 0) {
                for (const call of toolCalls) {
                  assistantToolCalls = resolveToolProgress(assistantToolCalls, call)
                  const idx = call.id ? assistantToolCalls.findIndex((item) => item.id === call.id) : -1
                  if (idx < 0) assistantToolCalls.push(call)
                  else assistantToolCalls[idx] = { ...assistantToolCalls[idx], ...call }
                }
              }
              if (subagent) {
                assistantSubagents = upsertSubagentEvent(assistantSubagents, subagent)
              }
              batcher.schedule(applyAccumulated)
            } catch {
              // skip invalid JSON
            }
          }
        }
      }

      // Flush remaining buffered text as a final segment
      if (streamTTS && voiceId && ttsBufferRef.current.trim()) {
        const clean = cleanTextForTTS(ttsBufferRef.current.trim())
        if (clean) {
          ttsPlayerRef.current!.add(clean, voiceId, fetchTTSSegment, 0)
        }
        ttsBufferRef.current = ''
      }

      batcher.flushNow()
      setMessages((prev) => {
        const next = [...prev]
        if (backendEmotion && next.length >= 2) {
          const prevEmotion = next[next.length - 2].emotion
          const backendNormalized = normalizeEmotion(backendEmotion)
          const mergedEmotion: Message['emotion'] = backendNormalized
            ? { ...prevEmotion, ...backendNormalized, phases: backendNormalized.phases ?? prevEmotion?.phases }
            : prevEmotion
          next[next.length - 2] = { ...next[next.length - 2], emotion: mergedEmotion }
        }
        const existing = next[next.length - 1]
        next[next.length - 1] = {
          role: 'assistant',
          content: assistantContent,
          reasoning: hasReasoning ? assistantReasoning : undefined,
          toolCalls: assistantToolCalls.length > 0 ? settleToolProgress(assistantToolCalls) : undefined,
          subagents: assistantSubagents.length > 0 ? assistantSubagents : undefined,
          todoList: assistantTodoList,
          observerEvents: observerEventsAcc.length > 0 ? observerEventsAcc : existing?.observerEvents,
          terminalEvents: existing?.terminalEvents,
          retrievalInfo: assistantRetrievalInfo !== undefined ? assistantRetrievalInfo : existing?.retrievalInfo,
          metrics: assistantMetrics ?? existing?.metrics,
        }
        persistMessages(conversationId, next)
        return next
      })
      // Refresh agent memory after agent run completes
      loadMemoryForConversation(conversationId)
      switchEmotionPhase('reading')
    } catch (err: any) {
      batcher.flushNow()
      stopAllTTS()
      settleLastAssistantTools()
      if (err.name !== 'AbortError') {
        setError(`${$t('common.error')}: ${err.message}`)
        setMessages((prev) => {
          const next: Message[] = [...prev, { role: 'assistant' as const, content: `${$t('common.error')}: ${err.message}` }]
          persistMessages(conversationId, next)
          return next
        })
      }
    } finally {
      sendInFlightRef.current = false
      setChatLoading(false)
      abortRef.current = null
      if (!phoneModeRef.current) {
        setTimeout(() => textareaRef.current?.focus(), 0)
      }
    }
  }

  async function resetChat() {
    await stopEmotionPolling()
    attachAbortRef.current?.abort()
    setMessages([])
    setError(null)
    setAgentMemories([])
    setEditingMemory(null)
    const oldConvId = sessionStorage.getItem(adminChatConversationIdKey(activeModelRef.current))
    if (oldConvId) {
      persistMessages(oldConvId, [])
      // Delete the conversation server-side: its artifacts/uploads would
      // otherwise persist forever once the client forgets the id.
      deleteChatConversation(getApiUrl(), oldConvId)
    }
    sessionStorage.removeItem(adminChatConversationIdKey(activeModelRef.current))
    getOrCreateChatConversationId(activeModelRef.current, userId)
    setMessageQueue([])
    setQueuePaused(false)
  }

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
    const convId = sessionStorage.getItem(adminChatConversationIdKey(activeModelRef.current))
    if (!convId) return
    fetch(`${getApiUrl()}/chat/runs/${encodeURIComponent(convId)}/cancel`, {
      method: 'POST',
      credentials: 'include',
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

  function newQueueId(): string {
    return (crypto as any).randomUUID ? (crypto as any).randomUUID() : `q-${Date.now()}-${Math.random()}`
  }

  function enqueueMessage(
    text: string,
    images?: string[],
    audioEmotions?: { audio_prosody?: EmotionContextPayload['audio_prosody'] | null },
    clearInput = true,
  ) {
    if (!text && !(images && images.length)) return
    setMessageQueue((q) => {
      if (q.length >= 20) return q
      return [...q, { id: newQueueId(), text, images, audioEmotions }]
    })
    if (clearInput) {
      setInput('')
      if (images) setPendingImages([])
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
  const isAgentModel = assistantConfigs.find((a) => a.name === model)?.agent_enabled === true

  async function injectQueued(item: QueuedChatMessage) {
    const convId = sessionStorage.getItem(adminChatConversationIdKey(activeModelRef.current))
    if (!convId) return
    try {
      const res = await fetch(`${getApiUrl()}/chat/runs/${encodeURIComponent(convId)}/input`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: item.text, images: item.images }),
      })
      const data = await res.json().catch(() => null)
      if (res.ok && data?.accepted) {
        setMessageQueue((q) => q.filter((m) => m.id !== item.id))
      } else {
        setError($t('chatQueue.injectFailed') + (data?.reason ? `: ${data.reason}` : ''))
      }
    } catch (e: any) {
      setError($t('chatQueue.injectFailed') + `: ${String(e?.message || e)}`)
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

  function stopGeneration() {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
    attachAbortRef.current?.abort()
    requestRunCancel()
    setQueuePaused(true)
  }

  /* ------------------------------------------------------------------ */
  /*  Conversation history                                                */
  /* ------------------------------------------------------------------ */
  async function fetchHistory() {
    if (!userId) return
    setHistoryLoading(true)
    setHistoryError(null)
    try {
      const result = await conversations.list({
        conversation_id: `dashboard-${userId}-`,
        page: 1,
        page_size: 50,
        sort_by: 'last_message_at',
        sort_order: 'desc',
      })
      setHistoryItems(result.items)
    } catch (e: any) {
      setHistoryError(String(e.message || e))
    } finally {
      setHistoryLoading(false)
    }
  }

  function normalizeConversationMessages(response: { messages: any[] }): Message[] {
    const loadedMessages: Message[] = []
    for (const msg of response.messages) {
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
              tool_name: tc.tool_name,
              id: tc.id,
              args: tc.args,
              ok: tc.ok,
              error: tc.error || undefined,
              duration_ms: tc.duration_ms || undefined,
              output: tc.output || undefined,
              kind: tc.kind,
              metadata: tc.metadata,
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

  async function loadConversation(conversationId: string) {
    setChatLoading(true)
    setError(null)
    setMessageQueue([])
    setQueuePaused(false)
    try {
      const response = await conversations.detail(conversationId)
      const loadedMessages = normalizeConversationMessages(response)
      const firstModel = response.messages.find((m) => m.model_name)?.model_name
      if (firstModel) {
        const isAssistant = assistants.includes(firstModel)
        const isChain = chains.some((c) => `chain:${c.id}` === firstModel)
        if (isAssistant || isChain) {
          setModel(firstModel)
          activeModelRef.current = firstModel
        }
      }
      sessionStorage.setItem(adminChatConversationIdKey(activeModelRef.current), conversationId)
      setMessages(loadedMessages)
      persistMessages(conversationId, loadedMessages)
      loadMemoryForConversation(conversationId)
      if (webcamActive) {
        startEmotionPolling('reading')
      }
      setHistoryOpen(false)
      setPreviewConversationId(null)
      attachToRunIfActive(conversationId)
    } catch (e: any) {
      setError(`${$t('common.error')}: ${e.message}`)
    } finally {
      setChatLoading(false)
    }
  }

  async function loadPreview(conversationId: string) {
    setPreviewLoading(true)
    setPreviewError(null)
    try {
      const response = await conversations.detail(conversationId)
      const loadedMessages = normalizeConversationMessages(response)
      setPreviewConversationId(conversationId)
      setPreviewMessages(loadedMessages)
    } catch (e: any) {
      setPreviewError(`${$t('common.error')}: ${e.message}`)
    } finally {
      setPreviewLoading(false)
    }
  }

  useEffect(() => {
    if (historyOpen && userId) {
      fetchHistory()
    }
  }, [historyOpen, userId])

  /* ------------------------------------------------------------------ */
  /*  Tool call modal                                                       */
  /* ------------------------------------------------------------------ */
  function ToolCallModal({
    tool,
    onClose,
  }: {
    tool: ToolCallLive | null
    onClose: () => void
  }) {
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
      console.log('[ToolCallModal] retrievedDocIds:', [...sources])
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
        const res = await knowledge.getOKFGraph(ragCollection)
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
            <button className="btn-icon" onClick={onClose} title={$t('common.close')}>
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
                  Arguments
                </div>
                <pre className="chat-toolcall-overlay-args" style={{ margin: 0, borderRadius: '8px' }}>
                  {JSON.stringify(tool.args, null, 2)}
                </pre>
              </div>
              {typeof tool.duration_ms === 'number' && (
                <div className="chat-toolcall-overlay-meta" style={{ marginBottom: '12px' }}>
                  Duration: {tool.duration_ms}ms
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
                    Output
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
        const res = await knowledge.getOKFGraph(okfCollection)
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
            <button className="btn-icon" onClick={onClose} title={$t('common.close')}>
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
    if (!html) return null
    return (
      <div className="modal-overlay" onClick={onClose}>
        <div className="modal-box preview-modal" onClick={(e) => e.stopPropagation()}>
          <div className="preview-header">
            <div>
              <h4>HTML Preview</h4>
            </div>
            <div className="preview-header-actions">
              <button className="btn-icon" onClick={onClose} title="Close">
                <X size={18} />
              </button>
            </div>
          </div>
          <div className="preview-content">
            <iframe className="preview-frame" sandbox="allow-scripts" srcDoc={html} title="HTML preview" />
          </div>
        </div>
      </div>
    )
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendChatMessage().catch((err) => {
        console.error('Send message error:', err)
      })
    }
  }

  function copyMessageContent(content: string, idx: number) {
    navigator.clipboard.writeText(content).then(() => copy(`msg-${idx}`))
  }

  const isEmpty = messages.length === 0

  return (
    <div className="chat-page">
      {/* Header */}
      <div className="chat-page-header">
        <div style={{ flex: 1, display: 'flex', alignItems: 'center' }}>
          <h2 style={{ margin: 0 }}>{$t('nav.chat')}</h2>
        </div>
        <select
          value={model}
          onChange={(e) => handleModelChange(e.target.value)}
          disabled={loadingModels}
          className="chat-model-select"
          style={{ margin: '0 auto' }}
        >
          {assistants.length === 0 && chains.length === 0 && !loadingModels && (
            <option value={model}>{model}</option>
          )}
          {assistants.length > 0 && (
            <optgroup label={$t('chat.assistants') || 'Assistenten'}>
              {assistants.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </optgroup>
          )}
          {chains.length > 0 && (
            <optgroup label={$t('chat.chains') || 'Chains'}>
              {chains.map((c) => (
                <option key={c.id} value={`chain:${c.id}`}>
                  {c.name}
                </option>
              ))}
            </optgroup>
          )}
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
          {userId && (
            <button
              className="chat-history-header-btn"
              onClick={() => { setHistoryOpen(true) }}
              title="Verlauf"
            >
              <History size={16} />
            </button>
          )}
          {(agentMemories.length > 0 || assistantConfigs.find((a) => a.name === model)?.agent_memory_enabled) && (
            <button
              className="chat-memory-header-btn"
              onClick={() => { setMemoryPanelOpen(true) }}
              title="Memory"
            >
              <Brain size={16} />
            </button>
          )}
          <button className="chat-reset-btn" onClick={resetChat} title={$t('widget.reset')}>
            <RotateCcw size={16} />
          </button>
        </div>
      </div>

      {/* Messages area */}
      <ChatLanguageContext.Provider value={language}>
      <HtmlPreviewContext.Provider value={{ open: setPreviewHtml, close: () => setPreviewHtml(null) }}>
        <div className="chat-messages-scroll" ref={messagesContainerRef}>
          <div className="chat-messages-col">
            {isEmpty && (
              <div className="chat-empty-state">
                <div className="chat-empty-icon">
                  <Sparkles size={32} />
                </div>
                <h3>{activeModelRef.current}</h3>
                {assistantConfigs.find((a) => a.name === activeModelRef.current)?.description && (
                  <p className="chat-empty-desc">
                    {assistantConfigs.find((a) => a.name === activeModelRef.current)!.description}
                  </p>
                )}
                <p>{language === 'de' ? 'Wie kann ich dir helfen?' : 'How can I help you today?'}</p>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`chat-message-row ${msg.role}${msg.systemNote ? ' system-note' : ''}`}>
                {msg.role === 'assistant' && (
                  <div className="chat-avatar assistant-avatar">
                    <Bot size={18} />
                  </div>
                )}

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
                          labels={{
                            download: $t('chatArtifact.download'),
                            preview: $t('chatArtifact.preview'),
                            hidePreview: $t('chatArtifact.hidePreview'),
                            downloadFailed: $t('chatArtifact.downloadFailed'),
                            previewFailed: $t('chatArtifact.previewFailed'),
                            exportAs: $t('chatArtifact.exportAs'),
                            pageOf: (c, t) => $t('chatArtifact.pageOf', { current: c, total: t }),
                          }}
                        />
                      ))}
                    </div>
                  )}

                  {/* Meta row: timestamp placeholder + copy + speak + emotion */}
                  <div className="chat-message-meta">
                    {msg.role === 'assistant' && (
                      <>
                        {audioConfig?.tts?.button_visible && (
                          <button
                            className={`chat-message-copy${playingMsgIdx === idx ? ' speaking' : ''}`}
                            onClick={() => speakMessage(msg.content, idx)}
                            title={playingMsgIdx === idx ? 'Stop' : 'Vorlesen'}
                          >
                            {playingMsgIdx === idx ? (
                              <>
                                <Square size={12} className="speak-stop-icon" /> Stop
                              </>
                            ) : (
                              <>
                                <Volume2 size={12} /> Vorlesen
                              </>
                            )}
                          </button>
                        )}
                        <button
                          className="chat-message-copy"
                          onClick={() => copyMessageContent(msg.content, idx)}
                          title={$t('common.copy')}
                        >
                          {copiedId === `msg-${idx}` ? (
                            <>
                              <Check size={12} /> {$t('common.copied')}
                            </>
                          ) : (
                            <>
                              <Copy size={12} /> {$t('common.copy')}
                            </>
                          )}
                        </button>
                        {msg.metrics ? (
                          <ResponseMetricsBadge metrics={msg.metrics} ragContextTokens={msg.retrievalInfo?.rag_context_tokens} lang={language} />
                        ) : (
                          chatLoading && idx === messages.length - 1 && (
                            <LiveTokenChip
                              live={msg.liveUsage}
                              streamedChars={(msg.content?.length ?? 0) + (msg.reasoning?.length ?? 0)}
                              lang={language}
                            />
                          )
                        )}
                      </>
                    )}
                    {msg.role === 'user' && msg.emotion?.fused_emotion && (
                      <span className="chat-message-emotion-tooltip-wrapper">
                        <span
                          className="chat-message-emotion"
                          title={`Source: ${msg.emotion.fused_emotion.source}, Confidence: ${msg.emotion.fused_emotion.confidence}`}
                        >
                          <Smile size={10} />
                          {msg.emotion.fused_emotion.primary_emotion} ({msg.emotion.fused_emotion.intensity})
                        </span>
                        <div className="chat-message-emotion-tooltip">
                          <div className="emotion-tooltip-header">Emotion History</div>
                          {msg.emotion.text_emotion && (
                            <div className="emotion-tooltip-row">
                              <span>Text:</span>
                              <span>{msg.emotion.text_emotion.primary_emotion} ({msg.emotion.text_emotion.confidence})</span>
                            </div>
                          )}
                          {msg.emotion.visual_pre_send && (
                            <div className="emotion-tooltip-row">
                              <span>Webcam:</span>
                              <span>{msg.emotion.visual_pre_send.primary_emotion} ({msg.emotion.visual_pre_send.confidence})</span>
                            </div>
                          )}
                          {msg.emotion.audio_text_emotion && (
                            <div className="emotion-tooltip-row">
                              <span>Audio Text:</span>
                              <span>{msg.emotion.audio_text_emotion.primary_emotion} ({msg.emotion.audio_text_emotion.confidence})</span>
                            </div>
                          )}
                          {msg.emotion.audio_prosody && (
                            <div className="emotion-tooltip-row">
                              <span>Audio Prosody:</span>
                              <span>{msg.emotion.audio_prosody.primary_emotion} ({msg.emotion.audio_prosody.confidence})</span>
                            </div>
                          )}
                          {msg.emotion.phases && msg.emotion.phases.length > 0 && (
                            <>
                              <div className="emotion-tooltip-header" style={{ marginTop: 6, borderTop: '1px solid var(--border)', paddingTop: 4 }}>
                                {language === 'de' ? 'Phasen' : 'Phases'}
                              </div>
                              {msg.emotion.phases.map((p) => {
                                const phaseLabels: Record<string, string> = {
                                  waiting_for_response: language === 'de' ? 'Warten auf Antwort' : 'While Waiting',
                                  reading: language === 'de' ? 'Beim Lesen' : 'While Reading',
                                  typing: language === 'de' ? 'Beim Tippen' : 'While Typing',
                                  speaking: language === 'de' ? 'Beim Sprechen' : 'While Speaking',
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
                            <span>Fused:</span>
                            <span>{msg.emotion.fused_emotion.primary_emotion} ({msg.emotion.fused_emotion.confidence}, {msg.emotion.fused_emotion.intensity})</span>
                          </div>
                        </div>
                      </span>
                    )}
                  </div>
                </div>

                {msg.role === 'user' && (
                  <div className="chat-avatar user-avatar">
                    <User size={18} />
                  </div>
                )}
              </div>
            ))}

            {error && (
              <div className="chat-error-row">
                <div className="chat-error-bubble">
                  {error}
                  <button className="chat-retry-btn" onClick={sendChatMessage}>
                    {$t('common.send')}
                  </button>
                </div>
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
        conversationId={activeConversationId || sessionStorage.getItem(adminChatConversationIdKey(activeModelRef.current)) || undefined}
        apiUrl={getApiUrl()}
        onClose={() => setSelectedSubagent(null)}
      />
      <RAGModal info={selectedRAGInfo} onClose={() => setSelectedRAGInfo(undefined)} />

      {/* Agent Memory Modal */}
      {memoryPanelOpen && (
        <div
          className="chat-memory-modal-overlay"
          onClick={() => { setMemoryPanelOpen(false); setEditingMemory(null) }}
        >
          <div
            className="chat-memory-modal-box"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="chat-history-modal-header">
              <h4>Agent Memory</h4>
              <button
                className="btn-icon"
                onClick={() => { setMemoryPanelOpen(false); setEditingMemory(null) }}
                title={$t('common.close')}
              >
                <X size={18} />
              </button>
            </div>
            <div className="chat-memory-modal-body">
              {agentMemories.length === 0 && (
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', padding: '16px 0' }}>
                  No agent memory for this conversation yet.
                </div>
              )}
              {agentMemories.map((mem) => (
                <div key={mem.assistant_name} className="chat-memory-modal-item">
                  <div className="chat-memory-modal-item-header">
                    <strong>{mem.assistant_name}</strong>
                    <div className="chat-memory-modal-item-actions">
                      {editingMemory?.assistant_name !== mem.assistant_name && (
                        <button
                          className="btn-secondary btn-small"
                          style={{ padding: '2px 8px', fontSize: 12 }}
                          onClick={() => setEditingMemory({ assistant_name: mem.assistant_name, memory: mem.memory })}
                        >
                          Edit
                        </button>
                      )}
                      <button
                        className="btn-danger btn-small"
                        style={{ padding: '2px 8px', fontSize: 12 }}
                        onClick={() => setMemDeleteTarget(mem.assistant_name)}
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>
                  {editingMemory?.assistant_name === mem.assistant_name ? (
                    <>
                      <textarea
                        value={editingMemory.memory}
                        onChange={(e) => setEditingMemory({ ...editingMemory, memory: e.target.value })}
                        style={{ width: '100%', minHeight: 200, fontFamily: 'inherit', fontSize: 13, padding: 8, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--card)', color: 'var(--text)' }}
                      />
                      <div style={{ display: 'flex', gap: 8, marginTop: 8, justifyContent: 'flex-end' }}>
                        <button className="btn-secondary btn-small" onClick={() => setEditingMemory(null)}>
                          Cancel
                        </button>
                        <button
                          className="btn-primary btn-small"
                          onClick={() => handleSaveChatMemory(mem.assistant_name, editingMemory.memory)}
                        >
                          <Save size={12} /> Save
                        </button>
                      </div>
                    </>
                  ) : (
                    <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12, background: 'var(--bg)', padding: 10, borderRadius: 6, margin: 0, color: 'var(--text)' }}>
                      {mem.memory}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Conversation History Modal */}
      {historyOpen && (
        <div
          className="chat-history-modal-overlay"
          onClick={() => setHistoryOpen(false)}
        >
          <div
            className="chat-history-modal-box"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="chat-history-modal-header">
              <h4>Conversation History</h4>
              <button
                className="btn-icon"
                onClick={() => setHistoryOpen(false)}
                title={$t('common.close')}
              >
                <X size={18} />
              </button>
            </div>
            <div className="chat-history-modal-body">
              {historyLoading && (
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', padding: '16px 0' }}>
                  {$t('common.loading')}
                </div>
              )}
              {historyError && (
                <div style={{ fontSize: 13, color: 'var(--danger)', padding: '16px 0' }}>
                  {historyError}
                </div>
              )}
              {!historyLoading && !historyError && historyItems.length === 0 && (
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', padding: '16px 0' }}>
                  No past conversations.
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
                      {item.model_name || '—'}
                    </div>
                    <div className="chat-history-modal-item-meta">
                      {item.message_count} messages · {new Date(item.last_message_at).toLocaleString(language === 'de' ? 'de-DE' : 'en-GB', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                  <button
                    className="btn-primary btn-small"
                    style={{ padding: '2px 8px', fontSize: 11, whiteSpace: 'nowrap' }}
                    onClick={(e) => { e.stopPropagation(); loadConversation(item.conversation_id) }}
                    disabled={chatLoading}
                  >
                    Load
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Conversation Preview Modal */}
      {previewConversationId && (
        <div
          className="chat-history-modal-overlay"
          onClick={() => setPreviewConversationId(null)}
        >
          <div
            className="chat-history-preview-box"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="chat-history-modal-header">
              <h4>{historyItems.find((i) => i.conversation_id === previewConversationId)?.model_name || 'Conversation Preview'}</h4>
              <button
                className="btn-icon"
                onClick={() => setPreviewConversationId(null)}
                title={$t('common.close')}
              >
                <X size={18} />
              </button>
            </div>
            <div className="chat-history-preview-body">
              {previewLoading && (
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', padding: '16px 0' }}>
                  {$t('common.loading')}
                </div>
              )}
              {previewError && (
                <div style={{ fontSize: 13, color: 'var(--danger)', padding: '16px 0' }}>
                  {previewError}
                </div>
              )}
              {!previewLoading && !previewError && previewMessages && previewMessages.map((msg, idx) => (
                <div key={idx} className={`chat-history-preview-message ${msg.role}`}>
                  {msg.role === 'assistant' && (
                    <div className="chat-avatar assistant-avatar" style={{ width: 28, height: 28, minWidth: 28 }}>
                      <Bot size={14} />
                    </div>
                  )}
                  {msg.role === 'user' && (
                    <div className="chat-avatar user-avatar" style={{ width: 28, height: 28, minWidth: 28 }}>
                      <User size={14} />
                    </div>
                  )}
                  <div className="chat-history-preview-bubble">
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
                    <MarkdownContent text={msg.content} style={{ fontSize: 13 }} />
                  </div>
                </div>
              ))}
            </div>
            <div className="chat-history-preview-footer">
              <button
                className="btn-primary"
                onClick={() => {
                  if (previewConversationId) {
                    loadConversation(previewConversationId)
                  }
                }}
                disabled={chatLoading}
              >
                In Chat laden
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Input bar */}
      <div className="chat-input-bar">
        <div
          className={`chat-input-inner${phoneMode ? ' phone-mode' : ' chat-input-dynamic'}`}
          style={{ position: 'relative' }}
          {...(uploadsEnabled ? { onDragOver: handleDragOver, onDragLeave: handleDragLeave, onDrop: handleDrop } : {})}
        >
          <QueuedMessagesBar
            items={messageQueue}
            paused={queuePaused}
            labels={{
              title: $t('chatQueue.title'),
              sendNow: $t('chatQueue.sendNow'),
              inject: isAgentModel ? $t('chatQueue.inject') : undefined,
              edit: $t('chatQueue.edit'),
              remove: $t('chatQueue.remove'),
              resume: $t('chatQueue.resume'),
              imageMessage: $t('chatQueue.imageMessage'),
            }}
            onSendNow={sendQueuedNow}
            onInject={isAgentModel ? injectQueued : undefined}
            onEdit={editQueued}
            onDelete={deleteQueued}
            onReorder={reorderQueued}
            onResume={() => setQueuePaused(false)}
            hidden={phoneMode}
          />
          {uploadsEnabled && dragOver && (
            <div
              style={{ position: "absolute", inset: 0, background: "rgba(0,122,255,0.1)", border: "2px dashed rgba(0,122,255,0.5)", borderRadius: "inherit", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 10, pointerEvents: "none" }}
            >
              <span style={{ color: "rgba(0,122,255,0.8)", fontSize: "14px" }}>{chatFileUploadEnabled ? 'Drop files here' : 'Drop images here'}</span>
            </div>
          )}
          <UploadTray
            pendingImages={chatImageUploadEnabled ? pendingImages : []}
            pendingFiles={pendingFiles}
            onRemoveImage={removePendingImage}
            onRemoveFile={removePendingFile}
            labels={{
              uploading: language === 'de' ? 'Wird hochgeladen…' : 'Uploading…',
              failed: language === 'de' ? 'Upload fehlgeschlagen' : 'Upload failed',
            }}
          />
          {uploadsEnabled && (
            <input
              ref={fileInputRef}
              type="file"
              accept={chatFileUploadEnabled ? ACCEPT_STRING : 'image/*'}
              multiple
              style={{ display: "none" }}
              onChange={(e) => {
                if (e.target.files) handleFiles(e.target.files)
                e.target.value = ""
              }}
            />
          )}
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
          {uploadsEnabled && !chatLoading && !isRecording && (
            <button
              className="chat-image-btn"
              title={chatFileUploadEnabled ? "File upload" : "Image upload"}
              onClick={() => fileInputRef.current?.click()}
              style={{ flexShrink: 0, background: "none", border: "none", cursor: "pointer", color: "var(--text-secondary, #888)", padding: "4px", display: "flex", alignItems: "center" }}
            >
              <ImagePlus size={18} />
            </button>
          )}
          <textarea
            ref={textareaRef}
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
            placeholder={language === 'de' ? 'Nachricht eingeben...' : 'Type your message...'}
            disabled={isRecording}
            className={`chat-textarea${phoneMode ? ' phone-mode-hidden' : ''}`}
          />
          {/* Phone mode toggle */}
          {audioConfig?.phone?.button_visible && (!chatLoading || phoneMode) && (
            <button
              className={`chat-phone-btn ${phoneMode ? 'active' : ''}`}
              title={phoneMode ? 'Telefonmodus beenden' : 'Telefonmodus'}
              onClick={togglePhoneMode}
            >
              {phoneMode ? <PhoneOff size={18} /> : <Phone size={18} />}
            </button>
          )}
          {/* Emotion toggle (webcam + audio) */}
          {emotionConfig?.enabled && (!chatLoading || phoneMode) && (
            <button
              className={`chat-emotion-btn ${webcamActive ? 'active' : ''} ${emotionRuntimeStatus === 'error' ? 'error' : ''}${(isRecording && !phoneMode) || (phoneMode && !emotionConfig?.enabled) ? ' phone-mode-hidden' : ''}`}
              title={webcamActive ? 'Emotion Detection aktiv' : 'Emotion Detection starten'}
              onClick={toggleEmotionDetection}
            >
              <Smile size={18} />
              {webcamActive && emotionRuntimeStatus === 'ready' && (
                <span className="chat-emotion-indicator" />
              )}
            </button>
          )}
          {/* Mic / STT button */}
          {audioConfig?.stt?.button_visible && !chatLoading && (
            <button
              className={`chat-mic-btn ${isRecording ? 'recording' : ''} ${awaitingUserSpeech ? 'awaiting' : ''}${phoneMode ? ' phone-mode-hidden' : ''}`}
              title={isRecording ? 'Aufnahme stoppen' : 'Spracheingabe'}
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
            <button
              onClick={stopGeneration}
              className={`chat-stop-indicator${phoneMode ? ' phone-mode-hidden' : ''}`}
              title={$t('widget.stop') || 'Stop'}
            >
              <span className="chat-stop-pulse" />
              <span className="chat-stop-icon">
                <Square size={14} fill="currentColor" />
              </span>
            </button>
          )}
          <button
            onClick={sendChatMessage}
            disabled={(!input.trim() && pendingImages.length === 0 && !pendingFiles.some(f => f.status === 'ready')) || isRecording}
            className={`chat-send-btn${phoneMode ? ' phone-mode-hidden' : ''}`}
            title={chatLoading ? $t('chatQueue.enqueue') : $t('common.send')}
          >
            {chatLoading ? <ListPlus size={18} /> : <Send size={18} />}
          </button>
        </div>
      </div>
      {wsEnabled && wsOpen && (
        <WorkspaceBrowser
          apiUrl={getApiUrl()}
          assistant={model}
          conversationId={activeConversationId}
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
        onConfirm={() => {
          setWebcamConsentOpen(false)
          setWebcamConsent(true)
          void startWebcamEmotionInner()
        }}
        onCancel={() => setWebcamConsentOpen(false)}
      />
      <ConfirmModal
        open={memDeleteTarget !== null}
        title={$t('common.delete')}
        message={memDeleteTarget ? `Delete agent memory for ${memDeleteTarget}?` : ''}
        onConfirm={() => {
          const name = memDeleteTarget
          setMemDeleteTarget(null)
          if (name) void handleDeleteChatMemory(name)
        }}
        onCancel={() => setMemDeleteTarget(null)}
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

