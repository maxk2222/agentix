/* Public Chat Auth & Conversations API */

function getApiBase(): string {
  const path = window.location.pathname
  const idx = path.lastIndexOf('/chat')
  if (idx === -1) return ''
  return path.slice(0, idx)
}

const API_BASE = window.location.origin + getApiBase()

/* ------------------------------------------------------------------ */
/*  Simple in-memory cache with TTL                                   */
/* ------------------------------------------------------------------ */
interface CacheEntry<T> {
  data: T
  timestamp: number
}

class ApiCache {
  private store = new Map<string, CacheEntry<unknown>>()

  get<T>(key: string, ttlMs: number): T | null {
    const entry = this.store.get(key)
    if (!entry) return null
    if (Date.now() - entry.timestamp > ttlMs) {
      this.store.delete(key)
      return null
    }
    return entry.data as T
  }

  set<T>(key: string, data: T): void {
    this.store.set(key, { data, timestamp: Date.now() })
  }

  invalidate(keyPrefix: string): void {
    for (const key of this.store.keys()) {
      if (key.startsWith(keyPrefix)) this.store.delete(key)
    }
  }

  clear(): void {
    this.store.clear()
  }
}

const _cache = new ApiCache()

const AUTH_CACHE_TTL = 30_000      // 30s
const CONV_CACHE_TTL = 120_000     // 2min
const MSG_CACHE_TTL = 15_000       // 15s — conversations keep changing while open

function _formatErrorDetail(detail: any): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((err: any) => {
        if (typeof err === 'string') return err
        const loc = err.loc?.join('.') || 'field'
        const msg = err.msg || JSON.stringify(err)
        return `${loc}: ${msg}`
      })
      .join('; ')
  }
  return JSON.stringify(detail)
}

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error')
    let message = text
    try {
      const data = JSON.parse(text)
      if (data.detail) message = _formatErrorDetail(data.detail)
    } catch {}
    throw new Error(message)
  }
  return res.json() as Promise<T>
}

export type ChatUserInfo = {
  id: string
  username: string
  display_name: string
  role: string
  email?: string | null
  groups?: string[]
}

export type MeResponse = {
  authenticated: boolean
  user?: ChatUserInfo
}

export type ConversationItem = {
  id: string
  conversation_id: string
  title: string | null
  model_name: string | null
  created_at: string
  updated_at: string
  message_count: number
}

export type ConversationMessage = {
  id: string
  conversation_id: string
  user_content?: string | null
  assistant_content?: string | null
  assistant_reasoning?: string | null
  model_name?: string | null
  retrieval_info?: unknown
  tool_calls?: Array<{
    id?: string | null
    tool_name: string
    kind?: 'tool' | 'subagent' | 'subagent_control'
    args: Record<string, unknown>
    ok: boolean
    error?: string | null
    duration_ms?: number | null
    output?: string | null
    metadata?: Record<string, any> | null
  }> | null
  emotion?: Record<string, unknown> | null
  created_at: string
}

export const publicAuth = {
  register: (username: string, password: string, display_name: string, email?: string) =>
    api<{ success: boolean; message: string }>('/public-chat/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username, password, display_name, ...(email ? { email } : {}) }),
    }),
  login: (username: string, password: string) =>
    api<{ success: boolean; user: ChatUserInfo }>('/public-chat/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),
  logout: () => {
    _cache.invalidate('auth:')
    return api<{ success: boolean }>('/public-chat/auth/logout', { method: 'POST' })
  },
  me: () => {
    const cached = _cache.get<MeResponse>('auth:me', AUTH_CACHE_TTL)
    if (cached) return Promise.resolve(cached)
    return api<MeResponse>('/public-chat/auth/me').then((data) => {
      _cache.set('auth:me', data)
      return data
    })
  },
  changePassword: (currentPassword: string, newPassword: string) =>
    api<{ success: boolean }>('/public-chat/auth/change-password', {
      method: 'POST',
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    }),
}

export const publicConversations = {
  list: () => {
    const cached = _cache.get<{ items: ConversationItem[] }>('conv:list', CONV_CACHE_TTL)
    if (cached) return Promise.resolve(cached)
    return api<{ items: ConversationItem[] }>('/public-chat/conversations').then((data) => {
      _cache.set('conv:list', data)
      return data
    })
  },
  messages: (conversationId: string) => {
    const key = `conv:msg:${conversationId}`
    const cached = _cache.get<{ messages: ConversationMessage[] }>(key, MSG_CACHE_TTL)
    if (cached) return Promise.resolve(cached)
    return api<{ messages: ConversationMessage[] }>(`/public-chat/conversations/${encodeURIComponent(conversationId)}/messages`).then((data) => {
      _cache.set(key, data)
      return data
    })
  },
  delete: (conversationId: string) => {
    _cache.invalidate('conv:')
    return api<{ success: boolean }>(`/public-chat/conversations/${encodeURIComponent(conversationId)}`, {
      method: 'DELETE',
    })
  },
  updateTitle: (conversationId: string, title: string) => {
    _cache.invalidate('conv:')
    return api<{ success: boolean }>(`/public-chat/conversations/${encodeURIComponent(conversationId)}/title`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    })
  },
  /* Read-only agent memory view for an owned conversation. */
  memory: (conversationId: string) =>
    api<{ conversation_id: string; memories: any[] }>(
      `/public-chat/conversations/${encodeURIComponent(conversationId)}/memory`,
    ),
  setMemory: (conversationId: string, assistantName: string, memory: string) =>
    api<{ success: boolean }>(
      `/public-chat/conversations/${encodeURIComponent(conversationId)}/memory`,
      { method: 'PUT', body: JSON.stringify({ assistant_name: assistantName, memory }) },
    ),
  deleteMemory: (conversationId: string, assistantName: string) =>
    api<{ success: boolean }>(
      `/public-chat/conversations/${encodeURIComponent(conversationId)}/memory?assistant_name=${encodeURIComponent(assistantName)}`,
      { method: 'DELETE' },
    ),
  /* Drop the conversation list + message caches — call after a turn
     completes or a conversation changes so stale data is never shown. */
  invalidate: () => _cache.invalidate('conv:'),
}

export const publicEmotion = {
  detectTextEmotion: (text: string) =>
    api<{
      primary_emotion: string
      confidence: number
      intensity: 'low' | 'medium' | 'high'
      source: string
      timestamp_ms: number
      signals?: string[]
    }>('/emotion/detect-text', {
      method: 'POST',
      body: JSON.stringify({ text }),
    }),
}

export type OKFConcept = {
  okf_concept_id: string
  okf_type: string
  okf_title: string
  source: string
  collection: string
  description: string
  date_downloaded: string
}

export const publicKnowledge = {
  getOKFGraph: (collection: string) =>
    api<{ concepts: OKFConcept[] }>(`/v1/knowledge/okf-graph?collection=${encodeURIComponent(collection)}`),
}
