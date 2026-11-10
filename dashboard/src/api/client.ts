function getAdminBase(): string {
  const path = window.location.pathname
  const idx = path.lastIndexOf('/dashboard/')
  if (idx === -1) return ''
  return path.slice(0, idx)
}

const API_BASE = getAdminBase()

const CACHE_TTL_MS = 60000
const pending = new Map<string, Promise<{ body: string; status: number; headers: Record<string, string> }>>()
const cache = new Map<string, { body: string; status: number; headers: Record<string, string>; ts: number }>()
// Bumped by every mutation — a GET that was in-flight while a write ran must
// not re-populate the cache with the pre-mutation body it may still carry.
let cacheEpoch = 0

async function fetchApi(url: string, options?: RequestInit): Promise<Response> {
  const method = (options?.method ?? 'GET').toUpperCase()
  const cacheable = (method === 'GET' || method === 'HEAD') && !url.includes('/admin/auth/')
  const key = `${method}:${url}:${JSON.stringify(options?.body ?? '')}`

  if (!cacheable) {
    // Any write can change arbitrary resources (config edits reshape model
    // lists, branding, permissions, …) — drop the whole short-lived cache.
    cacheEpoch++
    cache.clear()
  }

  if (cacheable) {
    const cached = cache.get(key)
    if (cached && Date.now() - cached.ts < CACHE_TTL_MS) {
      return new Response(cached.body, { status: cached.status, headers: cached.headers })
    }

    const existing = pending.get(key)
    if (existing) {
      const entry = await existing
      return new Response(entry.body, { status: entry.status, headers: entry.headers })
    }
  }

  const epoch = cacheEpoch
  const fetchPromise = fetch(url, { credentials: 'include', ...options }).then(async (res) => {
    const body = await res.text()
    return { body, status: res.status, headers: Object.fromEntries(res.headers.entries()) }
  })

  if (cacheable) {
    pending.set(key, fetchPromise)
    fetchPromise
      .then((entry) => {
        if (entry.status >= 200 && entry.status < 300 && epoch === cacheEpoch) {
          cache.set(key, { ...entry, ts: Date.now() })
        }
      })
      .catch(() => {
        // Fehlerhafte Requests nicht cachen
      })
      .finally(() => {
        pending.delete(key)
        setTimeout(() => cache.delete(key), CACHE_TTL_MS)
      })
  }

  const entry = await fetchPromise
  return new Response(entry.body, { status: entry.status, headers: entry.headers })
}

export function invalidateCache(pattern?: string): void {
  if (!pattern) {
    cache.clear()
    return
  }
  for (const key of cache.keys()) {
    if (key.includes(pattern)) cache.delete(key)
  }
}

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetchApi(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error')
    let message = text
    try {
      const data = JSON.parse(text)
      if (data.detail) message = data.detail
    } catch {}
    throw new Error(message)
  }
  return res.json() as Promise<T>
}

export type UserInfo = {
  id: string
  username: string
  display_name: string
  role: string
}

export type MeResponse = {
  authenticated: boolean
  role?: string
  user?: UserInfo
  permissions: string[]
  session_ttl_seconds?: number
}

export const auth = {
  login: (username: string, password: string) =>
    api<{ success: boolean }>('/admin/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),
  logout: () => api<{ success: boolean }>('/admin/auth/logout', { method: 'POST' }),
  me: () => api<MeResponse>('/admin/auth/me'),
  changePassword: (currentPassword: string, newPassword: string) =>
    api<{ success: boolean }>('/admin/auth/change-password', {
      method: 'POST',
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    }),
}

export type RangeParams = {
  range_hours: number
  bucket?: string
  model?: string
}

export type ReasoningConfig = {
  effort?: string
  enabled?: boolean
  budget?: number
  level?: string
}

export type ReasoningControl = {
  key: 'effort' | 'enabled' | 'budget' | 'level'
  type: 'select' | 'toggle' | 'number'
  options?: string[]
  min?: number
  max?: number
  default?: boolean
}

export type ReasoningSchema = {
  family: string
  wire: string
  controls: ReasoningControl[]
  assumed: boolean
  probed?: boolean
  canonical_keys: string[]
  model: string
}

export type AssistantEntry = {
  name: string
  description?: string
  model: string
  temperature: number
  collections?: string[]
  reasoning?: ReasoningConfig | null
  system_instructions: string
  context_window?: number | null
  max_output_tokens?: number | null
  rag_reserved_tokens?: number | null
  min_recent_turns?: number | null
  extra_parameters?: Record<string, unknown> | null
  extra_body?: Record<string, unknown> | null
  enable_thinking?: boolean | null
  provider_id?: string | null
  agent_enabled?: boolean
  tools?: Record<string, Record<string, unknown>>
  max_iterations?: number | null
  max_tool_calls?: number | null
  max_parallel_tool_calls?: number | null
  agent_memory_enabled?: boolean
  subagents?: SubagentsConfig | null
  observer?: ObserverConfig | null
  workspace?: AgentWorkspaceConfig | null
  api_accessible?: boolean
  audio_stt_enabled?: boolean
  audio_tts_enabled?: boolean
  audio_phone_enabled?: boolean
  audio_tts_auto_read?: boolean
  audio_tts_streaming_enabled?: boolean
  audio_voice_id?: string | null
  emotion_detection_enabled?: boolean
  emotion_text_enabled?: boolean
  emotion_webcam_enabled?: boolean
  emotion_prompt_steering_enabled?: boolean
  emotion_analytics_enabled?: boolean
}

export type SubagentsConfig = {
  enabled?: boolean
  max_concurrent?: number | null
  max_depth?: number | null
  max_total?: number | null
  report_answer_chars?: number | null
  /** Tool names withheld from spawned subagents; absent = backend defaults. */
  exclude_tools?: string[]
}

export type AgentWorkspaceConfig = {
  enabled?: boolean
  /** 'assistant' = shared for everyone, 'user' = per chat user, 'conversation' = per conversation. */
  scope?: 'assistant' | 'user' | 'conversation'
  /** Independently gated: files become listable/downloadable via the API. */
  api_enabled?: boolean
  /** Allow 'view' to return rendered page images — enable only for vision-capable models. */
  vision_read?: boolean
  max_file_mb?: number | null
  max_total_mb?: number | null
  max_files?: number | null
}

export type ObserverConfig = {
  enabled?: boolean
  /** Assistant whose model runs the checks; absent = `observer` assistant → default. */
  assistant?: string | null
  every_iterations?: number | null
  every_seconds?: number | null
  max_checks?: number | null
  max_nudges?: number | null
  check_timeout_seconds?: number | null
  include_subagents?: boolean
  instructions?: string | null
}

export type ObserverTodoOp = {
  action: string
  task_id?: string
  title?: string
  ok?: boolean
  detail?: string
}

/* Emitted at iteration boundaries while a run-observer check starts
   (kind=check_started) or a finished check is applied (assessment/action). */
export type ObserverEvent = {
  kind: 'check_started' | 'assessment' | 'action'
  iteration?: number
  trigger?: string[]
  check?: number
  assessment?: string
  nudge?: string
  nudge_dropped?: string
  todo_ops_applied?: ObserverTodoOp[]
  check_ms?: number
  model?: string
  ts?: number
}

export type SubagentActivityItem = {
  iteration: number
  tool_name: string
  args_preview?: string
  status: 'running' | 'ok' | 'error'
  duration_ms?: number | null
  error?: string | null
  output_preview?: string | null
}

/* Transient live-progress payload carried by `subagent` stream events while
   a nested tool call is in flight (backend emits `activity_item`). */
export type SubagentToolProgress = {
  id?: string | null
  tool_name: string
  stage?: string
  iteration?: number
  args_preview?: string
}

export type SubagentMeta = {
  id: string
  parent_id?: string | null
  owner_key?: string
  label?: string
  task: string
  assistant?: string
  depth?: number
  status: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  duration_ms?: number | null
  iterations?: number
  current_step?: string
  reasoning_preview?: string
  answer?: string
  error?: string | null
  activity?: SubagentActivityItem[]
  /** Total activity entries on the record — `activity` itself is capped. */
  activity_total?: number
  nested_tool_calls?: ToolCallItem[]
  nested_usage?: Record<string, number>
  activity_item?: SubagentToolProgress | null
  observer_events?: ObserverEvent[]
  /** Nudges injected into this subagent mid-run (parent agent or user). */
  nudges?: { source?: string; text: string; at?: number }[]
}

export type SubagentEvent = Partial<SubagentMeta> & {
  id: string
  stage: 'queued' | 'started' | 'progress' | 'finished' | 'cancelled'
  detail?: string
  ok?: boolean
}

export type AgentTodoTask = {
  id: string
  title: string
  note?: string
  status: 'pending' | 'in_progress' | 'completed' | 'skipped'
}

export type AgentTodoSnapshot = {
  version: number
  tasks: AgentTodoTask[]
}

export type InstructionsData = {
  assistants: AssistantEntry[]
}

export type CollectionInfo = {
  name: string
  files: string[]
  subfolders: CollectionInfo[]
  indexed_files?: string[]
  db_files?: string[]
  db_only?: boolean
  updated_at?: string
  vector_doc_count?: number
  file_updated_at?: Record<string, string>
  file_vector_doc_count?: Record<string, number>
}

export type WebDAVSource = {
  id: number
  name: string
  webdav_url: string
  remote_path: string
  username: string
  enabled: boolean
  convert_to_okf: boolean
  last_sync_at?: string
  last_sync_status?: string
  last_sync_error?: string
  created_at?: string
  updated_at?: string
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

export type ConversationsListParams = {
  date_from?: string
  date_to?: string
  conversation_id?: string
  model?: string
  min_user_messages?: number
  user_id?: string
  query?: string
  page?: number
  page_size?: number
  sort_by?: string
  sort_order?: string
}

export type ConversationSummary = {
  conversation_id: string
  message_count: number
  user_message_count: number
  assistant_message_count: number
  first_message_at: string
  last_message_at: string
  model_name?: string | null
  retrieval_rows: number
  title?: string | null
  conv_model_name?: string | null
  user_id?: string | null
  username?: string | null
  display_name?: string | null
  used_api_key?: boolean
  chat_metrics?: {
    request_count: number
    avg_response_time_ms: number
    avg_time_to_first_token_ms: number
    total_prompt_tokens: number
    total_completion_tokens: number
    total_tokens: number
    avg_prompt_tokens_per_second: number
    avg_completion_tokens_per_second: number
    total_input_chars: number
    total_output_chars: number
    total_rag_context_tokens: number
    total_reasoning_tokens: number
  } | null
  vectorstore_metrics?: {
    retrieval_count: number
    avg_retrieval_time_ms: number
    total_chunks: number
    error_rate: number
    total_rag_context_tokens: number
  } | null
  emotion_summary?: {
    dominant: string
    text_count: number
    webcam_count: number
  } | null
}

export type ToolCallItem = {
  id?: string | null
  iteration?: number
  tool_name: string
  kind?: 'tool' | 'subagent' | 'subagent_control'
  args: Record<string, unknown>
  ok: boolean
  error?: string | null
  duration_ms?: number | null
  output?: string | null
  metadata?: { subagent?: SubagentMeta; subagent_results?: SubagentMeta[] } | null
}

export type ConversationMessage = {
  id: string
  conversation_id: string
  user_content?: string | null
  assistant_content?: string | null
  assistant_reasoning?: string | null
  model_name?: string | null
  chat_history?: unknown
  retrieval_info?: unknown
  tool_calls?: ToolCallItem[] | null
  emotion?: {
    fused_emotion?: {
      primary_emotion: string
      confidence: number
      intensity: 'low' | 'medium' | 'high'
      source: string
      timestamp_ms: number
      signals?: string[]
    } | null
    text_emotion?: {
      primary_emotion: string
      confidence: number
      intensity: 'low' | 'medium' | 'high'
      source: string
      timestamp_ms: number
      signals?: string[]
    } | null
    visual_pre_send?: {
      primary_emotion: string
      confidence: number
      intensity: 'low' | 'medium' | 'high'
      source: string
      timestamp_ms: number
      sampled_frames?: number
      valid_face_frames?: number
      face_detection_rate?: number
    } | null
    audio_prosody?: {
      primary_emotion: string
      confidence: number
      intensity: 'low' | 'medium' | 'high'
      source: string
      timestamp_ms: number
    } | null
    phases?: Array<{
      phase: string
      primary_emotion: string
      confidence: number
      intensity: 'low' | 'medium' | 'high'
      source: string
      timestamp_ms: number
    }> | null
    consent?: {
      webcam_opt_in: boolean
      frames_sent_to_backend: boolean
      audio_opt_in: boolean
    }
  } | null
  created_at: string
}

export type ConversationsListResponse = {
  total: number
  page: number
  page_size: number
  total_pages: number
  items: ConversationSummary[]
}

export type ConversationUser = {
  user_id: string
  username: string | null
  display_name: string | null
  conversation_count: number
}

export type ConversationsExportParams = {
  date_from: string
  date_to: string
  conversation_id?: string
  model?: string
  min_user_messages?: number
  user_id?: string
  query?: string
  format: 'json' | 'csv' | 'txt'
}

export const config = {
  getInstructions: () => api<InstructionsData>('/admin/config/instructions'),
  saveInstructions: (data: InstructionsData) =>
    api<{ success: boolean }>('/admin/config/instructions', {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  downloadInstructionsYaml: () =>
    fetchApi(`${API_BASE}/admin/config/instructions/download`).then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => 'Unknown error')
        let message = text
        try {
          const data = JSON.parse(text)
          if (data.detail) message = data.detail
        } catch {}
        throw new Error(message)
      }
      return res.blob()
    }),
  uploadInstructionsYaml: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return fetchApi(`${API_BASE}/admin/config/instructions/upload`, {
      method: 'POST',
      body: fd,
    }).then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => 'Unknown error')
        let message = text
        try {
          const data = JSON.parse(text)
          if (data.detail) message = data.detail
        } catch {}
        throw new Error(message)
      }
      return res.json() as Promise<{ success: boolean }>
    })
  },
  apply: () => api<{ message: string }>('/admin/config/apply', { method: 'POST' }),
  listModels: () => api<string[]>('/admin/config/models'),
  listModelsByProvider: () => api<Record<string, { name: string; models: string[] }>>('/admin/config/provider-models'),
  instructionsHistory: (limit?: number) => {
    const qs = limit ? `?limit=${limit}` : ''
    return api<Array<{
      id: number
      created_at: string
      created_by: string
      is_current: boolean
    }>>(`/admin/config/instructions/history${qs}`)
  },
  rollbackInstructions: (versionId: number) =>
    api<{ success: boolean }>(`/admin/config/instructions/rollback/${versionId}`, { method: 'POST' }),
  getVersion: (versionId: number) =>
    api<{
      id: number
      yaml_content: string
      created_at: string
      created_by: string
      is_current: boolean
    }>(`/admin/config/instructions/version/${versionId}`),
  diffVersions: (fromVersionId: number, toVersionId: number) =>
    api<{
      diff: string
      structured_changes: Array<{
        type: 'added' | 'removed' | 'modified'
        assistant: string
        fields: Array<{
          field: string
          old_value: unknown
          new_value: unknown
          text_diff?: string
        }>
      }>
      from: { id: number; created_at: string; created_by: string; is_current: boolean }
      to: { id: number; created_at: string; created_by: string; is_current: boolean }
    }>('/admin/config/instructions/diff', {
      method: 'POST',
      body: JSON.stringify({ from_version_id: fromVersionId, to_version_id: toVersionId }),
    }),
}

export const knowledge = {
  listCollections: () => api<CollectionInfo[]>('/admin/knowledge/collections?include_counts=false'),
  getCollectionStats: (name: string) =>
    api<{ vector_doc_count: number; file_vector_doc_count: Record<string, number> }>(
      `/admin/knowledge/collections/${encodeURIComponent(name)}/stats`
    ),
  createCollection: (name: string) =>
    api<{ success: boolean; name: string }>(`/admin/knowledge/collections/${encodeURIComponent(name)}`, {
      method: 'POST',
    }),
  renameCollection: (name: string, newName: string) =>
    api<{ success: boolean; old_name: string; new_name: string }>(
      `/admin/knowledge/collections/${encodeURIComponent(name)}/rename`,
      {
        method: 'PUT',
        body: new URLSearchParams({ new_name: newName }),
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      }
    ),
  deleteCollection: (name: string) =>
    api<{ success: boolean }>(`/admin/knowledge/collections/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),
  rebuildCollection: (name: string) =>
    api<{ success: boolean; collection: string; files: string[]; indexing?: object }>(
      `/admin/knowledge/collections/${encodeURIComponent(name)}/rebuild`,
      { method: 'POST' }
    ),
  upload: (collection: string, files: File | File[]) => {
    const fileList = Array.isArray(files) ? files : [files]
    const fd = new FormData()
    fd.append('collection', collection)
    fileList.forEach((file) => fd.append('file', file))
    return fetchApi(`${API_BASE}/admin/knowledge/upload`, {
      method: 'POST',
      body: fd,
    }).then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => 'Unknown error')
        let message = text
        try {
          const data = JSON.parse(text)
          if (data.detail) message = data.detail
        } catch {}
        throw new Error(message)
      }
      return res.json() as Promise<{
        success: boolean;
        collection: string;
        filenames: string[];
        indexing?: {
          initialized?: boolean;
          sources_count?: number;
          chunks?: number;
          sources?: Array<{
            source: string;
            action: string;
            deleted?: number;
            added?: number;
          }>;
          error?: string;
        };
      }>
    })
  },
  delete: (collection: string, filename: string) =>
    api<{ success: boolean }>(`/admin/knowledge/${encodeURIComponent(collection)}/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
    }),
  previewUrl: (collection: string, filename: string) =>
    `${API_BASE}/admin/knowledge/preview/${encodeURIComponent(collection)}/${encodeURIComponent(filename)}`,
  readText: (collection: string, filename: string) =>
    api<{ content: string }>(`/admin/knowledge/text/${encodeURIComponent(collection)}/${encodeURIComponent(filename)}`),
  saveText: (collection: string, filename: string, content: string) =>
    api<{
      success: boolean;
      collection: string;
      filename: string;
      indexing?: {
        initialized?: boolean;
        sources_count?: number;
        chunks?: number;
        sources?: Array<{
          source: string;
          action: string;
          deleted?: number;
          added?: number;
        }>;
        error?: string;
      };
    }>(`/admin/knowledge/text/${encodeURIComponent(collection)}/${encodeURIComponent(filename)}`, {
      method: 'PUT',
      body: JSON.stringify({ content }),
      headers: { 'Content-Type': 'application/json' },
    }),

  // WebDAV sources
  listWebDAVSources: () => api<WebDAVSource[]>('/admin/knowledge/webdav'),
  createWebDAVSource: (payload: Omit<WebDAVSource, 'id' | 'last_sync_at' | 'last_sync_status' | 'last_sync_error' | 'created_at' | 'updated_at'>) =>
    api<WebDAVSource>('/admin/knowledge/webdav', {
      method: 'POST',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    }),
  updateWebDAVSource: (name: string, payload: Partial<Omit<WebDAVSource, 'id' | 'name' | 'last_sync_at' | 'last_sync_status' | 'last_sync_error' | 'created_at' | 'updated_at'>>) =>
    api<WebDAVSource>(`/admin/knowledge/webdav/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    }),
  deleteWebDAVSource: (name: string) =>
    api<{ success: boolean }>(`/admin/knowledge/webdav/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),
  syncWebDAVSource: (name: string) =>
    api<{ success: boolean; downloaded?: number; skipped?: number; deleted?: number; error?: string; cancelled?: boolean }>(
      `/admin/knowledge/webdav/${encodeURIComponent(name)}/sync`,
      { method: 'POST' }
    ),
  cancelWebDAVSync: (name: string) =>
    api<{ success: boolean }>(`/admin/knowledge/webdav/${encodeURIComponent(name)}/cancel`, { method: 'POST' }),
  testWebDAVConnection: (payload: { webdav_url: string; remote_path: string; username: string; password: string; name?: string }) =>
    api<{ success: boolean; file_count?: number; error?: string }>('/admin/knowledge/webdav/test', {
      method: 'POST',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    }),
  getWebDAVStatus: (name: string) =>
    api<{ name: string; last_sync_at: string | null; last_sync_status: string | null; last_sync_error: string | null; file_count: number; sync_progress?: { phase: string; current: number; total: number; message?: string } }>(
      `/admin/knowledge/webdav/${encodeURIComponent(name)}/status`
    ),

  // OKF graph
  getOKFGraph: (collection: string) =>
    api<{ concepts: OKFConcept[] }>(`/v1/knowledge/okf-graph?collection=${encodeURIComponent(collection)}`),
}

export const agentMemory = {
  check: (conversationIds: string[]) =>
    api<{ conversation_ids_with_memory: string[] }>('/admin/agent-memory/check', {
      method: 'POST',
      body: JSON.stringify({ conversation_ids: conversationIds }),
    }),
  get: (conversationId: string) =>
    api<{ conversation_id: string; memories: Array<{ assistant_name: string; memory: string; created_at: string | null; updated_at: string | null }> }>(
      `/admin/agent-memory/${encodeURIComponent(conversationId)}`
    ),
  set: (conversationId: string, assistantName: string, memory: string) =>
    api<{ success: boolean; conversation_id: string; assistant_name: string }>(
      `/admin/agent-memory/${encodeURIComponent(conversationId)}`,
      { method: 'PUT', body: JSON.stringify({ assistant_name: assistantName, memory }) }
    ),
  delete: (conversationId: string, assistantName: string) =>
    api<{ success: boolean; deleted: boolean }>(
      `/admin/agent-memory/${encodeURIComponent(conversationId)}?assistant_name=${encodeURIComponent(assistantName)}`,
      { method: 'DELETE' }
    ),
}

export const metrics = {
  summary: (range_hours: number) =>
    api<{
      total_requests: number
      total_messages: number
      total_tokens: number
      total_prompt_tokens: number
      total_answer_tokens: number
      avg_response_time_ms: number
      p50_response_time_ms: number
      p95_response_time_ms: number
      avg_time_to_first_token_ms: number
      avg_time_to_first_reasoning_token_ms: number
      avg_time_to_first_answer_token_ms: number
      p95_time_to_first_token_ms: number
      p50_time_to_first_answer_token_ms: number
      p95_time_to_first_answer_token_ms: number
      avg_prompt_tokens_per_second: number
      avg_completion_tokens_per_second: number
      avg_tokens_per_second: number
      error_rate: number
      total_rag_context_tokens: number
      total_reasoning_tokens: number
    }>(`/admin/metrics/summary?range_hours=${range_hours}`),
  byModel: (range_hours: number) =>
    api<
      {
        model_name: string
        requests: number
        total_messages: number
        avg_response_time_ms: number
        p95_response_time_ms: number
        avg_time_to_first_token_ms: number
        avg_time_to_first_reasoning_token_ms: number
        avg_time_to_first_answer_token_ms: number
        avg_prompt_tokens_per_second: number
        avg_completion_tokens_per_second: number
        total_tokens: number
        total_prompt_tokens: number
        total_answer_tokens: number
        error_rate: number
        total_rag_context_tokens: number
        total_reasoning_tokens: number
      }[]
    >(`/admin/metrics/by-model?range_hours=${range_hours}`),
  history: (params: RangeParams) => {
    const qs = new URLSearchParams()
    qs.set('range_hours', String(params.range_hours))
    if (params.bucket) qs.set('bucket', params.bucket)
    if (params.model) qs.set('model', params.model)
    return api<
      {
        bucket: string
        requests: number
        total_messages: number
        avg_response_time_ms: number
        p95_response_time_ms: number
        avg_time_to_first_token_ms: number
        avg_time_to_first_reasoning_token_ms: number
        avg_time_to_first_answer_token_ms: number
        avg_prompt_tokens_per_second: number
        avg_completion_tokens_per_second: number
        avg_tokens_per_second: number
        total_tokens: number
        total_prompt_tokens: number
        total_completion_tokens: number
        total_answer_tokens: number
        total_rag_context_tokens: number
        total_reasoning_tokens: number
      }[]
    >(`/admin/metrics/history?${qs.toString()}`)
  },
  usage: (range_hours: number) =>
    api<{
      hourly_activity: { dow: number; hour: number; count: number }[]
      error_breakdown: { error_type: string; count: number }[]
      stream_share_pct: number
      avg_message_count: number
      rag_share_pct: number
      reasoning_share_pct: number
      avg_agent_iterations: number
      top_tools: { name: string; count: number }[]
      top_users: { user_id: string | null; display_name: string | null; requests: number; total_tokens: number }[]
      top_collections: { name: string; count: number }[]
      retrieval_status: { ok: number; error: number; skipped: number }
    }>(`/admin/metrics/usage?range_hours=${range_hours}`),
  historyByApiKey: (params: RangeParams) => {
    const qs = new URLSearchParams()
    qs.set('range_hours', String(params.range_hours))
    if (params.bucket) qs.set('bucket', params.bucket)
    return api<(Record<string, number | string> & { bucket: string })[]>(`/admin/metrics/history/api-key?${qs.toString()}`)
  },
  models: (range_hours: number) =>
    api<string[]>(`/admin/metrics/models?range_hours=${range_hours}`),
  blockedSummary: (range_hours: number) =>
    api<{
      total_blocked: number
      auth_blocks: number
      size_blocks: number
      validation_blocks: number
      other_blocks: number
    }>(`/admin/metrics/blocked-summary?range_hours=${range_hours}`),
  blockedHistory: (params: RangeParams) => {
    const qs = new URLSearchParams()
    qs.set('range_hours', String(params.range_hours))
    if (params.bucket) qs.set('bucket', params.bucket)
    return api<
      {
        bucket: string
        blocked: number
        auth_blocks: number
        size_blocks: number
        validation_blocks: number
      }[]
    >(`/admin/metrics/blocked-history?${qs.toString()}`)
  },
}

export const dashboard = {
  overview: (range_hours: number, page_size = 200) =>
    api<{
      metrics_summary: {
        total_requests: number
        total_messages: number
        total_tokens: number
        total_prompt_tokens: number
        total_answer_tokens: number
        avg_response_time_ms: number
        p50_response_time_ms: number
        p95_response_time_ms: number
        avg_time_to_first_token_ms: number
        avg_time_to_first_reasoning_token_ms: number
        avg_time_to_first_answer_token_ms: number
        p95_time_to_first_token_ms: number
        p50_time_to_first_answer_token_ms: number
        p95_time_to_first_answer_token_ms: number
        avg_prompt_tokens_per_second: number
        avg_completion_tokens_per_second: number
        avg_tokens_per_second: number
        avg_time_to_first_generation_ms: number
        total_completion_tokens: number
        total_cache_tokens: number
        total_tool_result_tokens: number
        error_rate: number
        total_rag_context_tokens: number
        total_reasoning_tokens: number
      }
      usage_stats: {
        hourly_activity: { dow: number; hour: number; count: number }[]
        error_breakdown: { error_type: string; count: number }[]
        stream_share_pct: number
        avg_message_count: number
        rag_share_pct: number
        reasoning_share_pct: number
        avg_agent_iterations: number
        top_tools: { name: string; count: number }[]
        top_users: { user_id: string; display_name: string; requests: number; total_tokens: number }[]
        top_collections: { name: string; count: number }[]
        retrieval_status: { ok: number; error: number; skipped: number }
      }
      vectorstore_summary: {
        total_retrievals: number
        avg_retrieval_time_ms: number
        total_chunks: number
        error_rate: number
        total_rag_context_tokens: number
      }
      vectorstore_state: {
        total_collections: number
        total_documents: number
        collections: { name: string; documents: number; healthy: boolean }[]
      }
      conversations: ConversationsListResponse
    }>(`/admin/dashboard/overview?range_hours=${range_hours}&page_size=${page_size}`),
  liveActivity: () =>
    api<{
      generated_at: string
      runs: {
        conversation_id: string
        model: string
        kind: 'chat' | 'agent' | 'chain'
        source: string
        state: 'running' | 'done' | 'cancelled' | 'error'
        started_ts: string | null
        elapsed_s: number
        user_message: string
        phase: string
        phase_detail: string
        tokens: number | null
        llm_calls: number | null
        estimated: boolean
        tool_calls: number
        subagents_running: number
        subagents_total: number
        todos_open: number | null
        todos_total: number | null
      }[]
      automation: {
        triggers: { id: string; name: string; elapsed_s: number }[]
        backups: { id: number; kind: string; triggered_by: string; started_at: string | null }[]
        syncs: { name: string; since: string | null }[]
      }
    }>('/admin/dashboard/live-activity'),
  models: (range_hours: number) =>
    api<{
      metrics_by_model: {
        model_name: string
        requests: number
        total_messages: number
        avg_response_time_ms: number
        p95_response_time_ms: number
        avg_time_to_first_token_ms: number
        avg_time_to_first_reasoning_token_ms: number
        avg_time_to_first_answer_token_ms: number
        avg_prompt_tokens_per_second: number
        avg_completion_tokens_per_second: number
        total_tokens: number
        total_prompt_tokens: number
        total_answer_tokens: number
        error_rate: number
        total_rag_context_tokens: number
        total_reasoning_tokens: number
      }[]
      vectorstore_by_model: {
        model_name: string
        retrievals: number
        avg_retrieval_time_ms: number
        total_chunks: number
        error_rate: number
        total_rag_context_tokens: number
      }[]
    }>(`/admin/dashboard/models?range_hours=${range_hours}`),
  history: (params: RangeParams) => {
    const qs = new URLSearchParams()
    qs.set('range_hours', String(params.range_hours))
    if (params.bucket) qs.set('bucket', params.bucket)
    if (params.model) qs.set('model', params.model)
    return api<{
      metrics_history: {
        bucket: string
        requests: number
        total_messages: number
        avg_response_time_ms: number
        p95_response_time_ms: number
        avg_time_to_first_token_ms: number
        avg_time_to_first_reasoning_token_ms: number
        avg_time_to_first_answer_token_ms: number
        avg_prompt_tokens_per_second: number
        avg_completion_tokens_per_second: number
        avg_tokens_per_second: number
        total_tokens: number
        total_prompt_tokens: number
        total_completion_tokens: number
        total_answer_tokens: number
        total_rag_context_tokens: number
        total_reasoning_tokens: number
      }[]
      vectorstore_history: {
        bucket: string
        retrievals: number
        avg_retrieval_time_ms: number
        total_chunks: number
        total_rag_context_tokens: number
      }[]
      api_key_history: (Record<string, number | string> & { bucket: string })[]
      models: string[]
    }>(`/admin/dashboard/history?${qs.toString()}`)
  },
  blocked: (params: RangeParams) => {
    const qs = new URLSearchParams()
    qs.set('range_hours', String(params.range_hours))
    if (params.bucket) qs.set('bucket', params.bucket)
    return api<{
      blocked_summary: {
        total_blocked: number
        auth_blocks: number
        size_blocks: number
        validation_blocks: number
        other_blocks: number
      }
      blocked_history: {
        bucket: string
        blocked: number
        auth_blocks: number
        size_blocks: number
        validation_blocks: number
      }[]
    }>(`/admin/dashboard/blocked?${qs.toString()}`)
  },
}

export const conversations = {
  list: (params: ConversationsListParams) => {
    const qs = new URLSearchParams()
    if (params.date_from) qs.set('date_from', params.date_from)
    if (params.date_to) qs.set('date_to', params.date_to)
    if (params.conversation_id) qs.set('conversation_id', params.conversation_id)
    if (params.model) qs.set('model', params.model)
    if (params.min_user_messages !== undefined) qs.set('min_user_messages', String(params.min_user_messages))
    if (params.user_id) qs.set('user_id', params.user_id)
    if (params.query) qs.set('query', params.query)
    if (params.page !== undefined) qs.set('page', String(params.page))
    if (params.page_size !== undefined) qs.set('page_size', String(params.page_size))
    if (params.sort_by) qs.set('sort_by', params.sort_by)
    if (params.sort_order) qs.set('sort_order', params.sort_order)
    return api<ConversationsListResponse>(`/admin/conversations?${qs.toString()}`)
  },
  detail: (conversationId: string) =>
    api<{ conversation_id: string; messages: ConversationMessage[] }>(
      `/admin/conversations/${encodeURIComponent(conversationId)}`
    ),
  metadata: (conversationId: string) =>
    api<ConversationSummary>(
      `/admin/conversations/${encodeURIComponent(conversationId)}/metadata`
    ),
  models: (date_from?: string, date_to?: string) => {
    const qs = new URLSearchParams()
    if (date_from) qs.set('date_from', date_from)
    if (date_to) qs.set('date_to', date_to)
    return api<string[]>(`/admin/conversations/models?${qs.toString()}`)
  },
  listUsers: () =>
    api<{ users: ConversationUser[]; anonymous_count: number }>('/admin/conversations/users'),
  export: (params: ConversationsExportParams) => {
    const qs = new URLSearchParams()
    qs.set('date_from', params.date_from)
    qs.set('date_to', params.date_to)
    qs.set('format', params.format)
    if (params.conversation_id) qs.set('conversation_id', params.conversation_id)
    if (params.model) qs.set('model', params.model)
    if (params.min_user_messages !== undefined) qs.set('min_user_messages', String(params.min_user_messages))
    if (params.user_id) qs.set('user_id', params.user_id)
    if (params.query) qs.set('query', params.query)
    return fetchApi(`${API_BASE}/admin/conversations/export?${qs.toString()}`).then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => 'Unknown error')
        let message = text
        try {
          const data = JSON.parse(text)
          if (data.detail) message = data.detail
        } catch {}
        throw new Error(message)
      }
      return res.blob()
    })
  },
}

export const vectorstore = {
  summary: (range_hours: number) =>
    api<{
      total_retrievals: number
      avg_retrieval_time_ms: number
      total_chunks: number
      error_rate: number
      total_rag_context_tokens: number
    }>(`/admin/vectorstore/summary?range_hours=${range_hours}`),
  byModel: (range_hours: number) =>
    api<
      {
        model_name: string
        retrievals: number
        avg_retrieval_time_ms: number
        total_chunks: number
        error_rate: number
        total_rag_context_tokens: number
      }[]
    >(`/admin/vectorstore/by-model?range_hours=${range_hours}`),
  history: (params: RangeParams) => {
    const qs = new URLSearchParams()
    qs.set('range_hours', String(params.range_hours))
    if (params.bucket) qs.set('bucket', params.bucket)
    if (params.model) qs.set('model', params.model)
    return api<
      {
        bucket: string
        retrievals: number
        avg_retrieval_time_ms: number
        total_chunks: number
        total_rag_context_tokens: number
      }[]
    >(`/admin/vectorstore/history?${qs.toString()}`)
  },
  state: () =>
    api<{
      collections: { name: string; documents: number; healthy: boolean }[]
      total_collections: number
      total_documents: number
    }>('/admin/vectorstore/state'),
}

export type ProviderEntry = {
  id: string
  name: string
  base_url: string
  enabled: boolean
  is_default: boolean
  provider_type: string
  timeout_seconds: number
  max_retries: number
  extra_headers?: Record<string, string> | null
  last_checked_at?: string | null
  last_check_status?: string | null
  supports_native_tool_calls?: boolean | null
  tool_calling_checked_at?: string | null
  tool_calling_check_error?: string | null
  has_api_key: boolean
  source?: 'db' | 'env'
}

export type EffectiveConfigEntry = {
  value: unknown
  source: string
  editable: boolean
  default_value?: unknown
  env_value?: unknown | null
}

export type RuntimeConfigResponse = {
  stored: Record<string, unknown>
  effective: Record<string, EffectiveConfigEntry>
}

export type ConfigDescription = {
  label: string
  description: string
  type: string
  default: unknown
  category: string
  options?: { value: string; label: string }[]
}

export const runtimeConfig = {
  get: () => api<RuntimeConfigResponse>('/admin/config/runtime'),
  save: (values: Record<string, unknown>) =>
    api<{ success: boolean }>('/admin/config/runtime', {
      method: 'PUT',
      body: JSON.stringify({ values }),
    }),
  apply: () => api<{ message: string }>('/admin/config/runtime/apply', { method: 'POST' }),
  export: () =>
    fetchApi(`${API_BASE}/admin/config/runtime/export`).then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => 'Unknown error')
        throw new Error(text)
      }
      return res.blob()
    }),
  import: (values: Record<string, unknown>) =>
    api<{ success: boolean; imported_keys: string[] }>('/admin/config/runtime/import', {
      method: 'POST',
      body: JSON.stringify({ values }),
    }),
}

export const configDescriptions = {
  get: () => api<Record<string, ConfigDescription>>('/admin/config/descriptions'),
}

export type ProviderInput = Omit<ProviderEntry, 'last_checked_at' | 'last_check_status' | 'has_api_key'>

export const providers = {
  list: () => api<ProviderEntry[]>('/admin/config/providers'),
  get: (id: string) => api<ProviderEntry>(`/admin/config/providers/${encodeURIComponent(id)}`),
  create: (data: ProviderInput) =>
    api<{ success: boolean }>('/admin/config/providers', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  update: (id: string, data: ProviderInput) =>
    api<{ success: boolean }>(`/admin/config/providers/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  delete: (id: string) =>
    api<{ success: boolean }>(`/admin/config/providers/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
  rotateSecret: (id: string, apiKey: string) =>
    api<{ success: boolean }>(`/admin/config/providers/${encodeURIComponent(id)}/secret/rotate`, {
      method: 'POST',
      body: JSON.stringify({ api_key: apiKey }),
    }),
  check: (id: string) =>
    api<{ status: string; models: string[] }>(`/admin/config/providers/${encodeURIComponent(id)}/check`, {
      method: 'POST',
    }),
  detectToolCalling: (id: string, model?: string) =>
    api<{ success: boolean; model: string; supports_native?: boolean | null; used_content_json?: boolean; error?: string | null }>(
      `/admin/config/providers/${encodeURIComponent(id)}/detect-tool-calling`,
      { method: 'POST', body: JSON.stringify({ model }) }
    ),
  reasoningSchema: (model: string, providerId?: string) =>
    api<ReasoningSchema>(
      `/admin/config/reasoning-schema?model=${encodeURIComponent(model)}${providerId ? `&provider_id=${encodeURIComponent(providerId)}` : ''}`
    ),
  export: () =>
    fetchApi(`${API_BASE}/admin/config/providers/export`).then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => 'Unknown error')
        throw new Error(text)
      }
      return res.blob()
    }),
}

export type ApiKeyEntry = {
  id: string
  name: string
  allowed_models: string[]
  enabled: boolean
  rate_limit_rpm: number
  rate_limit_rpd: number
  source?: string
  created_at?: string | null
  updated_at?: string | null
}

export type ApiKeyPayload = {
  name: string
  allowed_models: string[]
  enabled: boolean
  rate_limit_rpm: number
  rate_limit_rpd: number
}

export type ApiKeyCreateResponse = ApiKeyEntry & {
  api_key: string
}

export const apiKeys = {
  list: () => api<ApiKeyEntry[]>('/admin/api-keys'),
  create: (data: ApiKeyPayload) =>
    api<ApiKeyCreateResponse>('/admin/api-keys', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  update: (id: string, data: ApiKeyPayload) =>
    api<ApiKeyEntry>(`/admin/api-keys/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  delete: (id: string) =>
    api<{ success: boolean }>(`/admin/api-keys/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
  models: () => api<string[]>('/admin/api-keys/models'),
}

export type DashboardUser = {
  id: string
  username: string
  display_name: string
  role: string
  is_active: boolean
  created_at?: string | null
  updated_at?: string | null
  last_login_at?: string | null
}

export const users = {
  list: () => api<DashboardUser[]>('/admin/users'),
  create: (data: { username: string; display_name: string; password: string; role: string }) =>
    api<DashboardUser>('/admin/users', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  update: (id: string, data: { username: string; display_name: string; role: string; is_active: boolean }) =>
    api<DashboardUser>(`/admin/users/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  setPassword: (id: string, password: string) =>
    api<{ success: boolean }>(`/admin/users/${encodeURIComponent(id)}/password`, {
      method: 'POST',
      body: JSON.stringify({ password }),
    }),
  disable: (id: string) =>
    api<{ success: boolean }>(`/admin/users/${encodeURIComponent(id)}/disable`, {
      method: 'POST',
    }),
  enable: (id: string) =>
    api<{ success: boolean }>(`/admin/users/${encodeURIComponent(id)}/enable`, {
      method: 'POST',
    }),
  delete: (id: string) =>
    api<{ success: boolean }>(`/admin/users/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
}

export type ToolEntry = {
  id: string
  name: string
  description: string
  enabled: boolean
  kind: string
  tool_schema: Record<string, unknown>
  config: Record<string, unknown>
  config_schema?: Record<string, unknown>[] | null
  code?: string | null
  venv_config?: Record<string, unknown>
  runtime_mode?: string
  created_at?: string | null
  updated_at?: string | null
}

export type ToolCreatePayload = Omit<ToolEntry, 'id' | 'created_at' | 'updated_at'>

export const tools = {
  list: (enabled_only = false) =>
    api<ToolEntry[]>(`/admin/tools?enabled_only=${enabled_only}`),
  create: (data: ToolCreatePayload) =>
    api<ToolEntry>('/admin/tools', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  get: (id: string) => api<ToolEntry>(`/admin/tools/${encodeURIComponent(id)}`),
  update: (id: string, data: ToolCreatePayload) =>
    api<ToolEntry>(`/admin/tools/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  delete: (id: string) =>
    api<{ success: boolean }>(`/admin/tools/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
  enable: (id: string) =>
    api<{ success: boolean }>(`/admin/tools/${encodeURIComponent(id)}/enable`, {
      method: 'POST',
    }),
  disable: (id: string) =>
    api<{ success: boolean }>(`/admin/tools/${encodeURIComponent(id)}/disable`, {
      method: 'POST',
    }),
  rebuildVenv: (id: string) =>
    api<{ success: boolean; venv_path: string }>(`/admin/tools/${encodeURIComponent(id)}/venv/rebuild`, {
      method: 'POST',
    }),
}

export type TriggerEntry = {
  id: string
  name: string
  assistant_name: string
  prompt: string
  schedule_type: string
  schedule_config: Record<string, unknown>
  enabled: boolean
  include_new_messages: boolean
  include_trigger_history: boolean
  max_history_turns: number
  watch_assistants: string[]
  target_type?: string
  target_name?: string
  timeout_seconds?: number
  is_running?: boolean
  last_run_at?: string | null
  next_run_at?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export type TriggerRunEntry = {
  id: number
  trigger_id: string
  started_at: string
  finished_at?: string | null
  status: string
  result_text?: string | null
  error_text?: string | null
  conversation_id?: string | null
}

export type TriggerCreatePayload = Omit<TriggerEntry, 'id' | 'created_at' | 'updated_at' | 'last_run_at' | 'next_run_at'>

export const triggers = {
  list: () => api<TriggerEntry[]>('/admin/triggers'),
  create: (data: TriggerCreatePayload) =>
    api<TriggerEntry>('/admin/triggers', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  get: (id: string) => api<TriggerEntry>(`/admin/triggers/${encodeURIComponent(id)}`),
  update: (id: string, data: TriggerCreatePayload) =>
    api<TriggerEntry>(`/admin/triggers/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  delete: (id: string) =>
    api<{ success: boolean }>(`/admin/triggers/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
  run: (id: string) =>
    api<{ success: boolean; run_id: number }>(`/admin/triggers/${encodeURIComponent(id)}/run`, {
      method: 'POST',
    }),
  runs: (id: string, limit = 50, status?: string, sort_order = 'desc') => {
    const qs = new URLSearchParams()
    qs.set('limit', String(limit))
    qs.set('sort_order', sort_order)
    if (status) qs.set('status', status)
    return api<TriggerRunEntry[]>(`/admin/triggers/${encodeURIComponent(id)}/runs?${qs.toString()}`)
  },
}

export type AgentChainEntry = {
  id: string
  name: string
  description: string
  enabled: boolean
  api_accessible: boolean
  flow_definition: Record<string, unknown>
  execution_options: Record<string, unknown>
  created_at?: string | null
  updated_at?: string | null
}

export type AgentChainCreatePayload = Omit<AgentChainEntry, 'id' | 'created_at' | 'updated_at'>

export type AgentChainValidateResponse = {
  valid: boolean
  errors: string[]
}

export type AgentChainNodeStatus = {
  status: 'done' | 'skipped' | 'error' | string
  chosen_route?: string
  error?: string
}

export type AgentChainTestRunResponse = {
  answer: string
  reasoning?: string
  node_outputs?: Record<string, string>
  node_status?: Record<string, AgentChainNodeStatus>
  tool_calls?: unknown[]
}

export const widgetDefaults = {
  get: () =>
    api<Record<string, unknown>>('/widget-defaults').catch(() => null),
  save: (config: Record<string, unknown>) =>
    api<{ success: boolean }>('/admin/widget-defaults', {
      method: 'POST',
      body: JSON.stringify({ config }),
    }),
}

export const branding = {
  get: () =>
    api<Record<string, unknown>>('/branding').catch(() => null),
  save: (config: Record<string, unknown>) =>
    api<{ success: boolean }>('/admin/branding', {
      method: 'POST',
      body: JSON.stringify({ config }),
    }),
}

export const publicChatConfig = {
  get: () =>
    api<Record<string, unknown>>('/public-chat/config').catch(() => null),
  save: (config: Record<string, unknown>) =>
    api<{ success: boolean }>('/admin/public-chat/config', {
      method: 'POST',
      body: JSON.stringify({ config }),
    }),
}

export type ChainRunEvent =
  | { type: 'run_start'; source?: string }
  | { type: 'node_start'; node_id: string; node_type?: string }
  | {
      type: 'node_end'
      node_id: string
      status: string
      chosen_route?: string
      error?: string | null
      duration_ms?: number
      preview?: string
    }
  | { type: 'run_end'; status: string; duration_ms?: number; answer?: string }
  | { type: 'run_result'; result: AgentChainTestRunResponse }
  | { type: 'run_error'; error: string }

export type AgentChainRunEntry = {
  id: string
  chain_id: string
  chain_name: string
  source: string
  conversation_id?: string | null
  started_at?: string | null
  duration_ms: number
  status: string
  input: string
  answer: string
  node_status: Record<string, { status: string; chosen_route?: string; error?: string }>
  node_outputs: Record<string, string>
}

export const agentChains = {
  list: () => api<AgentChainEntry[]>('/admin/agent-chains'),
  create: (data: AgentChainCreatePayload) =>
    api<AgentChainEntry>('/admin/agent-chains', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  get: (id: string) => api<AgentChainEntry>(`/admin/agent-chains/${encodeURIComponent(id)}`),
  update: (id: string, data: AgentChainCreatePayload) =>
    api<AgentChainEntry>(`/admin/agent-chains/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  delete: (id: string) =>
    api<{ success: boolean }>(`/admin/agent-chains/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
  validate: (flow_definition: Record<string, unknown>, execution_options?: Record<string, unknown>) =>
    api<AgentChainValidateResponse>('/admin/agent-chains/validate', {
      method: 'POST',
      body: JSON.stringify({ flow_definition, execution_options: execution_options || {} }),
    }),
  export: (id: string) =>
    fetchApi(`${API_BASE}/admin/agent-chains/${encodeURIComponent(id)}/export`).then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => 'Unknown error')
        throw new Error(text)
      }
      return res.json() as Promise<AgentChainCreatePayload>
    }),
  import: (data: AgentChainCreatePayload) =>
    api<AgentChainEntry>('/admin/agent-chains/import', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  testRun: (
    flow_definition: Record<string, unknown>,
    input: string,
    signal?: AbortSignal,
    execution_options?: Record<string, unknown>,
  ) =>
    api<AgentChainTestRunResponse>('/admin/agent-chains/test-run', {
      method: 'POST',
      body: JSON.stringify({ flow_definition, input, execution_options: execution_options || {} }),
      signal,
    }),

  /** Streaming test run — yields node_start/node_end/run_result events. */
  testRunStream: async function* (
    flow_definition: Record<string, unknown>,
    input: string,
    signal?: AbortSignal,
    execution_options?: Record<string, unknown>,
  ): AsyncGenerator<ChainRunEvent, void, void> {
    const res = await fetch(`${API_BASE}/admin/agent-chains/test-run/stream`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        flow_definition,
        input,
        execution_options: execution_options || {},
      }),
      signal,
    })
    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => 'Test run failed')
      throw new Error(text)
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    try {
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        let idx: number
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const raw = buf.slice(0, idx)
          buf = buf.slice(idx + 2)
          for (const line of raw.split('\n')) {
            if (!line.startsWith('data: ')) continue
            const payload = line.slice(6)
            if (payload === '[DONE]') return
            try {
              yield JSON.parse(payload) as ChainRunEvent
            } catch {
              // skip malformed event
            }
          }
        }
      }
    } finally {
      reader.cancel().catch(() => {})
    }
  },

  runs: (chainId: string, limit = 50) =>
    api<AgentChainRunEntry[]>(
      `/admin/agent-chains/${encodeURIComponent(chainId)}/runs?limit=${limit}`,
    ),
  run: (runId: string) =>
    api<AgentChainRunEntry>(`/admin/agent-chains/runs/${encodeURIComponent(runId)}`),
}

export type AudioSettings = {
  audio_tts_provider?: string | null
  audio_stt_provider?: string | null
  audio_stt_model?: string | null
  audio_stt_language?: string | null
  audio_tts_default_voice?: string | null
  audio_tts_kokoro_voice?: string | null
  audio_min_snr_db?: number | null
  audio_noise_gate_min?: number | null
  audio_noise_gate_max?: number | null
}

export type VoiceEntry = {
  id: string
  onnx_path: string
  config_path?: string | null
  has_config?: boolean
  voices_path?: string | null
  has_voices?: boolean
  active: boolean
}

export type VoicesListResponse = {
  voices_dir: string
  models_dir?: string
  voices: VoiceEntry[]
}

export const audioApi = {
  getSettings: () => api<AudioSettings>('/admin/audio/settings'),
  saveSettings: (settings: AudioSettings) =>
    api<{ success: boolean; updated_keys: string[] }>('/admin/audio/settings', {
      method: 'PUT',
      body: JSON.stringify(settings),
    }),
  listPiperVoices: () => api<VoicesListResponse>('/admin/audio/voices'),
  uploadPiperVoice: (onnxFile: File, configFile: File) => {
    const fd = new FormData()
    fd.append('onnx_file', onnxFile)
    fd.append('config_file', configFile)
    return fetchApi(`${API_BASE}/admin/audio/voices/upload`, {
      method: 'POST',
      body: fd,
    }).then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => 'Unknown error')
        let message = text
        try {
          const data = JSON.parse(text)
          if (data.detail) message = data.detail
        } catch {}
        throw new Error(message)
      }
      return res.json() as Promise<{ success: boolean; voice_id: string }>
    })
  },
  downloadPiperVoice: (voice_id: string, onnx_url: string, config_url: string) =>
    api<{ success: boolean; voice_id: string }>('/admin/audio/voices/download', {
      method: 'POST',
      body: JSON.stringify({ voice_id, onnx_url, config_url }),
    }),
  deletePiperVoice: (voiceId: string) =>
    api<{ success: boolean }>(`/admin/audio/voices/${encodeURIComponent(voiceId)}`, {
      method: 'DELETE',
    }),
  activatePiperVoice: (voiceId: string) =>
    api<{ success: boolean; voice_id: string }>(`/admin/audio/voices/${encodeURIComponent(voiceId)}/activate`, {
      method: 'POST',
    }),
  testPiperVoice: (voiceId: string, text: string) =>
    fetch(`${API_BASE}/admin/audio/voices/test`, {
      method: 'POST',
      credentials: 'include',
      body: (() => {
        const fd = new FormData()
        fd.append('voice_id', voiceId)
        fd.append('text', text)
        return fd
      })(),
    }).then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => 'Unknown error')
        let message = text
        try {
          const data = JSON.parse(text)
          if (data.detail) message = data.detail
        } catch {}
        throw new Error(message)
      }
      return res.blob()
    }),
  listKokoroVoices: () => api<VoicesListResponse>('/admin/audio/voices/kokoro'),
  uploadKokoroVoice: (onnxFile: File, npzFile: File) => {
    const fd = new FormData()
    fd.append('onnx_file', onnxFile)
    fd.append('npz_file', npzFile)
    return fetchApi(`${API_BASE}/admin/audio/voices/kokoro/upload`, {
      method: 'POST',
      body: fd,
    }).then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => 'Unknown error')
        let message = text
        try {
          const data = JSON.parse(text)
          if (data.detail) message = data.detail
        } catch {}
        throw new Error(message)
      }
      return res.json() as Promise<{ success: boolean; voice_id: string }>
    })
  },
  deleteKokoroVoice: (voiceId: string) =>
    api<{ success: boolean }>(`/admin/audio/voices/kokoro/${encodeURIComponent(voiceId)}`, {
      method: 'DELETE',
    }),
  activateKokoroVoice: (voiceId: string) =>
    api<{ success: boolean; voice_id: string }>(`/admin/audio/voices/kokoro/${encodeURIComponent(voiceId)}/activate`, {
      method: 'POST',
    }),
  testKokoroVoice: (voiceId: string, text: string) =>
    fetch(`${API_BASE}/admin/audio/voices/kokoro/test`, {
      method: 'POST',
      credentials: 'include',
      body: (() => {
        const fd = new FormData()
        fd.append('voice_id', voiceId)
        fd.append('text', text)
        return fd
      })(),
    }).then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => 'Unknown error')
        let message = text
        try {
          const data = JSON.parse(text)
          if (data.detail) message = data.detail
        } catch {}
        throw new Error(message)
      }
      return res.blob()
    }),
}

export interface EmotionModelEntry {
  filename: string
  size: number
  modified_at: number
}

export interface EmotionModelsResponse {
  models_dir: string
  face_detector: { models: EmotionModelEntry[]; active: string | null; default_filename: string }
  expression: { models: EmotionModelEntry[]; active: string | null; default_filename: string }
  audio_emotion: { models: EmotionModelEntry[]; active: string | null; default_filename: string }
  text_emotion: { models: EmotionModelEntry[]; active: string | null; default_filename: string }
}

export type ChatRoleMappings = Record<string, string[]>

export const chatRoles = {
  list: () => api<{ mappings: ChatRoleMappings }>('/admin/chat-roles'),
  save: (mappings: ChatRoleMappings) =>
    api<{ success: boolean }>('/admin/chat-roles', {
      method: 'POST',
      body: JSON.stringify({ mappings }),
    }),
}

export type ChatUserEntry = {
  id: string
  username: string
  display_name: string
  email?: string | null
  role: string
  is_active: boolean
  source: 'dashboard' | 'public'
  groups?: Array<{ id: string; name: string; color: string; assigned_via: string }>
  created_at: string | null
  updated_at: string | null
  last_login_at: string | null
}

export const chatUsers = {
  list: () => api<ChatUserEntry[]>('/admin/chat-users'),
  update: (userId: string, data: { display_name: string; role: string; is_active: boolean; email?: string | null }) =>
    api<ChatUserEntry>(`/admin/chat-users/${encodeURIComponent(userId)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  setPassword: (userId: string, password: string) =>
    api<{ success: boolean }>(`/admin/chat-users/${encodeURIComponent(userId)}/password`, {
      method: 'POST',
      body: JSON.stringify({ password }),
    }),
  delete: (userId: string) =>
    api<{ success: boolean }>(`/admin/chat-users/${encodeURIComponent(userId)}`, {
      method: 'DELETE',
    }),
}

export type ChatGroupEntry = {
  id: string
  name: string
  description: string
  color: string
  allowed_models: string[] | null
  auto_domains: string[]
  member_count: number
  created_at: string | null
  updated_at: string | null
  members?: Array<{ user_id: string; assigned_via: string }>
}

export type ChatGroupPayload = {
  name: string
  description: string
  color: string
  allowed_models: string[] | null
  auto_domains: string[]
}

export const chatGroups = {
  list: () => api<ChatGroupEntry[]>('/admin/chat-groups'),
  get: (groupId: string) =>
    api<ChatGroupEntry>(`/admin/chat-groups/${encodeURIComponent(groupId)}`),
  create: (data: ChatGroupPayload) =>
    api<ChatGroupEntry>('/admin/chat-groups', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  update: (groupId: string, data: ChatGroupPayload) =>
    api<ChatGroupEntry>(`/admin/chat-groups/${encodeURIComponent(groupId)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  delete: (groupId: string) =>
    api<{ success: boolean }>(`/admin/chat-groups/${encodeURIComponent(groupId)}`, {
      method: 'DELETE',
    }),
  setMembers: (groupId: string, userIds: string[]) =>
    api<{ success: boolean; member_count: number }>(
      `/admin/chat-groups/${encodeURIComponent(groupId)}/members`,
      { method: 'PUT', body: JSON.stringify({ user_ids: userIds }) },
    ),
  widgetLink: (groupId: string) =>
    api<{ token: string; path: string }>(
      `/admin/chat-groups/${encodeURIComponent(groupId)}/widget-link`,
    ),
  reapplyDomains: () =>
    api<{ users: number; assigned: number }>('/admin/chat-groups/reapply-domains', {
      method: 'POST',
    }),
}

export const emotionApi = {
  listModels: () => api<EmotionModelsResponse>('/admin/emotion/models'),
  uploadModel: (modelType: string, onnxFile: File) => {
    const fd = new FormData()
    fd.append('model_type', modelType)
    fd.append('onnx_file', onnxFile)
    return fetchApi(`${API_BASE}/admin/emotion/models/upload`, {
      method: 'POST',
      body: fd,
    }).then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => 'Unknown error')
        let message = text
        try {
          const data = JSON.parse(text)
          if (data.detail) message = data.detail
        } catch {}
        throw new Error(message)
      }
      return res.json() as Promise<{ success: boolean; filename: string }>
    })
  },
  activateModel: (modelType: string, filename: string) =>
    api<{ success: boolean; model_type: string; filename: string; url: string }>(
      `/admin/emotion/models/${encodeURIComponent(modelType)}/activate`,
      {
        method: 'POST',
        body: (() => {
          const fd = new FormData()
          fd.append('filename', filename)
          return fd
        })(),
      },
    ),
  deleteModel: (modelType: string, filename: string) =>
    api<{ success: boolean }>(
      `/admin/emotion/models/${encodeURIComponent(modelType)}/${encodeURIComponent(filename)}`,
      { method: 'DELETE' },
    ),
  downloadModel: (modelType: string, url: string) =>
    api<{ success: boolean; filename: string }>('/admin/emotion/models/download', {
      method: 'POST',
      body: JSON.stringify({ model_type: modelType, url }),
    }),
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

// ------------------------------------------------------------------------------
// Database backup & restore (/admin/backup)
// ------------------------------------------------------------------------------
export type BackupConfig = {
  enabled: boolean
  schedule_type: 'interval' | 'cron'
  schedule_minutes: number
  schedule_cron: string | null
  timezone: string
  webdav_url: string | null
  remote_path: string
  username: string | null
  password_set: boolean
  retention_count: number
  retention_days: number | null
  next_run_at: string | null
  last_run_at: string | null
  encryption_configured?: boolean
  pg_dump_available?: boolean
}

export type BackupConfigUpdate = Partial<{
  enabled: boolean
  schedule_type: 'interval' | 'cron'
  schedule_minutes: number
  schedule_cron: string
  timezone: string
  webdav_url: string
  remote_path: string
  username: string
  password: string
  retention_count: number
  retention_days: number
}>

export type BackupJob = {
  id: number
  kind: string
  status: string
  filename: string | null
  remote_path: string | null
  size_bytes: number | null
  sha256_plain: string | null
  sha256_enc: string | null
  duration_ms: number | null
  error: string | null
  started_at: string
  finished_at: string | null
  triggered_by: string | null
}

export type BackupSummary = {
  last_success_at: string | null
  last_error_at: string | null
  total_jobs: number
}

export type RemoteBackupFile = {
  name: string
  href: string
  size_bytes: number | null
  last_modified: string | null
  is_backup: boolean
  is_manifest: boolean
}

export const backups = {
  getConfig: () => api<BackupConfig>('/admin/backup/config'),
  updateConfig: (data: BackupConfigUpdate) =>
    api<BackupConfig>('/admin/backup/config', {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  testConnection: (data: { webdav_url: string; remote_path: string; username: string; password?: string }) =>
    api<{ success: boolean }>('/admin/backup/test', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  runNow: () =>
    api<{ success: boolean; job_id?: number; filename?: string; size_bytes?: number; error?: string }>(
      '/admin/backup/run',
      { method: 'POST' }
    ),
  cancel: () => api<{ success: boolean }>('/admin/backup/cancel', { method: 'POST' }),
  listJobs: (limit = 50, offset = 0) =>
    api<{ jobs: BackupJob[]; summary: BackupSummary }>(
      `/admin/backup/jobs?limit=${limit}&offset=${offset}`
    ),
  listRemote: () => api<{ files: RemoteBackupFile[] }>('/admin/backup/remote'),
  verify: (filename: string) =>
    api<{ valid: boolean; detail: string; size_bytes?: number }>(
      `/admin/backup/verify/${encodeURIComponent(filename)}`,
      { method: 'POST' }
    ),
  deleteRemote: (filename: string) =>
    api<{ success: boolean }>(`/admin/backup/remote/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
    }),
  restore: (filename: string, confirm: string) =>
    api<{ job_id: number; filename: string }>('/admin/backup/restore', {
      method: 'POST',
      body: JSON.stringify({ filename, confirm }),
    }),
}
