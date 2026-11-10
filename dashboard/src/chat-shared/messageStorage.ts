/* Storage-safe serialization for chat messages persisted to sessionStorage.

   Full-fidelity messages (raw tool outputs, subagent reports, RAG context,
   emotion phase frames) easily exceed the ~5MB quota on long answers — and
   an uncaught QuotaExceededError inside the persist effect crashed the whole
   page mid-stream. Persistence is best-effort: store a slimmed copy, shed
   history on quota errors, and never throw. */

const CONTENT_CAP = 200_000
const REASONING_CAP = 20_000
const TOOL_OUTPUT_CAP = 2_000
const TOOL_ERROR_CAP = 1_000
const TOOL_ARGS_CAP = 2_000
const PREVIEW_CAP = 500
const SUBAGENT_ANSWER_CAP = 4_000
const SUBAGENT_ACTIVITY_KEEP = 10
const OBSERVER_TEXT_CAP = 2_000
const OBSERVER_EVENTS_KEEP = 20
const FALLBACK_KEEP_COUNTS = [30, 10, 3]

function cap<T>(value: T, limit: number): T {
  return typeof value === 'string' && value.length > limit
    ? ((value.slice(0, limit) + '…') as T)
    : value
}

function slimSubagent(meta: any): any {
  if (!meta || typeof meta !== 'object') return meta
  const { nested_tool_calls, reasoning_preview, ...rest } = meta
  const out: any = {
    ...rest,
    answer: cap(rest.answer, SUBAGENT_ANSWER_CAP),
    task: cap(rest.task, 2_000),
    detail: cap(rest.detail, 2_000),
    error: cap(rest.error, 1_000),
  }
  if (Array.isArray(rest.activity)) {
    out.activity = rest.activity.slice(-SUBAGENT_ACTIVITY_KEEP).map((item: any) => ({
      ...item,
      output_preview: cap(item?.output_preview, PREVIEW_CAP),
      error: cap(item?.error, PREVIEW_CAP),
      args_preview: cap(item?.args_preview, PREVIEW_CAP),
    }))
  }
  if (Array.isArray(rest.observer_events)) {
    out.observer_events = rest.observer_events.slice(-OBSERVER_EVENTS_KEEP).map(slimObserverEvent)
  }
  return out
}

function slimObserverEvent(ev: any): any {
  if (!ev || typeof ev !== 'object') return ev
  return {
    ...ev,
    assessment: cap(ev.assessment, OBSERVER_TEXT_CAP),
    nudge: cap(ev.nudge, OBSERVER_TEXT_CAP),
    todo_ops_applied: Array.isArray(ev.todo_ops_applied)
      ? ev.todo_ops_applied.slice(-10).map((op: any) => ({ ...op, title: cap(op?.title, PREVIEW_CAP), detail: cap(op?.detail, PREVIEW_CAP) }))
      : ev.todo_ops_applied,
  }
}

function slimToolCall(call: any): any {
  if (!call || typeof call !== 'object') return call
  let args = call.args
  if (args && typeof args === 'object') {
    try {
      if (JSON.stringify(args).length > TOOL_ARGS_CAP) args = { truncated: true }
    } catch {
      /* keep args as-is */
    }
  }
  const metadata = call.metadata
    ? {
        ...call.metadata,
        subagent: call.metadata.subagent ? slimSubagent(call.metadata.subagent) : call.metadata.subagent,
        subagent_results: Array.isArray(call.metadata.subagent_results)
          ? call.metadata.subagent_results.map(slimSubagent)
          : call.metadata.subagent_results,
      }
    : call.metadata
  return {
    ...call,
    args,
    output: cap(call.output, TOOL_OUTPUT_CAP),
    error: cap(call.error, TOOL_ERROR_CAP),
    args_preview: cap(call.args_preview, PREVIEW_CAP),
    code: cap(call.code, TOOL_ARGS_CAP),
    metadata,
  }
}

export function slimMessageForStorage<T>(msg: T): T {
  if (!msg || typeof msg !== 'object') return msg
  const out: any = { ...(msg as any) }
  out.content = cap(out.content, CONTENT_CAP)
  out.reasoning = cap(out.reasoning, REASONING_CAP)
  if (Array.isArray(out.toolCalls)) out.toolCalls = out.toolCalls.map(slimToolCall)
  if (Array.isArray(out.subagents)) out.subagents = out.subagents.map(slimSubagent)
  if (Array.isArray(out.observerEvents)) out.observerEvents = out.observerEvents.slice(-OBSERVER_EVENTS_KEEP).map(slimObserverEvent)
  if (Array.isArray(out.terminalEvents)) out.terminalEvents = out.terminalEvents.map(slimToolCall)
  if (out.emotion && typeof out.emotion === 'object') {
    // Phase frames and per-signal emotion payloads are the bulkiest fields;
    // the fused result is all a restored view needs.
    out.emotion = { fused_emotion: out.emotion.fused_emotion ?? null, consent: out.emotion.consent }
  }
  // Live-stream counters are transient — never persist them into storage.
  delete out.liveUsage
  if (out.retrievalInfo && Array.isArray(out.retrievalInfo.rag_context)) {
    out.retrievalInfo = {
      ...out.retrievalInfo,
      rag_context: out.retrievalInfo.rag_context.map((doc: any) => ({ metadata: doc?.metadata })),
    }
  }
  return out
}

/* sessionStorage itself can throw SecurityError in sandboxed iframes /
   blocked-storage contexts — resolve it lazily inside a guard. */
function defaultStorage(): Storage | null {
  try {
    return window.sessionStorage
  } catch {
    return null
  }
}

/** Persist messages under `key`; on quota pressure shed oldest messages,
    then give up silently — persistence must never crash the app. */
export function persistChatMessages(key: string, messages: any[], storage: Storage | null = defaultStorage()) {
  if (!storage) return
  if (!messages.length) {
    try {
      storage.removeItem(key)
    } catch {
      /* ignore */
    }
    return
  }
  try {
    storage.setItem(key, JSON.stringify(messages.map(slimMessageForStorage)))
    return
  } catch {
    /* quota exceeded — shed history */
  }
  for (const keep of FALLBACK_KEEP_COUNTS) {
    try {
      storage.setItem(key, JSON.stringify(messages.slice(-keep).map(slimMessageForStorage)))
      return
    } catch {
      /* keep shrinking */
    }
  }
  try {
    storage.removeItem(key)
  } catch {
    /* best-effort only */
  }
}

/* Streaming writes every token; serializing the whole history each time is
   O(history) per event. Debounce per storage key — last write wins — and let
   callers flush pending writes on unmount so nothing is lost. pagehide is
   hooked once so pending writes (including removals) survive hard reloads,
   which do not run React unmount cleanups. */
const _pendingPersists = new Map<string, { timer: ReturnType<typeof setTimeout>; messages: any[] }>()
let _pagehideHooked = false

function _hookPagehideFlush() {
  if (_pagehideHooked || typeof window === 'undefined') return
  _pagehideHooked = true
  window.addEventListener('pagehide', flushPendingChatWrites)
}

export function persistChatMessagesDebounced(
  key: string,
  messages: any[],
  delayMs = 750,
) {
  _hookPagehideFlush()
  const pending = _pendingPersists.get(key)
  if (pending) clearTimeout(pending.timer)
  _pendingPersists.set(key, {
    timer: setTimeout(() => {
      _pendingPersists.delete(key)
      persistChatMessages(key, messages)
    }, delayMs),
    messages,
  })
}

/** Flush every pending debounced write immediately (e.g. on unmount). */
export function flushPendingChatWrites() {
  for (const [key, pending] of _pendingPersists) {
    clearTimeout(pending.timer)
    persistChatMessages(key, pending.messages)
  }
  _pendingPersists.clear()
}

/** Read + JSON.parse messages under `key`; returns [] on any failure. */
export function loadChatMessages(key: string, storage: Storage | null = defaultStorage()): any[] {
  if (!storage) return []
  try {
    const stored = storage.getItem(key)
    if (!stored) return []
    const parsed = JSON.parse(stored)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}
