import { useMemo, useState } from 'react'
import { Bot, Check, Clock3, Eye, LoaderCircle, MessageSquarePlus, SendHorizontal, X } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { SubagentEvent, SubagentMeta, ToolCallItem } from '../api/client'
import { ObserverEventList } from './ObserverTag'

const terminalStatuses = new Set(['done', 'error', 'cancelled'])

/** Last-write-wins ingest for live `subagent` stream events. Every event
   carries the complete record dict, so keeping the whole event log would grow
   memory quadratically — the render merge only ever needs the latest one per
   id (plus chronological position for stable ordering). */
export function upsertSubagentEvent(acc: SubagentEvent[], event: SubagentEvent): SubagentEvent[] {
  const idx = acc.findIndex(e => e.id === event.id)
  if (idx === -1) return [...acc, event]
  const next = acc.slice()
  next[idx] = event
  return next
}

function mergeEntry(previous: SubagentMeta | undefined, incoming: SubagentMeta): SubagentMeta {
  if (!previous) return incoming
  const preserveTerminal = terminalStatuses.has(previous.status) && !terminalStatuses.has(incoming.status)
  return {
    ...previous,
    ...incoming,
    ...(preserveTerminal ? {
      status: previous.status,
      answer: previous.answer,
      error: previous.error,
      duration_ms: previous.duration_ms,
      current_step: previous.current_step,
    } : {}),
    activity: incoming.activity?.length ? incoming.activity : previous.activity,
    nested_tool_calls: incoming.nested_tool_calls?.length ? incoming.nested_tool_calls : previous.nested_tool_calls,
    nested_usage: incoming.nested_usage || previous.nested_usage,
    observer_events: incoming.observer_events?.length ? incoming.observer_events : previous.observer_events,
    // Transient live-progress field: last write wins — including clearing it
    // once the follow-up event no longer carries an in-flight item.
    activity_item: incoming.activity_item,
  }
}

export function mergeSubagents(events: SubagentEvent[], calls: ToolCallItem[]): SubagentMeta[] {
  const entries = new Map<string, SubagentMeta>()
  for (const call of calls) {
    const meta = call.metadata?.subagent
    if (meta?.id) entries.set(meta.id, mergeEntry(entries.get(meta.id), meta))
    else if (call.tool_name === 'spawn_subagent') {
      const id = call.id || `spawn-${entries.size + 1}`
      entries.set(id, mergeEntry(entries.get(id), {
        id,
        task: String(call.args?.task || ''),
        label: String(call.args?.label || call.args?.task || id),
        status: call.ok ? 'queued' : 'error',
        error: call.error || undefined,
        answer: call.ok ? undefined : call.output || undefined,
      }))
    }
    for (const result of call.metadata?.subagent_results || []) {
      if (result.id) entries.set(result.id, mergeEntry(entries.get(result.id), result))
    }
  }
  for (const event of events) {
    const previous = entries.get(event.id)
    const status = event.status || (
      event.stage === 'finished' ? (event.ok === false ? 'error' : 'done')
        : event.stage === 'cancelled' ? 'cancelled'
          : event.stage === 'queued' ? 'queued' : 'running'
    )
    entries.set(event.id, mergeEntry(previous, {
      ...event,
      status,
      current_step: event.detail || event.current_step || previous?.current_step,
    } as SubagentMeta))
  }
  return [...entries.values()].sort((a, b) => (a.id || '').localeCompare(b.id || '', undefined, { numeric: true }))
}

function statusIcon(agent: SubagentMeta) {
  if (agent.status === 'done') return <Check size={13} />
  if (agent.status === 'error') return <X size={13} />
  if (agent.status === 'cancelled') return <X size={13} />
  return <LoaderCircle size={13} className="subagent-spinner" />
}

function statusLabel(status: string, language: string) {
  if (language !== 'de') return status
  return ({
    queued: 'wartet',
    running: 'läuft',
    done: 'fertig',
    error: 'Fehler',
    cancelled: 'abgebrochen',
  } as Record<string, string>)[status] || status
}

export function SubagentTagsRow({
  events,
  toolCalls,
  language = 'en',
  onSelect,
}: {
  events: SubagentEvent[]
  toolCalls: ToolCallItem[]
  language?: string
  onSelect: (agent: SubagentMeta) => void
}) {
  const entries = useMemo(() => mergeSubagents(events, toolCalls), [events, toolCalls])
  if (!entries.length) return null
  return (
    <div className="chat-subagents-row">
      {entries.map((agent) => (
        <button
          key={agent.id}
          type="button"
          className={`chat-subagent-badge ${agent.status}`}
          onClick={() => onSelect(agent)}
          title={`${statusLabel(agent.status, language)} · ${agent.task}`}
        >
          {statusIcon(agent)}
          <Bot size={13} />
          <span className="chat-subagent-badge-label">{agent.label || agent.task || agent.id}</span>
          {agent.depth && agent.depth > 1 && <span className="chat-subagent-depth">L{agent.depth}</span>}
          {agent.status === 'running' && agent.current_step && (
            <span className="chat-subagent-step">{agent.current_step}</span>
          )}
          {typeof agent.duration_ms === 'number' && (
            <span className="chat-subagent-duration">{(agent.duration_ms / 1000).toFixed(1)}s</span>
          )}
        </button>
      ))}
    </div>
  )
}

export function SubagentModal({
  agent,
  events = [],
  toolCalls = [],
  language = 'en',
  conversationId,
  apiUrl,
  requestHeaders,
  onClose,
}: {
  agent: SubagentMeta | null
  events?: SubagentEvent[]
  toolCalls?: ToolCallItem[]
  language?: string
  conversationId?: string
  apiUrl?: string
  requestHeaders?: Record<string, string>
  onClose: () => void
}) {
  const [nestedAgent, setNestedAgent] = useState<SubagentMeta | null>(null)
  const [nudgeText, setNudgeText] = useState('')
  const [nudgeState, setNudgeState] = useState<{ sending: boolean; sent?: boolean; error?: string }>({ sending: false })
  const entries = useMemo(() => mergeSubagents(events, toolCalls), [events, toolCalls])
  const current = nestedAgent
    ? entries.find((entry) => entry.id === nestedAgent.id) || nestedAgent
    : agent
      ? entries.find((entry) => entry.id === agent.id) || agent
      : null
  if (!current) return null
  const text = language === 'de'
    ? { close: 'Schließen', back: 'Zurück zu', task: 'Aufgabe', activity: 'Aktivität', output: 'Ausgabe', working: 'Wird bearbeitet…', empty: 'Keine Ausgabe vorhanden.', observer: 'Run-Observer', nudges: 'Nudges', nudge: 'Nudge senden', nudgePlaceholder: 'Anweisung an diesen Subagent…', nudgeSent: 'Nudge übermittelt.', nudgeFailed: 'Nudge fehlgeschlagen' }
    : { close: 'Close', back: 'Back to', task: 'Task', activity: 'Activity', output: 'Output', working: 'Working…', empty: 'No output was returned.', observer: 'Run observer', nudges: 'Nudges', nudge: 'Send nudge', nudgePlaceholder: 'Guidance for this subagent…', nudgeSent: 'Nudge delivered.', nudgeFailed: 'Nudge failed' }
  const canNudge = Boolean(
    conversationId && apiUrl && (current.status === 'running' || current.status === 'queued')
  )

  async function sendNudge() {
    const message = nudgeText.trim()
    if (!message || !conversationId || !apiUrl || !current) return
    setNudgeState({ sending: true })
    try {
      const res = await fetch(`${apiUrl}/chat/runs/${encodeURIComponent(conversationId)}/input`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json', ...(requestHeaders || {}) },
        body: JSON.stringify({ message, subagent_id: current.id }),
      })
      const data = await res.json().catch(() => null)
      if (res.ok && data?.accepted) {
        setNudgeState({ sending: false, sent: true })
        setNudgeText('')
      } else {
        setNudgeState({ sending: false, error: data?.reason || `HTTP ${res.status}` })
      }
    } catch (e: any) {
      setNudgeState({ sending: false, error: String(e?.message || e) })
    }
  }

  return (
    <>
      <div className="modal-overlay subagent-modal-overlay" onClick={onClose}>
        <section className="modal-box subagent-modal" onClick={(event) => event.stopPropagation()}>
          <header className="subagent-modal-header">
            <div className="subagent-modal-title">
              <span className="subagent-modal-icon"><Bot size={18} /></span>
              <div>
                <h3>{current.label || current.id}</h3>
                <div className="subagent-modal-subtitle">
                  {current.assistant || 'Agent'} · Depth {current.depth || 1} · {statusLabel(current.status, language)}
                  {typeof current.duration_ms === 'number' && ` · ${(current.duration_ms / 1000).toFixed(1)}s`}
                  {current.nested_usage?.total_tokens && ` · ${current.nested_usage.total_tokens.toLocaleString()} tokens`}
                </div>
              </div>
            </div>
            <button className="btn-icon" onClick={onClose} title={text.close}><X size={18} /></button>
          </header>

          {nestedAgent && (
            <button className="subagent-modal-back" onClick={() => setNestedAgent(null)}>
              ← {text.back} {agent?.label || agent?.id}
            </button>
          )}
          <div className="subagent-modal-body">
            <section className="subagent-detail-section">
              <h4>{text.task}</h4>
              <pre className="subagent-task">{current.task}</pre>
            </section>

            {(current.activity?.length || current.nested_tool_calls?.length || current.current_step) ? (
              <section className="subagent-detail-section">
                <h4><Clock3 size={14} /> {text.activity}</h4>
                {current.status === 'running' && current.current_step && (
                  <div className="subagent-current-step"><LoaderCircle size={13} className="subagent-spinner" />{current.current_step}</div>
                )}
                {current.reasoning_preview && current.status === 'running' && (
                  <p className="subagent-reasoning-preview">{current.reasoning_preview}</p>
                )}
                <div className="subagent-activity-list">
                  {(current.activity || []).map((item, index) => (
                    <div className="subagent-activity-item" key={`${item.tool_name}-${index}`}>
                      <span className={`subagent-activity-status ${item.status}`}>{item.status === 'ok' ? '✓' : item.status === 'error' ? '!' : '…'}</span>
                      <div className="subagent-activity-main">
                        <strong>{item.tool_name}</strong>
                        {item.args_preview && <code>{item.args_preview}</code>}
                        {(item.error || item.output_preview) && <small>{item.error || item.output_preview}</small>}
                      </div>
                      {typeof item.duration_ms === 'number' && <small>{item.duration_ms}ms</small>}
                    </div>
                  ))}
                  {(current.nested_tool_calls || []).filter((call) => call.tool_name === 'spawn_subagent').map((call, index) => {
                    const child = call.metadata?.subagent
                    if (!child) return null
                    return (
                      <button key={`${child.id}-${index}`} className="subagent-nested-link" onClick={() => setNestedAgent(child)}>
                        <Bot size={14} /> {child.label || child.id} · {child.status}
                      </button>
                    )
                  })}
                </div>
              </section>
            ) : null}

            {(current.observer_events?.length ?? 0) > 0 && (
              <section className="subagent-detail-section">
                <h4><Eye size={14} /> {text.observer}</h4>
                <ObserverEventList events={current.observer_events || []} language={language} />
              </section>
            )}

            {((current.nudges?.length ?? 0) > 0 || canNudge) && (
              <section className="subagent-detail-section">
                <h4><MessageSquarePlus size={14} /> {text.nudges}</h4>
                {(current.nudges?.length ?? 0) > 0 && (
                  <div className="subagent-nudge-list">
                    {(current.nudges || []).map((nudge, index) => (
                      <div className="subagent-nudge-item" key={index}>
                        <span className={`subagent-nudge-source ${nudge.source === 'parent' ? 'parent' : 'user'}`}>
                          {nudge.source === 'parent' ? 'parent' : 'user'}
                        </span>
                        <span className="subagent-nudge-text">{nudge.text}</span>
                      </div>
                    ))}
                  </div>
                )}
                {canNudge && (
                  <div className="subagent-nudge-form">
                    <input
                      className="subagent-nudge-input"
                      type="text"
                      value={nudgeText}
                      onChange={(event) => setNudgeText(event.target.value)}
                      onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); sendNudge() } }}
                      placeholder={text.nudgePlaceholder}
                      disabled={nudgeState.sending}
                    />
                    <button
                      type="button"
                      className="subagent-nudge-send"
                      title={text.nudge}
                      onClick={sendNudge}
                      disabled={nudgeState.sending || !nudgeText.trim()}
                    >
                      <SendHorizontal size={14} />
                    </button>
                  </div>
                )}
                {nudgeState.error && <div className="subagent-error">{text.nudgeFailed}: {nudgeState.error}</div>}
                {nudgeState.sent && !nudgeState.error && <div className="subagent-nudge-ok">{text.nudgeSent}</div>}
              </section>
            )}

            <section className="subagent-detail-section">
              <h4>{text.output}</h4>
              {current.error && <div className="subagent-error">{current.error}</div>}
              {current.answer ? (
                <div className="subagent-output chat-markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{current.answer}</ReactMarkdown>
                </div>
              ) : current.status === 'running' || current.status === 'queued' ? (
                <div className="subagent-waiting"><LoaderCircle size={15} className="subagent-spinner" />{text.working}</div>
              ) : <div className="subagent-empty">{text.empty}</div>}
            </section>
          </div>
        </section>
      </div>
    </>
  )
}
