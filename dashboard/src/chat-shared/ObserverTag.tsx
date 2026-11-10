import { useState } from 'react'
import { Eye, X } from 'lucide-react'
import type { ObserverEvent, ObserverTodoOp } from '../api/client'

const TRIGGER_LABELS: Record<string, { de: string; en: string }> = {
  periodic_iterations: { de: 'Intervall (Iterationen)', en: 'Interval (iterations)' },
  periodic_seconds: { de: 'Intervall (Zeit)', en: 'Interval (time)' },
  repeat_tool_calls: { de: 'Wiederholte Tool-Aufrufe', en: 'Repeated tool calls' },
  error_streak: { de: 'Fehlerserie', en: 'Error streak' },
  stale_todo: { de: 'Veralteter Plan', en: 'Stale plan' },
  no_todo: { de: 'Kein Plan', en: 'No plan' },
  budget_pressure: { de: 'Budget fast aufgebraucht', en: 'Budget nearly exhausted' },
  user_input_injected: { de: 'Neue Nutzereingabe', en: 'New user input' },
}

function triggerLabel(trigger: string, language: string): string {
  const entry = TRIGGER_LABELS[trigger]
  if (!entry) return trigger
  return language === 'de' ? entry.de : entry.en
}

function opLabel(op: ObserverTodoOp, language: string): string {
  const status = op.ok === false ? (language === 'de' ? 'fehlgeschlagen' : 'failed') : ''
  const title = op.title ? ` — ${op.title}` : ''
  return `${op.action}${op.task_id ? ` ${op.task_id}` : ''}${title}${status ? ` (${status})` : ''}`
}

/** Shared event list — rendered inside the observer modal and inside the
   subagent modal for nested-run observer events. */
export function ObserverEventList({
  events,
  language = 'en',
}: {
  events: ObserverEvent[]
  language?: string
}) {
  const text = language === 'de'
    ? { check: 'Check', iteration: 'Iteration', nudge: 'Nudge an den Agenten', todo: 'Plan-Änderungen', dropped: 'Nudge verworfen (Limit)' }
    : { check: 'Check', iteration: 'Iteration', nudge: 'Nudge to the agent', todo: 'Plan changes', dropped: 'Nudge dropped (cap)' }
  return (
    <div className="observer-event-list">
      {events.map((ev, idx) => (
        <div className={`observer-event observer-event-${ev.kind}`} key={idx}>
          <div className="observer-event-head">
            {ev.kind === 'check_started' ? (
              <span className="observer-event-title">
                {text.check} #{ev.check ?? idx + 1} · {text.iteration} {ev.iteration ?? '?'}
              </span>
            ) : (
              <span className="observer-event-title">
                {ev.kind === 'action'
                  ? (language === 'de' ? 'Eingriff' : 'Intervention')
                  : (language === 'de' ? 'Einschätzung' : 'Assessment')}
                {` · ${text.check} #${ev.check ?? idx + 1} · ${text.iteration} ${ev.iteration ?? '?'}`}
              </span>
            )}
            {typeof ev.check_ms === 'number' && (
              <span className="observer-event-meta">{Math.round(ev.check_ms)}ms</span>
            )}
          </div>
          {ev.trigger && ev.trigger.length > 0 && (
            <div className="observer-event-triggers">
              {ev.trigger.map((trigger) => (
                <span className="observer-trigger-tag" key={trigger}>{triggerLabel(trigger, language)}</span>
              ))}
            </div>
          )}
          {ev.assessment && <p className="observer-event-assessment">{ev.assessment}</p>}
          {ev.nudge && (
            <p className="observer-event-nudge">
              <strong>{text.nudge}:</strong> {ev.nudge}
            </p>
          )}
          {ev.nudge_dropped && <p className="observer-event-dropped">{text.dropped}</p>}
          {ev.todo_ops_applied && ev.todo_ops_applied.length > 0 && (
            <div className="observer-event-todo">
              <strong>{text.todo}:</strong>
              <ul>
                {ev.todo_ops_applied.map((op, opIdx) => (
                  <li key={opIdx} className={op.ok === false ? 'op-failed' : ''}>{opLabel(op, language)}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

/** Inline badge shown on assistant messages that produced observer events;
   opens a modal with the full check history. */
export function ObserverTag({
  events,
  language = 'en',
}: {
  events?: ObserverEvent[]
  language?: string
}) {
  const [open, setOpen] = useState(false)
  if (!events || events.length === 0) return null
  const interventions = events.filter((ev) => ev.kind === 'action').length
  const text = language === 'de'
    ? { title: 'Run-Observer', close: 'Schließen', subtitle: `${events.length} Checks · ${interventions} Eingriffe` }
    : { title: 'Run observer', close: 'Close', subtitle: `${events.length} checks · ${interventions} interventions` }
  return (
    <>
      <button
        type="button"
        className="chat-observer-badge"
        onClick={() => setOpen(true)}
        title={text.subtitle}
      >
        <Eye size={13} />
        <span>Observer</span>
        <span className="chat-observer-count">{events.length}</span>
      </button>
      {open && (
        <div className="modal-overlay observer-modal-overlay" onClick={() => setOpen(false)}>
          <section className="modal-box observer-modal" onClick={(event) => event.stopPropagation()}>
            <header className="subagent-modal-header">
              <div className="subagent-modal-title">
                <span className="subagent-modal-icon observer-modal-icon"><Eye size={18} /></span>
                <div>
                  <h3>{text.title}</h3>
                  <div className="subagent-modal-subtitle">{text.subtitle}</div>
                </div>
              </div>
              <button className="btn-icon" onClick={() => setOpen(false)} title={text.close}><X size={18} /></button>
            </header>
            <div className="subagent-modal-body">
              <ObserverEventList events={events} language={language} />
            </div>
          </section>
        </div>
      )}
    </>
  )
}
