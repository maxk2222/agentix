import { Check, ChevronDown, Circle, ListTodo, LoaderCircle, XCircle } from 'lucide-react'
import { useState } from 'react'
import type { AgentTodoSnapshot } from '../api/client'

export function AgentTodoPanel({
  snapshot,
  language = 'en',
}: {
  snapshot?: AgentTodoSnapshot
  language?: string
}) {
  const [expanded, setExpanded] = useState(true)
  if (!snapshot || snapshot.version < 1) return null

  const done = snapshot.tasks.filter((task) => task.status === 'completed').length
  const labels = language === 'de'
    ? { title: 'Arbeitsplan', empty: 'Keine offenen Aufgaben', pending: 'Ausstehend', active: 'In Arbeit', completed: 'Erledigt', skipped: 'Übersprungen' }
    : { title: 'Work plan', empty: 'No open tasks', pending: 'Pending', active: 'In progress', completed: 'Completed', skipped: 'Skipped' }

  return (
    <section className="chat-agent-todo" aria-label={labels.title} aria-live="polite">
      <button
        className="chat-agent-todo-header"
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}
      >
        <span className="chat-agent-todo-heading"><ListTodo size={15} />{labels.title}</span>
        <span className="chat-agent-todo-controls">
          <span className="chat-agent-todo-count">{done}/{snapshot.tasks.length}</span>
          <ChevronDown size={14} className={expanded ? '' : 'collapsed'} />
        </span>
      </button>
      {snapshot.tasks.length > 0 && (
        <div
          className="chat-agent-todo-progress"
          role="progressbar"
          aria-valuenow={done}
          aria-valuemin={0}
          aria-valuemax={snapshot.tasks.length}
        >
          <span style={{ width: `${Math.round((done / snapshot.tasks.length) * 100)}%` }} />
        </div>
      )}
      {expanded && (snapshot.tasks.length === 0 ? (
        <p className="chat-agent-todo-empty">{labels.empty}</p>
      ) : (
        <ul className="chat-agent-todo-list">
          {snapshot.tasks.map((task) => {
            const completed = task.status === 'completed'
            const active = task.status === 'in_progress'
            const skipped = task.status === 'skipped'
            const status = completed
              ? labels.completed
              : skipped
                ? labels.skipped
                : active
                  ? labels.active
                  : labels.pending
            return (
              <li className={`chat-agent-todo-task ${task.status}`} key={task.id}>
                <span className="chat-agent-todo-icon" aria-hidden="true">
                  {completed ? <Check size={14} /> : skipped ? <XCircle size={14} /> : active ? <LoaderCircle size={14} className="chat-agent-todo-spinner" /> : <Circle size={13} />}
                </span>
                <span className="chat-agent-todo-copy">
                  <span className="chat-agent-todo-task-title">{task.title}</span>
                  {task.note && <span className="chat-agent-todo-note">{task.note}</span>}
                </span>
                <span className="chat-agent-todo-status">{status}</span>
              </li>
            )
          })}
        </ul>
      )
      )}
    </section>
  )
}
