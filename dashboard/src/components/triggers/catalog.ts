import { BellRing, CalendarClock, Clock, type LucideIcon } from 'lucide-react'
import type { TriggerCreatePayload, TriggerEntry } from '../../api/client'

export type ScheduleType = 'interval' | 'cron' | 'message_threshold'

export const SCHEDULE_ICON: Record<string, LucideIcon> = {
  interval: Clock,
  cron: CalendarClock,
  message_threshold: BellRing,
}

export const SCHEDULE_ICON_CLASS: Record<string, string> = {
  interval: 'sched-interval',
  cron: 'sched-cron',
  message_threshold: 'sched-threshold',
}

export const isUnsaved = (t: TriggerEntry) => !t.id || t.id.startsWith('new-')

/** Full create/update payload — includes target_type/target_name (bug fix). */
export function toPayload(draft: TriggerEntry): TriggerCreatePayload {
  const isChain = draft.target_type === 'chain'
  const target = isChain
    ? draft.target_name || draft.assistant_name
    : draft.assistant_name || draft.target_name || ''
  return {
    name: draft.name,
    assistant_name: isChain ? target : draft.assistant_name || target,
    prompt: draft.prompt,
    schedule_type: draft.schedule_type,
    schedule_config: draft.schedule_config,
    enabled: draft.enabled,
    include_new_messages: draft.include_new_messages,
    include_trigger_history: draft.include_trigger_history,
    max_history_turns: draft.max_history_turns,
    watch_assistants: draft.watch_assistants,
    target_type: draft.target_type || 'assistant',
    target_name: target,
    timeout_seconds: draft.timeout_seconds ?? 600,
  }
}

export function targetLabel(t: TriggerEntry): string {
  const name = t.target_name || t.assistant_name
  return t.target_type === 'chain' ? `Chain: ${name}` : name
}

export function formatRemaining(
  nextRunAt: string,
  now: number,
  $t: (key: string) => string,
): string {
  const diff = new Date(nextRunAt).getTime() - now
  if (diff <= 0) return $t('triggers.overdue')
  const seconds = Math.floor(diff / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  const days = Math.floor(hours / 24)
  const parts: string[] = []
  if (days > 0) parts.push(`${days}d`)
  if (hours % 24 > 0 || (days > 0 && minutes % 60 > 0)) parts.push(`${hours % 24}h`)
  if (minutes % 60 > 0 || (hours > 0 && seconds % 60 > 0)) parts.push(`${minutes % 60}m`)
  if (days === 0 && hours === 0) parts.push(`${seconds % 60}s`)
  return `${$t('triggers.nextRunIn')} ${parts.join(' ')}`
}

export function durationText(started: string, finished?: string | null): string {
  if (!finished) return ''
  const diff = new Date(finished).getTime() - new Date(started).getTime()
  if (diff < 1000) return `${diff}ms`
  if (diff < 60000) return `${Math.round(diff / 1000)}s`
  return `${Math.round(diff / 60000)}min`
}

export function formatDateTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString('de-DE', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}
