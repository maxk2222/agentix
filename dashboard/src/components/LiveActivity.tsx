import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Activity, Bot, GitBranch, MessageSquare, Clock, Cpu,
  ListChecks, Wrench, Zap, CalendarClock, DatabaseBackup, RefreshCw,
} from 'lucide-react'
import { dashboard } from '../api/client'
import { formatNumber } from './charts'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'
import { fmtDuration } from '../utils/format'

type LiveRes = Awaited<ReturnType<typeof dashboard.liveActivity>>
type LiveRun = LiveRes['runs'][number]

const POLL_MS = 4000

const KIND_ICONS = { chat: MessageSquare, agent: Bot, chain: GitBranch } as const

function stateDotClass(state: LiveRun['state']): string {
  switch (state) {
    case 'running': return 'live-dot running'
    case 'cancelled': return 'live-dot cancelled'
    case 'error': return 'live-dot error'
    default: return 'live-dot done'
  }
}

function RunCard({ run, now0, $t, lang, onOpen }: {
  run: LiveRun
  now0: number
  $t: (k: string, p?: Record<string, string | number>) => string
  lang: 'de' | 'en'
  onOpen: () => void
}) {
  const elapsedMs = run.state === 'running'
    ? run.elapsed_s * 1000 + (Date.now() - now0)
    : run.elapsed_s * 1000
  const KindIcon = KIND_ICONS[run.kind] ?? MessageSquare
  const phaseKey = `overview.liveActivity.phase.${run.phase || 'starting'}`
  const phaseText = $t(phaseKey) === phaseKey
    ? (run.phase || 'starting')
    : $t(phaseKey)
  const startedAt = run.started_ts ? new Date(run.started_ts) : null
  return (
    <button
      type="button"
      className={`live-run-card ${run.state}`}
      onClick={onOpen}
      title={$t('overview.liveActivity.openConversation')}
    >
      <div className="live-run-head">
        <span className={stateDotClass(run.state)} aria-hidden />
        <span className="live-run-name" title={run.model}>
          {run.model.replace(/^chain:/, '')}
        </span>
        <span className="live-chip kind">
          <KindIcon size={11} />
          {$t(`overview.liveActivity.kind.${run.kind}`)}
        </span>
        {run.source && (
          <span className="live-chip source">{$t(`overview.liveActivity.source.${run.source}`)}</span>
        )}
        <span className="live-chip state">{$t(`overview.liveActivity.state.${run.state}`)}</span>
      </div>
      <div className="live-run-phase">
        <span className="live-phase-dot" />
        <span className={run.state === 'running' ? 'live-phase-text shimmer' : 'live-phase-text'}>
          {phaseText}{run.phase_detail ? ` · ${run.phase_detail}` : ''}
        </span>
      </div>
      <div className="live-run-stats">
        <span className="live-stat" title={$t('overview.liveActivity.elapsed')}>
          <Clock size={11} /> {fmtDuration(elapsedMs, lang)}
        </span>
        {startedAt && (
          <span className="live-stat muted">
            {startedAt.toLocaleTimeString(lang === 'de' ? 'de-DE' : 'en-GB', { hour: '2-digit', minute: '2-digit' })}
          </span>
        )}
        {run.tokens != null && (
          <span className="live-stat" title={$t('overview.liveActivity.tokens')}>
            <Zap size={11} />
            {run.estimated ? '≈' : ''}{formatNumber(run.tokens)} tok{run.llm_calls ? ` · ${run.llm_calls} calls` : ''}
          </span>
        )}
        {run.tool_calls > 0 && (
          <span className="live-stat" title={$t('overview.liveActivity.toolCalls')}>
            <Wrench size={11} /> {run.tool_calls}
          </span>
        )}
        {run.subagents_total > 0 && (
          <span className="live-stat" title={$t('overview.liveActivity.subagents')}>
            <Cpu size={11} /> {run.subagents_running}/{run.subagents_total}
          </span>
        )}
        {run.todos_total != null && (
          <span className="live-stat" title={$t('overview.liveActivity.todos')}>
            <ListChecks size={11} />
            {run.todos_total - (run.todos_open ?? 0)}/{run.todos_total}
          </span>
        )}
      </div>
      {run.user_message && (
        <div className="live-run-message" title={run.user_message}>{run.user_message}</div>
      )}
    </button>
  )
}

export default function LiveActivity() {
  const [res, setRes] = useState<LiveRes | null>(null)
  const [fetchedAt, setFetchedAt] = useState(0)
  const [, setTick] = useState(0)
  const { language } = useAppPreferences()
  const $t = useT(language)
  const navigate = useNavigate()
  const tickRef = useRef<number | null>(null)

  useEffect(() => {
    let cancelled = false
    async function load(silent = false) {
      try {
        const r = await dashboard.liveActivity()
        if (!cancelled) {
          setRes(r)
          setFetchedAt(Date.now())
        }
      } catch {
        // Permission/API failure — degrade to the quiet state rather than
        // pinning the section on a perpetual loading note.
        if (!cancelled && !silent) {
          setRes({
            generated_at: '',
            runs: [],
            automation: { triggers: [], backups: [], syncs: [] },
          })
        }
      }
    }
    load()
    const id = setInterval(() => {
      if (document.visibilityState === 'visible') load(true)
    }, POLL_MS)
    const onVis = () => { if (document.visibilityState === 'visible') load(true) }
    document.addEventListener('visibilitychange', onVis)
    return () => {
      cancelled = true
      clearInterval(id)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [])

  const hasRunning = useMemo(
    () => (res?.runs ?? []).some((r) => r.state === 'running'),
    [res],
  )
  useEffect(() => {
    if (hasRunning && tickRef.current === null) {
      tickRef.current = window.setInterval(() => setTick((t) => t + 1), 1000)
    } else if (!hasRunning && tickRef.current !== null) {
      clearInterval(tickRef.current)
      tickRef.current = null
    }
    return () => {
      if (tickRef.current !== null) {
        clearInterval(tickRef.current)
        tickRef.current = null
      }
    }
  }, [hasRunning])

  const automation = res?.automation
  const hasAutomation = Boolean(
    automation && (automation.triggers.length || automation.backups.length || automation.syncs.length),
  )
  const runs = res?.runs ?? []

  return (
    <div className="overview-section live-activity">
      <h3 className="section-header">
        <Activity size={18} /> {$t('overview.liveActivity.title')}
        {hasRunning && <span className="live-pill">{$t('overview.liveActivity.livePill')}</span>}
      </h3>
      {res === null ? (
        <div className="live-quiet loading"><span className="live-dot running" />{$t('overview.liveActivity.loading')}</div>
      ) : runs.length === 0 && !hasAutomation ? (
        <div className="live-quiet">
          <span className="live-dot done" />
          {$t('overview.liveActivity.quiet')}
        </div>
      ) : (
        <>
          {runs.length > 0 && (
            <div className="live-run-grid">
              {runs.map((run) => (
                <RunCard
                  key={run.conversation_id}
                  run={run}
                  now0={fetchedAt}
                  $t={$t}
                  lang={language}
                  onOpen={() => navigate(`/conversations?conversation_id=${encodeURIComponent(run.conversation_id)}`)}
                />
              ))}
            </div>
          )}
          {hasAutomation && automation && (
            <div className="live-automation">
              {automation.triggers.map((tr) => (
                <span key={tr.id} className="live-automation-chip" title={$t('overview.liveActivity.auto.triggerTip')}>
                  <CalendarClock size={12} />
                  {$t('overview.liveActivity.auto.trigger', { name: tr.name || tr.id.slice(0, 8) })}
                  · {fmtDuration(tr.elapsed_s * 1000, language)}
                </span>
              ))}
              {automation.backups.map((b) => (
                <span key={`b${b.id}`} className="live-automation-chip" title={$t('overview.liveActivity.auto.backupTip')}>
                  <DatabaseBackup size={12} />
                  {$t('overview.liveActivity.auto.backup')}{b.triggered_by ? ` · ${b.triggered_by}` : ''}
                </span>
              ))}
              {automation.syncs.map((s) => (
                <span key={`s${s.name}`} className="live-automation-chip" title={$t('overview.liveActivity.auto.syncTip')}>
                  <RefreshCw size={12} className="live-spin" />
                  {$t('overview.liveActivity.auto.sync', { name: s.name })}
                </span>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}
