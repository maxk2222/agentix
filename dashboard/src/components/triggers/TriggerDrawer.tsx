import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlertCircle, ArrowDown, ArrowUp, MessageCircle, Zap,
} from 'lucide-react'
import {
  triggers as triggersApi,
  type TriggerEntry,
  type TriggerRunEntry,
} from '../../api/client'
import UnsavedChangesBar, { useUnsavedGuard } from '../UnsavedChangesBar'
import SettingsDrawer from '../SettingsDrawer'
import MultiSelect from '../MultiSelect'
import { durationText, formatDateTime, isUnsaved } from './catalog'

type TabId = 'settings' | 'schedule' | 'history'

export function TriggerDrawer({
  trigger,
  assistants,
  chains,
  canWrite,
  initialTab,
  onSave,
  onClose,
  $t,
}: {
  trigger: TriggerEntry
  assistants: string[]
  chains: string[]
  canWrite: boolean
  initialTab?: TabId
  onSave: (t: TriggerEntry) => Promise<void>
  onClose: () => void
  $t: (key: string) => string
}) {
  const [draft, setDraft] = useState<TriggerEntry>({ ...trigger })
  const [tab, setTab] = useState<TabId>(initialTab || 'settings')
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)

  const update = <K extends keyof TriggerEntry>(field: K, value: TriggerEntry[K]) =>
    setDraft((d) => ({ ...d, [field]: value }))

  const isDirty = useMemo(
    () => JSON.stringify(draft) !== JSON.stringify(trigger),
    [draft, trigger],
  )
  useUnsavedGuard(isDirty && canWrite, $t('common.leaveWarning'))

  const handleSave = async () => {
    if (!canWrite) return
    setSaving(true)
    setSaveError(null)
    try {
      await onSave(draft)
    } catch (err) {
      setSaveError(String(err))
    } finally {
      setSaving(false)
    }
  }

  const isInterval = draft.schedule_type === 'interval'
  const isCron = draft.schedule_type === 'cron'
  const intervalConfig = (draft.schedule_config || {}) as Record<string, number>
  const cronExpr = (draft.schedule_config?.cron as string) || ''
  const thresholdConfig = (draft.schedule_config || {}) as Record<string, number>
  const savedTrigger = !isUnsaved(trigger)

  const tabs = [
    { id: 'settings', label: $t('tools.tabSettings'), show: true },
    { id: 'schedule', label: $t('triggers.tabSchedule'), show: true },
    { id: 'history', label: $t('triggers.history'), show: savedTrigger },
  ]

  return (
    <SettingsDrawer
      title={
        <>
          <Zap size={16} className="drawer-wrench" />
          <input
            type="text"
            className="drawer-name-input"
            value={draft.name}
            onChange={(e) => update('name', e.target.value)}
            disabled={!canWrite}
            placeholder={$t('triggers.name')}
          />
        </>
      }
      badges={
        <>
          <span className={`badge ${draft.enabled ? 'success' : 'neutral'}`}>
            {draft.enabled ? $t('common.on') : $t('common.off')}
          </span>
          <span className="badge neutral">{draft.schedule_type}</span>
        </>
      }
      tabs={tabs}
      activeTab={tab}
      onTabChange={(id) => setTab(id as TabId)}
      isDirty={isDirty && canWrite}
      onClose={onClose}
      footer={
        canWrite && tab !== 'history' ? (
          <UnsavedChangesBar
            dirty={isDirty}
            saving={saving}
            onSave={handleSave}
            onDiscard={() => setDraft({ ...trigger })}
          />
        ) : undefined
      }
    >
      {saveError && <div className="error-box">{saveError}</div>}

      {tab === 'settings' && (
        <div className="form-grid">
          <div className="form-group">
            <label>{$t('triggers.targetType')}</label>
            <select
              value={draft.target_type || 'assistant'}
              onChange={(e) => {
                const type = e.target.value
                setDraft((d) => ({ ...d, target_type: type, target_name: '', assistant_name: '' }))
              }}
              disabled={!canWrite}
            >
              <option value="assistant">{$t('triggers.assistant')}</option>
              <option value="chain">Chain</option>
            </select>
          </div>

          <div className="form-group">
            <label>{draft.target_type === 'chain' ? 'Chain' : $t('triggers.assistant')} *</label>
            <select
              value={draft.target_name || draft.assistant_name || ''}
              onChange={(e) => {
                const val = e.target.value
                setDraft((d) => ({ ...d, assistant_name: val, target_name: val }))
              }}
              disabled={!canWrite}
            >
              <option value="">—</option>
              {(draft.target_type === 'chain' ? chains : assistants).map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group wide">
            <label>{$t('triggers.prompt')} *</label>
            <textarea
              rows={5}
              value={draft.prompt}
              onChange={(e) => update('prompt', e.target.value)}
              disabled={!canWrite}
            />
          </div>

          <div className="form-group">
            <label>{$t('common.on')} / {$t('common.off')}</label>
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={draft.enabled}
                onChange={(e) => update('enabled', e.target.checked)}
                disabled={!canWrite}
              />
              <span>{draft.enabled ? $t('common.on') : $t('common.off')}</span>
            </label>
          </div>
        </div>
      )}

      {tab === 'schedule' && (
        <div className="form-grid">
          <div className="form-group">
            <label>{$t('triggers.scheduleType')}</label>
            <select
              value={draft.schedule_type}
              onChange={(e) => {
                const type = e.target.value
                const config =
                  type === 'interval'
                    ? { minutes: 60 }
                    : type === 'cron'
                      ? { cron: '0 9 * * *' }
                      : { check_interval_minutes: 5, message_threshold: 1 }
                setDraft((d) => ({ ...d, schedule_type: type, schedule_config: config }))
              }}
              disabled={!canWrite}
            >
              <option value="interval">Interval</option>
              <option value="cron">Cron</option>
              <option value="message_threshold">Message Threshold</option>
            </select>
            {draft.schedule_type === 'message_threshold' && !draft.include_new_messages && (
              <small className="field-help" style={{ color: 'var(--warning)' }}>
                {$t('triggers.thresholdNoContextHint')}
              </small>
            )}
          </div>

          {isInterval ? (
            <>
              <div className="form-group">
                <label>{$t('triggers.intervalMinutes')}</label>
                <input
                  type="number"
                  min={0}
                  value={intervalConfig.minutes || 0}
                  onChange={(e) =>
                    update('schedule_config', { ...intervalConfig, minutes: parseInt(e.target.value) || 0 })
                  }
                  disabled={!canWrite}
                />
              </div>
              <div className="form-group">
                <label>{$t('triggers.intervalHours')}</label>
                <input
                  type="number"
                  min={0}
                  value={intervalConfig.hours || 0}
                  onChange={(e) =>
                    update('schedule_config', { ...intervalConfig, hours: parseInt(e.target.value) || 0 })
                  }
                  disabled={!canWrite}
                />
              </div>
              <div className="form-group">
                <label>{$t('triggers.intervalDays')}</label>
                <input
                  type="number"
                  min={0}
                  value={intervalConfig.days || 0}
                  onChange={(e) =>
                    update('schedule_config', { ...intervalConfig, days: parseInt(e.target.value) || 0 })
                  }
                  disabled={!canWrite}
                />
              </div>
            </>
          ) : isCron ? (
            <div className="form-group wide">
              <label>Cron {$t('triggers.cronExpression')}</label>
              <input
                type="text"
                value={cronExpr}
                onChange={(e) => update('schedule_config', { cron: e.target.value })}
                disabled={!canWrite}
                placeholder="0 9 * * *"
              />
              <small className="field-help">e.g. 0 9 * * * = every day at 09:00</small>
            </div>
          ) : (
            <>
              <div className="form-group">
                <label>{$t('triggers.checkInterval')}</label>
                <input
                  type="number"
                  min={1}
                  value={thresholdConfig.check_interval_minutes || 5}
                  onChange={(e) =>
                    update('schedule_config', {
                      ...thresholdConfig,
                      check_interval_minutes: parseInt(e.target.value) || 5,
                    })
                  }
                  disabled={!canWrite}
                />
              </div>
              <div className="form-group">
                <label>{$t('triggers.messageThresholdCount')}</label>
                <input
                  type="number"
                  min={1}
                  value={thresholdConfig.message_threshold || 1}
                  onChange={(e) =>
                    update('schedule_config', {
                      ...thresholdConfig,
                      message_threshold: parseInt(e.target.value) || 1,
                    })
                  }
                  disabled={!canWrite}
                />
              </div>
            </>
          )}

          <div className="form-group wide">
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={draft.include_new_messages}
                onChange={(e) => update('include_new_messages', e.target.checked)}
                disabled={!canWrite}
              />
              <span>{$t('triggers.includeNewMessages')}</span>
            </label>
          </div>

          <div className="form-group wide">
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={draft.include_trigger_history}
                onChange={(e) => update('include_trigger_history', e.target.checked)}
                disabled={!canWrite}
              />
              <span>{$t('triggers.includeTriggerHistory')}</span>
            </label>
          </div>

          {draft.include_trigger_history && (
            <div className="form-group">
              <label>{$t('triggers.maxHistoryTurns')}</label>
              <input
                type="number"
                min={0}
                value={draft.max_history_turns}
                onChange={(e) => update('max_history_turns', parseInt(e.target.value) || 0)}
                disabled={!canWrite}
              />
              <small className="field-help">{$t('triggers.maxHistoryHelp')}</small>
            </div>
          )}

          <div className="form-group">
            <label>{$t('triggers.timeoutSeconds')}</label>
            <input
              type="number"
              min={10}
              max={3600}
              step={10}
              value={draft.timeout_seconds ?? 600}
              onChange={(e) => {
                const v = parseInt(e.target.value)
                update('timeout_seconds', Number.isNaN(v) ? 600 : Math.min(3600, Math.max(10, v)))
              }}
              disabled={!canWrite}
            />
            <small className="field-help">{$t('triggers.timeoutHelp')}</small>
          </div>

          {(draft.include_new_messages || draft.schedule_type === 'message_threshold') && (
            <div className="form-group wide">
              <label>{$t('triggers.watchAssistants')}</label>
              <MultiSelect
                value={draft.watch_assistants}
                options={assistants}
                onChange={(v) => update('watch_assistants', v)}
                disabled={!canWrite}
                placeholder={$t('triggers.watchAssistants')}
              />
              <small className="field-help">{$t('triggers.watchAssistantsHelp')}</small>
            </div>
          )}
        </div>
      )}

      {tab === 'history' && savedTrigger && (
        <RunHistory triggerId={trigger.id} $t={$t} />
      )}
    </SettingsDrawer>
  )
}

/* ------------------------------------------------------------------ */
/* Run history tab                                                    */
/* ------------------------------------------------------------------ */
function RunHistory({ triggerId, $t }: { triggerId: string; $t: (key: string) => string }) {
  const [runs, setRuns] = useState<TriggerRunEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<'all' | 'success' | 'error' | 'pending'>('all')
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc')
  const [expandedRun, setExpandedRun] = useState<number | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await triggersApi.runs(
        triggerId,
        50,
        statusFilter === 'all' ? undefined : statusFilter,
        sortOrder,
      )
      setRuns(data)
      setError(null)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }, [triggerId, statusFilter, sortOrder])

  useEffect(() => {
    load()
  }, [load])

  return (
    <div>
      <div className="run-toolbar">
        <div className="seg-control">
          {(['all', 'success', 'error', 'pending'] as const).map((s) => (
            <button
              key={s}
              className={`seg-btn ${statusFilter === s ? 'active' : ''}`}
              onClick={() => setStatusFilter(s)}
            >
              {s === 'all' ? $t('triggers.filterAll') : s === 'success' ? '✓' : s === 'error' ? '✗' : '⏳'}
            </button>
          ))}
        </div>
        <div className="seg-control">
          <button
            title={$t('triggers.sortNewest')}
            className={`seg-btn ${sortOrder === 'desc' ? 'active' : ''}`}
            onClick={() => setSortOrder('desc')}
          >
            <ArrowDown size={14} />
          </button>
          <button
            title={$t('triggers.sortOldest')}
            className={`seg-btn ${sortOrder === 'asc' ? 'active' : ''}`}
            onClick={() => setSortOrder('asc')}
          >
            <ArrowUp size={14} />
          </button>
        </div>
      </div>

      {loading && <div className="loading">{$t('common.loading')}</div>}
      {error && (
        <div className="error-box">
          <AlertCircle size={14} /> {error}
        </div>
      )}
      {!loading && runs.length === 0 && (
        <div className="text-muted">{$t('triggers.noRuns')}</div>
      )}

      {!loading && runs.length > 0 && (
        <div className="run-list">
          {runs.map((run) => (
            <div key={run.id} className="assistant-card run-item">
              <div
                className="run-item-head"
                onClick={() => setExpandedRun(expandedRun === run.id ? null : run.id)}
              >
                <div className="run-item-meta">
                  <span className="run-item-id">Run #{run.id}</span>
                  <span className={`badge ${run.status}`}>{run.status}</span>
                  {run.finished_at && (
                    <span className="run-item-duration">
                      {durationText(run.started_at, run.finished_at)}
                    </span>
                  )}
                </div>
                <span className="run-item-time">{formatDateTime(run.started_at)}</span>
              </div>

              <div className="run-item-preview">
                {run.error_text ? (
                  <span className="text-danger">{run.error_text.slice(0, 120)}…</span>
                ) : (
                  <span className="text-muted">{(run.result_text || '').slice(0, 120)}…</span>
                )}
              </div>

              {expandedRun === run.id && (
                <div className="run-item-detail">
                  {run.error_text && (
                    <div className="run-item-detail-block">
                      <div className="run-item-detail-label error">{$t('triggers.runResult')}</div>
                      <div className="run-item-detail-text error">{run.error_text}</div>
                    </div>
                  )}
                  {run.result_text && (
                    <div className="run-item-detail-block">
                      <div className="run-item-detail-label neutral">{$t('triggers.response')}</div>
                      <div className="run-item-detail-text">{run.result_text}</div>
                    </div>
                  )}
                  {run.conversation_id && (
                    <div className="run-item-actions">
                      <a
                        className="btn-secondary btn-small"
                        href={`/conversations?conversation_id=${encodeURIComponent(run.conversation_id)}`}
                      >
                        <MessageCircle size={14} /> {$t('triggers.openConversation')}
                      </a>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
