import { useEffect, useState, useCallback, useMemo } from 'react'
import { triggers as triggersApi, invalidateCache, type TriggerEntry } from '../api/client'
import { useAuth } from '../api/AuthContext'
import { useData } from '../api/DataContext'
import { useT } from '../i18n'
import { useAppPreferences } from '../ui/AppPreferences'
import { TriggerCard } from './triggers/TriggerCard'
import { TriggerDrawer } from './triggers/TriggerDrawer'
import { isUnsaved, toPayload, targetLabel } from './triggers/catalog'
import SearchToolbar from './SearchToolbar'
import ConfirmModal from './ConfirmModal'
import { AlertCircle, Plus } from 'lucide-react'

const DEFAULT_TRIGGER: TriggerEntry = {
  id: '',
  name: '',
  assistant_name: '',
  prompt: '',
  schedule_type: 'interval',
  schedule_config: { minutes: 60 },
  enabled: true,
  include_new_messages: false,
  include_trigger_history: false,
  max_history_turns: 10,
  watch_assistants: [],
  target_type: 'assistant',
  target_name: '',
}

const SCHED_FILTERS = ['all', 'interval', 'cron', 'message_threshold'] as const
type SchedFilter = (typeof SCHED_FILTERS)[number]

type DrawerState = { trigger: TriggerEntry; tab: 'settings' | 'history' }

export default function TriggersPage() {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const { hasPermission } = useAuth()
  const { getInstructions, getChains } = useData()
  const canWrite = hasPermission('triggers:write')
  const canRun = hasPermission('triggers:run')

  const [triggers, setTriggers] = useState<TriggerEntry[]>([])
  const [assistants, setAssistants] = useState<string[]>([])
  const [chains, setChains] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [schedFilter, setSchedFilter] = useState<SchedFilter>('all')
  const [drawer, setDrawer] = useState<DrawerState | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<TriggerEntry | null>(null)
  const [now, setNow] = useState<number>(Date.now())

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await triggersApi.list()
      setTriggers(data)
      setError(null)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
    getInstructions()
      .then((data) => setAssistants(data.assistants.map((a: { name: string }) => a.name)))
      .catch(() => setAssistants([]))
    getChains()
      .then((data) => setChains(data.map((c: { name: string }) => c.name)))
      .catch(() => setChains([]))
  }, [load, getInstructions, getChains])

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [])

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    return triggers.filter((t) => {
      if (schedFilter !== 'all' && t.schedule_type !== schedFilter) return false
      if (!q) return true
      return (
        t.name.toLowerCase().includes(q) ||
        (t.prompt || '').toLowerCase().includes(q) ||
        targetLabel(t).toLowerCase().includes(q)
      )
    })
  }, [triggers, search, schedFilter])

  /* ---- actions ---- */

  const handleDelete = async () => {
    if (!canWrite || !deleteTarget) return
    const trigger = deleteTarget
    if (isUnsaved(trigger)) {
      setDrawer(null)
      setDeleteTarget(null)
      return
    }
    try {
      await triggersApi.delete(trigger.id)
      invalidateCache('/admin/triggers')
      setDrawer(null)
      setDeleteTarget(null)
      await load()
    } catch (err) {
      setError(String(err))
      setDeleteTarget(null)
    }
  }

  const handleSave = async (draft: TriggerEntry) => {
    if (!canWrite) return
    const payload = toPayload(draft)
    if (isUnsaved(draft)) await triggersApi.create(payload)
    else await triggersApi.update(draft.id, payload)
    invalidateCache('/admin/triggers')
    setDrawer(null)
    await load()
  }

  const handleToggle = async (trigger: TriggerEntry) => {
    if (!canWrite || isUnsaved(trigger)) return
    try {
      await triggersApi.update(trigger.id, toPayload({ ...trigger, enabled: !trigger.enabled }))
      invalidateCache('/admin/triggers')
      load()
    } catch (err) {
      setError(String(err))
    }
  }

  const handleDuplicate = (trigger: TriggerEntry) => {
    const copy: TriggerEntry = {
      ...trigger,
      id: `new-${Date.now()}`,
      name: `${trigger.name} (${$t('configEditor.copySuffix')})`,
      enabled: true,
      last_run_at: null,
      next_run_at: null,
      created_at: null,
      updated_at: null,
      is_running: false,
    }
    setDrawer({ trigger: copy, tab: 'settings' })
  }

  const handleCreate = () => {
    setDrawer({ trigger: { ...DEFAULT_TRIGGER, id: `new-${Date.now()}` }, tab: 'settings' })
  }

  const handleRunNow = async (trigger: TriggerEntry) => {
    if (!canRun || isUnsaved(trigger)) return
    try {
      await triggersApi.run(trigger.id)
      load()
    } catch (err) {
      setError(String(err))
    }
  }

  if (loading) return <div className="loading">{$t('common.loading')}</div>

  return (
    <div>
      <div className="page-header">
        <h2>{$t('triggers.title')}</h2>
        {canWrite && (
          <button className="btn-primary" onClick={handleCreate}>
            <Plus size={14} /> {$t('triggers.create')}
          </button>
        )}
      </div>

      <SearchToolbar
        search={search}
        onSearch={setSearch}
        placeholder={$t('triggers.search')}
        chips={SCHED_FILTERS.map((s) => ({
          id: s,
          label: s === 'all' ? $t('triggers.filterAll') : s,
        }))}
        activeChip={schedFilter}
        onChip={(id) => setSchedFilter(id as SchedFilter)}
      />

      {error && (
        <div className="error-box">
          <AlertCircle size={14} /> {error}
        </div>
      )}

      {filtered.length === 0 ? (
        <div className="text-muted">{$t('triggers.empty')}</div>
      ) : (
        <div className="card-grid">
          {filtered.map((trigger) => (
            <TriggerCard
              key={trigger.id}
              trigger={trigger}
              now={now}
              canWrite={canWrite}
              canRun={canRun}
              onEdit={() => setDrawer({ trigger, tab: 'settings' })}
              onHistory={() => setDrawer({ trigger, tab: 'history' })}
              onDuplicate={() => handleDuplicate(trigger)}
              onDelete={() => setDeleteTarget(trigger)}
              onRunNow={() => handleRunNow(trigger)}
              onToggle={() => handleToggle(trigger)}
              $t={$t}
            />
          ))}
        </div>
      )}

      {drawer && (
        <TriggerDrawer
          trigger={drawer.trigger}
          assistants={assistants}
          chains={chains}
          canWrite={canWrite}
          initialTab={drawer.tab}
          onSave={handleSave}
          onClose={() => setDrawer(null)}
          $t={$t}
        />
      )}

      <ConfirmModal
        open={deleteTarget !== null}
        title={$t('common.delete')}
        message={deleteTarget ? `${$t('common.delete')} "${deleteTarget.name}"?` : ''}
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  )
}
