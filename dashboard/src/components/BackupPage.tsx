import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  Archive,
  CalendarClock,
  CheckCircle2,
  Globe,
  History,
  Loader,
  Lock,
  Play,
  Plug,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Trash2,
  User,
  XCircle,
} from 'lucide-react'
import {
  backups as backupsApi,
  invalidateCache,
  type BackupConfig,
  type BackupJob,
  type BackupSummary,
  type RemoteBackupFile,
} from '../api/client'
import { useAuth } from '../api/AuthContext'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'
import UnsavedChangesBar, { useUnsavedGuard } from './UnsavedChangesBar'
import ConfirmModal from './ConfirmModal'
import RestoreBackupModal from './RestoreBackupModal'

function formatBytes(n?: number | null): string {
  if (n === null || n === undefined) return '—'
  if (n < 1024) return `${n} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let v = n / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return `${v.toFixed(1)} ${units[i]}`
}

function formatDate(s?: string | null): string {
  if (!s) return '—'
  const d = new Date(s)
  return isNaN(d.getTime()) ? String(s) : d.toLocaleString("de-DE")
}

function formatDuration(ms?: number | null): string {
  if (ms === null || ms === undefined) return '—'
  if (ms < 1000) return `${ms} ms`
  const s = Math.round(ms / 1000)
  if (s < 60) return `${s} s`
  return `${Math.floor(s / 60)} min ${s % 60} s`
}

type FormState = {
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
  retention_days: string
}

function toForm(cfg: BackupConfig): FormState {
  return {
    enabled: cfg.enabled,
    schedule_type: cfg.schedule_type === 'cron' ? 'cron' : 'interval',
    schedule_minutes: cfg.schedule_minutes ?? 1440,
    schedule_cron: cfg.schedule_cron ?? '',
    timezone: cfg.timezone ?? 'UTC',
    webdav_url: cfg.webdav_url ?? '',
    remote_path: cfg.remote_path ?? '/agentix-backups',
    username: cfg.username ?? '',
    password: '',
    retention_count: cfg.retention_count ?? 10,
    retention_days: cfg.retention_days !== null && cfg.retention_days !== undefined ? String(cfg.retention_days) : '',
  }
}

export default function BackupPage() {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const { hasPermission } = useAuth()
  const canWrite = hasPermission('backup:write')
  const canRestore = hasPermission('backup:restore')

  const [config, setConfig] = useState<BackupConfig | null>(null)
  const [form, setForm] = useState<FormState | null>(null)
  const [jobs, setJobs] = useState<BackupJob[]>([])
  const [summary, setSummary] = useState<BackupSummary | null>(null)
  const [remote, setRemote] = useState<RemoteBackupFile[]>([])
  const [remoteError, setRemoteError] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<{ ok: boolean; msg: string } | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [verifyState, setVerifyState] = useState<Record<string, { busy: boolean; msg?: string; ok?: boolean }>>({})
  const [restoreTarget, setRestoreTarget] = useState<string | null>(null)
  const [restoring, setRestoring] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)

  const load = useCallback(async () => {
    setError('')
    try {
      const [cfg, jobsRes] = await Promise.all([backupsApi.getConfig(), backupsApi.listJobs(50, 0)])
      setConfig(cfg)
      setForm((prev) => (prev === null ? toForm(cfg) : prev))
      setJobs(jobsRes.jobs)
      setSummary(jobsRes.summary)
    } catch (e: any) {
      setError(e?.message || String(e))
    }
    try {
      const rem = await backupsApi.listRemote()
      setRemote(rem.files)
      setRemoteError('')
    } catch (e: any) {
      setRemote([])
      setRemoteError(e?.message || String(e))
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    load()
  }, [load])

  const isRunning = jobs.some((j) => j.status === 'running')

  const baseline = useMemo(() => (config ? toForm(config) : null), [config])
  const isDirty = useMemo(
    () =>
      form !== null &&
      baseline !== null &&
      JSON.stringify(form) !== JSON.stringify(baseline),
    [form, baseline],
  )
  useUnsavedGuard(isDirty && canWrite, $t('common.leaveWarning'))

  async function handleSave() {
    if (!form) return
    setSaving(true)
    setError('')
    setNotice('')
    try {
      const payload: Record<string, unknown> = {
        enabled: form.enabled,
        schedule_type: form.schedule_type,
        schedule_minutes: Number(form.schedule_minutes) || 1440,
        timezone: form.timezone.trim() || 'UTC',
        webdav_url: form.webdav_url.trim(),
        remote_path: form.remote_path.trim() || '/agentix-backups',
        username: form.username.trim(),
        retention_count: Number(form.retention_count) || 10,
      }
      if (form.schedule_type === 'cron' && form.schedule_cron.trim()) {
        payload.schedule_cron = form.schedule_cron.trim()
      }
      if (form.password) payload.password = form.password
      if (form.retention_days.trim()) payload.retention_days = Number(form.retention_days)
      const cfg = await backupsApi.updateConfig(payload)
      setConfig(cfg)
      setForm(toForm(cfg))
      setNotice($t('backup.saved'))
      invalidateCache('/admin/backup')
    } catch (err: any) {
      setError(err?.message || String(err))
    } finally {
      setSaving(false)
    }
  }

  async function handleTest() {
    if (!form) return
    setTesting(true)
    setTestResult(null)
    try {
      await backupsApi.testConnection({
        webdav_url: form.webdav_url.trim(),
        remote_path: form.remote_path.trim() || '/agentix-backups',
        username: form.username.trim(),
        password: form.password || undefined,
      })
      setTestResult({ ok: true, msg: $t('backup.testOk') })
    } catch (err: any) {
      setTestResult({ ok: false, msg: err?.message || String(err) })
    } finally {
      setTesting(false)
    }
  }

  async function handleRunNow() {
    setRunning(true)
    setError('')
    setNotice('')
    try {
      const res = await backupsApi.runNow()
      if (res.success) {
        setNotice(`${$t('backup.runOk')}: ${res.filename ?? ''}`)
      } else {
        setError(res.error || $t('backup.runFailed'))
      }
      invalidateCache('/admin/backup')
      await load()
    } catch (err: any) {
      setError(err?.message || String(err))
    } finally {
      setRunning(false)
    }
  }

  async function handleCancelJob() {
    try {
      await backupsApi.cancel()
      invalidateCache('/admin/backup')
      await load()
    } catch (err: any) {
      setError(err?.message || String(err))
    }
  }

  async function handleVerify(name: string) {
    setVerifyState((s) => ({ ...s, [name]: { busy: true } }))
    try {
      const res = await backupsApi.verify(name)
      setVerifyState((s) => ({ ...s, [name]: { busy: false, ok: res.valid, msg: res.detail } }))
    } catch (err: any) {
      setVerifyState((s) => ({ ...s, [name]: { busy: false, ok: false, msg: err?.message || String(err) } }))
    }
  }

  async function handleDeleteRemote() {
    if (!deleteTarget) return
    try {
      await backupsApi.deleteRemote(deleteTarget)
      setDeleteTarget(null)
      invalidateCache('/admin/backup')
      await load()
    } catch (err: any) {
      setError(err?.message || String(err))
      setDeleteTarget(null)
    }
  }

  async function handleRestore() {
    if (!restoreTarget) return
    setRestoring(true)
    setError('')
    try {
      await backupsApi.restore(restoreTarget, restoreTarget)
      setNotice($t('backup.restoreOk'))
      setRestoreTarget(null)
      invalidateCache('/admin/backup')
      await load()
    } catch (err: any) {
      setError(err?.message || String(err))
    } finally {
      setRestoring(false)
    }
  }

  if (loading) return <div className="loading">{$t('common.loading')}</div>

  return (
    <div>
      <div className="page-header">
        <h2>{$t('backup.title')}</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn-secondary" onClick={load} title={$t('common.refresh')}>
            <RefreshCw size={14} />
          </button>
          {canWrite && (
            <button className="btn-primary" onClick={handleRunNow} disabled={running || isRunning}>
              {running || isRunning ? <Loader size={14} className="spin" /> : <Play size={14} />}
              {$t('backup.runNow')}
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="error-box">
          <AlertCircle size={14} /> {error}
        </div>
      )}
      {notice && (
        <div className="alert-success">
          <CheckCircle2 size={14} /> {notice}
        </div>
      )}

      {config && (!config.encryption_configured || !config.pg_dump_available) && (
        <div className="error-box">
          <AlertCircle size={14} />
          {!config.encryption_configured && <span> {$t('backup.noEncryptionKey')}</span>}
          {!config.pg_dump_available && <span> {$t('backup.noPgDump')}</span>}
        </div>
      )}

      {/* Status KPIs */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-head">
            <span className="label">{$t('backup.lastSuccess')}</span>
            <span className="kpi-icon ok"><CheckCircle2 size={16} /></span>
          </div>
          <div className="value" style={{ fontSize: '1.1rem' }}>{formatDate(summary?.last_success_at)}</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-head">
            <span className="label">{$t('backup.lastError')}</span>
            <span className={`kpi-icon ${summary?.last_error_at ? 'err' : ''}`}><AlertCircle size={16} /></span>
          </div>
          <div className="value" style={{ fontSize: '1.1rem' }}>{formatDate(summary?.last_error_at)}</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-head">
            <span className="label">{$t('backup.nextRun')}</span>
            <span className="kpi-icon info"><CalendarClock size={16} /></span>
          </div>
          <div className="value" style={{ fontSize: '1.1rem' }}>
            {config?.enabled ? formatDate(config.next_run_at) : $t('common.off')}
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-head">
            <span className="label">{$t('backup.totalJobs')}</span>
            <span className="kpi-icon"><Archive size={16} /></span>
          </div>
          <div className="value">{summary?.total_jobs ?? 0}</div>
        </div>
      </div>

      {/* Configuration */}
      {form && (
        <div className="assistant-card" style={{ padding: 16, marginBottom: 16 }}>
          <h3 className="section-header" style={{ fontSize: 16 }}>
            <CalendarClock size={16} /> {$t('backup.configSection')}
          </h3>
          <label className="toggle-row" style={{ marginBottom: 12 }}>
            <input
              type="checkbox"
              checked={form.enabled}
              disabled={!canWrite}
              onChange={(e) => setForm((f) => f && { ...f, enabled: e.target.checked })}
            />
            <span>{$t('backup.enableAuto')}</span>
          </label>

          <div className="cfg-section">
            <div className="form-grid">
              <div className="form-group">
                <label>{$t('backup.scheduleType')}</label>
                <select
                  value={form.schedule_type}
                  disabled={!canWrite}
                  onChange={(e) => setForm((f) => f && { ...f, schedule_type: e.target.value as 'interval' | 'cron' })}
                >
                  <option value="interval">{$t('backup.interval')}</option>
                  <option value="cron">{$t('backup.cron')}</option>
                </select>
              </div>
              {form.schedule_type === 'interval' ? (
                <div className="form-group">
                  <label>{$t('backup.intervalMinutes')}</label>
                  <input
                    type="number"
                    min={15}
                    value={form.schedule_minutes}
                    disabled={!canWrite}
                    onChange={(e) => setForm((f) => f && { ...f, schedule_minutes: Number(e.target.value) })}
                  />
                </div>
              ) : (
                <div className="form-group">
                  <label>{$t('backup.cronExpr')}</label>
                  <input
                    type="text"
                    value={form.schedule_cron}
                    disabled={!canWrite}
                    placeholder="0 3 * * *"
                    onChange={(e) => setForm((f) => f && { ...f, schedule_cron: e.target.value })}
                  />
                </div>
              )}
              <div className="form-group">
                <label>{$t('backup.timezone')}</label>
                <input
                  type="text"
                  value={form.timezone}
                  disabled={!canWrite}
                  placeholder="UTC"
                  onChange={(e) => setForm((f) => f && { ...f, timezone: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>{$t('backup.retentionCount')}</label>
                <input
                  type="number"
                  min={1}
                  value={form.retention_count}
                  disabled={!canWrite}
                  onChange={(e) => setForm((f) => f && { ...f, retention_count: Number(e.target.value) })}
                />
              </div>
              <div className="form-group">
                <label>{$t('backup.retentionDays')}</label>
                <input
                  type="number"
                  min={1}
                  value={form.retention_days}
                  disabled={!canWrite}
                  placeholder={$t('backup.retentionDaysPlaceholder')}
                  onChange={(e) => setForm((f) => f && { ...f, retention_days: e.target.value })}
                />
              </div>
            </div>
          </div>

          <div className="cfg-section">
            <div className="cfg-section-title">{$t('backup.webdavSection')}</div>
            <div className="form-grid">
              <div className="form-group">
                <label>{$t('knowledge.webdavUrl')}</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Globe size={14} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
                  <input
                    type="text"
                    value={form.webdav_url}
                    disabled={!canWrite}
                    placeholder="https://sciebo.de/remote.php/dav/files/USER/"
                    style={{ flex: 1 }}
                    onChange={(e) => setForm((f) => f && { ...f, webdav_url: e.target.value })}
                  />
                </div>
              </div>
              <div className="form-group">
                <label>{$t('knowledge.remotePath')}</label>
                <input
                  type="text"
                  value={form.remote_path}
                  disabled={!canWrite}
                  placeholder="/agentix-backups"
                  onChange={(e) => setForm((f) => f && { ...f, remote_path: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label>{$t('knowledge.webdavUsername')}</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <User size={14} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
                  <input
                    type="text"
                    value={form.username}
                    disabled={!canWrite}
                    style={{ flex: 1 }}
                    onChange={(e) => setForm((f) => f && { ...f, username: e.target.value })}
                  />
                </div>
              </div>
              <div className="form-group">
                <label>
                  {$t('knowledge.webdavPassword')}
                  {config?.password_set && (
                    <span className="text-muted" style={{ fontSize: '0.75rem' }}> ({$t('backup.passwordSet')})</span>
                  )}
                </label>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Lock size={14} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
                  <input
                    type="password"
                    value={form.password}
                    disabled={!canWrite}
                    placeholder={config?.password_set ? $t('backup.keepPassword') : ''}
                    autoComplete="new-password"
                    style={{ flex: 1 }}
                    onChange={(e) => setForm((f) => f && { ...f, password: e.target.value })}
                  />
                </div>
              </div>
            </div>

            {canWrite && (
              <div style={{ display: 'flex', gap: 8, marginTop: 12, alignItems: 'center' }}>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={handleTest}
                  disabled={testing || !form.webdav_url.trim() || !form.username.trim()}
                >
                  <Plug size={14} /> {testing ? '…' : $t('knowledge.testConnection')}
                </button>
                {testResult && (
                  <span className={testResult.ok ? 'verify-ok' : 'verify-fail'}>
                    {testResult.msg}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Remote archives */}
      <div className="assistant-card" style={{ padding: 16, marginBottom: 16 }}>
        <h3 className="section-header" style={{ fontSize: 16 }}>
          <Archive size={16} /> {$t('backup.remoteSection')}
        </h3>
        {remoteError ? (
          <div className="text-muted">{remoteError}</div>
        ) : remote.filter((f) => f.is_backup).length === 0 ? (
          <div className="text-muted">{$t('backup.noRemote')}</div>
        ) : (
          <table className="data-table modern">
            <thead>
              <tr>
                <th>{$t('backup.colFile')}</th>
                <th>{$t('backup.colSize')}</th>
                <th>{$t('backup.colModified')}</th>
                <th>{$t('common.actions')}</th>
              </tr>
            </thead>
            <tbody>
              {remote
                .filter((f) => f.is_backup)
                .map((f) => (
                  <tr key={f.name}>
                    <td style={{ wordBreak: 'break-all' }}>{f.name}</td>
                    <td>{formatBytes(f.size_bytes)}</td>
                    <td>{formatDate(f.last_modified)}</td>
                    <td>
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                        <button
                          className="btn-icon"
                          title={$t('backup.verify')}
                          disabled={verifyState[f.name]?.busy}
                          onClick={() => handleVerify(f.name)}
                        >
                          {verifyState[f.name]?.busy ? <Loader size={14} className="spin" /> : <ShieldCheck size={14} />}
                        </button>
                        {canRestore && (
                          <button
                            className="btn-icon"
                            title={$t('backup.restore')}
                            onClick={() => setRestoreTarget(f.name)}
                          >
                            <RotateCcw size={14} />
                          </button>
                        )}
                        {canWrite && (
                          <button
                            className="btn-icon"
                            title={$t('common.delete')}
                            onClick={() => setDeleteTarget(f.name)}
                          >
                            <Trash2 size={14} />
                          </button>
                        )}
                        {verifyState[f.name]?.msg && (
                          <span className={verifyState[f.name]?.ok ? 'verify-ok' : 'verify-fail'}>
                            {verifyState[f.name]?.msg}
                          </span>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Job history */}
      <div className="assistant-card" style={{ padding: 16 }}>
        <h3 className="section-header" style={{ fontSize: 16 }}>
          <History size={16} /> {$t('backup.jobsSection')}
        </h3>
        {jobs.length === 0 ? (
          <div className="text-muted">{$t('backup.noJobs')}</div>
        ) : (
          <table className="data-table modern">
            <thead>
              <tr>
                <th>{$t('backup.colStarted')}</th>
                <th>{$t('backup.colKind')}</th>
                <th>{$t('backup.colStatus')}</th>
                <th>{$t('backup.colFile')}</th>
                <th>{$t('backup.colSize')}</th>
                <th>{$t('backup.colDuration')}</th>
                <th>{$t('backup.colError')}</th>
                {canWrite && <th />}
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id}>
                  <td>{formatDate(j.started_at)}</td>
                  <td>
                    <span className="badge neutral">{j.kind}</span>
                  </td>
                  <td>
                    <span
                      className={`badge ${
                        j.status === 'success'
                          ? 'success'
                          : j.status === 'running'
                            ? 'badge-running'
                            : j.status === 'error'
                              ? 'error'
                              : 'warning'
                      }`}
                    >
                      {j.status}
                    </span>
                  </td>
                  <td style={{ wordBreak: 'break-all' }}>{j.filename ?? '—'}</td>
                  <td>{formatBytes(j.size_bytes)}</td>
                  <td>{formatDuration(j.duration_ms)}</td>
                  <td style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis' }} title={j.error ?? ''}>
                    {j.error ?? '—'}
                  </td>
                  {canWrite && (
                    <td>
                      {j.status === 'running' && (
                        <button
                          className="btn-icon"
                          title={$t('backup.cancelJob')}
                          onClick={handleCancelJob}
                        >
                          <XCircle size={14} />
                        </button>
                      )}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <RestoreBackupModal
        open={restoreTarget !== null}
        filename={restoreTarget ?? ''}
        busy={restoring}
        onCancel={() => setRestoreTarget(null)}
        onConfirm={handleRestore}
      />

      <ConfirmModal
        open={deleteTarget !== null}
        title={$t('backup.deleteTitle')}
        message={deleteTarget ?? ''}
        onConfirm={handleDeleteRemote}
        onCancel={() => setDeleteTarget(null)}
      />

      {canWrite && (
        <UnsavedChangesBar
          dirty={isDirty}
          saving={saving}
          onSave={handleSave}
          onDiscard={() => setForm(baseline ? { ...baseline } : null)}
        />
      )}
    </div>
  )
}
