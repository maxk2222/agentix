import { useState, useEffect, useRef } from 'react'
import { Globe, FolderOpen, User, Lock, Plug } from 'lucide-react'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'
import { knowledge } from '../api/client'

export type WebDAVSourceFormData = {
  name: string
  webdav_url: string
  remote_path: string
  username: string
  password: string
  enabled: boolean
  convert_to_okf: boolean
}

export default function WebDAVSourceModal({
  open,
  initial,
  onConfirm,
  onCancel,
}: {
  open: boolean
  initial?: Partial<WebDAVSourceFormData>
  onConfirm: (data: WebDAVSourceFormData) => void
  onCancel: () => void
}) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const [form, setForm] = useState<WebDAVSourceFormData>({
    name: '',
    webdav_url: '',
    remote_path: '/',
    username: '',
    password: '',
    enabled: true,
    convert_to_okf: true,
  })
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null)
  const [testing, setTesting] = useState(false)
  const nameRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setForm({
        name: initial?.name ?? '',
        webdav_url: initial?.webdav_url ?? '',
        remote_path: initial?.remote_path ?? '/',
        username: initial?.username ?? '',
        password: initial?.password ?? '',
        enabled: initial?.enabled ?? true,
        convert_to_okf: initial?.convert_to_okf ?? true,
      })
      setErrors({})
      setTestResult(null)
      setTimeout(() => nameRef.current?.focus(), 50)
    }
  }, [open, initial])

  if (!open) return null

  const validate = (): boolean => {
    const e: Record<string, string> = {}
    const name = form.name.trim()
    if (!name) e.name = 'Name ist erforderlich'
    else if (name.includes('/') || name.includes('\\')) e.name = $t('knowledge.webdavNameError')

    const url = form.webdav_url.trim()
    if (!url) e.webdav_url = 'URL ist erforderlich'
    else if (!url.startsWith('https://')) e.webdav_url = $t('knowledge.webdavUrlError')

    if (!form.username.trim()) e.username = 'Benutzername ist erforderlich'
    if (!initial?.name && !form.password.trim()) e.password = 'Passwort ist erforderlich'

    setErrors(e)
    return Object.keys(e).length === 0
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!validate()) return
    onConfirm({ ...form, name: form.name.trim(), webdav_url: form.webdav_url.trim() })
  }

  const handleTest = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const res = await knowledge.testWebDAVConnection({
        webdav_url: form.webdav_url.trim(),
        remote_path: form.remote_path,
        username: form.username.trim(),
        password: form.password,
        name: initial?.name,
      })
      if (res.success) {
        setTestResult({ success: true, message: `${$t('knowledge.connectionSuccess')} (${res.file_count ?? 0} Dateien)` })
      } else {
        setTestResult({ success: false, message: `${$t('knowledge.connectionFailed')}: ${res.error ?? 'Unbekannt'}` })
      }
    } catch (err: any) {
      setTestResult({ success: false, message: `${$t('knowledge.connectionFailed')}: ${err.message ?? String(err)}` })
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-box" onClick={(ev) => ev.stopPropagation()} style={{ maxWidth: '520px' }}>
        <h4 style={{ marginBottom: '16px' }}>
          {initial?.name ? $t('knowledge.editWebDAV') : $t('knowledge.addWebDAV')}
        </h4>
        <form onSubmit={handleSubmit}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <label>
              <span style={{ fontSize: '13px', fontWeight: 500 }}>{$t('knowledge.enterName')}</span>
              <input
                ref={nameRef}
                type="text"
                value={form.name}
                onChange={(e) => setForm((s) => ({ ...s, name: e.target.value }))}
                placeholder={$t('knowledge.enterName')}
                disabled={!!initial?.name}
                style={{ width: '100%', marginTop: '4px' }}
              />
              {errors.name && <span style={{ color: 'var(--danger)', fontSize: '12px' }}>{errors.name}</span>}
            </label>

            <label>
              <span style={{ fontSize: '13px', fontWeight: 500 }}>{$t('knowledge.webdavUrl')}</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
                <Globe size={14} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
                <input
                  type="text"
                  value={form.webdav_url}
                  onChange={(e) => setForm((s) => ({ ...s, webdav_url: e.target.value }))}
                  placeholder={$t('knowledge.webdavUrlPlaceholder')}
                  style={{ flex: 1 }}
                />
              </div>
              {errors.webdav_url && <span style={{ color: 'var(--danger)', fontSize: '12px' }}>{errors.webdav_url}</span>}
            </label>

            <label>
              <span style={{ fontSize: '13px', fontWeight: 500 }}>{$t('knowledge.remotePath')}</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
                <FolderOpen size={14} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
                <input
                  type="text"
                  value={form.remote_path}
                  onChange={(e) => setForm((s) => ({ ...s, remote_path: e.target.value }))}
                  placeholder={$t('knowledge.remotePathPlaceholder')}
                  style={{ flex: 1 }}
                />
              </div>
            </label>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <label>
                <span style={{ fontSize: '13px', fontWeight: 500 }}>{$t('knowledge.webdavUsername')}</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
                  <User size={14} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
                  <input
                    type="text"
                    value={form.username}
                    onChange={(e) => setForm((s) => ({ ...s, username: e.target.value }))}
                    style={{ flex: 1 }}
                  />
                </div>
                {errors.username && <span style={{ color: 'var(--danger)', fontSize: '12px' }}>{errors.username}</span>}
              </label>

              <label>
                <span style={{ fontSize: '13px', fontWeight: 500 }}>{$t('knowledge.webdavPassword')}</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
                  <Lock size={14} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
                  <input
                    type="password"
                    value={form.password}
                    onChange={(e) => setForm((s) => ({ ...s, password: e.target.value }))}
                    placeholder={initial?.name ? 'Unverändert lassen' : ''}
                    style={{ flex: 1 }}
                  />
                </div>
                {errors.password && <span style={{ color: 'var(--danger)', fontSize: '12px' }}>{errors.password}</span>}
              </label>
            </div>

            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
              <input
                type="checkbox"
                checked={form.enabled}
                onChange={(e) => setForm((s) => ({ ...s, enabled: e.target.checked }))}
              />
              <span style={{ fontSize: '14px' }}>Aktiviert</span>
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
              <input
                type="checkbox"
                checked={form.convert_to_okf}
                onChange={(e) => setForm((s) => ({ ...s, convert_to_okf: e.target.checked }))}
              />
              <span style={{ fontSize: '14px' }}>{$t('knowledge.convertToOKF')}</span>
            </label>
          </div>

          <div className="modal-footer" style={{ marginTop: '20px', display: 'flex', gap: '8px', alignItems: 'center' }}>
            <button
              type="button"
              className="btn-secondary"
              onClick={handleTest}
              disabled={testing || !form.webdav_url.trim() || !form.username.trim()}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <Plug size={14} />
              {testing ? '...' : $t('knowledge.testConnection')}
            </button>
            <div style={{ flex: 1 }} />
            <button type="button" className="btn-secondary" onClick={onCancel}>
              {$t('common.cancel')}
            </button>
            <button type="submit" className="btn-primary">
              {initial?.name ? $t('common.save') : $t('common.add')}
            </button>
          </div>
          {testResult && (
            <div style={{
              marginTop: '8px',
              padding: '8px 12px',
              borderRadius: '6px',
              fontSize: '13px',
              background: testResult.success ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
              color: testResult.success ? '#22c55e' : '#ef4444',
            }}>
              {testResult.message}
            </div>
          )}
        </form>
      </div>
    </div>
  )
}
