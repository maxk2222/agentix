import { useState, useEffect } from 'react'
import { AlertTriangle, X } from 'lucide-react'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

export default function RestoreBackupModal({
  open,
  filename,
  onConfirm,
  onCancel,
  busy,
}: {
  open: boolean
  filename: string
  onConfirm: () => void
  onCancel: () => void
  busy?: boolean
}) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const [typed, setTyped] = useState('')

  useEffect(() => {
    if (open) setTyped('')
  }, [open])

  if (!open) return null

  const matches = typed.trim() === filename

  return (
    <div className="modal-overlay" onClick={busy ? undefined : onCancel}>
      <div className="modal-box confirm-modal" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div className="confirm-icon">
            <AlertTriangle size={40} />
          </div>
          <button className="btn-icon" onClick={onCancel} disabled={busy} title={$t('common.close')}>
            <X size={18} />
          </button>
        </div>
        <h4>{$t('backup.restoreTitle')}</h4>
        <p style={{ whiteSpace: 'pre-wrap' }}>{$t('backup.restoreWarning')}</p>
        <p style={{ fontSize: '13px' }}>
          {$t('backup.restoreTypePrompt')}{' '}
          <code style={{ wordBreak: 'break-all' }}>{filename}</code>
        </p>
        <input
          type="text"
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          placeholder={filename}
          disabled={busy}
          autoFocus
          style={{ width: '100%', marginBottom: '12px' }}
        />
        <div className="confirm-actions">
          <button type="button" className="btn-secondary" onClick={onCancel} disabled={busy}>
            {$t('common.cancel')}
          </button>
          <button
            type="button"
            className="btn-danger"
            onClick={onConfirm}
            disabled={!matches || busy}
          >
            {busy ? $t('backup.restoring') : $t('backup.restoreConfirm')}
          </button>
        </div>
      </div>
    </div>
  )
}
