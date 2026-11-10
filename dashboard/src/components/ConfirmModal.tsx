import { AlertTriangle } from 'lucide-react'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

export default function ConfirmModal({
  open,
  title,
  message,
  onConfirm,
  onCancel,
  confirmText,
  cancelText,
  confirmDisabled,
}: {
  open: boolean
  title: string
  message: string
  onConfirm: () => void
  onCancel: () => void
  confirmText?: string
  cancelText?: string
  confirmDisabled?: boolean
}) {
  const { language } = useAppPreferences()
  const $t = useT(language)

  if (!open) return null

  return (
    <div className="modal-overlay" onClick={confirmDisabled ? undefined : onCancel}>
      <div className="modal-box confirm-modal" onClick={(e) => e.stopPropagation()}>
        <div className="confirm-icon">
          <AlertTriangle size={40} />
        </div>
        <h4>{title}</h4>
        <p style={{ whiteSpace: 'pre-wrap' }}>{message}</p>
        <div className="confirm-actions">
          <button type="button" className="btn-secondary" onClick={onCancel} disabled={confirmDisabled}>
            {cancelText || $t('common.cancel')}
          </button>
          <button type="button" className="btn-danger" onClick={onConfirm} disabled={confirmDisabled}>
            {confirmText || $t('common.confirmDelete')}
          </button>
        </div>
      </div>
    </div>
  )
}
