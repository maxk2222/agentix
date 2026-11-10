import { createPortal } from 'react-dom'
import { Info } from 'lucide-react'

/**
 * Single-button notice dialog — the styled replacement for window.alert().
 * Purely presentational: all strings come from the caller so the same
 * component works in the dashboard and the standalone public chat.
 */
export default function AlertModal({
  open,
  title,
  message,
  okText = 'OK',
  onClose,
  danger,
}: {
  open: boolean
  title: string
  message: string
  okText?: string
  onClose: () => void
  /** Red styling for error notices. */
  danger?: boolean
}) {
  if (!open) return null

  // Portal to <body>: clipped/filtered/transformed ancestors (e.g. the
  // floating workspace panel) would otherwise cut off the fixed overlay.
  return createPortal(
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box confirm-modal" onClick={(e) => e.stopPropagation()}>
        <div className="confirm-icon" style={danger ? { color: 'var(--danger)' } : undefined}>
          <Info size={40} />
        </div>
        <h4>{title}</h4>
        <p style={{ whiteSpace: 'pre-wrap' }}>{message}</p>
        <div className="confirm-actions">
          <button
            type="button"
            className={danger ? 'btn-danger' : 'btn-primary'}
            onClick={onClose}
            autoFocus
          >
            {okText}
          </button>
        </div>
      </div>
    </div>,
    document.body
  )
}
