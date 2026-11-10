import { useEffect, useRef, useState } from 'react'
import { Edit3 } from 'lucide-react'

/**
 * Text-input dialog — the styled replacement for window.prompt().
 * Purely presentational: all strings come from the caller so the same
 * component works in the dashboard and the standalone public chat.
 */
export default function PromptModal({
  open,
  title,
  message,
  initialValue = '',
  placeholder,
  confirmText = 'OK',
  cancelText,
  onSubmit,
  onCancel,
}: {
  open: boolean
  title: string
  message?: string
  initialValue?: string
  placeholder?: string
  confirmText?: string
  cancelText?: string
  onSubmit: (value: string) => void
  onCancel: () => void
}) {
  const [value, setValue] = useState(initialValue)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setValue(initialValue)
      // Autofocus after the overlay mounts
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [open, initialValue])

  if (!open) return null

  const submit = () => {
    const v = value.trim()
    if (v) onSubmit(v)
  }

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-box confirm-modal" onClick={(e) => e.stopPropagation()}>
        <div className="confirm-icon">
          <Edit3 size={40} />
        </div>
        <h4>{title}</h4>
        {message && <p style={{ whiteSpace: 'pre-wrap' }}>{message}</p>}
        <form
          onSubmit={(e) => {
            e.preventDefault()
            submit()
          }}
        >
          <input
            ref={inputRef}
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={placeholder}
            style={{ width: '100%', marginBottom: 14 }}
          />
        </form>
        <div className="confirm-actions">
          <button type="button" className="btn-secondary" onClick={onCancel}>
            {cancelText || 'Cancel'}
          </button>
          <button type="button" className="btn-primary" onClick={submit} disabled={!value.trim()}>
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  )
}
