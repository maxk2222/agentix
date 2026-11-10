import { useEffect, useRef, useState } from 'react'
import { AlertCircle, Save, RotateCcw, X } from 'lucide-react'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'
import ConfirmModal from './ConfirmModal'

/**
 * Warns before leaving the page (browser close/reload) or navigating to another
 * in-app route while `dirty` is true.
 *
 * In-app navigation can't be blocked synchronously by a styled modal, so the
 * guard cancels the click and dispatches `NAV_CONFIRM_EVENT`. `UnsavedChangesBar`
 * (rendered by the same page while dirty) listens for it and shows a ConfirmModal;
 * on confirm it sets `navConfirmed` and navigates via `location.href`.
 */
type NavConfirmRequest = { url: string; message: string }
const NAV_CONFIRM_EVENT = 'agentix:nav-confirm'
let navConfirmed = false

export function useUnsavedGuard(dirty: boolean, warningMessage: string) {
  const dirtyRef = useRef(dirty)
  dirtyRef.current = dirty
  const msgRef = useRef(warningMessage)
  msgRef.current = warningMessage

  useEffect(() => {
    if (!dirty) return

    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (navConfirmed) return
      e.preventDefault()
      e.returnValue = ''
    }

    const onClickCapture = (e: MouseEvent) => {
      if (!dirtyRef.current) return
      const anchor = (e.target as HTMLElement | null)
        ?.closest?.('a[href]') as HTMLAnchorElement | null
      if (!anchor || anchor.target === '_blank' || anchor.hasAttribute('download')) return
      let url: URL
      try {
        url = new URL(anchor.href, window.location.href)
      } catch {
        return
      }
      if (url.origin !== window.location.origin) return
      if (url.pathname === window.location.pathname) return
      e.preventDefault()
      e.stopPropagation()
      window.dispatchEvent(
        new CustomEvent<NavConfirmRequest>(NAV_CONFIRM_EVENT, {
          detail: { url: url.href, message: msgRef.current },
        })
      )
    }

    window.addEventListener('beforeunload', onBeforeUnload)
    document.addEventListener('click', onClickCapture, true)
    return () => {
      window.removeEventListener('beforeunload', onBeforeUnload)
      document.removeEventListener('click', onClickCapture, true)
    }
  }, [dirty])
}

export default function UnsavedChangesBar({
  dirty,
  saving = false,
  onSave,
  onDiscard,
}: {
  dirty: boolean
  saving?: boolean
  onSave: () => void
  onDiscard?: () => void
}) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [pendingNav, setPendingNav] = useState<NavConfirmRequest | null>(null)

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<NavConfirmRequest>).detail
      if (detail?.url) setPendingNav(detail)
    }
    window.addEventListener(NAV_CONFIRM_EVENT, handler)
    return () => window.removeEventListener(NAV_CONFIRM_EVENT, handler)
  }, [])

  if (!dirty) return null

  return (
    <>
      <div className="unsaved-bar">
        <span className="unsaved-bar-label">
          <AlertCircle size={14} /> {$t('common.unsavedChanges')}
        </span>
        {onDiscard && (
          <button
            type="button"
            className="btn-secondary"
            onClick={() => setConfirmOpen(true)}
            disabled={saving}
          >
            <X size={14} /> {$t('common.discard')}
          </button>
        )}
        <button
          type="button"
          className="btn-primary"
          onClick={onSave}
          disabled={saving}
        >
          {saving ? <RotateCcw size={14} className="spin" /> : <Save size={14} />}{' '}
          {saving ? $t('common.saving') : $t('common.save')}
        </button>
      </div>
      {onDiscard && (
        <ConfirmModal
          open={confirmOpen}
          title={$t('common.discard')}
          message={$t('common.discardConfirm')}
          confirmText={$t('common.discard')}
          onConfirm={() => {
            setConfirmOpen(false)
            onDiscard()
          }}
          onCancel={() => setConfirmOpen(false)}
        />
      )}
      <ConfirmModal
        open={pendingNav !== null}
        title={$t('common.unsavedChanges')}
        message={pendingNav?.message || ''}
        confirmText={$t('common.discard')}
        onConfirm={() => {
          const url = pendingNav?.url
          setPendingNav(null)
          if (url) {
            navConfirmed = true
            window.location.href = url
          }
        }}
        onCancel={() => setPendingNav(null)}
      />
    </>
  )
}
