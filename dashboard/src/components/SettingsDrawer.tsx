import { useState, type ReactNode } from 'react'
import { X } from 'lucide-react'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'
import ConfirmModal from './ConfirmModal'

export type DrawerTab = {
  id: string
  label: string
  show?: boolean
}

export default function SettingsDrawer({
  title,
  badges,
  tabs,
  activeTab,
  onTabChange,
  isDirty = false,
  confirmDiscard = true,
  onClose,
  footer,
  children,
}: {
  title: ReactNode
  badges?: ReactNode
  tabs?: DrawerTab[]
  activeTab?: string
  onTabChange?: (id: string) => void
  isDirty?: boolean
  confirmDiscard?: boolean
  onClose: () => void
  footer?: ReactNode
  children: ReactNode
}) {
  const { language } = useAppPreferences()
  const $t = useT(language)

  const [discardConfirmOpen, setDiscardConfirmOpen] = useState(false)

  const handleClose = () => {
    if (isDirty && confirmDiscard) {
      setDiscardConfirmOpen(true)
      return
    }
    onClose()
  }

  const visibleTabs = (tabs || []).filter((t) => t.show !== false)

  return (
    <div className="drawer-overlay" onClick={handleClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-head">
          <div className="drawer-title">
            {title}
            {badges}
          </div>
          <button className="btn-icon" onClick={handleClose} title={$t('common.cancel')}>
            <X size={16} />
          </button>
        </div>

        {visibleTabs.length > 1 && (
          <div className="drawer-tabs">
            {visibleTabs.map((t) => (
              <button
                key={t.id}
                className={`drawer-tab ${activeTab === t.id ? 'active' : ''}`}
                onClick={() => onTabChange?.(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>
        )}

        <div className="drawer-body">{children}</div>
        {footer}
        <ConfirmModal
          open={discardConfirmOpen}
          title={$t('common.discard')}
          message={$t('common.discardConfirm')}
          confirmText={$t('common.discard')}
          onConfirm={() => {
            setDiscardConfirmOpen(false)
            onClose()
          }}
          onCancel={() => setDiscardConfirmOpen(false)}
        />
      </div>
    </div>
  )
}
