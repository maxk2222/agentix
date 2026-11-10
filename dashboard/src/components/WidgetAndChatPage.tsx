import { useState } from 'react'
import { MessageSquare, Globe } from 'lucide-react'
import WidgetPage from './WidgetPage'
import PublicChatConfigPage from './PublicChatConfigPage'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

type TabId = 'widget' | 'public-chat'

export default function WidgetAndChatPage() {
  const [activeTab, setActiveTab] = useState<TabId>('widget')
  const { language } = useAppPreferences()
  const $t = useT(language)

  const tabs: { id: TabId; label: string; icon: React.ComponentType<any> }[] = [
    { id: 'widget', label: $t('widgetAndChat.widgetTab'), icon: MessageSquare },
    { id: 'public-chat', label: $t('widgetAndChat.publicChatTab'), icon: Globe },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="tabs-header" style={{ display: 'flex', gap: 0, borderBottom: '1px solid var(--border)' }}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            style={{
              padding: '12px 20px',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              background: activeTab === tab.id ? 'var(--bg)' : 'transparent',
              border: 'none',
              borderBottom: activeTab === tab.id ? '2px solid var(--primary)' : '2px solid transparent',
              color: activeTab === tab.id ? 'var(--primary)' : 'var(--muted)',
              cursor: 'pointer',
              fontWeight: activeTab === tab.id ? 600 : 400,
              fontSize: 14,
            }}
          >
            <tab.icon size={16} />
            {tab.label}
          </button>
        ))}
      </div>
      <div style={{ flex: 1, overflow: 'auto' }}>
        {activeTab === 'widget' && <WidgetPage />}
        {activeTab === 'public-chat' && <PublicChatConfigPage />}
      </div>
    </div>
  )
}
