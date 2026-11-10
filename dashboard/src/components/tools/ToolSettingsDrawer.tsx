import { useEffect, useMemo, useState } from 'react'
import { knowledge, tools as toolsApi, type ToolEntry } from '../../api/client'
import UnsavedChangesBar, { useUnsavedGuard } from '../UnsavedChangesBar'
import SettingsDrawer from '../SettingsDrawer'
import AlertModal from '../AlertModal'
import { ConfigFieldRenderer } from './ConfigFieldRenderer'
import type { ConfigField } from './catalog'

type TabId = 'settings' | 'advanced' | 'code' | 'schema'

export function ToolSettingsDrawer({
  tool,
  onSave,
  onClose,
  canWrite,
  $t,
}: {
  tool: ToolEntry
  onSave: (t: ToolEntry) => Promise<void>
  onClose: () => void
  canWrite: boolean
  $t: (key: string) => string
}) {
  const [draft, setDraft] = useState<ToolEntry>({ ...tool })
  const [tab, setTab] = useState<TabId>('settings')
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [notice, setNotice] = useState<{ message: string; danger?: boolean } | null>(null)
  const [collections, setCollections] = useState<string[]>([])
  const [collectionsLoading, setCollectionsLoading] = useState(false)

  const schema = (tool.config_schema as ConfigField[] | null | undefined) ?? []
  const isBuiltin = draft.kind === 'builtin'
  const isCustom = draft.kind === 'custom'
  const isAssistant = draft.kind === 'assistant'
  const needsCollections = schema.some((f) => f.type === 'collections')

  useEffect(() => {
    if (!needsCollections) return
    setCollectionsLoading(true)
    knowledge.listCollections()
      .then((cols) => setCollections(cols.map((c) => c.name)))
      .catch(() => setCollections([]))
      .finally(() => setCollectionsLoading(false))
  }, [needsCollections])

  const update = <K extends keyof ToolEntry>(field: K, value: ToolEntry[K]) =>
    setDraft((d) => ({ ...d, [field]: value }))

  const isDirty = useMemo(
    () => JSON.stringify(draft) !== JSON.stringify(tool),
    [draft, tool],
  )
  useUnsavedGuard(isDirty && canWrite, $t('common.leaveWarning'))

  const handleSave = async () => {
    if (!canWrite) return
    setSaving(true)
    setSaveError(null)
    try {
      await onSave(draft)
    } catch (err) {
      setSaveError(String(err))
    } finally {
      setSaving(false)
    }
  }

  const tabs: { id: string; label: string; show: boolean }[] = [
    { id: 'settings', label: $t('tools.tabSettings'), show: !isAssistant },
    { id: 'advanced', label: $t('tools.tabAdvanced'), show: !isAssistant },
    { id: 'code', label: $t('tools.tabCode'), show: isCustom },
    { id: 'schema', label: $t('tools.tabSchema'), show: !isAssistant },
  ]

  const configStr = JSON.stringify(draft.config || {}, null, 2)
  const schemaStr = JSON.stringify(draft.tool_schema || {}, null, 2)

  return (
    <>
    <SettingsDrawer
      title={
        <>
          <WrenchIcon />
          <input
            type="text"
            className="drawer-name-input"
            value={draft.name}
            onChange={(e) => update('name', e.target.value)}
            disabled={!canWrite || isBuiltin || isAssistant}
            placeholder={$t('tools.name')}
          />
        </>
      }
      badges={
        <>
          <span className={`badge ${draft.kind}`}>{draft.kind}</span>
          {isCustom && (
            <span className={`badge ${(draft.runtime_mode || 'venv') === 'docker' ? 'docker' : 'venv'}`}>
              {(draft.runtime_mode || 'venv') === 'docker' ? 'docker' : 'venv'}
            </span>
          )}
        </>
      }
      tabs={tabs}
      activeTab={tab}
      onTabChange={(id) => setTab(id as TabId)}
      isDirty={isDirty && canWrite}
      onClose={onClose}
      footer={
        canWrite && !isAssistant ? (
          <UnsavedChangesBar
            dirty={isDirty}
            saving={saving}
            onSave={handleSave}
            onDiscard={() => setDraft({ ...tool })}
          />
        ) : undefined
      }
    >
      {saveError && <div className="error-box">{saveError}</div>}
      <div className="form-group wide">
        <label>{$t('tools.description')}</label>
        <textarea
          rows={2}
          value={draft.description}
          onChange={(e) => update('description', e.target.value)}
          disabled={!canWrite || isAssistant}
        />
      </div>

      {isAssistant && (
        <div className="info-box">
          {$t('tools.assistantInfo')}
        </div>
      )}

      {tab === 'settings' && !isAssistant && (
        schema.length > 0 ? (
          <ConfigFieldRenderer
            fields={schema}
            config={draft.config || {}}
            onConfigChange={(next) => update('config', next)}
            canWrite={canWrite}
            collections={collections}
            collectionsLoading={collectionsLoading}
            tool={draft}
            $t={$t}
          />
        ) : (
          <div className="form-group wide">
            <label>{$t('tools.config')}</label>
            <small className="field-help">{$t('tools.fieldHelp.config')}</small>
            <textarea
              rows={8}
              className="code-textarea"
              value={configStr}
              onChange={(e) => {
                try {
                  update('config', JSON.parse(e.target.value))
                } catch { /* keep typing */ }
              }}
              disabled={!canWrite}
              placeholder='{ "key": "value" }'
            />
          </div>
        )
      )}

      {tab === 'advanced' && !isAssistant && (
        <div className="form-group wide">
          <label>{$t('tools.config')} (JSON)</label>
          <small className="field-help">{$t('tools.advancedHint')}</small>
          <textarea
            rows={14}
            className="code-textarea"
            value={configStr}
            onChange={(e) => {
              try {
                update('config', JSON.parse(e.target.value))
              } catch { /* keep typing */ }
            }}
            disabled={!canWrite}
            placeholder='{ "key": "value" }'
          />
        </div>
      )}

      {tab === 'code' && isCustom && (
        <>
          <div className="form-group">
            <label>{$t('tools.runtimeMode')}</label>
            <label className={`rc-toggle ${!canWrite ? 'locked' : ''}`}>
              <input
                type="checkbox"
                checked={draft.runtime_mode === 'docker'}
                onChange={(e) => update('runtime_mode', e.target.checked ? 'docker' : 'venv')}
                disabled={!canWrite}
              />
              <span className="rc-toggle-track"><span className="rc-toggle-knob" /></span>
              <span className="rc-toggle-label">
                {draft.runtime_mode === 'docker' ? $t('tools.runtimeDocker') : $t('tools.runtimeVenv')}
              </span>
            </label>
          </div>
          <div className="form-group wide">
            <label>{$t('tools.code')}</label>
            <small className="field-help">{$t('tools.fieldHelp.code')}</small>
            <textarea
              rows={10}
              className="code-textarea"
              value={draft.code || ''}
              onChange={(e) => update('code', e.target.value)}
              disabled={!canWrite}
              placeholder="def run(params):\n    return {'result': params}"
            />
          </div>
          <div className="form-group">
            <label>{$t('tools.pythonVersion')}</label>
            <input
              type="text"
              value={(draft.venv_config?.python_version as string) || ''}
              onChange={(e) =>
                update('venv_config', { ...(draft.venv_config || {}), python_version: e.target.value })
              }
              disabled={!canWrite}
              placeholder="e.g. 3.11"
            />
          </div>
          <div className="form-group wide">
            <label>{$t('tools.requirements')}</label>
            <textarea
              rows={4}
              className="code-textarea"
              value={(draft.venv_config?.requirements as string) || ''}
              onChange={(e) =>
                update('venv_config', { ...(draft.venv_config || {}), requirements: e.target.value })
              }
              disabled={!canWrite}
              placeholder={'requests\nbeautifulsoup4'}
            />
          </div>
          {draft.id && (
            <div className="form-group wide">
              <button
                className="btn-secondary"
                disabled={!canWrite || saving}
                onClick={async () => {
                  try {
                    await toolsApi.rebuildVenv(draft.id)
                    setNotice({ message: $t('tools.rebuildVenvSuccess') })
                  } catch (err) {
                    setNotice({ danger: true, message: String(err) })
                  }
                }}
              >
                {$t('tools.rebuildVenv')}
              </button>
            </div>
          )}
        </>
      )}

      {tab === 'schema' && !isAssistant && (
        <div className="form-group wide">
          <label>{$t('tools.toolSchema')}</label>
          <small className="field-help">{$t('tools.fieldHelp.toolSchema')}</small>
          <textarea
            rows={12}
            className="code-textarea"
            value={schemaStr}
            onChange={(e) => {
              try {
                update('tool_schema', JSON.parse(e.target.value))
              } catch { /* keep typing */ }
            }}
            disabled={!canWrite}
            placeholder={$t('tools.toolSchemaPlaceholder')}
          />
        </div>
      )}
    </SettingsDrawer>
    <AlertModal
      open={notice !== null}
      title={notice?.danger ? 'Error' : 'Info'}
      message={notice?.message || ''}
      danger={notice?.danger}
      onClose={() => setNotice(null)}
    />
    </>
  )
}

function WrenchIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="drawer-wrench">
      <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
    </svg>
  )
}
