import {
  CheckCircle,
  ClipboardPaste,
  Copy,
  CopyPlus,
  Download,
  History,
  LayoutGrid,
  Play,
  Redo2,
  Undo2,
  X,
} from 'lucide-react'

function ToolButton({
  icon: Icon,
  title,
  onClick,
  disabled,
  active,
}: {
  icon: any
  title: string
  onClick: () => void
  disabled?: boolean
  active?: boolean
}) {
  return (
    <button
      type="button"
      className={`chain-tool-btn${active ? ' active' : ''}`}
      onClick={onClick}
      disabled={disabled}
      title={title}
    >
      <Icon size={14} />
    </button>
  )
}

export function EditorToolbar({
  $t,
  canWrite,
  isNew,
  canUndo,
  canRedo,
  hasSelection,
  hasClipboard,
  templateNames,
  testOpen,
  runsOpen,
  onToggleRuns,
  onValidate,
  onUndo,
  onRedo,
  onCopy,
  onPaste,
  onDuplicate,
  onAutoLayout,
  onApplyTemplate,
  onToggleTest,
  onExport,
  onClose,
}: {
  $t: (k: string) => string
  canWrite: boolean
  isNew: boolean
  canUndo: boolean
  canRedo: boolean
  hasSelection: boolean
  hasClipboard: boolean
  templateNames: string[]
  testOpen: boolean
  runsOpen?: boolean
  onToggleRuns?: () => void
  onValidate: () => void
  onUndo: () => void
  onRedo: () => void
  onCopy: () => void
  onPaste: () => void
  onDuplicate: () => void
  onAutoLayout: () => void
  onApplyTemplate: (name: string) => void
  onToggleTest: () => void
  onExport: () => void
  onClose: () => void
}) {
  return (
    <div className="chain-toolbar">
      <ToolButton icon={CheckCircle} title={$t('agentChains.validate')} onClick={onValidate} />
      <span className="chain-toolbar-divider" />
      <ToolButton icon={Undo2} title={$t('agentChains.toolbar.undo')} onClick={onUndo} disabled={!canUndo || !canWrite} />
      <ToolButton icon={Redo2} title={$t('agentChains.toolbar.redo')} onClick={onRedo} disabled={!canRedo || !canWrite} />
      <span className="chain-toolbar-divider" />
      <ToolButton icon={Copy} title={$t('agentChains.toolbar.copy')} onClick={onCopy} disabled={!hasSelection || !canWrite} />
      <ToolButton icon={ClipboardPaste} title={$t('agentChains.toolbar.paste')} onClick={onPaste} disabled={!hasClipboard || !canWrite} />
      <ToolButton icon={CopyPlus} title={$t('agentChains.toolbar.duplicate')} onClick={onDuplicate} disabled={!hasSelection || !canWrite} />
      <span className="chain-toolbar-divider" />
      <ToolButton icon={LayoutGrid} title={$t('agentChains.toolbar.layout')} onClick={onAutoLayout} disabled={!canWrite} />
      {canWrite && (
        <select
          className="chain-template-select"
          value=""
          title={$t('agentChains.toolbar.template')}
          onChange={(e) => {
            const v = e.target.value
            if (v) onApplyTemplate(v)
            e.target.value = ''
          }}
        >
          <option value="" disabled>
            {$t('agentChains.toolbar.template')}
          </option>
          {templateNames.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      )}
      <span className="chain-toolbar-divider" />
      {canWrite && (
        <ToolButton icon={Play} title={$t('agentChains.test.title')} onClick={onToggleTest} active={testOpen} />
      )}
      {!isNew && onToggleRuns && (
        <ToolButton icon={History} title={$t('agentChains.runs.title')} onClick={onToggleRuns} active={runsOpen} />
      )}
      {!isNew && <ToolButton icon={Download} title={$t('agentChains.export')} onClick={onExport} />}
      <ToolButton icon={X} title={$t('agentChains.cancel')} onClick={onClose} />
    </div>
  )
}
