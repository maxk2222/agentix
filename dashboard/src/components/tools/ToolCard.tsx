import {
  AlertTriangle, BookOpen, Copy, Cpu, FileCode2, Globe, Mail,
  Plug, Trash2, Wrench, Workflow, type LucideIcon,
} from 'lucide-react'
import type { ToolEntry } from '../../api/client'
import { categoryFor, configuredCount, needsConfig, type ToolCategoryId } from './catalog'
import EntityCard from '../EntityCard'
import ActionsMenu, { type ActionMenuItem } from '../ActionsMenu'

const CATEGORY_ICON: Record<ToolCategoryId, LucideIcon> = {
  knowledge: BookOpen,
  web: Globe,
  files: FileCode2,
  comms: Mail,
  internals: Cpu,
  mcp: Plug,
  custom: Wrench,
  assistants: Workflow,
  other: Wrench,
}

export function ToolCard({
  tool,
  canWrite,
  onEdit,
  onDuplicate,
  onDelete,
  onToggle,
  $t,
}: {
  tool: ToolEntry
  canWrite: boolean
  onEdit: () => void
  onDuplicate: () => void
  onDelete: () => void
  onToggle: () => void
  $t: (key: string) => string
}) {
  const Icon = CATEGORY_ICON[categoryFor(tool)]
  const needs = needsConfig(tool)
  const cfg = configuredCount(tool)
  const isAssistant = tool.kind === 'assistant'
  const isBuiltin = tool.kind === 'builtin'

  const menuItems: ActionMenuItem[] = [
    { icon: Wrench, label: $t('tools.edit'), onClick: onEdit },
    { icon: Copy, label: $t('tools.duplicate'), hidden: isAssistant, onClick: onDuplicate },
    {
      icon: Trash2,
      label: $t('tools.delete'),
      danger: true,
      hidden: !canWrite || isBuiltin || isAssistant,
      onClick: onDelete,
    },
  ]

  return (
    <EntityCard
      icon={Icon}
      iconClass={`cat-${categoryFor(tool)}`}
      title={tool.name}
      off={!tool.enabled}
      badges={
        <>
          {isAssistant ? (
            Boolean((tool.config as Record<string, unknown>)?.agent_enabled) ? (
              <span className="badge agent">Agent</span>
            ) : (
              <span className="badge assistant">Assistant</span>
            )
          ) : (
            <span className={`badge ${tool.kind}`}>{tool.kind}</span>
          )}
          {tool.kind === 'custom' && (
            <span className={`badge ${(tool.runtime_mode || 'venv') === 'docker' ? 'docker' : 'venv'}`}>
              {(tool.runtime_mode || 'venv') === 'docker' ? 'docker' : 'venv'}
            </span>
          )}
          {!tool.enabled && <span className="rc-badge disabled">{$t('tools.disabled')}</span>}
          {needs && (
            <span className="badge badge-needs-config" title={$t('tools.needsConfigHint')}>
              <AlertTriangle size={11} /> {$t('tools.needsConfig')}
            </span>
          )}
        </>
      }
      toggle={
        !isAssistant
          ? {
              checked: tool.enabled,
              disabled: !canWrite,
              title: tool.enabled ? $t('tools.enabled') : $t('tools.disabled'),
              onChange: onToggle,
            }
          : undefined
      }
      menu={<ActionsMenu items={menuItems} title={$t('tools.actions')} />}
      description={tool.description}
      footerLeft={
        cfg.total > 0 ? (
          <span className="text-muted">
            {$t('tools.configuredCount').replace('{set}', String(cfg.set)).replace('{total}', String(cfg.total))}
          </span>
        ) : undefined
      }
      footerRight={
        <button className="btn-link" onClick={onEdit}>
          {$t('tools.configure')} →
        </button>
      }
    />
  )
}
