import { ArrowRight, Copy, Settings2, Trash2, Workflow } from 'lucide-react'
import type { AgentChainEntry } from '../../api/client'
import EntityCard from '../EntityCard'
import ActionsMenu, { type ActionMenuItem } from '../ActionsMenu'

export function ChainCard({
  chain,
  canWrite,
  onEdit,
  onDuplicate,
  onDelete,
  onExport,
  onToggle,
  $t,
}: {
  chain: AgentChainEntry
  canWrite: boolean
  onEdit: () => void
  onDuplicate: () => void
  onDelete: () => void
  onExport: () => void
  onToggle: () => void
  $t: (key: string) => string
}) {
  const flow = chain.flow_definition as { nodes?: unknown[]; edges?: unknown[] } | undefined
  const stats = $t('agentChains.nodeStats')
    .replace('{nodes}', String(flow?.nodes?.length ?? 0))
    .replace('{edges}', String(flow?.edges?.length ?? 0))

  const menuItems: ActionMenuItem[] = [
    { icon: Settings2, label: $t('agentChains.edit'), onClick: onEdit },
    { icon: ArrowRight, label: $t('agentChains.export'), onClick: onExport },
    { icon: Copy, label: $t('common.duplicate'), hidden: !canWrite, onClick: onDuplicate },
    { icon: Trash2, label: $t('common.delete'), danger: true, hidden: !canWrite, onClick: onDelete },
  ]

  return (
    <EntityCard
      icon={Workflow}
      iconClass="chain"
      title={chain.name}
      off={!chain.enabled}
      badges={
        <>
          {chain.enabled ? (
            <span className="badge success">{$t('agentChains.enabled')}</span>
          ) : (
            <span className="rc-badge disabled">{$t('agentChains.disabled')}</span>
          )}
          {chain.api_accessible && <span className="badge info">API</span>}
        </>
      }
      toggle={{
        checked: chain.enabled,
        disabled: !canWrite,
        title: chain.enabled ? $t('agentChains.enabled') : $t('agentChains.disabled'),
        onChange: onToggle,
      }}
      menu={<ActionsMenu items={menuItems} title={$t('common.actions')} />}
      description={chain.description}
      footerLeft={<span className="text-muted">{stats}</span>}
      footerRight={
        <button className="btn-link" onClick={onEdit}>
          {$t('agentChains.edit')} →
        </button>
      }
      onClick={onEdit}
    />
  )
}
