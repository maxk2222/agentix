import { Copy, History, Loader, Play, Settings2, Trash2 } from 'lucide-react'
import type { TriggerEntry } from '../../api/client'
import EntityCard from '../EntityCard'
import ActionsMenu, { type ActionMenuItem } from '../ActionsMenu'
import {
  SCHEDULE_ICON,
  SCHEDULE_ICON_CLASS,
  formatRemaining,
  targetLabel,
} from './catalog'

export function TriggerCard({
  trigger,
  now,
  canWrite,
  canRun,
  onEdit,
  onHistory,
  onDuplicate,
  onDelete,
  onRunNow,
  onToggle,
  $t,
}: {
  trigger: TriggerEntry
  now: number
  canWrite: boolean
  canRun: boolean
  onEdit: () => void
  onHistory: () => void
  onDuplicate: () => void
  onDelete: () => void
  onRunNow: () => void
  onToggle: () => void
  $t: (key: string) => string
}) {
  const Icon = SCHEDULE_ICON[trigger.schedule_type] || SCHEDULE_ICON.interval
  const iconClass = SCHEDULE_ICON_CLASS[trigger.schedule_type] || 'sched-interval'

  const menuItems: ActionMenuItem[] = [
    { icon: Settings2, label: $t('triggers.edit'), onClick: onEdit },
    { icon: Play, label: $t('triggers.runNow'), hidden: !canRun, onClick: onRunNow },
    { icon: History, label: $t('triggers.history'), onClick: onHistory },
    { icon: Copy, label: $t('common.duplicate'), hidden: !canWrite, onClick: onDuplicate },
    { icon: Trash2, label: $t('common.delete'), danger: true, hidden: !canWrite, onClick: onDelete },
  ]

  return (
    <EntityCard
      icon={Icon}
      iconClass={iconClass}
      title={trigger.name || $t('triggers.create')}
      off={!trigger.enabled}
      badges={
        <>
          {trigger.enabled ? (
            <span className="badge badge-current">{$t('common.on')}</span>
          ) : (
            <span className="rc-badge disabled">{$t('common.off')}</span>
          )}
          <span className="badge neutral" title={targetLabel(trigger)}>
            {targetLabel(trigger)}
          </span>
          <span className="badge neutral" title={JSON.stringify(trigger.schedule_config)}>
            <Icon size={10} /> {trigger.schedule_type}
          </span>
          {trigger.is_running && (
            <span className="badge badge-running">
              <Loader size={10} /> {$t('triggers.running')}
            </span>
          )}
        </>
      }
      toggle={{
        checked: trigger.enabled,
        disabled: !canWrite,
        title: trigger.enabled ? $t('common.on') : $t('common.off'),
        onChange: onToggle,
      }}
      menu={<ActionsMenu items={menuItems} title={$t('common.actions')} />}
      description={trigger.prompt}
      footerLeft={
        trigger.enabled && trigger.next_run_at ? (
          <span className="text-muted">
            {formatRemaining(trigger.next_run_at, now, $t)}
          </span>
        ) : trigger.last_run_at ? (
          <span className="text-muted">
            {$t('triggers.lastRun')}: {new Date(trigger.last_run_at).toLocaleString("de-DE")}
          </span>
        ) : undefined
      }
      footerRight={
        <button className="btn-link" onClick={onEdit}>
          {$t('triggers.edit')} →
        </button>
      }
      onClick={onEdit}
    />
  )
}
