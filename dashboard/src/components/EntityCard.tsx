import type { ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'

export type EntityCardToggle = {
  checked: boolean
  disabled?: boolean
  title?: string
  onChange: () => void
}

export default function EntityCard({
  icon: Icon,
  iconClass,
  title,
  off,
  badges,
  toggle,
  menu,
  description,
  footerLeft,
  footerRight,
  onClick,
}: {
  icon: LucideIcon
  iconClass?: string
  title: string
  off?: boolean
  badges?: ReactNode
  toggle?: EntityCardToggle
  menu?: ReactNode
  description?: ReactNode
  footerLeft?: ReactNode
  footerRight?: ReactNode
  onClick?: () => void
}) {
  return (
    <div
      className={`entity-card ${off ? 'off' : ''} ${onClick ? 'clickable' : ''}`}
      onClick={onClick}
    >
      <div className="entity-card-head">
        <span className={`entity-icon ${iconClass || ''}`}>
          <Icon size={17} />
        </span>
        <div className="entity-card-title">
          <span className="entity-card-name" title={title}>
            {title}
          </span>
          {badges && <div className="entity-card-badges">{badges}</div>}
        </div>
        {toggle && (
          <label
            className={`rc-toggle sm ${toggle.disabled ? 'locked' : ''}`}
            title={toggle.title}
            onClick={(e) => e.stopPropagation()}
          >
            <input
              type="checkbox"
              checked={toggle.checked}
              onChange={toggle.onChange}
              disabled={toggle.disabled}
            />
            <span className="rc-toggle-track">
              <span className="rc-toggle-knob" />
            </span>
          </label>
        )}
        {menu}
      </div>

      {description !== undefined && (
        <p
          className="entity-card-desc"
          title={typeof description === 'string' ? description : undefined}
        >
          {description || '—'}
        </p>
      )}

      {(footerLeft || footerRight) && (
        <div className="entity-card-foot">
          {footerLeft ?? <span />}
          {footerRight}
        </div>
      )}
    </div>
  )
}
