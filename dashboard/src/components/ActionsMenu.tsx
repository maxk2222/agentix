import { useEffect, useRef, useState } from 'react'
import { MoreVertical, type LucideIcon } from 'lucide-react'

export type ActionMenuItem = {
  icon: LucideIcon
  label: string
  danger?: boolean
  hidden?: boolean
  onClick: () => void
}

export default function ActionsMenu({
  items,
  title,
}: {
  items: ActionMenuItem[]
  title?: string
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])

  const visible = items.filter((i) => !i.hidden)
  if (visible.length === 0) return null

  return (
    <div className="dropdown-wrap" ref={ref}>
      <button
        className="btn-icon"
        onClick={(e) => {
          e.stopPropagation()
          setOpen((v) => !v)
        }}
        title={title}
      >
        <MoreVertical size={14} />
      </button>
      {open && (
        <div className="dropdown-menu">
          {visible.map((item) => (
            <button
              key={item.label}
              className={`dropdown-item ${item.danger ? 'danger' : ''}`}
              onClick={(e) => {
                e.stopPropagation()
                setOpen(false)
                item.onClick()
              }}
            >
              <item.icon size={14} /> {item.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
