import type { ReactNode } from 'react'
import { Search } from 'lucide-react'

export type FilterChip = {
  id: string
  label: string
}

export default function SearchToolbar({
  search,
  onSearch,
  placeholder,
  chips,
  activeChip,
  onChip,
  children,
}: {
  search: string
  onSearch: (v: string) => void
  placeholder?: string
  chips?: FilterChip[]
  activeChip?: string
  onChip?: (id: string) => void
  children?: ReactNode
}) {
  return (
    <div className="list-toolbar">
      <div className="search-box">
        <Search size={14} />
        <input
          type="text"
          value={search}
          placeholder={placeholder}
          onChange={(e) => onSearch(e.target.value)}
        />
      </div>
      {chips && chips.length > 0 && (
        <div className="filter-chips">
          {chips.map((c) => (
            <button
              key={c.id}
              className={`filter-chip ${activeChip === c.id ? 'active' : ''}`}
              onClick={() => onChip?.(c.id)}
            >
              {c.label}
            </button>
          ))}
        </div>
      )}
      {children}
    </div>
  )
}
