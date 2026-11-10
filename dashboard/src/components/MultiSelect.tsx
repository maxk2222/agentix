import { useEffect, useRef, useState } from 'react'
import { ChevronDown, X } from 'lucide-react'

export default function MultiSelect({
  value,
  options,
  onChange,
  disabled,
  placeholder,
}: {
  value: string[]
  options: string[]
  onChange: (v: string[]) => void
  disabled?: boolean
  placeholder?: string
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [])

  const available = options.filter((o) => !value.includes(o))

  const remove = (v: string) => onChange(value.filter((x) => x !== v))
  const add = (v: string) => {
    if (!value.includes(v)) onChange([...value, v])
    setOpen(false)
  }

  return (
    <div ref={ref} className="multi-select">
      <div
        className={`multi-select-box ${disabled ? 'disabled' : ''}`}
        onClick={() => !disabled && setOpen(!open)}
      >
        {value.length === 0 && (
          <span className="multi-select-placeholder">{placeholder}</span>
        )}
        {value.map((v) => (
          <span key={v} className="multi-select-chip">
            {v}
            {!disabled && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  remove(v)
                }}
              >
                <X size={12} />
              </button>
            )}
          </span>
        ))}
        {!disabled && <ChevronDown size={14} className="multi-select-chevron" />}
      </div>

      {open && available.length > 0 && (
        <div className="multi-select-dropdown">
          {available.map((opt) => (
            <div key={opt} className="multi-select-option" onClick={() => add(opt)}>
              {opt}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
