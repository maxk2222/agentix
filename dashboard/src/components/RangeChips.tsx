import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

const RANGES = [24, 168, 720, 8760] as const

export default function RangeChips({
  value,
  onChange,
}: {
  value: number
  onChange: (hours: number) => void
}) {
  const { language } = useAppPreferences()
  const $t = useT(language)

  const labels: Record<number, string> = {
    24: $t('overview.last24h'),
    168: $t('overview.last7d'),
    720: $t('overview.last30d'),
    8760: $t('overview.last365d'),
  }

  return (
    <div className="filter-chips">
      {RANGES.map((r) => (
        <button
          key={r}
          className={`filter-chip ${value === r ? 'active' : ''}`}
          onClick={() => onChange(r)}
        >
          {labels[r]}
        </button>
      ))}
    </div>
  )
}
