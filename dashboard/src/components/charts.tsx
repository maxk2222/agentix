import type { TooltipProps } from 'recharts'

export const API_KEY_COLORS = [
  '#2563eb', '#059669', '#d97706', '#dc2626', '#7c3aed',
  '#0ea5e9', '#f97316', '#f59e0b', '#ec4899', '#14b8a6',
]

export const CHART_COLORS = {
  primary: '#2563eb',
  primarySoft: 'rgba(37, 99, 235, 0.12)',
  success: '#059669',
  successSoft: 'rgba(5, 150, 105, 0.12)',
  warning: '#d97706',
  warningSoft: 'rgba(217, 119, 6, 0.12)',
  danger: '#dc2626',
  dangerSoft: 'rgba(220, 38, 38, 0.12)',
  info: '#0ea5e9',
  infoSoft: 'rgba(14, 165, 233, 0.12)',
  orange: '#f97316',
  orangeSoft: 'rgba(249, 115, 22, 0.12)',
  purple: '#7c3aed',
  purpleSoft: 'rgba(124, 58, 237, 0.12)',
  amber: '#f59e0b',
  amberSoft: 'rgba(245, 158, 11, 0.12)',
  slate: '#6b7280',
  grid: 'var(--chart-grid)',
  axis: 'var(--chart-axis)',
}

export function formatNumber(value: number | string): string {
  const num = typeof value === 'string' ? parseFloat(value) : value
  if (Number.isNaN(num)) return String(value)
  if (Math.abs(num) >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`
  if (Math.abs(num) >= 1_000) return `${(num / 1_000).toFixed(1)}K`
  return num.toLocaleString(undefined, { maximumFractionDigits: 1 })
}

export function ChartTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-label">{label}</div>
      <div className="chart-tooltip-rows">
        {payload.map((entry, i) => (
          <div key={i} className="chart-tooltip-row">
            <span
              className="chart-tooltip-dot"
              style={{ background: entry.color }}
            />
            <span className="chart-tooltip-name">{entry.name}</span>
            <span className="chart-tooltip-value">
              {entry.value !== undefined && entry.value !== null
                ? formatNumber(entry.value)
                : '—'}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function EmptyChart({ message }: { message?: string }) {
  return (
    <div className="chart-empty">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 3v18h18" />
        <path d="M7 16l4-4 4 4 6-6" />
      </svg>
      <p>{message}</p>
    </div>
  )
}

export const chartAxisProps = {
  stroke: CHART_COLORS.axis,
  tickLine: false,
  axisLine: { stroke: CHART_COLORS.grid },
  tick: { fill: CHART_COLORS.axis, fontSize: 12 },
}

export const chartGridProps = {
  strokeDasharray: '3 3',
  stroke: CHART_COLORS.grid,
  vertical: false,
}

export const chartLegendProps = {
  wrapperStyle: { paddingTop: 16 },
  iconType: 'circle' as const,
  iconSize: 8,
}
