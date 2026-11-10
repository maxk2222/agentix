import { useEffect, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { dashboard } from '../api/client'
import { ChartTooltip, CHART_COLORS, chartAxisProps, chartGridProps, chartLegendProps } from './charts'
import { ShieldAlert, Lock, FileWarning, AlertCircle, Ban } from 'lucide-react'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

export default function Blocked() {
  const [range, setRange] = useState(24)
  const [bucket, setBucket] = useState<'hour' | 'day'>('hour')
  const [summary, setSummary] = useState<{
    total_blocked: number
    auth_blocks: number
    size_blocks: number
    validation_blocks: number
    other_blocks: number
  } | null>(null)
  const [history, setHistory] = useState<
    {
      bucket: string
      blocked: number
      auth_blocks: number
      size_blocks: number
      validation_blocks: number
    }[]
  >([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const { language } = useAppPreferences()
  const $t = useT(language)

  useEffect(() => {
    let cancelled = false
    async function load(silent = false) {
      if (!silent) {
        setLoading(true)
        setError('')
      }
      try {
        const res = await dashboard.blocked({ range_hours: range, bucket })
        if (!cancelled) {
          setSummary(res.blocked_summary)
          setHistory(res.blocked_history)
        }
      } catch (e: any) {
        if (!cancelled) setError(String(e.message || e))
      } finally {
        if (!cancelled && !silent) setLoading(false)
      }
    }
    load()
    const id = setInterval(() => {
      if (document.visibilityState === 'visible') load(true)
    }, 60000)
    return () => { cancelled = true; clearInterval(id) }
  }, [range, bucket])

  const kpis = summary
    ? [
        { label: $t('blocked.title'), value: summary.total_blocked, icon: ShieldAlert, color: '#dc2626' },
        { label: 'Auth', value: summary.auth_blocks, icon: Lock, color: '#7c3aed' },
        { label: 'Size', value: summary.size_blocks, icon: FileWarning, color: '#ea580c' },
        { label: 'Validation', value: summary.validation_blocks, icon: AlertCircle, color: '#059669' },
        { label: 'Other', value: summary.other_blocks, icon: Ban, color: '#6b7280' },
      ]
    : []

  const formatLabel = (v: string) => {
    const d = new Date(v)
    if (bucket === 'hour') return d.toLocaleTimeString(language === 'de' ? 'de-DE' : 'en-GB', { hour: '2-digit', minute: '2-digit' })
    return d.toLocaleDateString(language === 'de' ? 'de-DE' : 'en-GB', { day: '2-digit', month: '2-digit' })
  }

  return (
    <div>
      <div className="page-header">
        <h2>{$t('blocked.title')}</h2>
        <div className="filter-bar">
          <select value={range} onChange={(e) => setRange(Number(e.target.value))}>
            <option value={24}>{$t('overview.last24h')}</option>
            <option value={168}>{$t('overview.last7d')}</option>
            <option value={720}>{$t('overview.last30d')}</option>
            <option value={8760}>{$t('overview.last365d')}</option>
          </select>
          <select value={bucket} onChange={(e) => setBucket(e.target.value as 'hour' | 'day')}>
            <option value="hour">{$t('common.hours')}</option>
            <option value="day">{$t('common.days')}</option>
          </select>
        </div>
      </div>
      {error && <div className="error-box">{error}</div>}
      {loading && <div className="loading">{$t('common.loading')}</div>}
      {!loading && (
        <>
          <div className="kpi-grid">
            {kpis.map((k) => {
              const Icon = k.icon
              return (
                <div className="kpi-card" key={k.label}>
                  <div className="label">{k.label}</div>
                  <div className="value" style={{ color: k.color }}>{k.value}</div>
                  <div className="sub">
                    <Icon size={14} style={{ verticalAlign: 'middle', marginRight: 4, color: k.color }} />
                    {$t('blocked.title')}
                  </div>
                </div>
              )
            })}
          </div>
          <div className="chart-card">
            <h3>{$t('blocked.title')} {$t('blocked.overTime')}</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={history}>
                <CartesianGrid {...chartGridProps} />
                <XAxis dataKey="bucket" tickFormatter={formatLabel} {...chartAxisProps} />
                <YAxis {...chartAxisProps} />
                <Tooltip content={<ChartTooltip />} />
                <Legend {...chartLegendProps} />
                <Bar dataKey="auth_blocks" name="Auth" fill={CHART_COLORS.purple} radius={[4, 4, 0, 0]} />
                <Bar dataKey="size_blocks" name="Size" fill={CHART_COLORS.orange} radius={[4, 4, 0, 0]} />
                <Bar dataKey="validation_blocks" name="Validation" fill={CHART_COLORS.success} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  )
}
