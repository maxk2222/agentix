import { useEffect, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  Area,
  AreaChart,
  PieChart,
  Pie,
  Cell,
} from 'recharts'
import { ChartTooltip, CHART_COLORS, API_KEY_COLORS, chartAxisProps, chartGridProps, chartLegendProps, EmptyChart, formatNumber } from './charts'
import RangeChips from './RangeChips'
import { dashboard, metrics } from '../api/client'
import { fmtDuration } from '../utils/format'
import {
  Clock, Cpu, ShieldAlert, Lock, FileWarning, AlertCircle, Ban,
  Activity, Zap, Timer, Gauge, MessageSquare, PieChart as PieIcon,
  Database, Wrench, Users, Layers, Radio, Brain,
} from 'lucide-react'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

type TabId = 'history' | 'usage' | 'models' | 'blocked'

type MetricsSummary = Awaited<ReturnType<typeof metrics.summary>>
type MetricsUsage = Awaited<ReturnType<typeof metrics.usage>>
type MetricsByModel = Awaited<ReturnType<typeof metrics.byModel>>

function zeroFillBuckets<T extends { bucket: string }>(
  data: T[],
  rangeHours: number,
  bucketType: 'hour' | 'day'
): T[] {
  if (data.length === 0) return data

  const now = new Date()
  const start = new Date(now.getTime() - rangeHours * 60 * 60 * 1000)

  const normalize = (s: string) => {
    if (!s || s.length < 10) return s
    const parts = s.split('T')
    if (parts.length !== 2) return s
    const datePart = parts[0]
    const timePart = parts[1]
    const hh = timePart.substring(0, 2)
    if (bucketType === 'hour') {
      return `${datePart}T${hh}:00:00Z`
    }
    return `${datePart}T00:00:00Z`
  }

  const buckets: string[] = []
  if (bucketType === 'hour') {
    const current = new Date(start)
    current.setUTCMinutes(0, 0, 0)
    while (current <= now) {
      const yyyy = current.getUTCFullYear()
      const MM = String(current.getUTCMonth() + 1).padStart(2, '0')
      const dd = String(current.getUTCDate()).padStart(2, '0')
      const hh = String(current.getUTCHours()).padStart(2, '0')
      buckets.push(`${yyyy}-${MM}-${dd}T${hh}:00:00Z`)
      current.setUTCHours(current.getUTCHours() + 1)
    }
  } else {
    const current = new Date(start)
    current.setUTCHours(0, 0, 0, 0)
    while (current <= now) {
      const yyyy = current.getUTCFullYear()
      const MM = String(current.getUTCMonth() + 1).padStart(2, '0')
      const dd = String(current.getUTCDate()).padStart(2, '0')
      buckets.push(`${yyyy}-${MM}-${dd}T00:00:00Z`)
      current.setUTCDate(current.getUTCDate() + 1)
    }
  }

  const map = new Map<string, T>()
  for (const item of data) {
    map.set(normalize(item.bucket), item)
  }

  const keys = Object.keys(data[0] || {}).filter((k) => k !== 'bucket') as (keyof T)[]
  const zeroItem = {} as T
  for (const key of keys) {
    (zeroItem as any)[key] = 0
  }

  return buckets.map((b) => {
    const existing = map.get(b)
    if (existing) return existing
    return { ...zeroItem, bucket: b }
  })
}

export default function MetricsPage() {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const [activeTab, setActiveTab] = useState<TabId>('history')
  const [range, setRange] = useState(24)

  const tabs = [
    { id: 'history' as TabId, label: $t('metrics.tabHistory'), icon: Clock },
    { id: 'usage' as TabId, label: $t('metrics.tabUsage'), icon: PieIcon },
    { id: 'models' as TabId, label: $t('metrics.tabModels'), icon: Cpu },
    { id: 'blocked' as TabId, label: $t('metrics.tabBlocked'), icon: ShieldAlert },
  ]

  return (
    <div>
      <div className="page-header">
        <h2>{$t('metrics.title')}</h2>
        <RangeChips value={range} onChange={setRange} />
      </div>

      <div className="rc-tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`rc-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <tab.icon size={16} />
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'history' && <HistoryTab range={range} />}
      {activeTab === 'usage' && <UsageTab range={range} />}
      {activeTab === 'models' && <ModelsTab range={range} />}
      {activeTab === 'blocked' && <BlockedTab range={range} />}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Shared bits                                                       */
/* ------------------------------------------------------------------ */

function KpiCard({ label, value, sub, icon: Icon, color }: {
  label: string
  value: string
  sub?: string
  icon: React.ComponentType<{ size?: number | string }>
  color: string
}) {
  return (
    <div className="kpi-card">
      <div className="kpi-head">
        <span className="label">{label}</span>
        <span className="kpi-icon" style={{ color, background: `${color}1f` }}>
          <Icon size={16} />
        </span>
      </div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  )
}

function Donut({ data, colors }: { data: { name: string; value: number }[]; colors?: string[] }) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  if (data.length === 0) return <EmptyChart message={$t('common.noDataForPeriod')} />
  const palette = colors || API_KEY_COLORS
  return (
    <ResponsiveContainer width="100%" height={230}>
      <PieChart>
        <Pie data={data} dataKey="value" nameKey="name" innerRadius={55} outerRadius={85} paddingAngle={3} strokeWidth={0}>
          {data.map((_, i) => (
            <Cell key={i} fill={palette[i % palette.length]} />
          ))}
        </Pie>
        <Tooltip content={<ChartTooltip />} />
        <Legend {...chartLegendProps} layout="vertical" align="right" verticalAlign="middle" />
      </PieChart>
    </ResponsiveContainer>
  )
}

function TopList({ items }: { items: { name: string; count: number; hint?: string }[] }) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  if (items.length === 0) return <EmptyChart message={$t('common.noDataForPeriod')} />
  const max = Math.max(...items.map((i) => i.count), 1)
  return (
    <div className="top-list">
      {items.map((item, i) => (
        <div className="top-list-item" key={`${item.name}-${i}`}>
          <span className="top-list-rank">{i + 1}</span>
          <span className="top-list-name" title={item.hint || item.name}>{item.name}</span>
          <span className="top-list-bar">
            <span style={{ width: `${(item.count / max) * 100}%` }} />
          </span>
          <span className="top-list-count">{formatNumber(item.count)}</span>
        </div>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  History Tab                                                       */
/* ------------------------------------------------------------------ */

function HistoryTab({ range }: { range: number }) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const [bucket, setBucket] = useState<'hour' | 'day'>('hour')
  const [assistant, setAssistant] = useState('')
  const [assistants, setAssistants] = useState<string[]>([])
  const [summary, setSummary] = useState<MetricsSummary | null>(null)
  const [rows, setRows] = useState<
    {
      bucket: string
      requests: number
      total_messages: number
      avg_response_time_ms: number
      p95_response_time_ms: number
      avg_time_to_first_token_ms: number
      avg_time_to_first_reasoning_token_ms: number
      avg_time_to_first_answer_token_ms: number
      avg_prompt_tokens_per_second: number
      avg_completion_tokens_per_second: number
      avg_tokens_per_second: number
      total_tokens: number
      total_prompt_tokens: number
      total_completion_tokens: number
      total_answer_tokens: number
      total_rag_context_tokens: number
      total_reasoning_tokens: number
    }[]
  >([])
  const [vsRows, setVsRows] = useState<
    {
      bucket: string
      retrievals: number
      avg_retrieval_time_ms: number
      total_chunks: number
      total_rag_context_tokens: number
    }[]
  >([])
  const [apiKeyRows, setApiKeyRows] = useState<(Record<string, number | string> & { bucket: string })[]>([])
  const [apiKeyNames, setApiKeyNames] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    async function load(silent = false) {
      if (!silent) {
        setLoading(true)
        setError('')
      }
      try {
        const [res, sum] = await Promise.all([
          dashboard.history({ range_hours: range, bucket, model: assistant || undefined }),
          metrics.summary(range),
        ])
        if (!cancelled) {
          setRows(zeroFillBuckets(res.metrics_history, range, bucket))
          setVsRows(zeroFillBuckets(res.vectorstore_history, range, bucket))
          const akKeys = res.api_key_history.length > 0 ? Object.keys(res.api_key_history[0]).filter((k) => k !== 'bucket') : []
          setApiKeyNames(akKeys)
          setApiKeyRows(zeroFillBuckets(res.api_key_history, range, bucket))
          setAssistants(res.models || [])
          setSummary(sum)
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
  }, [range, bucket, assistant])

  const formatLabel = (v: string) => {
    const d = new Date(v)
    if (bucket === 'hour') return d.toLocaleTimeString(language === 'de' ? 'de-DE' : 'en-GB', { hour: '2-digit', minute: '2-digit' })
    return d.toLocaleDateString(language === 'de' ? 'de-DE' : 'en-GB', { day: '2-digit', month: '2-digit' })
  }

  return (
    <>
      <div className="filter-bar" style={{ marginBottom: 16 }}>
        <div className="seg-control">
          <button
            className={`seg-btn ${bucket === 'hour' ? 'active' : ''}`}
            onClick={() => setBucket('hour')}
          >
            {$t('common.hours')}
          </button>
          <button
            className={`seg-btn ${bucket === 'day' ? 'active' : ''}`}
            onClick={() => setBucket('day')}
          >
            {$t('common.days')}
          </button>
        </div>
        <select value={assistant} onChange={(e) => setAssistant(e.target.value)}>
          <option value="">{$t('history.allAssistants')}</option>
          {assistants.map((a) => (
            <option key={a} value={a}>{a}</option>
          ))}
        </select>
      </div>

      {error && <div className="error-box">{error}</div>}
      {loading && <div className="loading">{$t('common.loading')}</div>}
      {!loading && (
        <>
          {summary && (
            <div className="kpi-grid">
              <KpiCard label={$t('overview.requests')} value={formatNumber(summary.total_requests)} icon={Activity} color={CHART_COLORS.primary} />
              <KpiCard
                label={$t('overview.totalTokens')}
                value={formatNumber(summary.total_tokens)}
                sub={$t('overview.tokensBreakdown', {
                  p: formatNumber(summary.total_prompt_tokens),
                  a: formatNumber(summary.total_answer_tokens),
                  r: formatNumber(summary.total_reasoning_tokens),
                })}
                icon={Database}
                color={CHART_COLORS.success}
              />
              <KpiCard label={$t('overview.avgTTFT')} value={fmtDuration(summary.avg_time_to_first_token_ms, language)} icon={Zap} color={CHART_COLORS.amber} />
              <KpiCard label={$t('overview.avgTTFA')} value={fmtDuration(summary.avg_time_to_first_answer_token_ms, language)} icon={Timer} color={CHART_COLORS.purple} />
              <KpiCard
                label={$t('overview.avgResponseTime')}
                value={fmtDuration(summary.avg_response_time_ms, language)}
                sub={`p95 ${fmtDuration(summary.p95_response_time_ms, language)}`}
                icon={Clock}
                color={CHART_COLORS.info}
              />
              <KpiCard
                label={$t('metrics.errorRate')}
                value={`${(summary.error_rate * 100).toFixed(1)}%`}
                icon={Gauge}
                color={summary.error_rate > 0.05 ? CHART_COLORS.danger : CHART_COLORS.slate}
              />
            </div>
          )}

          <div className="charts-grid">
            <div className="chart-card">
              <h3>{$t('history.requestsAndMessages')}</h3>
              {rows.length === 0 ? (
                <EmptyChart message={$t('common.noDataForPeriod')} />
              ) : (
                <ResponsiveContainer width="100%" height={340}>
                  <AreaChart data={rows}>
                    <defs>
                      <linearGradient id="reqGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={CHART_COLORS.primary} stopOpacity={0.15} />
                        <stop offset="95%" stopColor={CHART_COLORS.primary} stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="msgGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={CHART_COLORS.info} stopOpacity={0.15} />
                        <stop offset="95%" stopColor={CHART_COLORS.info} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid {...chartGridProps} />
                    <XAxis dataKey="bucket" tickFormatter={formatLabel} {...chartAxisProps} />
                    <YAxis {...chartAxisProps} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend {...chartLegendProps} />
                    <Area type="monotone" dataKey="requests" name={$t('overview.requests')} stroke={CHART_COLORS.primary} strokeWidth={2.5} fill="url(#reqGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                    <Area type="monotone" dataKey="total_messages" name={$t('overview.messages')} stroke={CHART_COLORS.info} strokeWidth={2.5} fill="url(#msgGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>

            <div className="chart-card">
              <h3>{$t('history.tokenBreakdown')}</h3>
              {rows.length === 0 ? (
                <EmptyChart message={$t('common.noDataForPeriod')} />
              ) : (
                <ResponsiveContainer width="100%" height={340}>
                  <AreaChart data={rows}>
                    <defs>
                      <linearGradient id="promptTokGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={CHART_COLORS.primary} stopOpacity={0.14} />
                        <stop offset="95%" stopColor={CHART_COLORS.primary} stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="answerTokGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={CHART_COLORS.success} stopOpacity={0.14} />
                        <stop offset="95%" stopColor={CHART_COLORS.success} stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="reasonTokGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={CHART_COLORS.purple} stopOpacity={0.14} />
                        <stop offset="95%" stopColor={CHART_COLORS.purple} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid {...chartGridProps} />
                    <XAxis dataKey="bucket" tickFormatter={formatLabel} {...chartAxisProps} />
                    <YAxis {...chartAxisProps} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend {...chartLegendProps} />
                    <Area stackId="tok" type="monotone" dataKey="total_prompt_tokens" name={$t('history.promptTokens')} stroke={CHART_COLORS.primary} strokeWidth={2} fill="url(#promptTokGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                    <Area stackId="tok" type="monotone" dataKey="total_answer_tokens" name={$t('history.answerTokens')} stroke={CHART_COLORS.success} strokeWidth={2} fill="url(#answerTokGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                    <Area stackId="tok" type="monotone" dataKey="total_reasoning_tokens" name={$t('history.reasoningTokens')} stroke={CHART_COLORS.purple} strokeWidth={2} fill="url(#reasonTokGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                    <Line type="monotone" dataKey="total_rag_context_tokens" name={$t('history.ragEstimated')} stroke={CHART_COLORS.warning} strokeWidth={2} strokeDasharray="6 3" dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>

            <div className="chart-card">
              <h3>{$t('history.latency')}</h3>
              {rows.length === 0 ? (
                <EmptyChart message={$t('common.noDataForPeriod')} />
              ) : (
                <ResponsiveContainer width="100%" height={340}>
                  <AreaChart data={rows.map((r) => ({
                    ...r,
                    avg_time_to_first_reasoning_token_ms: r.total_reasoning_tokens > 0 ? r.avg_time_to_first_reasoning_token_ms : null,
                  }))}>
                    <defs>
                      <linearGradient id="ttftGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={CHART_COLORS.amber} stopOpacity={0.12} />
                        <stop offset="95%" stopColor={CHART_COLORS.amber} stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="ttfaGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={CHART_COLORS.purple} stopOpacity={0.12} />
                        <stop offset="95%" stopColor={CHART_COLORS.purple} stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="respGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={CHART_COLORS.danger} stopOpacity={0.12} />
                        <stop offset="95%" stopColor={CHART_COLORS.danger} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid {...chartGridProps} />
                    <XAxis dataKey="bucket" tickFormatter={formatLabel} {...chartAxisProps} />
                    <YAxis {...chartAxisProps} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend {...chartLegendProps} />
                    <Area type="monotone" dataKey="avg_time_to_first_token_ms" name={$t('history.ttftFirst')} stroke={CHART_COLORS.amber} strokeWidth={2.5} fill="url(#ttftGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                    <Area type="monotone" dataKey="avg_time_to_first_answer_token_ms" name={$t('history.ttfaAnswer')} stroke={CHART_COLORS.purple} strokeWidth={2.5} fill="url(#ttfaGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                    <Line type="monotone" dataKey="avg_time_to_first_reasoning_token_ms" name={$t('history.ttftReasoning')} stroke={CHART_COLORS.info} strokeWidth={2} strokeDasharray="4 3" dot={false} connectNulls={false} />
                    <Area type="monotone" dataKey="avg_response_time_ms" name={$t('history.responseTime')} stroke={CHART_COLORS.danger} strokeWidth={2.5} fill="url(#respGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                    <Line type="monotone" dataKey="p95_response_time_ms" name={$t('history.p95Response')} stroke={CHART_COLORS.slate} strokeWidth={2} strokeDasharray="8 4" dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>

            <div className="chart-card">
              <h3>{$t('history.speed')}</h3>
              {rows.length === 0 ? (
                <EmptyChart message={$t('common.noDataForPeriod')} />
              ) : (
                <ResponsiveContainer width="100%" height={340}>
                  <AreaChart data={rows}>
                    <defs>
                      <linearGradient id="inputTpsGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={CHART_COLORS.primary} stopOpacity={0.12} />
                        <stop offset="95%" stopColor={CHART_COLORS.primary} stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="answerTpsGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={CHART_COLORS.success} stopOpacity={0.12} />
                        <stop offset="95%" stopColor={CHART_COLORS.success} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid {...chartGridProps} />
                    <XAxis dataKey="bucket" tickFormatter={formatLabel} {...chartAxisProps} />
                    <YAxis {...chartAxisProps} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend {...chartLegendProps} />
                    <Area type="monotone" dataKey="avg_prompt_tokens_per_second" name={$t('history.inputPerSec')} stroke={CHART_COLORS.primary} strokeWidth={2.5} fill="url(#inputTpsGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                    <Area type="monotone" dataKey="avg_completion_tokens_per_second" name={$t('history.answerPerSec')} stroke={CHART_COLORS.success} strokeWidth={2.5} fill="url(#answerTpsGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                    <Line type="monotone" dataKey="avg_tokens_per_second" name={$t('history.finalPerSec')} stroke={CHART_COLORS.purple} strokeWidth={2} strokeDasharray="6 3" dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>

            <div className="chart-card">
              <h3>{$t('history.ragPerformance')}</h3>
              {vsRows.length === 0 ? (
                <EmptyChart message={$t('common.noDataForPeriod')} />
              ) : (
                <ResponsiveContainer width="100%" height={340}>
                  <AreaChart data={vsRows}>
                    <defs>
                      <linearGradient id="chunkGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={CHART_COLORS.orange} stopOpacity={0.12} />
                        <stop offset="95%" stopColor={CHART_COLORS.orange} stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="retrievalGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={CHART_COLORS.info} stopOpacity={0.12} />
                        <stop offset="95%" stopColor={CHART_COLORS.info} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid {...chartGridProps} />
                    <XAxis dataKey="bucket" tickFormatter={formatLabel} {...chartAxisProps} />
                    <YAxis yAxisId="left" {...chartAxisProps} />
                    <YAxis yAxisId="right" orientation="right" {...chartAxisProps} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend {...chartLegendProps} />
                    <Area yAxisId="left" type="monotone" dataKey="total_chunks" name={$t('models.chunks')} stroke={CHART_COLORS.orange} strokeWidth={2.5} fill="url(#chunkGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                    <Area yAxisId="right" type="monotone" dataKey="avg_retrieval_time_ms" name={$t('overview.avgRetrievalTime')} stroke={CHART_COLORS.info} strokeWidth={2.5} fill="url(#retrievalGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>

            <div className="chart-card">
              <h3>{$t('history.tokensPerApiKey')}</h3>
              {apiKeyRows.length === 0 || apiKeyNames.length === 0 ? (
                <EmptyChart message={$t('history.noApiKeyData')} />
              ) : (
                <ResponsiveContainer width="100%" height={340}>
                  <LineChart data={apiKeyRows}>
                    <CartesianGrid {...chartGridProps} />
                    <XAxis dataKey="bucket" tickFormatter={formatLabel} {...chartAxisProps} />
                    <YAxis {...chartAxisProps} />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend {...chartLegendProps} />
                    {apiKeyNames.map((key, i) => (
                      <Line
                        key={key}
                        type="monotone"
                        dataKey={key}
                        name={key}
                        stroke={API_KEY_COLORS[i % API_KEY_COLORS.length]}
                        strokeWidth={2.5}
                        dot={false}
                        activeDot={{ r: 4, strokeWidth: 0 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </>
      )}
    </>
  )
}

/* ------------------------------------------------------------------ */
/*  Usage Tab                                                         */
/* ------------------------------------------------------------------ */

function UsageTab({ range }: { range: number }) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const [usage, setUsage] = useState<MetricsUsage | null>(null)
  const [byModel, setByModel] = useState<MetricsByModel>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    async function load(silent = false) {
      if (!silent) {
        setLoading(true)
        setError('')
      }
      try {
        const [u, m] = await Promise.all([metrics.usage(range), metrics.byModel(range)])
        if (!cancelled) {
          setUsage(u)
          setByModel(m)
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
  }, [range])

  const weekdayFmt = new Intl.DateTimeFormat(language === 'de' ? 'de-DE' : 'en-GB', { weekday: 'short' })
  // Jan 1 2024 is a Monday — index by position.
  const dayLabels = Array.from({ length: 7 }, (_, i) => weekdayFmt.format(new Date(2024, 0, i + 1)))

  const heatmap = (() => {
    const counts = new Map<string, number>()
    let max = 0
    for (const c of usage?.hourly_activity || []) {
      counts.set(`${c.dow}-${c.hour}`, c.count)
      if (c.count > max) max = c.count
    }
    return { counts, max }
  })()

  const assistantDist = (() => {
    const sorted = [...byModel].sort((a, b) => b.requests - a.requests)
    const top = sorted.slice(0, 7)
    const rest = sorted.slice(7)
    const data = top.map((m) => ({ name: m.model_name, value: m.requests }))
    if (rest.length > 0) {
      data.push({ name: $t('metrics.other'), value: rest.reduce((s, m) => s + m.requests, 0) })
    }
    return data
  })()

  const retrievalData = usage
    ? [
        { name: 'ok', value: usage.retrieval_status.ok },
        { name: 'error', value: usage.retrieval_status.error },
        { name: 'skipped', value: usage.retrieval_status.skipped },
      ].filter((d) => d.value > 0)
    : []

  return (
    <>
      {error && <div className="error-box">{error}</div>}
      {loading && <div className="loading">{$t('common.loading')}</div>}
      {!loading && usage && (
        <>
          <div className="kpi-grid">
            <KpiCard label={$t('metrics.streamShare')} value={`${usage.stream_share_pct.toFixed(0)}%`} icon={Radio} color={CHART_COLORS.info} />
            <KpiCard label={$t('metrics.avgMessages')} value={usage.avg_message_count.toFixed(1)} icon={MessageSquare} color={CHART_COLORS.primary} />
            <KpiCard
              label={$t('metrics.ragShare')}
              value={`${usage.rag_share_pct.toFixed(1)}%`}
              sub={$t('metrics.estimated')}
              icon={Database}
              color={CHART_COLORS.warning}
            />
            <KpiCard
              label={$t('metrics.reasoningShare')}
              value={`${usage.reasoning_share_pct.toFixed(1)}%`}
              sub={$t('overview.reasoningShareSub')}
              icon={Brain}
              color={CHART_COLORS.purple}
            />
            <KpiCard label={$t('metrics.avgAgentIterations')} value={usage.avg_agent_iterations.toFixed(1)} icon={Layers} color={CHART_COLORS.orange} />
          </div>

          <div className="chart-card">
            <h3>{$t('metrics.hourlyActivity')}</h3>
            {usage.hourly_activity.length === 0 ? (
              <EmptyChart message={$t('common.noDataForPeriod')} />
            ) : (
              <div className="heatmap">
                <div className="heatmap-row heatmap-header">
                  <span className="heatmap-day" />
                  {Array.from({ length: 24 }, (_, h) => (
                    <span key={h} className="heatmap-hour-label">
                      {h % 4 === 0 ? h : ''}
                    </span>
                  ))}
                </div>
                {[1, 2, 3, 4, 5, 6, 0].map((dow, idx) => (
                  <div className="heatmap-row" key={dow}>
                    <span className="heatmap-day">{dayLabels[idx]}</span>
                    {Array.from({ length: 24 }, (_, h) => {
                      const count = heatmap.counts.get(`${dow}-${h}`) || 0
                      const alpha = count > 0 && heatmap.max > 0
                        ? 0.08 + 0.92 * (count / heatmap.max)
                        : 0
                      return (
                        <span
                          key={h}
                          className="heatmap-cell"
                          title={`${dayLabels[idx]} ${String(h).padStart(2, '0')}:00 — ${count}`}
                          style={alpha > 0 ? { background: `rgba(37, 99, 235, ${alpha.toFixed(2)})` } : undefined}
                        />
                      )
                    })}
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="usage-grid">
            <div className="chart-card">
              <h3>{$t('metrics.assistantDistribution')}</h3>
              <Donut data={assistantDist} />
            </div>
            <div className="chart-card">
              <h3>{$t('metrics.errorBreakdown')}</h3>
              <Donut data={usage.error_breakdown.map((e) => ({ name: e.error_type, value: e.count }))} />
            </div>
            <div className="chart-card">
              <h3>{$t('metrics.retrievalStatus')}</h3>
              <Donut
                data={retrievalData}
                colors={[CHART_COLORS.success, CHART_COLORS.danger, CHART_COLORS.slate]}
              />
            </div>
          </div>

          <div className="usage-grid">
            <div className="chart-card">
              <h3><Wrench size={15} style={{ marginRight: 6, verticalAlign: -2 }} />{$t('metrics.topTools')}</h3>
              <TopList items={usage.top_tools.map((t) => ({ name: t.name, count: t.count }))} />
            </div>
            <div className="chart-card">
              <h3><Users size={15} style={{ marginRight: 6, verticalAlign: -2 }} />{$t('metrics.topUsers')}</h3>
              <TopList
                items={usage.top_users.map((u) => ({
                  name: u.display_name || u.user_id || $t('metrics.unknownUser'),
                  count: u.requests,
                  hint: `${formatNumber(u.total_tokens)} ${$t('overview.totalTokens')}`,
                }))}
              />
            </div>
            <div className="chart-card">
              <h3><Layers size={15} style={{ marginRight: 6, verticalAlign: -2 }} />{$t('metrics.topCollections')}</h3>
              <TopList items={usage.top_collections.map((c) => ({ name: c.name, count: c.count }))} />
            </div>
          </div>
        </>
      )}
    </>
  )
}

/* ------------------------------------------------------------------ */
/*  Models Tab                                                        */
/* ------------------------------------------------------------------ */

function ModelsTab({ range }: { range: number }) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const [rows, setRows] = useState<
    {
      model_name: string
      requests: number
      avg_response_time_ms: number
      p95_response_time_ms: number
      avg_time_to_first_token_ms: number
      avg_time_to_first_answer_token_ms: number
      avg_prompt_tokens_per_second: number
      avg_completion_tokens_per_second: number
      total_tokens: number
      total_answer_tokens: number
      error_rate: number
      total_rag_context_tokens: number
      total_reasoning_tokens: number
    }[]
  >([])
  const [vsRows, setVsRows] = useState<
    Record<
      string,
      {
        model_name: string
        retrievals: number
        avg_retrieval_time_ms: number
        total_chunks: number
        error_rate: number
        total_rag_context_tokens: number
      }
    >
  >({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    async function load(silent = false) {
      if (!silent) {
        setLoading(true)
        setError('')
      }
      try {
        const res = await dashboard.models(range)
        if (!cancelled) {
          setRows(res.metrics_by_model)
          const vsMap: Record<string, any> = {}
          for (const v of res.vectorstore_by_model) {
            vsMap[v.model_name] = v
          }
          setVsRows(vsMap)
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
  }, [range])

  return (
    <>
      {error && <div className="error-box">{error}</div>}
      {loading && <div className="loading">{$t('common.loading')}</div>}
      {!loading && (
        <div className="table-card">
          <table className="data-table modern">
            <thead>
              <tr>
                <th>{$t('models.model')}</th>
                <th>{$t('models.requests')}</th>
                <th>{$t('models.avgResponseTime')}</th>
                <th>{$t('models.p95Response')}</th>
                <th>{$t('models.avgTTFT')}</th>
                <th>{$t('models.avgTTFA')}</th>
                <th>{$t('models.inputPerSec')}</th>
                <th>{$t('models.answerPerSec')}</th>
                <th>{$t('models.totalTokens')}</th>
                <th>{$t('models.answerTokens')}</th>
                <th>{$t('models.reasoningTokens')}</th>
                <th>{$t('models.errorRate')}</th>
                <th>{$t('models.retrievals')}</th>
                <th>{$t('models.avgRetrievalTime')}</th>
                <th>{$t('models.chunks')}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const vs = vsRows[r.model_name]
                return (
                  <tr key={r.model_name}>
                    <td style={{ fontWeight: 600 }}>{r.model_name}</td>
                    <td>{formatNumber(r.requests)}</td>
                    <td>{fmtDuration(r.avg_response_time_ms, language)}</td>
                    <td>{fmtDuration(r.p95_response_time_ms, language)}</td>
                    <td>{fmtDuration(r.avg_time_to_first_token_ms, language)}</td>
                    <td>{fmtDuration(r.avg_time_to_first_answer_token_ms, language)}</td>
                    <td>{r.avg_prompt_tokens_per_second.toFixed(1)}</td>
                    <td>{r.avg_completion_tokens_per_second.toFixed(1)}</td>
                    <td>{formatNumber(r.total_tokens)}</td>
                    <td>{formatNumber(r.total_answer_tokens)}</td>
                    <td>{formatNumber(r.total_reasoning_tokens)}</td>
                    <td>
                      <span className={`badge ${r.error_rate > 0.05 ? 'error' : r.error_rate > 0 ? 'warning' : 'success'}`}>
                        {(r.error_rate * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td>{vs ? formatNumber(vs.retrievals) : 0}</td>
                    <td>{vs ? fmtDuration(vs.avg_retrieval_time_ms, language) : '—'}</td>
                    <td>{vs ? formatNumber(vs.total_chunks) : '—'}</td>
                  </tr>
                )
              })}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={15} style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                    {$t('models.noData')}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

/* ------------------------------------------------------------------ */
/*  Blocked Tab                                                       */
/* ------------------------------------------------------------------ */

function BlockedTab({ range }: { range: number }) {
  const { language } = useAppPreferences()
  const $t = useT(language)
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
    <>
      <div className="filter-bar" style={{ marginBottom: 16 }}>
        <div className="seg-control">
          <button
            className={`seg-btn ${bucket === 'hour' ? 'active' : ''}`}
            onClick={() => setBucket('hour')}
          >
            {$t('common.hours')}
          </button>
          <button
            className={`seg-btn ${bucket === 'day' ? 'active' : ''}`}
            onClick={() => setBucket('day')}
          >
            {$t('common.days')}
          </button>
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
                  <div className="kpi-head">
                    <span className="label">{k.label}</span>
                    <span className="kpi-icon" style={{ color: k.color, background: `${k.color}1f` }}>
                      <Icon size={16} />
                    </span>
                  </div>
                  <div className="value" style={{ color: k.color }}>{formatNumber(k.value)}</div>
                  <div className="sub">{$t('blocked.title')}</div>
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
    </>
  )
}
