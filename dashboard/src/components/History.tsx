import { useEffect, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart,
} from 'recharts'
import { ChartTooltip, CHART_COLORS, API_KEY_COLORS, chartAxisProps, chartGridProps, chartLegendProps, EmptyChart } from './charts'
import { dashboard } from '../api/client'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

export default function History() {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const [range, setRange] = useState(24)
  const [bucket, setBucket] = useState<'hour' | 'day'>('hour')
  const [assistant, setAssistant] = useState('')
  const [assistants, setAssistants] = useState<string[]>([])
  const [rows, setRows] = useState<
    {
      bucket: string
      requests: number
      avg_response_time_ms: number
      avg_time_to_first_token_ms: number
      avg_time_to_first_reasoning_token_ms: number
      avg_prompt_tokens_per_second: number
      avg_completion_tokens_per_second: number
      total_tokens: number
      total_completion_tokens: number
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

  function zeroFillBuckets<T extends { bucket: string }>(
    data: T[],
    rangeHours: number,
    bucketType: 'hour' | 'day'
  ): T[] {
    if (data.length === 0) return data

    const now = new Date()
    const start = new Date(now.getTime() - rangeHours * 60 * 60 * 1000)

    // Normalisiere einen Bucket-String zu YYYY-MM-DDTHH:00:00Z als UTC-Schlüssel
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

  useEffect(() => {
    let cancelled = false
    async function load(silent = false) {
      if (!silent) {
        setLoading(true)
        setError('')
      }
      try {
        const res = await dashboard.history({ range_hours: range, bucket, model: assistant || undefined })
        if (!cancelled) {
          setRows(zeroFillBuckets(res.metrics_history, range, bucket))
          setVsRows(zeroFillBuckets(res.vectorstore_history, range, bucket))
          const akKeys = res.api_key_history.length > 0 ? Object.keys(res.api_key_history[0]).filter((k) => k !== 'bucket') : []
          setApiKeyNames(akKeys)
          setApiKeyRows(zeroFillBuckets(res.api_key_history, range, bucket))
          setAssistants(res.models || [])
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
    <div>
      <div className="page-header">
        <h2>{$t('history.title')}</h2>
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
          <select value={assistant} onChange={(e) => setAssistant(e.target.value)}>
            <option value="">{$t('history.allAssistants')}</option>
            {assistants.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </div>
      </div>
      {error && <div className="error-box">{error}</div>}
      {loading && <div className="loading">{$t('history.loading')}</div>}
      {!loading && (
        <>
          <div className="chart-card">
            <h3>{$t('history.requestsAndTokens')}</h3>
            {rows.length === 0 ? (
              <EmptyChart message={$t('common.noDataForPeriod')} />
            ) : (
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={rows}>
                  <defs>
                    <linearGradient id="reqGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART_COLORS.primary} stopOpacity={0.15} />
                      <stop offset="95%" stopColor={CHART_COLORS.primary} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="tokGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART_COLORS.success} stopOpacity={0.15} />
                      <stop offset="95%" stopColor={CHART_COLORS.success} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid {...chartGridProps} />
                  <XAxis dataKey="bucket" tickFormatter={formatLabel} {...chartAxisProps} />
                  <YAxis yAxisId="left" {...chartAxisProps} />
                  <YAxis yAxisId="right" orientation="right" {...chartAxisProps} />
                  <Tooltip content={<ChartTooltip />} />
                  <Legend {...chartLegendProps} />
                  <Area yAxisId="left" type="monotone" dataKey="requests" name="Requests" stroke={CHART_COLORS.primary} strokeWidth={2.5} fill="url(#reqGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                  <Area yAxisId="right" type="monotone" dataKey="total_tokens" name={$t('overview.totalTokens')} stroke={CHART_COLORS.success} strokeWidth={2.5} fill="url(#tokGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                  <Line yAxisId="right" type="monotone" dataKey="total_rag_context_tokens" name={$t('overview.ragTokens')} stroke={CHART_COLORS.warning} strokeWidth={2} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="chart-card">
            <h3>{$t('history.tokensPerRequest')}</h3>
            {rows.length === 0 ? (
              <EmptyChart message={$t('common.noDataForPeriod')} />
            ) : (
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={rows.map((r) => ({
                  ...r,
                  avg_tokens_per_request: r.requests > 0 ? Math.round(r.total_tokens / r.requests) : 0,
                  avg_rag_tokens_per_request: r.requests > 0 ? Math.round(r.total_rag_context_tokens / r.requests) : 0,
                  avg_completion_tokens_per_request: r.requests > 0 ? Math.round(r.total_completion_tokens / r.requests) : 0,
                  avg_reasoning_tokens_per_request: r.requests > 0 ? Math.round(r.total_reasoning_tokens / r.requests) : 0,
                }))}>
                  <defs>
                    <linearGradient id="tokReqGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART_COLORS.orange} stopOpacity={0.12} />
                      <stop offset="95%" stopColor={CHART_COLORS.orange} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="ragReqGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART_COLORS.warning} stopOpacity={0.12} />
                      <stop offset="95%" stopColor={CHART_COLORS.warning} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="completionReqGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART_COLORS.success} stopOpacity={0.12} />
                      <stop offset="95%" stopColor={CHART_COLORS.success} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="reasoningReqGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART_COLORS.purple} stopOpacity={0.12} />
                      <stop offset="95%" stopColor={CHART_COLORS.purple} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid {...chartGridProps} />
                  <XAxis dataKey="bucket" tickFormatter={formatLabel} {...chartAxisProps} />
                  <YAxis yAxisId="left" {...chartAxisProps} />
                  <YAxis yAxisId="right" orientation="right" {...chartAxisProps} />
                  <Tooltip content={<ChartTooltip />} />
                  <Legend {...chartLegendProps} />
                  <Area yAxisId="left" type="monotone" dataKey="avg_tokens_per_request" name={$t('history.tokenPerRequestShort')} stroke={CHART_COLORS.orange} strokeWidth={2.5} fill="url(#tokReqGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                  <Area yAxisId="left" type="monotone" dataKey="avg_rag_tokens_per_request" name={$t('history.ragPerRequestShort')} stroke={CHART_COLORS.warning} strokeWidth={2.5} fill="url(#ragReqGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                  <Area yAxisId="right" type="monotone" dataKey="avg_completion_tokens_per_request" name={$t('history.responsePerRequestShort')} stroke={CHART_COLORS.success} strokeWidth={2.5} fill="url(#completionReqGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                  <Area yAxisId="right" type="monotone" dataKey="avg_reasoning_tokens_per_request" name={$t('history.reasoningPerRequestShort')} stroke={CHART_COLORS.purple} strokeWidth={2.5} fill="url(#reasoningReqGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="chart-card">
            <h3>{$t('history.latency')}</h3>
            {rows.length === 0 ? (
              <EmptyChart message={$t('common.noDataForPeriod')} />
            ) : (
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={rows.map((r) => ({
                  ...r,
                  avg_time_to_first_reasoning_token_ms: r.total_reasoning_tokens > 0 ? r.avg_time_to_first_reasoning_token_ms : null,
                }))}>
                  <defs>
                    <linearGradient id="ttftGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART_COLORS.amber} stopOpacity={0.12} />
                      <stop offset="95%" stopColor={CHART_COLORS.amber} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="ttftReasoningGrad" x1="0" y1="0" x2="0" y2="1">
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
                  <Area type="monotone" dataKey="avg_time_to_first_token_ms" name={$t('history.ttftContext')} stroke={CHART_COLORS.amber} strokeWidth={2.5} fill="url(#ttftGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                  <Area type="monotone" dataKey="avg_time_to_first_reasoning_token_ms" name={$t('history.ttftReasoning')} stroke={CHART_COLORS.purple} strokeWidth={2.5} fill="url(#ttftReasoningGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} connectNulls={false} />
                  <Area type="monotone" dataKey="avg_response_time_ms" name={$t('history.responseTime')} stroke={CHART_COLORS.danger} strokeWidth={2.5} fill="url(#respGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="chart-card">
            <h3>{$t('history.promptCompletionTokens')}</h3>
            {rows.length === 0 ? (
              <EmptyChart message={$t('common.noDataForPeriod')} />
            ) : (
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={rows}>
                  <defs>
                    <linearGradient id="promptGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART_COLORS.primary} stopOpacity={0.12} />
                      <stop offset="95%" stopColor={CHART_COLORS.primary} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="completionGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART_COLORS.success} stopOpacity={0.12} />
                      <stop offset="95%" stopColor={CHART_COLORS.success} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid {...chartGridProps} />
                  <XAxis dataKey="bucket" tickFormatter={formatLabel} {...chartAxisProps} />
                  <YAxis yAxisId="left" {...chartAxisProps} />
                  <YAxis yAxisId="right" orientation="right" {...chartAxisProps} />
                  <Tooltip content={<ChartTooltip />} />
                  <Legend {...chartLegendProps} />
                  <Area yAxisId="left" type="monotone" dataKey="avg_prompt_tokens_per_second" name={$t('overview.avgPromptTokens')} stroke={CHART_COLORS.primary} strokeWidth={2.5} fill="url(#promptGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                  <Area yAxisId="right" type="monotone" dataKey="avg_completion_tokens_per_second" name={$t('overview.avgCompletionTokens')} stroke={CHART_COLORS.success} strokeWidth={2.5} fill="url(#completionGrad)" dot={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="chart-card">
            <h3>{$t('history.ragPerformance')}</h3>
            {vsRows.length === 0 ? (
              <EmptyChart message={$t('common.noDataForPeriod')} />
            ) : (
              <ResponsiveContainer width="100%" height={320}>
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
              <ResponsiveContainer width="100%" height={320}>
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
        </>
      )}
    </div>
  )
}
