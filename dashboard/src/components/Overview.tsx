import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { CHART_COLORS, formatNumber } from './charts'
import RangeChips from './RangeChips'
import { dashboard } from '../api/client'
import {
  Activity, Zap, Timer, Database, FileStack, MessageSquare, Hash,
  Users, Radio, Wrench, Layers, Brain,
} from 'lucide-react'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'
import { fmtDuration } from '../utils/format'
import LiveActivity from './LiveActivity'

type OverviewRes = Awaited<ReturnType<typeof dashboard.overview>>

type IconType = typeof Activity

function Kpi({ label, value, sub, icon: Icon, color }: {
  label: string
  value: string | number
  sub?: string
  icon: IconType
  color?: string
}) {
  return (
    <div className="kpi-card">
      <div className="kpi-head">
        <span className="label">{label}</span>
        <span className="kpi-icon" style={color ? { color } : undefined}><Icon size={16} /></span>
      </div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  )
}

function Section({ icon: Icon, title, children }: {
  icon: IconType
  title: string
  children: React.ReactNode
}) {
  return (
    <div className="overview-section">
      <h3 className="section-header"><Icon size={18} /> {title}</h3>
      {children}
    </div>
  )
}

export default function Overview() {
  const [range, setRange] = useState(24)
  const [res, setRes] = useState<OverviewRes | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [refreshedAt, setRefreshedAt] = useState<Date | null>(null)
  const { language } = useAppPreferences()
  const $t = useT(language)
  const navigate = useNavigate()

  useEffect(() => {
    let cancelled = false
    async function load(silent = false) {
      if (!silent) {
        setLoading(true)
        setError('')
      }
      try {
        const r = await dashboard.overview(range)
        if (!cancelled) {
          setRes(r)
          setRefreshedAt(new Date())
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

  const data = res?.metrics_summary ?? null
  const usage = res?.usage_stats ?? null
  const vsData = res?.vectorstore_summary ?? null
  const vsState = res?.vectorstore_state ?? null
  const convData = res?.conversations ?? null

  const activeAgents = new Set(
    (convData?.items ?? []).map((item) => item.model_name).filter(Boolean),
  ).size

  const weekdayFmt = new Intl.DateTimeFormat(language === 'de' ? 'de-DE' : 'en-GB', { weekday: 'short' })
  // Jan 1 2024 is a Monday — index by position.
  const dayLabels = Array.from({ length: 7 }, (_, i) => weekdayFmt.format(new Date(2024, 0, i + 1)))

  const heatmap = (() => {
    const counts = new Map<string, number>()
    let max = 0
    for (const c of usage?.hourly_activity ?? []) {
      counts.set(`${c.dow}-${c.hour}`, c.count)
      if (c.count > max) max = c.count
    }
    return { counts, max }
  })()

  const tokenParts = data
    ? [
        { key: 'tokenPrompt', value: data.total_prompt_tokens, color: CHART_COLORS.primary },
        { key: 'tokenAnswer', value: data.total_answer_tokens, color: CHART_COLORS.success },
        { key: 'tokenReasoning', value: data.total_reasoning_tokens, color: CHART_COLORS.purple },
        { key: 'tokenCache', value: data.total_cache_tokens, color: CHART_COLORS.info },
        { key: 'tokenToolResults', value: data.total_tool_result_tokens, color: CHART_COLORS.warning },
      ].filter((p) => p.value > 0)
    : []
  const tokenSum = tokenParts.reduce((s, p) => s + p.value, 0)

  const previewRows = convData?.items.slice(0, 5) ?? []

  function formatDateTime(value: string): string {
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) return value
    return date.toLocaleString(language === 'de' ? 'de-DE' : 'en-GB', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return (
    <div>
      <div className="page-header">
        <h2>
          {$t('overview.title')}
          {refreshedAt && !loading && (
            <span
              className="live-pill"
              title={refreshedAt.toLocaleTimeString(language === 'de' ? 'de-DE' : 'en-GB')}
            >
              {$t('overview.live')}
            </span>
          )}
        </h2>
        <RangeChips value={range} onChange={setRange} />
      </div>
      {error && <div className="error-box">{error}</div>}
      <LiveActivity />
      {loading && !res && (
        <div className="kpi-grid">
          {Array.from({ length: 5 }, (_, i) => (
            <div className="kpi-card skeleton-card" key={i}>
              <div className="skeleton skeleton-line" style={{ width: '55%' }} />
              <div className="skeleton skeleton-value" />
              <div className="skeleton skeleton-line" style={{ width: '70%' }} />
            </div>
          ))}
        </div>
      )}
      {res && (
        <>
          <Section icon={Timer} title={$t('overview.performance')}>
            {data && (
              <div className="kpi-grid">
                <Kpi
                  label={$t('overview.medianResponseTime')}
                  value={fmtDuration(data.p50_response_time_ms, language)}
                  sub={$t('overview.medianSub', {
                    avg: fmtDuration(data.avg_response_time_ms, language),
                    p95: fmtDuration(data.p95_response_time_ms, language),
                  })}
                  icon={Timer}
                  color={CHART_COLORS.danger}
                />
                <Kpi
                  label={$t('overview.avgTTFT')}
                  value={fmtDuration(data.avg_time_to_first_generation_ms, language)}
                  sub={$t('overview.ttftReasoning')}
                  icon={Zap}
                  color={CHART_COLORS.amber}
                />
                <Kpi
                  label={$t('metrics.answerStart')}
                  value={fmtDuration(data.p50_time_to_first_answer_token_ms, language)}
                  sub={$t('overview.medianSub', {
                    avg: fmtDuration(data.avg_time_to_first_answer_token_ms, language),
                    p95: fmtDuration(data.p95_time_to_first_answer_token_ms, language),
                  })}
                  icon={Radio}
                  color={CHART_COLORS.purple}
                />
                <Kpi
                  label={$t('overview.avgTps')}
                  value={`${data.avg_tokens_per_second.toFixed(1)} tok/s`}
                  sub={`${formatNumber(data.total_tokens)} ${$t('overview.totalTokens').toLowerCase()}`}
                  icon={Activity}
                  color={CHART_COLORS.primary}
                />
                {vsData && (
                  <Kpi
                    label={$t('overview.avgRetrievalTime')}
                    value={fmtDuration(vsData.avg_retrieval_time_ms, language)}
                    sub={`${formatNumber(vsData.total_retrievals)} ${$t('overview.retrievals')}`}
                    icon={Database}
                    color={CHART_COLORS.info}
                  />
                )}
              </div>
            )}
          </Section>

          <Section icon={Activity} title={$t('overview.usage')}>
            <div className="kpi-grid">
              {data && (
                <>
                  <Kpi
                    label={$t('overview.requests')}
                    value={formatNumber(data.total_requests)}
                    sub={`${$t('metrics.errorRate')} ${(data.error_rate * 100).toFixed(1)}%`}
                    icon={Activity}
                    color={CHART_COLORS.primary}
                  />
                  <Kpi
                    label={$t('overview.totalTokens')}
                    value={formatNumber(data.total_tokens)}
                    sub={$t('overview.tokensBreakdown', {
                      p: formatNumber(data.total_prompt_tokens),
                      a: formatNumber(data.total_answer_tokens),
                      r: formatNumber(data.total_reasoning_tokens),
                    })}
                    icon={Zap}
                    color={CHART_COLORS.warning}
                  />
                  <Kpi
                    label={$t('overview.totalMessages')}
                    value={formatNumber(data.total_messages)}
                    sub={usage ? `Ø ${usage.avg_message_count.toFixed(1)} / Request` : undefined}
                    icon={Hash}
                    color={CHART_COLORS.success}
                  />
                </>
              )}
              {convData && (
                <Kpi
                  label={$t('overview.newConversations')}
                  value={formatNumber(convData.total)}
                  icon={MessageSquare}
                  color={CHART_COLORS.info}
                />
              )}
              {usage && (
                <>
                  <Kpi
                    label={$t('metrics.reasoningShare')}
                    value={`${usage.reasoning_share_pct.toFixed(1)}%`}
                    sub={$t('overview.reasoningShareSub')}
                    icon={Brain}
                    color={CHART_COLORS.purple}
                  />
                  <Kpi
                    label={$t('overview.activeAgents')}
                    value={activeAgents}
                    icon={Users}
                    color={CHART_COLORS.purple}
                  />
                </>
              )}
            </div>

            <div className="usage-grid" style={{ marginTop: 16 }}>
              <div className="chart-card">
                <h3>{$t('metrics.hourlyActivity')}</h3>
                {heatmap.counts.size === 0 ? (
                  <div className="chart-empty">{$t('common.noDataForPeriod')}</div>
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

              <div className="chart-card">
                <h3>{$t('overview.tokenBreakdownTitle')}</h3>
                {tokenSum === 0 ? (
                  <div className="chart-empty">{$t('common.noDataForPeriod')}</div>
                ) : (
                  <>
                    <div className="token-bar">
                      {tokenParts.map((p) => (
                        <div
                          key={p.key}
                          style={{ width: `${(p.value / tokenSum) * 100}%`, background: p.color }}
                          title={`${$t(`overview.${p.key}`)}: ${formatNumber(p.value)}`}
                        />
                      ))}
                    </div>
                    <div className="token-legend">
                      {tokenParts.map((p) => (
                        <div className="token-legend-item" key={p.key}>
                          <span className="token-dot" style={{ background: p.color }} />
                          <span className="token-name">{$t(`overview.${p.key}`)}</span>
                          <span className="token-value">
                            {formatNumber(p.value)}
                            <span className="token-pct">{` · ${((p.value / tokenSum) * 100).toFixed(0)}%`}</span>
                          </span>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
          </Section>

          <Section icon={Wrench} title={$t('overview.agentTools')}>
            <div className="usage-grid">
              <div>
                {usage && (
                  <div className="kpi-grid">
                    <Kpi
                      label={$t('metrics.avgAgentIterations')}
                      value={usage.avg_agent_iterations.toFixed(1)}
                      icon={Layers}
                      color={CHART_COLORS.orange}
                    />
                    <Kpi
                      label={$t('metrics.ragShare')}
                      value={`${usage.rag_share_pct.toFixed(1)}%`}
                      sub={$t('metrics.estimated')}
                      icon={Database}
                      color={CHART_COLORS.warning}
                    />
                  </div>
                )}
              </div>
              <div className="chart-card">
                <h3><Wrench size={15} style={{ marginRight: 6, verticalAlign: -2 }} />{$t('metrics.topTools')}</h3>
                {!usage || usage.top_tools.length === 0 ? (
                  <div className="chart-empty">{$t('common.noDataForPeriod')}</div>
                ) : (
                  <div className="top-list">
                    {usage.top_tools.slice(0, 8).map((t, i) => (
                      <div className="top-list-item" key={`${t.name}-${i}`}>
                        <span className="top-list-rank">{i + 1}</span>
                        <span className="top-list-name" title={t.name}>{t.name}</span>
                        <span className="top-list-bar">
                          <span style={{ width: `${(t.count / Math.max(usage.top_tools[0]?.count ?? 1, 1)) * 100}%` }} />
                        </span>
                        <span className="top-list-count">{formatNumber(t.count)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </Section>

          {(vsData || vsState) && (
            <Section icon={Database} title={$t('overview.knowledge')}>
              <div className="usage-grid">
                <div>
                  <div className="kpi-grid">
                    {vsState && (
                      <>
                        <Kpi
                          label={$t('overview.collections')}
                          value={vsState.total_collections}
                          sub={$t('common.all')}
                          icon={Database}
                          color={CHART_COLORS.info}
                        />
                        <Kpi
                          label={$t('overview.totalDocuments')}
                          value={formatNumber(vsState.total_documents)}
                          sub="Vectorstore"
                          icon={FileStack}
                          color={CHART_COLORS.primary}
                        />
                      </>
                    )}
                    {vsData && (
                      <>
                        <Kpi
                          label={$t('overview.totalChunks')}
                          value={formatNumber(vsData.total_chunks)}
                          sub={$t('overview.retrievals')}
                          icon={Zap}
                          color={CHART_COLORS.success}
                        />
                        <Kpi
                          label={$t('overview.ragTokens')}
                          value={formatNumber(vsData.total_rag_context_tokens)}
                          sub={$t('metrics.estimated')}
                          icon={Zap}
                          color={CHART_COLORS.warning}
                        />
                      </>
                    )}
                  </div>
                </div>
                <div className="chart-card">
                  <h3>{$t('overview.collections')}</h3>
                  {!vsState || vsState.collections.length === 0 ? (
                    <div className="chart-empty">{$t('common.noDataForPeriod')}</div>
                  ) : (
                    <div className="collection-health-list">
                      {vsState.collections.map((c) => (
                        <div className="collection-health-item" key={c.name}>
                          <span
                            className={`health-dot ${c.healthy ? 'ok' : 'err'}`}
                            title={c.healthy ? $t('overview.healthy') : $t('overview.unhealthy')}
                          />
                          <span className="collection-name" title={c.name}>{c.name}</span>
                          <span className="collection-docs">
                            {formatNumber(c.documents)} {$t('overview.documents')}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </Section>
          )}

          <Section icon={MessageSquare} title={$t('overview.activity')}>
            {previewRows.length === 0 ? (
              <div className="chart-empty">{$t('common.noDataForPeriod')}</div>
            ) : (
              <div className="table-card">
                <table className="data-table modern">
                  <thead>
                    <tr>
                      <th>{$t('overview.conversationId')}</th>
                      <th>{$t('overview.model')}</th>
                      <th>{$t('overview.messages')}</th>
                      <th>{$t('overview.lastMessage')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {previewRows.map((row) => (
                      <tr
                        key={row.conversation_id}
                        style={{ cursor: 'pointer' }}
                        onClick={() =>
                          navigate(`/conversations?conversation_id=${encodeURIComponent(row.conversation_id)}`)
                        }
                      >
                        <td className="conversation-id-cell">{row.conversation_id}</td>
                        <td>{row.model_name || '—'}</td>
                        <td>{row.message_count}</td>
                        <td>{formatDateTime(row.last_message_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>
        </>
      )}
    </div>
  )
}
