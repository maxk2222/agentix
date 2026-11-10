import { useEffect, useState } from 'react'
import { dashboard } from '../api/client'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

type MetricsRow = {
  model_name: string
  requests: number
  avg_response_time_ms: number
  avg_time_to_first_token_ms: number
  avg_prompt_tokens_per_second: number
  avg_completion_tokens_per_second: number
  total_tokens: number
  error_rate: number
  total_rag_context_tokens: number
}

type VectorstoreRow = {
  model_name: string
  retrievals: number
  avg_retrieval_time_ms: number
  total_chunks: number
  error_rate: number
  total_rag_context_tokens: number
}

export default function Models() {
  const [range, setRange] = useState(24)
  const [rows, setRows] = useState<MetricsRow[]>([])
  const [vsRows, setVsRows] = useState<Record<string, VectorstoreRow>>({})
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
        const res = await dashboard.models(range)
        if (!cancelled) {
          setRows(res.metrics_by_model)
          const vsMap: Record<string, VectorstoreRow> = {}
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
    <div>
      <div className="page-header">
        <h2>{$t('models.title')}</h2>
        <div className="filter-bar">
          <select value={range} onChange={(e) => setRange(Number(e.target.value))}>
            <option value={24}>{$t('overview.last24h')}</option>
            <option value={168}>{$t('overview.last7d')}</option>
            <option value={720}>{$t('overview.last30d')}</option>
            <option value={8760}>{$t('overview.last365d')}</option>
          </select>
        </div>
      </div>
      {error && <div className="error-box">{error}</div>}
      {loading && <div className="loading">{$t('common.loading')}</div>}
      {!loading && (
        <div className="table-card">
          <table>
            <thead>
              <tr>
                <th>{$t('models.model')}</th>
                <th>{$t('models.requests')}</th>
                <th>{$t('models.avgResponseTime')}</th>
                <th>{$t('models.avgTTFT')}</th>
                <th>{$t('models.avgPromptTokens')}</th>
                <th>{$t('models.avgCompletionTokens')}</th>
                <th>{$t('models.totalTokens')}</th>
                <th>{$t('models.ragTokens')}</th>
                <th>{$t('models.errorRate')}</th>
                <th>{$t('models.retrievals')}</th>
                <th>{$t('models.avgRetrievalTime')}</th>
                <th>{$t('models.chunks')}</th>
                <th>{$t('models.ragTokens')} (VS)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const vs = vsRows[r.model_name]
                return (
                  <tr key={r.model_name}>
                    <td style={{ fontWeight: 600 }}>{r.model_name}</td>
                    <td>{r.requests}</td>
                    <td>{r.avg_response_time_ms.toFixed(0)}</td>
                    <td>{r.avg_time_to_first_token_ms.toFixed(0)}</td>
                    <td>{r.avg_prompt_tokens_per_second.toFixed(1)}</td>
                    <td>{r.avg_completion_tokens_per_second.toFixed(1)}</td>
                    <td>{r.total_tokens.toLocaleString()}</td>
                    <td>{r.total_rag_context_tokens.toLocaleString()}</td>
                    <td>{(r.error_rate * 100).toFixed(1)}%</td>
                    <td>{vs ? vs.retrievals : 0}</td>
                    <td>{vs ? vs.avg_retrieval_time_ms.toFixed(0) : '—'}</td>
                    <td>{vs ? vs.total_chunks.toLocaleString() : '—'}</td>
                    <td>{vs ? vs.total_rag_context_tokens.toLocaleString() : '—'}</td>
                  </tr>
                )
              })}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={12} style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                    {$t('models.noData')}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
