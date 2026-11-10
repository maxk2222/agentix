import { useEffect, useRef, useState } from 'react'
import { Gauge } from 'lucide-react'
import { fmtDuration } from '../utils/format'
import { CHART_COLORS } from './charts'

export interface PerCallUsage {
  p?: number
  c?: number
  r?: number
  ms?: number
}

/* Cumulative snapshot emitted by the agent runner after each completed LLM
   call (`live_usage` stream delta) — drives the live token chip while the
   final `metrics` trailer is not available yet. */
export interface LiveUsage {
  llm_calls?: number
  prompt_tokens?: number
  completion_tokens?: number
  reasoning_tokens?: number
  total_tokens?: number
  per_call_completion?: number[]
  estimated?: boolean
}

export interface ResponseMetrics {
  prompt_tokens?: number | null
  completion_tokens?: number | null
  answer_tokens?: number | null
  total_tokens?: number | null
  reasoning_tokens?: number | null
  cache_tokens?: number | null
  tool_result_tokens?: number | null
  rag_context_tokens?: number | null
  observer_tokens?: number | null
  compression_tokens?: number | null
  subagent_tokens?: number | null
  final_prompt_tokens?: number | null
  final_completion_tokens?: number | null
  final_reasoning_tokens?: number | null
  final_answer_ms?: number | null
  response_time_ms?: number | null
  time_to_first_token_ms?: number | null
  time_to_first_answer_token_ms?: number | null
  time_to_first_reasoning_token_ms?: number | null
  tokens_per_second?: number | null
  agent_iterations?: number | null
  llm_calls?: number | null
  subagent_count?: number | null
  per_call?: PerCallUsage[] | null
  estimated?: boolean | null
}

const LABELS = {
  de: {
    metrics: 'Metriken',
    finalAnswer: 'Finale Antwort',
    total: 'Gesamt',
    llmCalls: 'LLM-Calls',
    timing: 'Timing',
    prompt: 'Prompt-Tokens',
    answer: 'Antwort-Tokens',
    reasoning: 'Reasoning-Tokens',
    cache: 'Cache-Tokens',
    toolResult: 'Tool-Ergebnis-Tokens',
    ragContext: 'RAG-Kontext-Tokens',
    speed: 'Antwort-Speed',
    time: 'Gesamtzeit',
    answerTime: 'Antwortzeit',
    ttft: 'Zeit bis 1. Token',
    answerStart: 'Antwortbeginn',
    ofWhich: 'davon',
    subagents: 'Subagenten',
    observer: 'Observer',
    compression: 'Kompression',
    inPrompt: 'im Prompt enthalten',
    est: 'geschätzt',
    composition: 'Token-Verteilung',
    perCall: 'Tokens pro LLM-Call',
    liveCalls: 'Calls',
    aux: 'Subagent/Observer/Kompression',
  },
  en: {
    metrics: 'Metrics',
    finalAnswer: 'Final answer',
    total: 'Total',
    llmCalls: 'LLM calls',
    timing: 'Timing',
    prompt: 'Prompt tokens',
    answer: 'Answer tokens',
    reasoning: 'Reasoning tokens',
    cache: 'Cache tokens',
    toolResult: 'Tool result tokens',
    ragContext: 'RAG context tokens',
    speed: 'Answer speed',
    time: 'Total time',
    answerTime: 'Answer time',
    ttft: 'Time to first token',
    answerStart: 'Answer start',
    ofWhich: 'of which',
    subagents: 'Subagents',
    observer: 'Observer',
    compression: 'Compression',
    inPrompt: 'incl. in prompt',
    est: 'estimated',
    composition: 'Token composition',
    perCall: 'Tokens per LLM call',
    liveCalls: 'calls',
    aux: 'Subagent/observer/compression',
  },
} as const

function fmtInt(n: number | null | undefined, lang: 'de' | 'en'): string {
  if (n === null || n === undefined) return '—'
  return Math.round(n).toLocaleString(lang === 'de' ? 'de-DE' : 'en-GB')
}

function fmtCompact(n: number | null | undefined, lang?: 'de' | 'en'): string {
  if (n === null || n === undefined) return '—'
  if (n >= 10000) {
    const str = (n / 1000).toFixed(1)
    return `${lang === 'de' ? str.replace('.', ',') : str}k`
  }
  return `${Math.round(n)}`
}

const fmtMs = fmtDuration

function fmtTps(tps: number | null | undefined, lang: 'de' | 'en'): string {
  if (tps === null || tps === undefined) return '—'
  const str = tps >= 100 ? Math.round(tps).toString() : tps.toFixed(1)
  return `${lang === 'de' ? str.replace('.', ',') : str} tok/s`
}

type Row = { label: string; value: string }

type LabelSet = (typeof LABELS)['de'] | (typeof LABELS)['en']

/* Stacked composition bar: share of each token class in the run total. */
function TokenMixBar({ metrics, labels, lang }: { metrics: ResponseMetrics; labels: LabelSet; lang: 'de' | 'en' }) {
  const parts: Array<{ label: string; value: number; color: string }> = []
  const prompt = metrics.prompt_tokens ?? 0
  const answer = metrics.answer_tokens ?? Math.max((metrics.completion_tokens ?? 0) - (metrics.reasoning_tokens ?? 0), 0)
  const reasoning = metrics.reasoning_tokens ?? 0
  const cache = metrics.cache_tokens ?? 0
  const aux = (metrics.subagent_tokens ?? 0) + (metrics.observer_tokens ?? 0) + (metrics.compression_tokens ?? 0)
  if (prompt) parts.push({ label: labels.prompt, value: prompt, color: CHART_COLORS.primary })
  if (answer) parts.push({ label: labels.answer, value: answer, color: CHART_COLORS.success })
  if (reasoning) parts.push({ label: labels.reasoning, value: reasoning, color: CHART_COLORS.purple })
  if (cache) parts.push({ label: labels.cache, value: cache, color: CHART_COLORS.info })
  if (aux) parts.push({ label: labels.aux, value: aux, color: CHART_COLORS.warning })
  const total = parts.reduce((s, p) => s + p.value, 0)
  if (!parts.length || total <= 0) return null
  return (
    <span className="msg-metrics-chart-block">
      <span className="msg-metrics-section-title">{labels.composition}</span>
      <span className="mtr-mix">
        <span className="mtr-mix-bar">
          {parts.map((p) => (
            <span key={p.label} className="mtr-mix-seg" style={{ width: `${(p.value / total) * 100}%`, background: p.color }} />
          ))}
        </span>
        <span className="mtr-mix-legend">
          {parts.map((p) => (
            <span key={p.label} className="mtr-mix-key" title={`${p.label}: ${p.value.toLocaleString()}`}>
              <span className="mtr-mix-dot" style={{ background: p.color }} />
              {fmtCompact(p.value, lang)}
            </span>
          ))}
        </span>
      </span>
    </span>
  )
}

/* Mini column chart — completion tokens per LLM call across the run. */
function CallChart({ perCall, lang }: { perCall: PerCallUsage[]; lang: 'de' | 'en' }) {
  const vals = perCall.map((c) => c.c ?? 0)
  const max = Math.max(...vals, 1)
  const W = 216
  const H = 40
  const n = vals.length
  const gap = 2
  const bw = Math.max((W - gap * (n - 1)) / n, 1.5)
  const callLabel = lang === 'de' ? 'Call' : 'call'
  return (
    <svg className="mtr-callchart" viewBox={`0 0 ${W} ${H}`} role="img" aria-hidden="true">
      {perCall.map((c, i) => {
        const v = c.c ?? 0
        const h = Math.max((v / max) * (H - 4), v > 0 ? 2 : 0.5)
        const x = i * (bw + gap)
        const detail = `${fmtInt(v, lang)} out · ${fmtInt(c.p, lang)} in${c.r ? ` · ${fmtInt(c.r, lang)} reasoning` : ''}${c.ms ? ` · ${fmtMs(c.ms, lang)}` : ''}`
        return (
          <rect key={i} x={x} y={H - h} width={bw} height={h} rx={1.5} fill={CHART_COLORS.primary} opacity={0.85}>
            <title>{`#${i + 1} ${callLabel}: ${detail}`}</title>
          </rect>
        )
      })}
    </svg>
  )
}

/* Timeline: ttft / answer-start markers on the total response time scale. */
function TimingBar({ metrics, lang, labels }: { metrics: ResponseMetrics; lang: 'de' | 'en'; labels: LabelSet }) {
  const total = metrics.response_time_ms
  const ttft = metrics.time_to_first_token_ms
  const answerStart = metrics.time_to_first_answer_token_ms
  if (total === null || total === undefined || total <= 0) return null
  const marks: Array<{ at: number; color: string; label: string }> = []
  if (ttft !== null && ttft !== undefined && ttft <= total) marks.push({ at: ttft, color: CHART_COLORS.info, label: labels.ttft })
  if (answerStart !== null && answerStart !== undefined && answerStart <= total) marks.push({ at: answerStart, color: CHART_COLORS.success, label: labels.answerStart })
  if (!marks.length) return null
  const W = 216
  const trackY = 7
  return (
    <span className="mtr-timeline-block">
      <svg className="mtr-timeline" viewBox={`0 0 ${W} 14`} role="img" aria-hidden="true">
        <rect x={0} y={trackY - 2} width={W} height={4} rx={2} fill={CHART_COLORS.primarySoft} />
        {marks.map((m, i) => (
          <circle key={i} cx={(m.at / total) * W} cy={trackY} r={3} fill={m.color}>
            <title>{`${m.label}: ${fmtMs(m.at, lang)}`}</title>
          </circle>
        ))}
        <circle cx={W} cy={trackY} r={3} fill={CHART_COLORS.primary}>
          <title>{`${labels.time}: ${fmtMs(total, lang)}`}</title>
        </circle>
      </svg>
      <span className="mtr-mix-legend">
        {marks.map((m) => (
          <span key={m.label} className="mtr-mix-key">
            <span className="mtr-mix-dot" style={{ background: m.color }} />
            {m.label}
          </span>
        ))}
        <span className="mtr-mix-key">
          <span className="mtr-mix-dot" style={{ background: CHART_COLORS.primary }} />
          {labels.time}
        </span>
      </span>
    </span>
  )
}

/* Sparkline for the live chip — completion tokens per call so far. */
function LiveSpark({ perCall }: { perCall: number[] }) {
  const vals = perCall.slice(-24)
  if (vals.length < 2) return null
  const max = Math.max(...vals, 1)
  const W = 56
  const H = 14
  const gap = 1
  const bw = Math.max((W - gap * (vals.length - 1)) / vals.length, 1)
  return (
    <svg className="live-token-spark" width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
      {vals.map((v, i) => {
        const h = Math.max((v / max) * (H - 2), v > 0 ? 1.5 : 0.5)
        return <rect key={i} x={i * (bw + gap)} y={H - h} width={bw} height={h} rx={0.8} fill="currentColor" opacity={0.7} />
      })}
    </svg>
  )
}

/* Live token counter shown on the streaming assistant message. Tokens come
   from `live_usage` events (exact per completed call); while a call is in
   flight the streamed text length feeds a `~` estimate and a live tok/s rate. */
export function LiveTokenChip({ live, streamedChars, lang }: { live?: LiveUsage | null; streamedChars: number; lang?: string }) {
  const l10n: 'de' | 'en' = lang === 'en' ? 'en' : 'de'
  const l = lang === 'en' ? LABELS.en : LABELS.de
  const [now, setNow] = useState(() => Date.now())
  const startedRef = useRef(Date.now())
  const firstContentRef = useRef<number | null>(null)
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 500)
    return () => clearInterval(t)
  }, [])
  useEffect(() => {
    if (firstContentRef.current === null && streamedChars > 0) firstContentRef.current = Date.now()
  }, [streamedChars])

  const elapsedMs = now - startedRef.current
  const estOut = Math.round(streamedChars / 4)
  const activeSec = firstContentRef.current !== null ? Math.max((now - firstContentRef.current) / 1000, 0.5) : null
  const estTps = activeSec !== null && estOut > 0 ? estOut / activeSec : null
  const reported = live?.total_tokens ?? 0
  const est = live?.estimated ? '~' : ''
  const calls = live?.llm_calls ?? 0

  const breakdown = reported > 0 && live
    ? `${l.prompt}: ${fmtInt(live.prompt_tokens, l10n)} · ${l.answer}: ${fmtInt(live.completion_tokens, l10n)}${live.reasoning_tokens ? ` · ${l.reasoning}: ${fmtInt(live.reasoning_tokens, l10n)}` : ''}`
    : undefined

  return (
    <span className="live-token-chip" title={breakdown}>
      <span className="live-token-dot" />
      <span className="live-token-item">{fmtMs(elapsedMs, l10n)}</span>
      {reported > 0 && (
        <>
          <span className="live-token-item live-token-strong">{est}{fmtCompact(reported, l10n)} tok</span>
          {calls > 0 && <span className="live-token-item">{calls} {l.liveCalls}</span>}
        </>
      )}
      {reported === 0 && estOut > 0 && (
        <span className="live-token-item">~{fmtCompact(estOut, l10n)} tok</span>
      )}
      {estTps !== null && estTps > 0 && (
        <span className="live-token-item live-token-speed">~{fmtTps(estTps, l10n)}</span>
      )}
      {live?.per_call_completion && live.per_call_completion.length > 1 && (
        <LiveSpark perCall={live.per_call_completion} />
      )}
    </span>
  )
}

export function ResponseMetricsBadge({ metrics, ragContextTokens, lang }: { metrics: ResponseMetrics; ragContextTokens?: number | null; lang?: string }) {
  const l = lang === 'en' ? LABELS.en : LABELS.de
  const l10n: 'de' | 'en' = lang === 'en' ? 'en' : 'de'
  // `estimated` marks runs where some LLM calls reported no provider usage and
  // their token counts were estimated — show '~' on every derived number.
  const est = Boolean(metrics.estimated)
  const fmtT = (n: number | null | undefined) => `${est ? '~' : ''}${fmtInt(n, l10n)}`
  const summary = `${fmtMs(metrics.response_time_ms, l10n)} · ${est ? '~' : ''}${fmtCompact(metrics.total_tokens, l10n)} tok · ${est ? '~' : ''}${fmtTps(metrics.tokens_per_second, l10n)}`

  const sections: Array<{ title: string; rows: Row[] }> = []

  // Final-answer call — what the user actually sees being generated.
  const finalRows: Row[] = []
  if (metrics.final_prompt_tokens !== null && metrics.final_prompt_tokens !== undefined) {
    finalRows.push({ label: l.prompt, value: fmtT(metrics.final_prompt_tokens) })
  }
  const finalAnswerTokens =
    metrics.final_completion_tokens !== null && metrics.final_completion_tokens !== undefined
      ? Math.max(metrics.final_completion_tokens - (metrics.final_reasoning_tokens ?? 0), 0)
      : null
  if (finalAnswerTokens !== null) {
    finalRows.push({ label: l.answer, value: fmtT(finalAnswerTokens) })
  }
  if (metrics.final_reasoning_tokens) {
    finalRows.push({ label: l.reasoning, value: fmtT(metrics.final_reasoning_tokens) })
  }
  if (metrics.final_answer_ms !== null && metrics.final_answer_ms !== undefined) {
    finalRows.push({ label: l.answerTime, value: fmtMs(metrics.final_answer_ms, l10n) })
  }
  if (finalRows.length) sections.push({ title: l.finalAnswer, rows: finalRows })

  // Whole-run total — every LLM call incl. subagents, observer, compression.
  const callCount = metrics.llm_calls ?? metrics.agent_iterations
  const totalTitle =
    callCount !== null && callCount !== undefined && callCount > 1
      ? `${l.total} · ${callCount} ${l.llmCalls}`
      : l.total
  const totalRows: Row[] = [
    { label: l.prompt, value: fmtT(metrics.prompt_tokens) },
    { label: l.answer, value: fmtT(metrics.answer_tokens ?? metrics.completion_tokens) },
  ]
  if (metrics.reasoning_tokens) {
    totalRows.push({ label: l.reasoning, value: fmtT(metrics.reasoning_tokens) })
  }
  if (metrics.cache_tokens) {
    totalRows.push({ label: l.cache, value: fmtT(metrics.cache_tokens) })
  }
  totalRows.push({ label: l.total, value: fmtT(metrics.total_tokens) })
  const parts: Row[] = []
  if (metrics.subagent_tokens) parts.push({ label: l.subagents, value: fmtT(metrics.subagent_tokens) })
  if (metrics.observer_tokens) parts.push({ label: l.observer, value: fmtT(metrics.observer_tokens) })
  if (metrics.compression_tokens) parts.push({ label: l.compression, value: fmtT(metrics.compression_tokens) })
  if (parts.length) {
    totalRows.push({
      label: l.ofWhich,
      value: parts.map((p) => `${p.label} ${p.value}`).join(' · '),
    })
  }
  const ragTokens = ragContextTokens ?? metrics.rag_context_tokens
  if (metrics.tool_result_tokens) {
    totalRows.push({ label: l.toolResult, value: `${fmtT(metrics.tool_result_tokens)} (${l.inPrompt})` })
  }
  if (ragTokens) {
    totalRows.push({ label: l.ragContext, value: `${fmtInt(ragTokens, l10n)} (${l.inPrompt}, ${l.est})` })
  }
  sections.push({ title: totalTitle, rows: totalRows })

  // Timings.
  const timingRows: Row[] = []
  if (metrics.time_to_first_token_ms !== null && metrics.time_to_first_token_ms !== undefined) {
    timingRows.push({ label: l.ttft, value: fmtMs(metrics.time_to_first_token_ms, l10n) })
  }
  if (metrics.time_to_first_answer_token_ms !== null && metrics.time_to_first_answer_token_ms !== undefined) {
    timingRows.push({ label: l.answerStart, value: fmtMs(metrics.time_to_first_answer_token_ms, l10n) })
  }
  timingRows.push(
    { label: l.time, value: fmtMs(metrics.response_time_ms, l10n) },
    { label: l.speed, value: `${est ? '~' : ''}${fmtTps(metrics.tokens_per_second, l10n)}` },
  )
  sections.push({ title: l.timing, rows: timingRows })

  return (
    <span className="msg-metrics">
      <button type="button" className="msg-metrics-trigger" aria-label={l.metrics}>
        <Gauge size={12} />
        <span>{summary}</span>
      </button>
      <span className="msg-metrics-popover" role="tooltip">
        <span className="msg-metrics-title">{l.metrics}</span>
        <TokenMixBar metrics={metrics} labels={l} lang={l10n} />
        {metrics.per_call && metrics.per_call.length > 1 && (
          <span className="msg-metrics-chart-block">
            <span className="msg-metrics-section-title">{l.perCall}</span>
            <CallChart perCall={metrics.per_call} lang={l10n} />
          </span>
        )}
        {sections.map((s) => (
          <span className="msg-metrics-section" key={s.title}>
            <span className="msg-metrics-section-title">{s.title}</span>
            {s.title === l.timing && <TimingBar metrics={metrics} lang={l10n} labels={l} />}
            <span className="msg-metrics-grid">
              {s.rows.map((r) => (
                <span className="msg-metrics-row" key={r.label}>
                  <span className="msg-metrics-label">{r.label}</span>
                  <span className="msg-metrics-value">{r.value}</span>
                </span>
              ))}
            </span>
          </span>
        ))}
      </span>
    </span>
  )
}
