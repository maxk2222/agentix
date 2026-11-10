import { createContext, memo, useContext, useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Check, ChevronDown, ChevronUp, Copy, LoaderCircle, Maximize, Sparkles, Wrench, X } from 'lucide-react'
import type { SubagentEvent, SubagentMeta } from '../api/client'
import { aggregateToolUsage } from './toolUsage'

/* ------------------------------------------------------------------ */
/*  Shared types                                                        */
/* ------------------------------------------------------------------ */
export interface ToolExecutionEvent {
  tool: string
  stage: 'started' | 'result'
  code?: string
  packages?: string[]
  output?: string
}

export interface ArtifactFile {
  name: string
  conversation_id: string
  kind: string
  mime: string
  size: number
  updated_at?: number | null
}

export function collectArtifactFiles(toolCalls?: ToolCallLive[] | null): ArtifactFile[] {
  // Last occurrence wins per filename: a re-emitted entry carries the new
  // size/updated_at after the artifact was regenerated in place.
  const byKey = new Map<string, ArtifactFile>()
  for (const call of toolCalls || []) {
    for (const file of call.metadata?.files || []) {
      byKey.set(`${file.conversation_id}/${file.name}`, file)
    }
  }
  return Array.from(byKey.values())
}

export interface ToolCallLive {
  id?: string | null
  tool_name: string
  args: Record<string, unknown>
  ok: boolean
  error?: string | null
  duration_ms?: number | null
  output?: string | null
  iteration?: number
  kind?: 'tool' | 'subagent' | 'subagent_control'
  metadata?: { subagent?: SubagentMeta; subagent_results?: SubagentMeta[]; files?: ArtifactFile[] } | null
  /* live-progress placeholders created from tool_progress stream events */
  running?: boolean
  args_preview?: string
  _rawId?: string
}

export interface ToolProgressItem {
  id?: string | null
  tool_name: string
  stage?: string
  iteration?: number
  args_preview?: string
}

/* Internal coordination tools are rendered by dedicated UI (todo panel,
   subagent tags) and never shown as generic tool chips. */
export const INTERNAL_TOOL_NAMES = new Set(['agent_todo', 'spawn_subagent', 'wait_subagent', 'subagent_status'])

/* Stable empty array so memoized children don't re-render on `|| []` churn. */
const EMPTY_TOOL_CALLS: ToolCallLive[] = []

/* ------------------------------------------------------------------ */
/*  Contexts                                                            */
/* ------------------------------------------------------------------ */
export const ChatLanguageContext = createContext<string>('en')

export interface CodeBlockInfo {
  language: string
  content: string
  /* Character offset of the opening fence in the source markdown; lets a
     rendered code component locate its own block deterministically even
     when two blocks share identical content. */
  offset: number
}

export const CodeBlocksContext = createContext<CodeBlockInfo[]>([])

export interface HtmlPreviewContextValue {
  open: (html: string) => void
  close: () => void
}

export const HtmlPreviewContext = createContext<HtmlPreviewContextValue>({
  open: () => {},
  close: () => {},
})

/* ------------------------------------------------------------------ */
/*  Labels                                                              */
/* ------------------------------------------------------------------ */
const CHAT_LABELS: Record<string, Record<string, string>> = {
  en: {
    reasoning: 'Reasoning',
    thinking: 'Thinking',
    iteration: 'Iteration',
    iterations: 'iterations',
    tools: 'tools',
    preamble: 'Prelude',
    codeSandbox: 'Code Sandbox',
    code: 'Code',
    web: 'Web',
    copy: 'Copy',
    fullscreen: 'Fullscreen preview',
  },
  de: {
    reasoning: 'Reasoning',
    thinking: 'Denkt nach',
    iteration: 'Iteration',
    iterations: 'Iterationen',
    tools: 'Tools',
    preamble: 'Einleitung',
    codeSandbox: 'Code-Sandbox',
    code: 'Code',
    web: 'Web',
    copy: 'Kopieren',
    fullscreen: 'Vollbild-Vorschau',
  },
}

export function chatLabels(language: string): Record<string, string> {
  return CHAT_LABELS[language] || CHAT_LABELS.en
}

/* ------------------------------------------------------------------ */
/*  Code-block helpers                                                  */
/* ------------------------------------------------------------------ */
export function extractCodeBlocks(markdown: string): CodeBlockInfo[] {
  const blocks: CodeBlockInfo[] = []
  const regex = /```(\w*)\n([\s\S]*?)```/g
  let match
  while ((match = regex.exec(markdown)) !== null) {
    blocks.push({ language: match[1].trim().toLowerCase(), content: match[2], offset: match.index })
  }
  return blocks
}

/* A css/js block belongs to its nearest html block (ties prefer the
   preceding one); blocks in languages outside the web preview set are
   never assigned. This keeps separately emitted snippets independent
   instead of merging every block of the whole message into one doc. */
const WEB_ONLY_LANGS = new Set(['css', 'js', 'javascript'])

function buildHtmlDoc(blocks: CodeBlockInfo[], anchor: number): string {
  const owned: CodeBlockInfo[] = []
  for (let i = 0; i < blocks.length; i++) {
    const lang = blocks[i].language
    if (lang === 'html') {
      if (i === anchor) owned.push(blocks[i])
      continue
    }
    if (!WEB_ONLY_LANGS.has(lang)) continue
    let prev = -1
    let next = -1
    for (let j = i - 1; j >= 0; j--) {
      if (blocks[j].language === 'html') {
        prev = j
        break
      }
    }
    for (let j = i + 1; j < blocks.length; j++) {
      if (blocks[j].language === 'html') {
        next = j
        break
      }
    }
    const owner =
      prev === -1 ? next : next === -1 ? prev : i - prev <= next - i ? prev : next
    if (owner === anchor) owned.push(blocks[i])
  }
  const css = owned
    .filter((b) => b.language === 'css')
    .map((b) => b.content)
    .join('\n')
  const js = owned
    .filter((b) => b.language === 'javascript' || b.language === 'js')
    .map((b) => b.content)
    .join('\n')
  const html = owned
    .filter((b) => b.language === 'html')
    .map((b) => b.content)
    .join('\n')

  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
${css}
</style>
<script>
${js}
</script>
</head>
<body>
${html}
</body>
</html>`
}

/* ------------------------------------------------------------------ */
/*  Typing indicator                                                    */
/* ------------------------------------------------------------------ */
export function TypingIndicator() {
  return (
    <div className="chat-typing-indicator">
      <span />
      <span />
      <span />
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Code block with copy + html preview                                 */
/* ------------------------------------------------------------------ */
export function CodeBlock({
  children,
  className,
  node,
}: {
  children: React.ReactNode
  className?: string
  node?: { position?: { start?: { offset?: number } } }
}) {
  const language = useContext(ChatLanguageContext)
  const labels = chatLabels(language)
  const [copied, setCopied] = useState(false)
  const [view, setView] = useState<'code' | 'web'>('code')
  const lang = (className ? className.replace('language-', '') : '').toLowerCase()
  const allBlocks = useContext(CodeBlocksContext)
  const htmlPreview = useContext(HtmlPreviewContext)
  const isHtml = lang === 'html'

  if (!className && !String(children).includes('\n')) {
    return <code>{children}</code>
  }

  const handleCopy = async () => {
    const text = String(children).replace(/\n$/, '')
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // ignore
    }
  }

  /* Locate this component's own extracted block: offset match via the
     markdown AST position, content match as fallback. Computed without a
     hook so the early inline-code return above can't break hook order. */
  const blockIndex = (() => {
    const offset = node?.position?.start?.offset
    if (typeof offset === 'number') {
      const i = allBlocks.findIndex((b) => b.offset === offset)
      if (i !== -1) return i
    }
    const text = String(children).replace(/\n$/, '')
    return allBlocks.findIndex(
      (b) => b.language === 'html' && b.content.replace(/\n$/, '') === text,
    )
  })()

  const htmlDoc = isHtml && blockIndex !== -1 ? buildHtmlDoc(allBlocks, blockIndex) : ''

  return (
    <div className="chat-code-block">
      {isHtml && (
        <div className="chat-code-toggle">
          <button className={view === 'code' ? 'active' : ''} onClick={() => setView('code')} style={{ fontFamily: 'inherit' }}>
            {labels.code}
          </button>
          <button className={view === 'web' ? 'active' : ''} onClick={() => setView('web')} style={{ fontFamily: 'inherit' }}>
            {labels.web}
          </button>
          <button className="chat-code-maximize" onClick={() => htmlPreview.open(htmlDoc)} style={{ fontFamily: 'inherit' }}>
            <Maximize size={14} /> {labels.fullscreen}
          </button>
        </div>
      )}
      {isHtml && view === 'web' ? (
        <div style={{ position: 'relative', maxWidth: '100%' }}>
          <iframe
            className="chat-html-iframe"
            sandbox="allow-scripts"
            srcDoc={htmlDoc}
            title="HTML preview"
          />
        </div>
      ) : (
        <>
          {lang && <span className="chat-code-lang">{lang}</span>}
          <button className="chat-code-copy" onClick={handleCopy} title={labels.copy}>
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </button>
          <pre className={className}>
            <code className={className}>{children}</code>
          </pre>
        </>
      )}
    </div>
  )
}

export function PrePassthrough({ children }: { children?: React.ReactNode }) {
  return <>{children}</>
}

/* ------------------------------------------------------------------ */
/*  Tool-call chip                                                      */
/* ------------------------------------------------------------------ */
export function ToolCallBadge({ tool, onClick }: { tool: ToolCallLive; onClick?: () => void }) {
  const state = tool.running ? 'running' : tool.ok ? 'ok' : 'err'
  return (
    <button
      type="button"
      className={`chat-toolcall-badge ${state}`}
      title={tool.error || tool.args_preview || undefined}
      onClick={onClick}
    >
      {tool.running ? (
        <LoaderCircle size={12} className="chat-spin" />
      ) : tool.ok ? (
        <Check size={12} />
      ) : (
        <X size={12} />
      )}
      <span className="chat-toolcall-name">
        {tool.tool_name === 'wait' ? `wait ${tool.args?.seconds ?? '?'}s` : tool.tool_name}
      </span>
      {tool.running && tool.args_preview && (
        <span className="chat-toolcall-preview">{tool.args_preview}</span>
      )}
      {typeof tool.duration_ms === 'number' && (
        <span className="chat-toolcall-dur">{(tool.duration_ms / 1000).toFixed(1)}s</span>
      )}
    </button>
  )
}

/* Aggregated tool-usage badges ("rag_search 5") — live counts over the
   answer stream, covering the root agent and its whole subagent tree.
   Recomputed per render from the streamed accumulators, so resume/replay
   never double-counts. */
export const ToolUsageBadges = memo(function ToolUsageBadges({
  toolCalls,
  subagents,
}: {
  toolCalls?: ToolCallLive[]
  subagents?: SubagentEvent[]
}) {
  const language = useContext(ChatLanguageContext)
  const stats = useMemo(
    () => aggregateToolUsage(toolCalls, subagents),
    [toolCalls, subagents],
  )
  if (!stats.length) return null
  return (
    <div className="chat-tool-usage-row">
      {stats.map((stat) => (
        <span
          key={stat.name}
          className={`chat-tool-usage-badge${stat.running ? ' running' : ''}`}
          title={
            language === 'de'
              ? `${stat.count} Aufrufe${stat.running ? `, ${stat.running} läuft` : ''}`
              : `${stat.count} calls${stat.running ? `, ${stat.running} running` : ''}`
          }
        >
          {stat.running ? (
            <LoaderCircle size={11} className="chat-spin" />
          ) : (
            <Wrench size={11} />
          )}
          <span className="chat-tool-usage-name">{stat.name}</span>
          <span className="chat-tool-usage-count">{stat.count}</span>
        </span>
      ))}
    </div>
  )
})

/* ------------------------------------------------------------------ */
/*  Terminal box (code sandbox events)                                  */
/* ------------------------------------------------------------------ */
export function TerminalBox({ events }: { events: ToolExecutionEvent[] }) {
  const language = useContext(ChatLanguageContext)
  const labels = chatLabels(language)
  const [open, setOpen] = useState(false)

  return (
    <div className="chat-activity-card chat-terminal-box">
      <button className="chat-activity-toggle chat-terminal-toggle" onClick={() => setOpen(!open)}>
        <span className="chat-activity-title">
          <ChevronDown size={12} className={open ? 'rotated' : ''} /> {labels.codeSandbox}
        </span>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>
      {open && (
        <div className="chat-terminal-body">
          {events.map((ev, idx) => (
            <div key={idx} className={`chat-terminal-event ${ev.stage}`}>
              {ev.stage === 'started' && (
                <>
                  <div className="chat-terminal-label">
                    <span className="chat-terminal-prompt">{'>'}</span> python sandbox
                  </div>
                  {ev.packages && ev.packages.length > 0 && (
                    <div className="chat-terminal-label">
                      <span className="chat-terminal-prompt">{'>'}</span> pip install {ev.packages.join(' ')}
                    </div>
                  )}
                  <pre className="chat-terminal-code">{ev.code}</pre>
                </>
              )}
              {ev.stage === 'result' && (
                <pre className="chat-terminal-output">{ev.output}</pre>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* Memoized markdown body — while a long answer streams, only the message
   whose text actually changed re-parses; every other bubble keeps its
   rendered output. Also memoizes the code-block extraction. */
export const MarkdownContent = memo(function MarkdownContent({
  text,
  style,
}: {
  text: string
  style?: CSSProperties
}) {
  const blocks = useMemo(() => extractCodeBlocks(text), [text])
  return (
    <div className="chat-markdown" style={style}>
      <CodeBlocksContext.Provider value={blocks}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{ code: CodeBlock as any, pre: PrePassthrough as any }}
        >
          {text}
        </ReactMarkdown>
      </CodeBlocksContext.Provider>
    </div>
  )
})

/* ------------------------------------------------------------------ */
/*  Reasoning panel: iteration badges + single scrollable body          */
/* ------------------------------------------------------------------ */
export interface ReasoningSection {
  key: string
  iteration: number | null
  text: string
}

export function parseReasoningSections(reasoning: string): ReasoningSection[] {
  // Strip the legacy "Assistant is thinking..." preamble that older messages
  // may still carry in their persisted reasoning content.
  const cleaned = reasoning.replace(/^\s*Assistant is thinking\.\.\.\s*/, '')
  // Cap at 3 digits: model reasoning can emit "Iteration 2025..." (a year) —
  // the backend emits the real counter via structured fields now, this is a
  // display fallback only.
  const re = /\*\*Iteration (\d{1,3})\.\.\.\*\*/g
  const markers: Array<{ n: number; start: number; end: number }> = []
  let match
  while ((match = re.exec(cleaned)) !== null) {
    markers.push({ n: Number(match[1]), start: match.index, end: re.lastIndex })
  }
  if (!markers.length) {
    const text = cleaned.trim()
    return text ? [{ key: 'main', iteration: null, text }] : []
  }
  const preamble = cleaned.slice(0, markers[0].start).trim()
  const sections: ReasoningSection[] = []
  markers.forEach((m, i) => {
    const stop = i + 1 < markers.length ? markers[i + 1].start : cleaned.length
    let text = cleaned.slice(m.end, stop).trim()
    if (i === 0 && preamble) text = `${preamble}\n\n${text}`.trim()
    sections.push({ key: `it-${m.n}`, iteration: m.n, text })
  })
  return sections
}

/* Merge a tool_progress placeholder into the live call list. Returns the
   updated array (or the original if a matching entry already exists). */
export function addToolProgress(calls: ToolCallLive[], progress: ToolProgressItem): ToolCallLive[] {
  const rawId = progress.id || ''
  const exists = calls.some(
    (c) =>
      c.running &&
      ((rawId && c._rawId === rawId) ||
        (!rawId && c.tool_name === progress.tool_name && c.iteration === progress.iteration)),
  )
  if (exists) return calls
  return [
    ...calls,
    {
      id: `prog-${rawId || `${progress.iteration ?? 'x'}-${progress.tool_name}`}`,
      _rawId: rawId,
      tool_name: progress.tool_name,
      iteration: progress.iteration,
      args_preview: progress.args_preview,
      running: true,
      ok: true,
      args: {},
    },
  ]
}

/* Drop running placeholders that match an incoming final tool-call record. */
export function resolveToolProgress(calls: ToolCallLive[], incoming: ToolCallLive): ToolCallLive[] {
  const finalId = incoming.id || ''
  return calls.filter((c) => {
    if (!c.running) return true
    if (finalId && c._rawId && finalId.endsWith(`:${c._rawId}`)) return false
    if (finalId && c._rawId && finalId === c._rawId) return false
    // Fallback: match by tool name + iteration when ids are incomparable.
    return !(c.tool_name === incoming.tool_name && c.iteration === incoming.iteration)
  })
}

/* Clear all running flags (e.g. when the run finishes or errors out). */
export function settleToolProgress(calls: ToolCallLive[]): ToolCallLive[] {
  let changed = false
  const next = calls.map((c) => {
    if (!c.running) return c
    changed = true
    return { ...c, running: false }
  })
  return changed ? next : calls
}

export const ReasoningPanel = memo(function ReasoningPanel({
  reasoning,
  toolCalls,
  isThinking,
  onToolCallClick,
}: {
  reasoning: string
  toolCalls?: ToolCallLive[]
  isThinking: boolean
  onToolCallClick?: (tool: ToolCallLive) => void
}) {
  const calls = toolCalls ?? EMPTY_TOOL_CALLS
  const language = useContext(ChatLanguageContext)
  const labels = chatLabels(language)
  const sections = useMemo(() => parseReasoningSections(reasoning), [reasoning])
  const iterations = sections.filter((s) => s.iteration !== null)
  const [open, setOpen] = useState(isThinking)
  // null = follow the latest iteration; a key = pinned to that iteration.
  const [selected, setSelected] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const userScrolledUpRef = useRef(false)

  useEffect(() => {
    if (!isThinking) setOpen(false)
  }, [isThinking])

  // The pinned iteration may be trimmed/merged away — fall back to follow mode.
  useEffect(() => {
    if (selected && !sections.some((s) => s.key === selected)) setSelected(null)
  }, [selected, sections])

  const lastSection = sections.length ? sections[sections.length - 1] : null
  const displayed =
    (selected ? sections.find((s) => s.key === selected) : undefined) ?? lastSection
  const following = displayed !== null && lastSection !== null && displayed.key === lastSection.key

  // While live (follow mode), pin the scroll body to the bottom so the current
  // reasoning output stays visible; a pinned historical view is left alone.
  useEffect(() => {
    const el = scrollRef.current
    if (open && isThinking && following && el && !userScrolledUpRef.current) {
      el.scrollTop = el.scrollHeight
    }
  }, [reasoning, open, isThinking, following])

  const handlePanelScroll = () => {
    const el = scrollRef.current
    if (!el) return
    userScrolledUpRef.current = el.scrollHeight - el.scrollTop - el.clientHeight > 60
  }

  const selectIteration = (key: string) => {
    setOpen(true)
    if (selected === key) {
      // Re-click the active badge → return to auto-follow on the latest output.
      setSelected(null)
      userScrolledUpRef.current = false
      return
    }
    setSelected(key)
    requestAnimationFrame(() => {
      const el = scrollRef.current
      if (el) el.scrollTop = 0
    })
  }

  const visibleCalls = calls.filter((tc) => !INTERNAL_TOOL_NAMES.has(tc.tool_name))
  const ungroupedCalls = visibleCalls.filter((tc) => tc.iteration === undefined || tc.iteration === null)
  const displayedTools = displayed?.iteration != null
    ? visibleCalls.filter((tc) => tc.iteration === displayed.iteration)
    : []
  const toolCount = visibleCalls.length
  const iterationCount = iterations.length
  const isLive = isThinking && following

  return (
    <div className={`chat-activity-card chat-reasoning-panel ${open ? 'open' : ''}`}>
      <button className="chat-activity-toggle chat-reasoning-toggle" onClick={() => setOpen(!open)}>
        {isThinking ? (
          <span className="chat-activity-title chat-reasoning-thinking">
            <span className="chat-reasoning-dot" />
            {labels.thinking}
          </span>
        ) : (
          <span className="chat-activity-title">
            <Sparkles size={12} /> {labels.reasoning}
            {iterationCount > 0 && (
              <span className="chat-reasoning-summary">
                {iterationCount} {labels.iterations}
                {toolCount > 0 && ` · ${toolCount} ${labels.tools}`}
              </span>
            )}
          </span>
        )}
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {open && (
        <div className="chat-reasoning-panel-body">
          {iterations.length > 1 && (
            <div className="chat-reasoning-steps" role="navigation">
              {iterations.map((s) => (
                <button
                  key={s.key}
                  type="button"
                  className={`chat-reasoning-step ${displayed && s.key === displayed.key ? 'active' : ''}`}
                  onClick={() => selectIteration(s.key)}
                >
                  {s.iteration}
                </button>
              ))}
            </div>
          )}
          <div className="chat-reasoning-scroll" ref={scrollRef} onScroll={handlePanelScroll}>
            {displayed && (
              <section className={`chat-reasoning-iter ${isLive ? 'active' : ''}`}>
                <div className="chat-reasoning-iter-header static">
                  <span className="chat-reasoning-iter-rail" />
                  <span className="chat-reasoning-iter-label">
                    {displayed.iteration !== null
                      ? `${labels.iteration} ${displayed.iteration}`
                      : labels.reasoning}
                  </span>
                  {isLive && <span className="chat-reasoning-iter-pulse" />}
                </div>
                <div className="chat-reasoning-iter-body">
                  {displayed.text && (
                    <div className="chat-reasoning-text">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayed.text}</ReactMarkdown>
                    </div>
                  )}
                  {displayedTools.length > 0 && (
                    <div className="chat-reasoning-tools">
                      {displayedTools.map((tc, i) => (
                        <ToolCallBadge key={tc.id || i} tool={tc} onClick={() => onToolCallClick?.(tc)} />
                      ))}
                    </div>
                  )}
                </div>
              </section>
            )}
            {ungroupedCalls.length > 0 && (
              <section className="chat-reasoning-iter">
                <div className="chat-reasoning-iter-header static">
                  <span className="chat-reasoning-iter-rail" />
                  <span className="chat-reasoning-iter-label">{labels.tools}</span>
                </div>
                <div className="chat-reasoning-iter-body">
                  <div className="chat-reasoning-tools">
                    {ungroupedCalls.map((tc, i) => (
                      <ToolCallBadge key={tc.id || i} tool={tc} onClick={() => onToolCallClick?.(tc)} />
                    ))}
                  </div>
                </div>
              </section>
            )}
          </div>
        </div>
      )}
    </div>
  )
})
