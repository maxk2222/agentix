import { useState, useCallback } from 'react'
import { X, RotateCcw, ChevronDown, ChevronRight, Maximize2, Minimize2, FileText } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

interface VersionInfo {
  id: number
  created_at: string
  created_by: string
  is_current: boolean
}

interface FieldChange {
  field: string
  old_value: unknown
  new_value: unknown
  text_diff?: string
}

interface StructuredChange {
  type: 'added' | 'removed' | 'modified'
  assistant: string
  fields: FieldChange[]
}

interface DiffResult {
  diff: string
  structured_changes?: StructuredChange[]
  from: VersionInfo
  to: VersionInfo
}

interface Props {
  open: boolean
  onClose: () => void
  diffResult: DiffResult | null
  onRollback: (versionId: number) => void
}

/* ------------------------------------------------------------------ */
/* YAML Syntax Highlighting                                           */
/* ------------------------------------------------------------------ */
function highlightYaml(text: string): JSX.Element {
  const tokens: Array<{ text: string; className: string | null }> = []
  let remaining = text

  const patterns = [
    { regex: /^#.*$/, className: 'diff-yaml-comment' },
    { regex: /^\s*[\w-]+\s*:(?=\s|$)/, className: 'diff-yaml-key' },
    { regex: /^\s*-\s*/, className: 'diff-yaml-key' },
    { regex: /"[^"]*"|'[^']*'/, className: 'diff-yaml-string' },
    { regex: /\b\d+(\.\d+)?\b/, className: 'diff-yaml-number' },
  ]

  while (remaining.length > 0) {
    let matched = false
    for (const { regex, className } of patterns) {
      const m = remaining.match(regex)
      if (m && m.index === 0) {
        tokens.push({ text: m[0], className })
        remaining = remaining.slice(m[0].length)
        matched = true
        break
      }
    }
    if (!matched) {
      // Take one char as plain text
      tokens.push({ text: remaining[0], className: null })
      remaining = remaining.slice(1)
    }
  }

  return (
    <>
      {tokens.map((t, i) =>
        t.className ? (
          <span key={i} className={t.className}>{t.text}</span>
        ) : (
          <span key={i}>{t.text}</span>
        )
      )}
    </>
  )
}

/* ------------------------------------------------------------------ */
/* Split Diff Parsing                                                 */
/* ------------------------------------------------------------------ */
interface SplitCell {
  text: string
  lineNum: number | null
  type: 'context' | 'remove' | 'add'
}

interface SplitHunk {
  header: string
  oldStart: number
  newStart: number
  rows: Array<{ left: SplitCell | null; right: SplitCell | null }>
}

function parseHeader(header: string): { oldStart: number; oldLen: number; newStart: number; newLen: number } {
  const m = header.match(/@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/)
  if (!m) return { oldStart: 1, oldLen: 0, newStart: 1, newLen: 0 }
  return {
    oldStart: parseInt(m[1], 10),
    oldLen: m[2] ? parseInt(m[2], 10) : 1,
    newStart: parseInt(m[3], 10),
    newLen: m[4] ? parseInt(m[4], 10) : 1,
  }
}

function parseSplitDiff(diffText: string): SplitHunk[] {
  const lines = diffText.split('\n')
  const hunks: SplitHunk[] = []
  let currentHunk: SplitHunk | null = null
  let oldLine = 1
  let newLine = 1

  for (const raw of lines) {
    if (raw.startsWith('@@')) {
      const parsed = parseHeader(raw)
      currentHunk = {
        header: raw,
        oldStart: parsed.oldStart,
        newStart: parsed.newStart,
        rows: [],
      }
      oldLine = parsed.oldStart
      newLine = parsed.newStart
      hunks.push(currentHunk)
      continue
    }
    if (!currentHunk) continue

    if (raw.startsWith('-')) {
      currentHunk.rows.push({
        left: { text: raw.slice(1), lineNum: oldLine, type: 'remove' },
        right: null,
      })
      oldLine++
    } else if (raw.startsWith('+')) {
      currentHunk.rows.push({
        left: null,
        right: { text: raw.slice(1), lineNum: newLine, type: 'add' },
      })
      newLine++
    } else if (raw.startsWith(' ') || raw === '') {
      const text = raw.startsWith(' ') ? raw.slice(1) : raw
      currentHunk.rows.push({
        left: { text, lineNum: oldLine, type: 'context' },
        right: { text, lineNum: newLine, type: 'context' },
      })
      oldLine++
      newLine++
    }
  }

  return hunks
}

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */
function formatValue(val: unknown): string {
  if (val === null || val === undefined) return '—'
  if (typeof val === 'string') return val
  if (Array.isArray(val)) return val.length === 0 ? '[]' : val.join(', ')
  if (typeof val === 'object') {
    return Object.entries(val as Record<string, unknown>)
      .map(([key, value]) => `${key}: ${formatValue(value)}`)
      .join(', ')
  }
  return String(val)
}

function MiniDiff({ diffText }: { diffText: string }) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const hunks = parseSplitDiff(diffText)
  const [collapsed, setCollapsed] = useState<Set<number>>(new Set())

  const toggleHunk = useCallback((idx: number) => {
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }, [])

  if (hunks.length === 0) return <div className="diff-empty-state mini">{$t('versionDiff.noDiff')}</div>

  return (
    <div className="diff-split-view mini">
      {hunks.map((hunk, hi) => {
        const isCollapsed = collapsed.has(hi)
        return (
          <div key={hi} className={`diff-split-hunk ${isCollapsed ? 'collapsed' : ''}`}>
            <div className="diff-split-hunk-header" onClick={() => toggleHunk(hi)}>
              <span>{hunk.header}</span>
              <span className="diff-split-hunk-toggle">
                {isCollapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
              </span>
            </div>
            <div className="diff-split-columns">
              <div className="diff-split-body">
                {hunk.rows.map((row, ri) => (
                  <div key={ri} className="diff-split-row">
                    {row.left && (
                      <div className={`diff-split-cell ${row.left.type}`}>
                        <span className="diff-line-content">{highlightYaml(row.left.text)}</span>
                      </div>
                    )}
                    {row.right && (
                      <div className={`diff-split-cell ${row.right.type}`}>
                        <span className="diff-line-content">{highlightYaml(row.right.text)}</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function StructuredDiffView({ changes }: { changes: StructuredChange[] }) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const [expandedAssistants, setExpandedAssistants] = useState<Set<number>>(
    new Set(changes.map((_, i) => i))
  )

  const toggleAssistant = useCallback((idx: number) => {
    setExpandedAssistants((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }, [])

  return (
    <div className="structured-diff-view">
      {changes.map((change, ci) => {
        const isExpanded = expandedAssistants.has(ci)

        return (
          <div key={ci} className={`structured-change ${change.type}`}>
            <div
              className="structured-change-header"
              onClick={() => toggleAssistant(ci)}
            >
              <span className="structured-change-title">
                {change.type === 'added' && <span className="badge-add">+ {$t('versionDiff.newShort')}</span>}
                {change.type === 'removed' && <span className="badge-remove">− {$t('versionDiff.removedShort')}</span>}
                {change.type === 'modified' && <span className="badge-modify">~ {$t('versionDiff.modifiedShort')}</span>}
                <strong>{change.assistant}</strong>
              </span>
              <span className="diff-split-hunk-toggle">
                {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              </span>
            </div>

            {isExpanded && (
              <div className="structured-change-body">
                {change.type === 'added' && (
                  <div className="field-change-row">
                    <span className="field-change-label">{$t('versionDiff.assistantAdded')}</span>
                  </div>
                )}
                {change.type === 'removed' && (
                  <div className="field-change-row">
                    <span className="field-change-label">{$t('versionDiff.assistantRemoved')}</span>
                  </div>
                )}
                {change.type === 'modified' && (
                  <>
                    {change.fields.map((field, fi) => {
                      const isInstructions = field.field === 'system_instructions'
                      return (
                        <div key={fi} className="field-change">
                          <div className="field-change-header">
                            <span className="field-name">{field.field}</span>
                          </div>
                          {isInstructions ? (
                            <div className="instructions-diff">
                              <div className="instructions-diff-preview">
                                <div className="instructions-diff-old">
                                  <div className="instructions-diff-label">{$t('versionDiff.before')}</div>
                                  <div className="markdown-body">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                      {typeof field.old_value === 'string' ? field.old_value : ''}
                                    </ReactMarkdown>
                                  </div>
                                </div>
                                <div className="instructions-diff-new">
                                  <div className="instructions-diff-label">{$t('versionDiff.after')}</div>
                                  <div className="markdown-body">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                      {typeof field.new_value === 'string' ? field.new_value : ''}
                                    </ReactMarkdown>
                                  </div>
                                </div>
                              </div>
                              {field.text_diff && (
                                <div className="instructions-text-diff">
                                  <div className="instructions-diff-label">{$t('versionDiff.lineDiff')}</div>
                                  <MiniDiff diffText={field.text_diff} />
                                </div>
                              )}
                            </div>
                          ) : (
                            <div className="field-change-values">
                              <div className="field-value-old">
                                <span className="field-value-label">{$t('versionDiff.old')}</span>
                                <span>{formatValue(field.old_value)}</span>
                              </div>
                              <div className="field-value-new">
                                <span className="field-value-label">{$t('versionDiff.new')}</span>
                                <span>{formatValue(field.new_value)}</span>
                              </div>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Component                                                          */
/* ------------------------------------------------------------------ */
export default function VersionDiffModal({ open, onClose, diffResult, onRollback }: Props) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const [rollbackLoading, setRollbackLoading] = useState(false)
  const [collapsed, setCollapsed] = useState<Set<number>>(new Set())
  const [allCollapsed, setAllCollapsed] = useState(false)
  const [rawDiffOpen, setRawDiffOpen] = useState(false)

  const hunks = diffResult ? parseSplitDiff(diffResult.diff) : []
  const canRollback = diffResult ? !diffResult.to.is_current : false
  const hasChanges = hunks.length > 0
  const structuredChanges = diffResult?.structured_changes

  const toggleHunk = useCallback((idx: number) => {
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }, [])

  const expandAll = useCallback(() => {
    setCollapsed(new Set())
    setAllCollapsed(false)
  }, [])

  const collapseAll = useCallback(() => {
    setCollapsed(new Set(hunks.map((_, i) => i)))
    setAllCollapsed(true)
  }, [hunks])

  if (!open || !diffResult) return null

  const handleRollback = async () => {
    setRollbackLoading(true)
    try {
      await onRollback(diffResult.to.id)
    } finally {
      setRollbackLoading(false)
    }
  }

  const renderCell = (cell: SplitCell | null) => {
    if (!cell) {
      return (
        <div className="diff-split-cell empty">
          <span className="diff-line-num" />
          <span className="diff-line-content" />
        </div>
      )
    }
    return (
      <div className={`diff-split-cell ${cell.type}`}>
        <span className="diff-line-num">{cell.lineNum ?? ''}</span>
        <span className="diff-line-content">{highlightYaml(cell.text)}</span>
      </div>
    )
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content wide" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>
            {$t('versionDiff.title')}: #{diffResult.from.id} → #{diffResult.to.id}
          </h3>
          <button className="btn-icon" onClick={onClose} title={$t('common.close')}>
            <X size={18} />
          </button>
        </div>

        <div className="version-meta-bar">
          <div>
            <strong>{$t('versionDiff.from')}</strong> Version #{diffResult.from.id} —{' '}
            {new Date(diffResult.from.created_at).toLocaleString('de-DE')} —{' '}
            {diffResult.from.created_by || '—'}
            {diffResult.from.is_current && <span className="badge-current"> {$t('versionDiff.currentBadge')}</span>}
          </div>
          <div>
            <strong>{$t('versionDiff.to')}</strong> Version #{diffResult.to.id} —{' '}
            {new Date(diffResult.to.created_at).toLocaleString('de-DE')} —{' '}
            {diffResult.to.created_by || '—'}
            {diffResult.to.is_current && <span className="badge-current"> {$t('versionDiff.currentBadge')}</span>}
          </div>
        </div>

        {/* Structured diff (primary view) */}
        {structuredChanges && structuredChanges.length > 0 && (
          <StructuredDiffView changes={structuredChanges} />
        )}

        {/* Raw YAML diff (collapsible fallback) */}
        <div className="raw-diff-section">
          <button
            className="btn-text raw-diff-toggle"
            onClick={() => setRawDiffOpen((prev) => !prev)}
          >
            <FileText size={14} />
            {rawDiffOpen ? $t('versionDiff.yamlHide') : $t('versionDiff.yamlShow')}
            {rawDiffOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>

          {rawDiffOpen && (
            <>
              {hasChanges && (
                <div className="diff-actions-bar">
                  <button onClick={allCollapsed ? expandAll : collapseAll}>
                    {allCollapsed ? (
                      <>
                        <Maximize2 size={12} /> {$t('versionDiff.expandAll')}
                      </>
                    ) : (
                      <>
                        <Minimize2 size={12} /> {$t('versionDiff.collapseAll')}
                      </>
                    )}
                  </button>
                </div>
              )}

              <div className="diff-split-view">
                {!hasChanges ? (
                  <div className="diff-empty-state">{$t('versionDiff.noDiff')}</div>
                ) : (
                  hunks.map((hunk, hi) => {
                    const isCollapsed = collapsed.has(hi)
                    return (
                      <div
                        key={hi}
                        className={`diff-split-hunk ${isCollapsed ? 'collapsed' : ''}`}
                      >
                        <div
                          className="diff-split-hunk-header"
                          onClick={() => toggleHunk(hi)}
                        >
                          <span>{hunk.header}</span>
                          <span className="diff-split-hunk-toggle">
                            {isCollapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
                          </span>
                        </div>
                        <div className="diff-split-columns">
                          <div className="diff-split-header-row">
                            <div className="diff-split-col-header remove">
                              − {$t('versionDiff.older')} (Zeile {hunk.oldStart})
                            </div>
                            <div className="diff-split-col-header add">
                              + {$t('versionDiff.newer')} (Zeile {hunk.newStart})
                            </div>
                          </div>
                          <div className="diff-split-body">
                            {hunk.rows.map((row, ri) => (
                              <div key={ri} className="diff-split-row">
                                {renderCell(row.left)}
                                {renderCell(row.right)}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )
                  })
                )}
              </div>
            </>
          )}
        </div>

        <div className="modal-footer">
          {canRollback && (
            <button
              className="btn-primary"
              onClick={handleRollback}
              disabled={rollbackLoading}
            >
              {rollbackLoading ? (
                <>
                  <RotateCcw size={14} className="spin" /> {$t('versionDiff.rollingBack')}
                </>
              ) : (
                <>
                  <RotateCcw size={14} /> {$t('versionDiff.rollBackTo', { version: diffResult.to.id })}
                </>
              )}
            </button>
          )}
          <button className="btn-secondary" onClick={onClose}>
            {$t('common.close')}
          </button>
        </div>
      </div>
    </div>
  )
}
