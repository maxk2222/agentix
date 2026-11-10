import { useEffect, useState, useRef, useCallback } from 'react'
import {
  config,
  providers,
  tools as toolsApi,
  invalidateCache,
  type AssistantEntry,
  type InstructionsData,
  type ProviderEntry,
  type ReasoningConfig,
  type ReasoningControl,
  type ReasoningSchema,
  type ToolEntry,
} from '../api/client'
import ConfirmModal from './ConfirmModal'
import VersionDiffModal from './VersionDiffModal'
import UnsavedChangesBar, { useUnsavedGuard } from './UnsavedChangesBar'
import {
  Save,
  Trash2,
  Plus,
  Download,
  Upload,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  CheckCircle,
  RotateCcw,
  HelpCircle,
  MoreVertical,
  Search,
  Wrench,
  Copy,
  Eye,
  Code,
  GitCompare,
  History,
} from 'lucide-react'
import {
  MDXEditor,
  headingsPlugin,
  listsPlugin,
  quotePlugin,
  thematicBreakPlugin,
  markdownShortcutPlugin,
  linkPlugin,
  tablePlugin,
  codeBlockPlugin,
  toolbarPlugin,
  UndoRedo,
  BoldItalicUnderlineToggles,
  BlockTypeSelect,
  ListsToggle,
  CreateLink,
  InsertCodeBlock,
  InsertThematicBreak,
} from '@mdxeditor/editor'
import '@mdxeditor/editor/style.css'
import { useAuth } from '../api/AuthContext'
import { useData } from '../api/DataContext'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

const DEFAULT_ASSISTANT: AssistantEntry = {
  name: '',
  model: '',
  temperature: 0.3,
  system_instructions: '',
  collections: [],
  agent_enabled: false,
  tools: {},
  max_tool_calls: 5,
  agent_memory_enabled: false,
  subagents: null,
  observer: null,
  workspace: null,
  api_accessible: true,
  audio_stt_enabled: false,
  audio_tts_enabled: false,
  audio_phone_enabled: false,
  audio_tts_auto_read: false,
  audio_tts_streaming_enabled: true,
  audio_voice_id: null,
  emotion_detection_enabled: false,
  emotion_text_enabled: true,
  emotion_webcam_enabled: false,
  emotion_prompt_steering_enabled: true,
  emotion_analytics_enabled: true,
  extra_body: null,
  enable_thinking: null,
}

function isSensitiveKey(key: string): boolean {
  const lower = key.toLowerCase()
  return ['api_key', 'apikey', 'key', 'secret', 'token'].some((s) => lower.includes(s))
}

function maskSensitiveValue(key: string, value: unknown): string {
  if (isSensitiveKey(key)) return '***'
  return JSON.stringify(value)
}

/* ------------------------------------------------------------------ */
/* Tooltip                                                             */
/* ------------------------------------------------------------------ */
function Tooltip({
  label,
  description,
  defaultValue,
  $t,
}: {
  label: string
  description: string
  defaultValue: string
  $t: (key: string) => string
}) {
  return (
    <span className="tooltip-wrap">
      {label} <small>{$t('common.optional')}</small>
      <span className="tooltip-icon">
        <HelpCircle size={13} />
      </span>
      <span className="tooltip-box">
        <strong>{label}</strong>
        <span className="tooltip-line">
          <em>{$t('common.default')}</em> {defaultValue}
        </span>
        <span className="tooltip-line">{description}</span>
      </span>
    </span>
  )
}

/* ------------------------------------------------------------------ */
/* MultiSelect                                                         */
/* ------------------------------------------------------------------ */
function MultiSelect({
  options,
  deprecatedOptions = [],
  selected,
  onChange,
  emptyText = '—',
  disabled = false,
}: {
  options: string[]
  deprecatedOptions?: string[]
  selected: string[]
  onChange: (values: string[]) => void
  emptyText?: string
  disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])

  const toggle = (val: string) => {
    if (disabled) return
    const next = selected.includes(val)
      ? selected.filter((s) => s !== val)
      : [...selected, val]
    onChange(next)
  }

  return (
    <div className="multi-select" ref={ref}>
      <button type="button" className="multi-select-trigger" onClick={() => !disabled && setOpen((v) => !v)} disabled={disabled}>
        {selected.length === 0
          ? '—'
          : selected.join(', ')}
      </button>
      {open && (
        <div className="multi-select-dropdown">
          {options.length === 0 && deprecatedOptions.length === 0 && (
            <span className="multi-select-empty">{emptyText}</span>
          )}
          {options.map((opt) => (
            <label key={opt} className="multi-select-option">
              <input
                type="checkbox"
                checked={selected.includes(opt)}
                onChange={() => toggle(opt)}
              />
              <span>{opt}</span>
            </label>
          ))}
          {deprecatedOptions.map((opt) => (
            <label key={opt} className="multi-select-option deprecated">
              <input
                type="checkbox"
                checked={selected.includes(opt)}
                onChange={() => toggle(opt)}
              />
              <span>{opt} (deprecated)</span>
            </label>
          ))}
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Field metadata                                                      */
/* ------------------------------------------------------------------ */
function getFieldMeta($t: (key: string) => string): Record<string, { description: string; defaultValue: string }> {
  return {
    description: {
      description: $t('configEditor.fieldDescriptions.description'),
      defaultValue: '""',
    },
    temperature: {
      description: $t('configEditor.fieldDescriptions.temperature'),
      defaultValue: '0.3',
    },
    collections: {
      description: $t('configEditor.fieldDescriptions.collections'),
      defaultValue: '[]',
    },
    reasoning: {
      description: $t('configEditor.fieldDescriptions.reasoning'),
      defaultValue: 'null',
    },
    enable_thinking: {
      description: $t('configEditor.fieldDescriptions.enable_thinking'),
      defaultValue: 'null',
    },
    context_window: {
      description: $t('configEditor.fieldDescriptions.context_window'),
      defaultValue: 'null',
    },
    max_output_tokens: {
      description: $t('configEditor.fieldDescriptions.max_output_tokens'),
      defaultValue: 'null',
    },
    rag_reserved_tokens: {
      description: $t('configEditor.fieldDescriptions.rag_reserved_tokens'),
      defaultValue: 'null',
    },
    min_recent_turns: {
      description: $t('configEditor.fieldDescriptions.min_recent_turns'),
      defaultValue: 'null',
    },
  }
}

/* ------------------------------------------------------------------ */
/* AssistantActionsMenu                                                */
/* ------------------------------------------------------------------ */
function AssistantActionsMenu({
  onDuplicate,
  onDelete,
  $t,
}: {
  onDuplicate: () => void
  onDelete: () => void
  $t: (key: string) => string
}) {
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!menuOpen) return
    const close = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [menuOpen])

  return (
    <div className="dropdown-wrap" ref={menuRef}>
      <button
        className="btn-icon"
        onClick={(e) => {
          e.stopPropagation()
          setMenuOpen((v) => !v)
        }}
        title={$t('common.actions')}
      >
        <MoreVertical size={14} />
      </button>
      {menuOpen && (
        <div className="dropdown-menu">
          <button
            className="dropdown-item"
            onClick={(e) => {
              e.stopPropagation()
              setMenuOpen(false)
              onDuplicate()
            }}
          >
            <Copy size={14} /> {$t('common.duplicate')}
          </button>
          <button
            className="dropdown-item danger"
            onClick={(e) => {
              e.stopPropagation()
              setMenuOpen(false)
              onDelete()
            }}
          >
            <Trash2 size={14} /> {$t('common.delete')}
          </button>
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* KeyValueEditor                                                      */
/* ------------------------------------------------------------------ */
function KeyValueEditor({
  entries,
  onChange,
  disabled = false,
  $t,
}: {
  entries: [string, string][]
  onChange: (entries: [string, string][]) => void
  disabled?: boolean
  $t: (key: string) => string
}) {
  const add = () => {
    if (disabled) return
    const existingKeys = new Set(entries.map(([k]) => k))
    let i = 1
    while (existingKeys.has(`param_${i}`)) i++
    onChange([...entries, [`param_${i}`, '']])
  }
  const remove = (i: number) => {
    if (disabled) return
    onChange(entries.filter((_, idx) => idx !== i))
  }
  const update = (i: number, key: string, value: string) => {
    if (disabled) return
    const next = entries.map((e, idx) => (idx === i ? [key, value] as [string, string] : e))
    onChange(next)
  }

  return (
    <div className="key-value-editor" style={{ opacity: disabled ? 0.6 : 1 }}>
      {entries.map(([k, v], i) => (
        <div key={i} className="key-value-row">
          <input
            type="text"
            placeholder="Key"
            value={k}
            onChange={(e) => update(i, e.target.value, v)}
            disabled={disabled}
          />
          <input
            type="text"
            placeholder="Value"
            value={v}
            onChange={(e) => update(i, k, e.target.value)}
            disabled={disabled}
          />
          <button type="button" className="btn-icon danger" onClick={() => remove(i)} title={$t('configEditor.remove')} disabled={disabled}>
            <Trash2 size={14} />
          </button>
        </div>
      ))}
      <button type="button" className="btn-secondary" onClick={add} disabled={disabled}>
        <Plus size={14} /> {$t('common.add')}
      </button>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* ToolAddDropdown                                                     */
/* ------------------------------------------------------------------ */
const GROUP_ORDER = ['builtin', 'assistant', 'custom']
const GROUP_LABELS: Record<string, string> = {
  builtin: 'Built-in',
  assistant: 'Assistant',
  custom: 'Custom',
}
const GROUP_COLORS: Record<string, { bg: string; text: string }> = {
  builtin: { bg: '#e8f0fe', text: '#2563eb' },
  assistant: { bg: '#f3e8ff', text: '#7c3aed' },
  custom: { bg: '#f3f4f6', text: '#6b7280' },
}
const AGENT_COLORS = { bg: '#ffedd5', text: '#ea580c' }

function ToolAddDropdown({
  availableEntries,
  onSelect,
}: {
  availableEntries: ToolEntry[]
  onSelect: (name: string) => void
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const ref = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
        setQuery('')
      }
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false)
        setQuery('')
      }
      if (e.key === 'Enter' && filtered.length > 0) {
        onSelect(filtered[0].name)
        setOpen(false)
        setQuery('')
      }
    }
    document.addEventListener('mousedown', close)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', close)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  useEffect(() => {
    if (open && inputRef.current) {
      inputRef.current.focus()
    }
  }, [open])

  const filtered = availableEntries.filter((e) => {
    const q = query.toLowerCase()
    return e.name.toLowerCase().includes(q) || e.description.toLowerCase().includes(q)
  })

  const grouped = filtered.reduce<Record<string, ToolEntry[]>>((acc, entry) => {
    const k = entry.kind || 'custom'
    if (!acc[k]) acc[k] = []
    acc[k].push(entry)
    return acc
  }, {})

  // Sort within each group alphabetically by name
  Object.keys(grouped).forEach((k) => {
    grouped[k].sort((a, b) => a.name.localeCompare(b.name))
  })

  const groups = GROUP_ORDER.filter((k) => grouped[k] && grouped[k].length > 0)

  if (availableEntries.length === 0) {
    return (
      <small style={{ color: 'var(--text-secondary)' }}>All available tools added.</small>
    )
  }

  return (
    <div className="multi-select tool-dropdown" ref={ref} style={{ maxWidth: 400 }}>
      <button type="button" className="multi-select-trigger" onClick={() => setOpen((v) => !v)} style={{ fontSize: '13px', padding: '6px 10px' }}>
        <Plus size={14} style={{ verticalAlign: 'text-bottom', marginRight: 4 }} /> Tool hinzufügen
      </button>
      {open && (
        <div className="multi-select-dropdown tool-dropdown-panel">
          <div className="tool-dropdown-search">
            <Search size={14} className="tool-dropdown-search-icon" />
            <input
              ref={inputRef}
              type="text"
              placeholder="Search tools..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onClick={(e) => e.stopPropagation()}
            />
          </div>
          <div className="tool-dropdown-scroll">
            {filtered.length === 0 ? (
              <div className="tool-dropdown-empty">No tools match your search.</div>
            ) : (
              groups.map((kind) => (
                <div key={kind}>
                  <div className="tool-dropdown-group">{GROUP_LABELS[kind] || kind}</div>
                  {grouped[kind].map((entry) => {
                    const isAgent = kind === 'assistant' && Boolean((entry.config as Record<string, unknown>)?.agent_enabled)
                    const label = isAgent ? 'Agent' : (GROUP_LABELS[kind] || kind)
                    const colors = isAgent ? AGENT_COLORS : (GROUP_COLORS[kind] || GROUP_COLORS.custom)
                    return (
                      <button
                        key={entry.name}
                        type="button"
                        className="tool-dropdown-item"
                        onClick={() => {
                          onSelect(entry.name)
                          setOpen(false)
                          setQuery('')
                        }}
                      >
                        <div className="tool-dropdown-item-top">
                          <span className="tool-dropdown-name">{entry.name}</span>
                          <span
                            className="tool-dropdown-badge"
                            style={{ background: colors.bg, color: colors.text }}
                          >
                            {kind === 'builtin' && <Wrench size={10} style={{ marginRight: 3 }} />}
                            {label}
                          </span>
                        </div>
                        <span className="tool-dropdown-desc">{entry.description}</span>
                      </button>
                    )
                  })}
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* InlineMarkdownEditor                                               */
/* ------------------------------------------------------------------ */
function InlineMarkdownEditor({
  value,
  onChange,
  placeholder,
  disabled = false,
}: {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  disabled?: boolean
}) {
  // MDX parses bare "{" as JSX expressions ({{context}} placeholders, unfenced
  // JSON), and the rich-mode round-trip silently drops them — so any content
  // with braces outside fenced/inline code edits must stay in the lossless
  // code view.
  const bracesOutsideCode = (md: string) =>
    /[{]/.test(md.replace(/```[\s\S]*?```/g, '').replace(/`[^`\n]*`/g, ''))
  const [mode, setMode] = useState<'rich' | 'code'>(() => (bracesOutsideCode(value) ? 'code' : 'rich'))
  const [markdown, setMarkdown] = useState(value)
  const editorRef = useRef<any>(null)
  const lastValueRef = useRef(value)

  useEffect(() => {
    if (value !== lastValueRef.current) {
      lastValueRef.current = value
      setMarkdown(value)
      if (bracesOutsideCode(value)) {
        setMode('code')
      } else {
        editorRef.current?.setMarkdown(value)
      }
    }
  }, [value])

  return (
    <div className="inline-markdown-editor" style={{ opacity: disabled ? 0.6 : 1 }}>
      {!disabled && (
        <div className="editor-mode-toggle">
          <button
            className={mode === 'rich' ? 'active' : ''}
            onClick={() => setMode('rich')}
            disabled={bracesOutsideCode(markdown)}
            title={bracesOutsideCode(markdown) ? 'Rich text is unavailable for content with {…} placeholders' : 'Rich Text'}
          >
            <Eye size={14} />
          </button>
          <button
            className={mode === 'code' ? 'active' : ''}
            onClick={() => setMode('code')}
            title="Markdown"
          >
            <Code size={14} />
          </button>
        </div>
      )}
      {mode === 'rich' && !disabled ? (
        <MDXEditor
          ref={editorRef}
          markdown={markdown}
          onChange={(md) => {
            setMarkdown(md)
            onChange(md)
          }}
          plugins={[
            headingsPlugin(),
            listsPlugin(),
            quotePlugin(),
            thematicBreakPlugin(),
            markdownShortcutPlugin(),
            linkPlugin(),
            tablePlugin(),
            codeBlockPlugin(),
            toolbarPlugin({
              toolbarContents: () => (
                <>
                  <UndoRedo />
                  <BlockTypeSelect />
                  <BoldItalicUnderlineToggles />
                  <ListsToggle />
                  <CreateLink />
                  <InsertCodeBlock />
                  <InsertThematicBreak />
                </>
              ),
            }),
          ]}
        />
      ) : (
        <textarea
          className="editor-textarea"
          value={markdown}
          onChange={(e) => {
            if (disabled) return
            setMarkdown(e.target.value)
            onChange(e.target.value)
          }}
          placeholder={placeholder}
          spellCheck={false}
          readOnly={disabled}
        />
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Model-aware reasoning controls                                       */
/* ------------------------------------------------------------------ */

const reasoningSchemaCache = new Map<string, Promise<ReasoningSchema>>()

function fetchReasoningSchema(model: string, providerId?: string): Promise<ReasoningSchema> {
  const key = `${providerId || ''}:${model}`
  let cached = reasoningSchemaCache.get(key)
  if (!cached) {
    cached = providers.reasoningSchema(model, providerId || undefined)
      .then((schema) => {
        // Only persist live-probed descriptors — a rule-based answer may be
        // replaced once the background capability probe lands, so the next
        // fetch must re-ask instead of serving the stale cached schema.
        if (!schema.probed) reasoningSchemaCache.delete(key)
        return schema
      })
      .catch((err) => {
        reasoningSchemaCache.delete(key)
        throw err
      })
    reasoningSchemaCache.set(key, cached)
  }
  return cached
}

const FALLBACK_SCHEMA: ReasoningSchema = {
  family: 'generic',
  wire: 'generic',
  controls: [{ key: 'effort', type: 'select', options: ['low', 'medium', 'high'] }],
  assumed: true,
  canonical_keys: ['effort', 'enabled', 'budget', 'level'],
  model: '',
}

function ReasoningControls({
  model,
  providerId,
  reasoning,
  legacyEnableThinking,
  onChange,
  onClearLegacy,
  disabled,
  $t,
}: {
  model: string
  providerId?: string
  reasoning: ReasoningConfig | null | undefined
  legacyEnableThinking?: boolean | null
  onChange: (reasoning: ReasoningConfig | null) => void
  onClearLegacy: () => void
  disabled?: boolean
  $t: (key: string) => string
}) {
  const [schema, setSchema] = useState<ReasoningSchema | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let alive = true
    let retryTimer: ReturnType<typeof setTimeout> | undefined
    setFailed(false)
    if (!model) {
      setSchema(null)
      return
    }
    const run = (isRetry: boolean) => {
      fetchReasoningSchema(model, providerId)
        .then((s) => {
          if (!alive) return
          setSchema(s)
          // The provider capability probe runs in the background — re-ask
          // once shortly after a non-probed answer so detected controls show
          // up without reopening the editor.
          if (!s.probed && providerId && !isRetry) {
            retryTimer = setTimeout(() => run(true), 3000)
          }
        })
        .catch(() => {
          if (alive && !isRetry) {
            setSchema(null)
            setFailed(true)
          }
        })
    }
    run(false)
    return () => {
      alive = false
      if (retryTimer) clearTimeout(retryTimer)
    }
  }, [model, providerId])

  const effective = schema || FALLBACK_SCHEMA
  const controls: ReasoningControl[] = effective.controls ?? []
  if (!model || controls.length === 0) {
    return (
      <div className="form-group">
        <label>
          <Tooltip {...{ description: $t('configEditor.fieldDescriptions.reasoning'), defaultValue: 'null' }} label="Reasoning" $t={$t} />
        </label>
        <small style={{ color: 'var(--text-secondary)' }}>
          {$t(model ? 'configEditor.reasoningNoControls' : 'configEditor.reasoningNoModel')}
        </small>
      </div>
    )
  }

  const r: ReasoningConfig = reasoning || {}
  const setKey = (key: string, value: unknown) => {
    const next: ReasoningConfig = { ...r }
    if (value === undefined || value === '' || value === null) {
      delete (next as Record<string, unknown>)[key]
    } else {
      (next as Record<string, unknown>)[key] = value
    }
    onChange(Object.keys(next).length ? next : null)
    // Writing reasoning.enabled supersedes the legacy enable_thinking field
    if (key === 'enabled') onClearLegacy()
  }

  const enabledEffective =
    r.enabled !== undefined ? r.enabled : legacyEnableThinking ?? undefined

  return (
    <div className="form-group wide reasoning-controls">
      <label>
        <Tooltip {...{ description: $t('configEditor.fieldDescriptions.reasoning'), defaultValue: 'null' }} label="Reasoning" $t={$t} />
        {effective.assumed && (
          <span className="assistant-badge badge-unknown" style={{ marginLeft: 8 }}>
            {$t('configEditor.reasoningAssumed')}
          </span>
        )}
        {failed && (
          <span className="assistant-badge badge-unknown" style={{ marginLeft: 8 }}>
            {$t('configEditor.reasoningSchemaFailed')}
          </span>
        )}
        {schema && (
          <span
            className={`assistant-badge reasoning-family-badge${effective.probed ? ' badge-probed' : ''}`}
            style={{ marginLeft: 8 }}
            title={$t(effective.probed ? 'configEditor.reasoningProbed' : 'configEditor.reasoningFamily')}
          >
            {effective.family}
          </span>
        )}
      </label>
      <div className="reasoning-controls-grid">
        {controls.map((ctl) => {
          if (ctl.type === 'toggle') {
            return (
              <div key={ctl.key} className="form-group">
                <label>{$t(`configEditor.reasoning${ctl.key[0].toUpperCase()}${ctl.key.slice(1)}`)}</label>
                <select
                  value={enabledEffective === undefined ? '' : enabledEffective ? 'true' : 'false'}
                  onChange={(e) => setKey('enabled', e.target.value === '' ? undefined : e.target.value === 'true')}
                  disabled={disabled}
                >
                  <option value="">
                    {ctl.default === true
                      ? $t('configEditor.reasoningDefaultOn')
                      : ctl.default === false
                        ? $t('configEditor.reasoningDefaultOff')
                        : $t('configEditor.reasoningDefault')}
                  </option>
                  <option value="true">{$t('configEditor.thinkingOn')}</option>
                  <option value="false">{$t('configEditor.thinkingOff')}</option>
                </select>
              </div>
            )
          }
          if (ctl.type === 'select') {
            return (
              <div key={ctl.key} className="form-group">
                <label>{$t(`configEditor.reasoning${ctl.key[0].toUpperCase()}${ctl.key.slice(1)}`)}</label>
                <select
                  value={typeof (r as Record<string, unknown>)[ctl.key] === 'string' ? ((r as Record<string, unknown>)[ctl.key] as string) : ''}
                  onChange={(e) => setKey(ctl.key, e.target.value || undefined)}
                  disabled={disabled}
                >
                  <option value="">{$t('configEditor.reasoningDefault')}</option>
                  {(ctl.options || []).map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </div>
            )
          }
          // number
          return (
            <div key={ctl.key} className="form-group">
              <label>{$t(`configEditor.reasoning${ctl.key[0].toUpperCase()}${ctl.key.slice(1)}`)}</label>
              <input
                type="number"
                min={ctl.min}
                max={ctl.max}
                placeholder={ctl.min !== undefined && ctl.max !== undefined ? `${ctl.min}–${ctl.max}` : ''}
                value={typeof (r as Record<string, unknown>)[ctl.key] === 'number' ? ((r as Record<string, unknown>)[ctl.key] as number) : ''}
                onChange={(e) => {
                  const v = e.target.value
                  setKey(ctl.key, v === '' ? undefined : Number(v))
                }}
                disabled={disabled}
              />
            </div>
          )
        })}
      </div>
    </div>
  )
}

function resolveProviderForAssistant(
  assistant: AssistantEntry,
  providersList: ProviderEntry[],
  modelsByProvider: Record<string, { name: string; models: string[] }>
): ProviderEntry | undefined {
  if (assistant.provider_id) {
    return providersList.find((p) => p.id === assistant.provider_id)
  }
  if (assistant.model) {
    for (const [pid, pm] of Object.entries(modelsByProvider)) {
      if (pm.models.includes(assistant.model)) {
        return providersList.find((p) => p.id === pid)
      }
    }
  }
  return undefined
}

function AssistantBadges({
  assistant,
  provider,
}: {
  assistant: AssistantEntry
  provider?: ProviderEntry
}) {
  const badges: { label: string; className: string }[] = []
  if (assistant.agent_enabled) {
    badges.push({ label: 'Agent', className: 'badge-agent' })
    if (assistant.subagents?.enabled) {
      badges.push({ label: 'Subagents', className: 'badge-subagents' })
    }
    if (provider) {
      if (provider.supports_native_tool_calls === true) {
        badges.push({ label: 'Native', className: 'badge-native' })
      } else if (provider.supports_native_tool_calls === false) {
        badges.push({ label: 'Parsed', className: 'badge-parsed' })
      } else {
        badges.push({ label: '?', className: 'badge-unknown' })
      }
    } else {
      badges.push({ label: '?', className: 'badge-unknown' })
    }
  } else {
    badges.push({ label: 'Assistant', className: 'badge-assistant' })
  }
  if (assistant.api_accessible === false) {
    badges.push({ label: 'Internal', className: 'badge-internal' })
  }
  if (assistant.collections && assistant.collections.length > 0) {
    badges.push({ label: 'RAG', className: 'badge-rag' })
  }
  return (
    <span className="assistant-badges">
      {badges.map((b) => (
        <span key={b.label} className={`assistant-badge ${b.className}`}>
          {b.label}
        </span>
      ))}
    </span>
  )
}

export default function ConfigEditor() {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const { hasPermission } = useAuth()
  const { getInstructions, getCollections, getProviders, refreshInstructions, refreshProviders, invalidateInstructions } = useData()
  const canWrite = hasPermission('instructions:write')
  const fieldMeta = getFieldMeta($t)

  const [data, setData] = useState<InstructionsData>({ assistants: [] })
  const [extraBodyInputs, setExtraBodyInputs] = useState<Record<string, string>>({})
  const [extraBodyErrors, setExtraBodyErrors] = useState<Record<string, string>>({})
  const [availableCollections, setAvailableCollections] = useState<string[]>([])
  const [availableModelsByProvider, setAvailableModelsByProvider] = useState<Record<string, { name: string; models: string[] }>>({})
  const [availableToolEntries, setAvailableToolEntries] = useState<ToolEntry[]>([])
  const [providerEntries, setProviderEntries] = useState<ProviderEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [applyStatus, setApplyStatus] = useState<'idle' | 'loading' | 'done' | 'error'>('idle')
  const [transferStatus, setTransferStatus] = useState<'idle' | 'downloading' | 'uploading' | 'uploaded' | 'error'>('idle')
  const [expanded, setExpanded] = useState<Set<number>>(new Set())
  const [assistantTab, setAssistantTab] = useState<Record<number, string>>({})
  const [audioExpanded, setAudioExpanded] = useState<Set<number>>(new Set())
  const [emotionExpanded, setEmotionExpanded] = useState<Set<number>>(new Set())
  const [toolRowsExpanded, setToolRowsExpanded] = useState<Set<string>>(new Set())
  const [deleteModal, setDeleteModal] = useState<{ idx: number; name: string } | null>(null)
  const [rollbackTarget, setRollbackTarget] = useState<number | null>(null)
  const [history, setHistory] = useState<Array<{ id: number; created_at: string; created_by: string; is_current: boolean }>>([])
  const [historyOpen, setHistoryOpen] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [selectedVersions, setSelectedVersions] = useState<[number | null, number | null]>([null, null])
  const [diffModalOpen, setDiffModalOpen] = useState(false)
  const [diffResult, setDiffResult] = useState<{
    diff: string
    from: { id: number; created_at: string; created_by: string; is_current: boolean }
    to: { id: number; created_at: string; created_by: string; is_current: boolean }
  } | null>(null)
  const [diffLoading, setDiffLoading] = useState(false)
  const loadedRef = useRef(false)
  const initialDataRef = useRef<string>(JSON.stringify({ assistants: [] }))
  const uploadInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    Promise.all([
      getInstructions(),
      getCollections(),
      getProviders().catch(() => []),
      config.listModelsByProvider().catch(() => ({})),
      toolsApi.list(true).catch(() => []),
    ])
      .then(([res, cols, providersList, modelsByProvider, toolsList]) => {
        const assistants = (Array.isArray(res.assistants) ? res.assistants : []).map(
          (a: Partial<AssistantEntry>) => ({ ...DEFAULT_ASSISTANT, ...a })
        )
        const payload: InstructionsData = { assistants }
        setData(payload)
        setAvailableCollections(cols.map((c) => c.name))
        setProviderEntries(providersList as ProviderEntry[])
        setAvailableModelsByProvider(modelsByProvider)
        setAvailableToolEntries(toolsList as ToolEntry[])
        initialDataRef.current = JSON.stringify(payload)
        loadedRef.current = true
        setLoading(false)
      })
      .catch((err) => {
        setError(String(err))
        setLoading(false)
      })
  }, [])

  const isDirty = loadedRef.current && JSON.stringify(data) !== initialDataRef.current

  useUnsavedGuard(isDirty, $t('common.leaveWarning'))

  const handleSave = useCallback(() => {
    setSaveStatus('saving')
    config
      .saveInstructions(data)
      .then(() => {
        setSaveStatus('saved')
        initialDataRef.current = JSON.stringify(data)
        invalidateInstructions()
        invalidateCache('/admin/config/instructions')
        setTimeout(() => setSaveStatus('idle'), 2000)
      })
      .catch((err) => {
        setSaveStatus('error')
        setError(String(err))
      })
  }, [data, invalidateInstructions])

  const handleDiscard = useCallback(() => {
    setData(JSON.parse(initialDataRef.current))
  }, [])

  const updateAssistant = useCallback(
    (idx: number, patch: Partial<AssistantEntry>) => {
      setData((prev) => ({
        ...prev,
        assistants: prev.assistants.map((a, i) =>
          i === idx ? { ...a, ...patch } : a
        ),
      }))
    },
    []
  )

  const addAssistant = useCallback(() => {
    setData((prev) => ({
      ...prev,
      assistants: [...prev.assistants, { ...DEFAULT_ASSISTANT }],
    }))
    setExpanded((prev) => {
      const s = new Set(prev)
      s.add(data.assistants.length)
      return s
    })
  }, [data.assistants.length])

  const removeAssistant = useCallback(
    (idx: number) => {
      setData((prev) => ({
        ...prev,
        assistants: prev.assistants.filter((_, i) => i !== idx),
      }))
      setDeleteModal(null)
    },
    []
  )

  const duplicateAssistant = useCallback(
    (idx: number) => {
      const original = data.assistants[idx]
      if (!original) return
      const baseName = original.name || $t('configEditor.newAssistant')
      let newName = `${baseName} ${$t('configEditor.copySuffix')}`
      let counter = 2
      const existingNames = new Set(data.assistants.map((a) => a.name))
      while (existingNames.has(newName)) {
        newName = `${baseName} ${$t('configEditor.copySuffix')} ${counter}`
        counter++
      }
      const copy = { ...original, name: newName }
      setData((prev) => ({
        ...prev,
        assistants: [...prev.assistants, copy],
      }))
      setExpanded((prev) => {
        const s = new Set(prev)
        s.add(data.assistants.length)
        return s
      })
    },
    [data.assistants]
  )

  const toggleExpand = useCallback((idx: number) => {
    setExpanded((prev) => {
      const s = new Set(prev)
      if (s.has(idx)) s.delete(idx)
      else s.add(idx)
      return s
    })
  }, [])

  const toggleAudio = useCallback((idx: number) => {
    setAudioExpanded((prev) => {
      const s = new Set(prev)
      if (s.has(idx)) s.delete(idx)
      else s.add(idx)
      return s
    })
  }, [])

  const toggleEmotion = useCallback((idx: number) => {
    setEmotionExpanded((prev) => {
      const s = new Set(prev)
      if (s.has(idx)) s.delete(idx)
      else s.add(idx)
      return s
    })
  }, [])

  const handleApply = useCallback(() => {
    setApplyStatus('loading')
    config
      .apply()
      .then(() => {
        setApplyStatus('done')
        setTimeout(() => setApplyStatus('idle'), 3000)
      })
      .catch((err) => {
        setApplyStatus('error')
        setError(String(err))
      })
  }, [])

  const loadHistory = useCallback(() => {
    setHistoryLoading(true)
    config
      .instructionsHistory(10)
      .then((res) => {
        setHistory(res)
        setHistoryLoading(false)
        // Pre-select current version as the "newer" comparison target
        const current = res.find((h) => h.is_current)
        if (current) {
          setSelectedVersions([null, current.id])
        }
      })
      .catch(() => setHistoryLoading(false))
  }, [])

  const handleRollback = useCallback(
    (versionId: number) => {
      setRollbackTarget(versionId)
    },
    []
  )

  const doRollback = useCallback(
    (versionId: number) => {
      config
        .rollbackInstructions(versionId)
        .then(() => {
          invalidateCache('/admin/config/instructions')
          return refreshInstructions()
        })
        .then((res) => {
          const assistants = (Array.isArray(res.assistants) ? res.assistants : []).map(
            (a: Partial<AssistantEntry>) => ({ ...DEFAULT_ASSISTANT, ...a })
          )
          const payload: InstructionsData = { assistants }
          setData(payload)
          initialDataRef.current = JSON.stringify(payload)
          setExpanded(new Set(assistants.map((_, idx) => idx)))
          setHistory((prev) =>
            prev.map((h) => ({ ...h, is_current: h.id === versionId }))
          )
        })
        .catch((err) => setError(String(err)))
    },
    []
  )

  const handleDownloadInstructions = useCallback(() => {
    setTransferStatus('downloading')
    config
      .downloadInstructionsYaml()
      .then((blob) => {
        const url = window.URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = 'instructions.yaml'
        document.body.appendChild(link)
        link.click()
        link.remove()
        window.URL.revokeObjectURL(url)
        setTransferStatus('idle')
      })
      .catch((err) => {
        setTransferStatus('error')
        setError(String(err))
      })
  }, [])

  const handleUploadInstructions = useCallback((file: File) => {
    setTransferStatus('uploading')
    config
      .uploadInstructionsYaml(file)
      .then(() => {
          invalidateCache('/admin/config/instructions')
          return refreshInstructions()
        })
      .then((res) => {
        const assistants = (Array.isArray(res.assistants) ? res.assistants : []).map(
          (a: Partial<AssistantEntry>) => ({ ...DEFAULT_ASSISTANT, ...a })
        )
        const payload: InstructionsData = { assistants }
        setData(payload)
        initialDataRef.current = JSON.stringify(payload)
        setExpanded(new Set(assistants.map((_, idx) => idx)))
        setTransferStatus('uploaded')
        setTimeout(() => setTransferStatus('idle'), 3000)
      })
      .catch((err) => {
        setTransferStatus('error')
        setError(String(err))
      })
  }, [])

  if (loading) return <div className="loading">{$t('common.loading')}</div>
  if (error && data.assistants.length === 0)
    return <div className="error-box">{error}</div>

  return (
    <div>
      <div className="page-header">
        <h2>{$t('config.title')}</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {saveStatus === 'saving' && (
            <span className="status-badge saving">
              <Save size={14} /> {$t('common.saving')}
            </span>
          )}
          {saveStatus === 'saved' && (
            <span className="status-badge saved">
              <CheckCircle size={14} /> {$t('common.saved')}
            </span>
          )}
          {saveStatus === 'error' && (
            <span className="status-badge error">
              <AlertCircle size={14} /> {$t('common.error')}
            </span>
          )}
          {transferStatus === 'uploaded' && (
            <span className="status-badge saved">
              <CheckCircle size={14} /> {$t('config.uploadCompleted')}
            </span>
          )}
          {transferStatus === 'error' && (
            <span className="status-badge error">
              <AlertCircle size={14} /> {$t('config.uploadError')}
            </span>
          )}
          <button
            className="btn-secondary"
            onClick={handleDownloadInstructions}
            disabled={transferStatus === 'downloading' || transferStatus === 'uploading'}
          >
            <Download size={14} />{' '}
            {transferStatus === 'downloading' ? $t('common.loading') : $t('common.download')}
          </button>
          <input
            ref={uploadInputRef}
            type="file"
            accept=".yaml,.yml,application/x-yaml,text/yaml,text/plain"
            style={{ display: 'none' }}
            onChange={(e) => {
              const file = e.target.files?.[0]
              e.target.value = ''
              if (file) handleUploadInstructions(file)
            }}
          />
          {canWrite && (
            <button
              className="btn-secondary"
              onClick={() => uploadInputRef.current?.click()}
              disabled={transferStatus === 'downloading' || transferStatus === 'uploading'}
            >
              <Upload size={14} />{' '}
              {transferStatus === 'uploading' ? $t('common.loading') : $t('common.upload')}
            </button>
          )}
          {canWrite && (
            <button
              className="btn-primary"
              onClick={handleApply}
              disabled={applyStatus === 'loading'}
            >
              {applyStatus === 'loading' ? (
                <>
                  <RotateCcw size={14} className="spin" /> {$t('common.applying')}
                </>
              ) : (
                <>
                  <RotateCcw size={14} /> {$t('common.apply')}
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}
      {applyStatus === 'done' && (
        <div className="success-box">
          {$t('config.configInitialized')}
        </div>
      )}

      <div className="config-section">
        <div
          className="collections-header"
          style={{ cursor: 'pointer' }}
          onClick={() => {
            const next = !historyOpen
            setHistoryOpen(next)
            if (next && history.length === 0) loadHistory()
          }}
        >
          <h3>
            <ChevronDown
              size={16}
              style={{
                transform: historyOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s',
                verticalAlign: 'middle',
                marginRight: 6,
              }}
            />
            {$t('config.versionHistory')}
          </h3>
        </div>
        {historyOpen && (
          <div style={{ marginTop: 8 }}>
            {historyLoading ? (
              <span className="loading">{$t('common.loading')}</span>
            ) : history.length === 0 ? (
              <span className="text-muted">{$t('config.noVersions')}</span>
            ) : (
              <>
                <div className="version-selection-bar">
                  {diffLoading ? (
                    <>
                      <RotateCcw size={14} className="spin" />
                      <span>{$t('config.diffLoading')}</span>
                    </>
                  ) : (
                    <>
                      <GitCompare size={14} />
                      {selectedVersions[0] !== null && selectedVersions[1] !== null ? (
                        <span>
                          {$t('config.version')} #{selectedVersions[0]} → #{$t('config.version')} #{selectedVersions[1]}
                        </span>
                      ) : selectedVersions[1] !== null ? (
                        <span>{$t('config.selectVersionsHint')}</span>
                      ) : (
                        <span>{$t('config.selectVersionsHint')}</span>
                      )}
                    </>
                  )}
                  {(selectedVersions[0] !== null || selectedVersions[1] !== null) && (
                    <button
                      className="btn-text"
                      onClick={() => {
                        const current = history.find((h) => h.is_current)
                        setSelectedVersions([null, current?.id ?? null])
                      }}
                      style={{ marginLeft: 'auto' }}
                    >
                      {$t('config.resetSelection')}
                    </button>
                  )}
                </div>
                <div className="version-timeline-list">
                  {history.map((item) => {
                    const isOlder = selectedVersions[0] === item.id
                    const isNewer = selectedVersions[1] === item.id
                    return (
                      <div
                        key={item.id}
                        className={[
                          'version-timeline-item',
                          item.is_current ? 'current' : '',
                          isOlder ? 'selected-older' : '',
                          isNewer ? 'selected-newer' : '',
                        ].join(' ')}
                        onClick={() => {
                          setSelectedVersions((prev) => {
                            const [older, newer] = prev
                            if (older === item.id) return [null, newer]
                            if (newer === item.id) return [older, null]

                            let next: [number | null, number | null]
                            if (older === null && newer === null) {
                              next = [item.id, null]
                            } else if (older === null) {
                              next = [item.id, newer]
                            } else if (newer === null) {
                              next = [older, item.id]
                            } else {
                              // Both already set → start fresh with this as the only selection
                              next = [item.id, null]
                            }

                            // Auto-trigger diff whenever we reach 2 selections
                            if (next[0] !== null && next[1] !== null) {
                              const ids = [next[0], next[1]].sort((a, b) => a - b) as [number, number]
                              setDiffLoading(true)
                              config
                                .diffVersions(ids[0], ids[1])
                                .then((res) => {
                                  setDiffResult(res)
                                  setDiffModalOpen(true)
                                  setDiffLoading(false)
                                })
                                .catch((err) => {
                                  setError(String(err))
                                  setDiffLoading(false)
                                })
                            }

                            return next
                          })
                        }}
                      >
                        <div className="version-timeline-meta">
                          <span className="version-id">
                            {isOlder ? (
                              <span className="version-timeline-badge older">← {$t('config.olderVersion')}</span>
                            ) : isNewer ? (
                              <span className="version-timeline-badge newer">→ {$t('config.newerVersion')}</span>
                            ) : item.is_current ? (
                              <span className="version-timeline-badge current">{$t('config.current')}</span>
                            ) : (
                              <span className="version-timeline-badge">#{item.id}</span>
                            )}
                          </span>
                          <span className="version-date">
                            <History size={12} style={{ verticalAlign: 'text-bottom', marginRight: 4 }} />
                            {new Date(item.created_at).toLocaleString('de-DE')}
                          </span>
                          <span className="version-author">{item.created_by || '—'}</span>
                        </div>
                        <div className="version-timeline-actions">
                          {!item.is_current && canWrite && (
                            <button
                              className="btn-secondary"
                              onClick={(e) => {
                                e.stopPropagation()
                                handleRollback(item.id)
                              }}
                            >
                              <RotateCcw size={14} /> {$t('config.rollback')}
                            </button>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </>
            )}
          </div>
        )}
      </div>

      <div className="config-section">
        <div className="collections-header">
          <h3>{$t('configEditor.assistantsCount', { count: data.assistants.length })}</h3>
          {canWrite && (
            <button className="btn-secondary" onClick={addAssistant}>
              <Plus size={16} /> {$t('configEditor.newAssistant')}
            </button>
          )}
        </div>

        {data.assistants.map((assistant, idx) => {
          const isOpen = expanded.has(idx)
          const aTab = assistantTab[idx] ?? 'basis'
          const setATab = (t: string) => setAssistantTab((prev) => ({ ...prev, [idx]: t }))
          const ebKey = assistant.name?.trim() || `new-${idx}`
          const assistantProvider = resolveProviderForAssistant(
            assistant,
            providerEntries,
            availableModelsByProvider
          )
          return (
            <div key={idx} className="assistant-card">
              <div
                className="assistant-header"
                onClick={() => toggleExpand(idx)}
              >
                <div className="assistant-title">
                  <span className="assistant-name">
                    {assistant.name || 'Neuer Assistant'}
                  </span>
                  <AssistantBadges assistant={assistant} provider={assistantProvider} />
                  <span className="assistant-model">{assistant.model}</span>
                </div>
                <div className="assistant-actions">
                  {canWrite && (
                    <AssistantActionsMenu
                      onDuplicate={() => duplicateAssistant(idx)}
                      onDelete={() => setDeleteModal({ idx, name: assistant.name || $t('configEditor.newAssistant') })}
                      $t={$t}
                    />
                  )}
                  {isOpen ? (
                    <ChevronUp size={18} />
                  ) : (
                    <ChevronDown size={18} />
                  )}
                </div>
              </div>

              {isOpen && (
                <div className="assistant-body">
                  <div className="assistant-tabs" role="tablist">
                    {([
                      ['basis', 'configEditor.tabBasis'],
                      ['model', 'configEditor.tabModel'],
                      ['agent', 'configEditor.tabAgent'],
                      ['tools', 'configEditor.tabTools', Object.keys(assistant.tools || {}).length || null],
                      ['subagents', 'configEditor.tabSubagents'],
                      ['workspace', 'configEditor.tabWorkspace'],
                      ['media', 'configEditor.tabMedia'],
                    ] as Array<[string, string, number?] | [string, string]>).map(([key, label, badge]) => (
                      <button
                        key={key}
                        type="button"
                        role="tab"
                        aria-selected={aTab === key}
                        className={`assistant-tab${aTab === key ? ' active' : ''}`}
                        onClick={() => setATab(key)}
                      >
                        {$t(label)}
                        {badge ? <span className="assistant-tab-badge">{badge}</span> : null}
                      </button>
                    ))}
                  </div>
                  <div className="form-grid">
                  {aTab === 'basis' && (<>
                    <div className="form-group">
                      <label>Name *</label>
                      <input
                        type="text"
                        value={assistant.name}
                        onChange={(e) =>
                          updateAssistant(idx, { name: e.target.value })
                        }
                        placeholder="z. B. CyberSecurityChatbot"
                        disabled={!canWrite}
                      />
                    </div>

                    <div className="form-group">
                      <label>Modell *</label>
                      <select
                        value={assistant.model}
                        onChange={(e) =>
                          updateAssistant(idx, { model: e.target.value })
                        }
                        disabled={!canWrite}
                      >
                        <option value="">— Modell wählen —</option>
                        {Object.entries(availableModelsByProvider).map(([providerId, pm]) => (
                          <optgroup key={providerId} label={pm.name}>
                            {pm.models.map((m) => (
                              <option key={m} value={m}>
                                {m}
                              </option>
                            ))}
                          </optgroup>
                        ))}
                        {(() => {
                          const all = Object.values(availableModelsByProvider).flatMap((pm) => pm.models)
                          return !all.includes(assistant.model) && assistant.model ? (
                            <option value={assistant.model}>{assistant.model}</option>
                          ) : null
                        })()}
                      </select>
                    </div>

                    {assistant.agent_enabled && assistantProvider && (
                      <div className="form-group wide">
                        <label>Tool Calling Status</label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                          {assistantProvider.supports_native_tool_calls === true && (
                            <span className="assistant-badge badge-native">Native</span>
                          )}
                          {assistantProvider.supports_native_tool_calls === false && (
                            <span className="assistant-badge badge-parsed">Parsed (JSON)</span>
                          )}
                          {assistantProvider.supports_native_tool_calls === null && (
                            <span className="assistant-badge badge-unknown">Unbekannt</span>
                          )}
                          {assistantProvider.tool_calling_checked_at && (
                            <small style={{ color: 'var(--text-secondary)' }}>
                              Geprüft: {new Date(assistantProvider.tool_calling_checked_at).toLocaleString('de-DE')}
                            </small>
                          )}
                          {canWrite && (
                            <button
                              type="button"
                              className="btn-secondary"
                              style={{ padding: '4px 10px', fontSize: '12px' }}
                              onClick={async () => {
                                try {
                                  await providers.detectToolCalling(assistantProvider.id, assistant.model || undefined)
                                  const refreshed = await refreshProviders()
                                  setProviderEntries(refreshed)
                                } catch (err) {
                                  setError(String(err))
                                }
                              }}
                            >
                              Neu prüfen
                            </button>
                          )}
                        </div>
                      </div>
                    )}

                    <div className="form-group wide">
                      <label>
                        <Tooltip {...fieldMeta.description} label="Beschreibung" $t={$t} />
                      </label>
                      <textarea
                        rows={3}
                        value={assistant.description || ''}
                        onChange={(e) =>
                          updateAssistant(idx, {
                            description: e.target.value,
                          })
                        }
                        disabled={!canWrite}
                      />
                    </div>
                  </>)}

                  {aTab === 'model' && (<>
                      <div className="form-grid" style={{ gridColumn: '1 / -1' }}>
                        <div className="form-group">
                          <label>
                            <Tooltip {...fieldMeta.temperature} label="Temperatur" $t={$t} />
                          </label>
                          <input
                            type="number"
                        step={0.1}
                        min={0}
                        max={2}
                        value={assistant.temperature}
                        onChange={(e) =>
                          updateAssistant(idx, {
                            temperature:
                              e.target.value === ''
                                ? 0
                                : parseFloat(e.target.value),
                          })
                        }
                        disabled={!canWrite}
                      />
                    </div>

                    <div className="form-group">
                      <label>
                        <Tooltip {...fieldMeta.collections} label="Collections" $t={$t} />
                      </label>
                      <MultiSelect
                        options={availableCollections}
                        deprecatedOptions={(assistant.collections || []).filter(
                          (c) => !availableCollections.includes(c)
                        )}
                        selected={assistant.collections || []}
                        onChange={(vals) =>
                          updateAssistant(idx, { collections: vals })
                        }
                        emptyText={$t('config.noCollections')}
                        disabled={!canWrite}
                      />
                    </div>

                    <ReasoningControls
                      model={assistant.model}
                      providerId={assistant.provider_id || assistantProvider?.id}
                      reasoning={assistant.reasoning}
                      legacyEnableThinking={assistant.enable_thinking}
                      onChange={(r) => updateAssistant(idx, { reasoning: r })}
                      onClearLegacy={() => updateAssistant(idx, { enable_thinking: null })}
                      disabled={!canWrite}
                      $t={$t}
                    />
                      </div>
                  </>)}

                  {aTab === 'agent' && (<>
                    <div className="form-group">
                      <label>
                        Agent Mode
                        <small style={{ marginLeft: 6, color: '#888' }}>({$t('common.optional')})</small>
                      </label>
                      <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={assistant.agent_enabled || false}
                          onChange={(e) =>
                            updateAssistant(idx, { agent_enabled: e.target.checked })
                          }
                          disabled={!canWrite}
                        />
                        <span>Enable agent mode</span>
                      </label>
                    </div>
                  </>)}

                  {aTab === 'basis' && (<>
                    <div className="form-group">
                      <label>
                        {$t('configEditor.apiAccessible')}
                        <small style={{ marginLeft: 6, color: '#888' }}>({$t('common.optional')})</small>
                      </label>
                      <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={assistant.api_accessible !== false}
                          onChange={(e) =>
                            updateAssistant(idx, { api_accessible: e.target.checked })
                          }
                          disabled={!canWrite}
                        />
                        <span>{$t('configEditor.apiAccessibleHint')}</span>
                      </label>
                    </div>
                  </>)}

                  {aTab === 'agent' && (<>
                    {assistant.agent_enabled && (
                      <>
                        <div className="form-group">
                          <label>
                            Agent Memory
                            <small style={{ marginLeft: 6, color: '#888' }}>(optional)</small>
                          </label>
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={assistant.agent_memory_enabled || false}
                              onChange={(e) =>
                                updateAssistant(idx, { agent_memory_enabled: e.target.checked })
                              }
                              disabled={!canWrite}
                            />
                            <span>Enable agent memory</span>
                          </label>
                          <small className="field-hint" style={{ marginTop: 4, display: 'block' }}>
                            When active, a persistent conversation memory is automatically available. The agent can store facts, preferences, and intermediate results across tool calls. Use <code>{'{{agent_memory}}'}</code> in the system instructions to place the memory at a specific position; otherwise it is appended automatically.
                          </small>
                        </div>
                      </>
                    )}
                  </>)}

                  {aTab === 'subagents' && (<>
                    {assistant.agent_enabled && (
                      <>
                        <div className="form-group wide subagents-config-section">
                          <label>{$t('configEditor.subagentsTitle')}</label>
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={Boolean(assistant.subagents?.enabled)}
                              onChange={(e) =>
                                updateAssistant(idx, {
                                  subagents: e.target.checked
                                    ? { enabled: true, max_concurrent: 2, max_depth: 1, max_total: 8 }
                                    : null,
                                })
                              }
                              disabled={!canWrite}
                            />
                            <span>{$t('configEditor.enableSubagents')}</span>
                          </label>
                          <small className="field-hint" style={{ display: 'block', margin: '5px 0 12px' }}>
                            {$t('configEditor.subagentsDescription')}
                          </small>
                          {assistant.subagents?.enabled && (
                            <>
                              <div className="subagents-config-grid">
                                {([
                                  ['max_concurrent', 'configEditor.maxConcurrentSubagents', 1, 10, 2],
                                  ['max_total', 'configEditor.maxTotalSubagents', 1, 500, 8],
                                  ['max_depth', 'configEditor.maxSubagentDepth', 1, 5, 1],
                                  ['report_answer_chars', 'configEditor.subagentReportChars', 1000, 60000, 20000],
                                ] as const).map(([field, label, min, max, fallback]) => (
                                  <div className="form-group" key={field}>
                                    <label>{$t(label)}</label>
                                    <input
                                      type="number"
                                      min={min}
                                      max={max}
                                      value={assistant.subagents?.[field] ?? fallback}
                                      onChange={(e) => {
                                        const value = Number.parseInt(e.target.value, 10)
                                        updateAssistant(idx, {
                                          subagents: {
                                            ...(assistant.subagents || { enabled: true }),
                                            [field]: Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback,
                                          },
                                        })
                                      }}
                                      disabled={!canWrite}
                                    />
                                  </div>
                                ))}
                              </div>
                              <div className="form-group" style={{ marginTop: 8 }}>
                                <label>{$t('configEditor.subagentExcludeTools')}</label>
                                <input
                                  type="text"
                                  key={`exclude-${idx}-${(assistant.subagents?.exclude_tools || []).join(',')}`}
                                  placeholder="email_sender, code_execution"
                                  defaultValue={(assistant.subagents?.exclude_tools || []).join(', ')}
                                  onBlur={(e) => {
                                    const names = e.target.value
                                      .split(',')
                                      .map((s) => s.trim())
                                      .filter(Boolean)
                                    updateAssistant(idx, {
                                      subagents: {
                                        ...(assistant.subagents || { enabled: true }),
                                        exclude_tools: names,
                                      },
                                    })
                                  }}
                                  disabled={!canWrite}
                                />
                                <small style={{ color: 'var(--text-secondary)' }}>
                                  {$t('configEditor.subagentExcludeToolsHint')}
                                </small>
                              </div>
                            </>
                          )}
                        </div>

                        <div className="form-group wide observer-config-section">
                          <label>{$t('configEditor.observerTitle')}</label>
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={Boolean(assistant.observer?.enabled)}
                              onChange={(e) =>
                                updateAssistant(idx, {
                                  observer: e.target.checked ? { enabled: true } : null,
                                })
                              }
                              disabled={!canWrite}
                            />
                            <span>{$t('configEditor.enableObserver')}</span>
                          </label>
                          <small className="field-hint" style={{ display: 'block', margin: '5px 0 12px' }}>
                            {$t('configEditor.observerDescription')}
                          </small>
                          {assistant.observer?.enabled && (
                            <>
                              <div className="subagents-config-grid">
                                {([
                                  ['every_iterations', 'configEditor.observerEveryIterations', 1, 20, 4],
                                  ['every_seconds', 'configEditor.observerEverySeconds', 30, 3600, 180],
                                  ['max_checks', 'configEditor.observerMaxChecks', 1, 50, 8],
                                  ['max_nudges', 'configEditor.observerMaxNudges', 1, 20, 4],
                                  ['check_timeout_seconds', 'configEditor.observerCheckTimeout', 10, 300, 60],
                                ] as const).map(([field, label, min, max, fallback]) => (
                                  <div className="form-group" key={field}>
                                    <label>{$t(label)}</label>
                                    <input
                                      type="number"
                                      min={min}
                                      max={max}
                                      value={assistant.observer?.[field] ?? fallback}
                                      onChange={(e) => {
                                        const value = Number(e.target.value)
                                        updateAssistant(idx, {
                                          observer: {
                                            ...(assistant.observer || { enabled: true }),
                                            [field]: Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback,
                                          },
                                        })
                                      }}
                                      disabled={!canWrite}
                                    />
                                  </div>
                                ))}
                              </div>
                              <div className="form-group" style={{ marginTop: 8 }}>
                                <label>{$t('configEditor.observerAssistant')}</label>
                                <input
                                  type="text"
                                  list={`observer-assistants-${idx}`}
                                  placeholder="observer"
                                  value={assistant.observer?.assistant ?? ''}
                                  onChange={(e) =>
                                    updateAssistant(idx, {
                                      observer: {
                                        ...(assistant.observer || { enabled: true }),
                                        assistant: e.target.value.trim() || null,
                                      },
                                    })
                                  }
                                  disabled={!canWrite}
                                />
                                <datalist id={`observer-assistants-${idx}`}>
                                  {data.assistants.map((a) => a.name).filter(Boolean).map((name) => (
                                    <option key={name} value={name} />
                                  ))}
                                </datalist>
                                <small style={{ color: 'var(--text-secondary)' }}>
                                  {$t('configEditor.observerAssistantHint')}
                                </small>
                              </div>
                              <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', marginTop: 8 }}>
                                <input
                                  type="checkbox"
                                  checked={assistant.observer?.include_subagents !== false}
                                  onChange={(e) =>
                                    updateAssistant(idx, {
                                      observer: {
                                        ...(assistant.observer || { enabled: true }),
                                        include_subagents: e.target.checked,
                                      },
                                    })
                                  }
                                  disabled={!canWrite}
                                />
                                <span>{$t('configEditor.observerIncludeSubagents')}</span>
                              </label>
                              <div className="form-group" style={{ marginTop: 8 }}>
                                <label>{$t('configEditor.observerInstructions')}</label>
                                <textarea
                                  rows={2}
                                  value={assistant.observer?.instructions ?? ''}
                                  onChange={(e) =>
                                    updateAssistant(idx, {
                                      observer: {
                                        ...(assistant.observer || { enabled: true }),
                                        instructions: e.target.value || null,
                                      },
                                    })
                                  }
                                  disabled={!canWrite}
                                />
                                <small style={{ color: 'var(--text-secondary)' }}>
                                  {$t('configEditor.observerInstructionsHint')}
                                </small>
                              </div>
                            </>
                          )}
                        </div>

                      </>
                    )}
                    {!assistant.agent_enabled && (
                      <small className="field-hint">{$t('configEditor.agentModeRequiredHint')}</small>
                    )}
                  </>)}

                  {aTab === 'tools' && (<>
                    {!assistant.agent_enabled && (
                      <small className="field-hint">{$t('configEditor.agentModeRequiredHint')}</small>
                    )}
                    <div className="form-group wide">
                      <label>
                        {$t('configEditor.toolsTitle')}
                        {Object.keys(assistant.tools || {}).length > 0 && (
                          <span className="assistant-tab-badge" style={{ marginLeft: 6 }}>
                            {Object.keys(assistant.tools || {}).length}
                          </span>
                        )}
                      </label>
                      {Object.keys(assistant.tools || {}).length === 0 && (
                        <small className="field-hint">{$t('configEditor.noToolsHint')}</small>
                      )}
                      {(() => {
                        const sel = Object.keys(assistant.tools || {})
                        const byKind: Record<string, string[]> = {}
                        for (const name of sel) {
                          const kind = availableToolEntries.find((e) => e.name === name)?.kind || 'custom'
                          ;(byKind[kind] = byKind[kind] || []).push(name)
                        }
                        const kinds = GROUP_ORDER.filter((k) => byKind[k]?.length)
                        return kinds.map((kind) => (
                          <div key={kind} className="tool-group">
                            <div className="tool-group-header">{GROUP_LABELS[kind] || kind}</div>
                            {byKind[kind].map((toolName) => {
                              const rowKey = `${idx}:${toolName}`
                              const rowOpen = toolRowsExpanded.has(rowKey)
                              const toolConfig = (assistant.tools || {})[toolName] || {}
                              const overrideCount = Object.keys(toolConfig).length
                              const entry = availableToolEntries.find((e) => e.name === toolName)
                              return (
                                <div key={toolName} className="tool-row">
                                  <div className="tool-row-head">
                                    <button
                                      type="button"
                                      className="tool-row-toggle"
                                      onClick={() =>
                                        setToolRowsExpanded((prev) => {
                                          const s = new Set(prev)
                                          if (s.has(rowKey)) s.delete(rowKey)
                                          else s.add(rowKey)
                                          return s
                                        })
                                      }
                                    >
                                      {rowOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                    </button>
                                    <span className="tool-row-name">{toolName}</span>
                                    {overrideCount > 0 && (
                                      <span className="assistant-tab-badge" title={$t('configEditor.toolOverridesTitle')}>
                                        {overrideCount}
                                      </span>
                                    )}
                                    <span className="tool-row-desc">{entry?.description || ''}</span>
                                    {canWrite && (
                                      <button
                                        type="button"
                                        className="btn-text tool-row-remove"
                                        onClick={() => {
                                          const next = { ...(assistant.tools || {}) }
                                          delete next[toolName]
                                          updateAssistant(idx, { tools: next })
                                        }}
                                      >
                                        <Trash2 size={14} />
                                      </button>
                                    )}
                                  </div>
                                  {rowOpen && (
                                    <div className="tool-row-body">
                                      <div className="tool-row-defaults">
                                        {(() => {
                                          const defaults = Object.entries(entry?.config || {})
                                          if (defaults.length === 0) return $t('configEditor.toolNoGlobalConfig')
                                          return `${$t('configEditor.toolGlobalDefaults')}: ${defaults.map(([k, v]) => `${k}=${maskSensitiveValue(k, v)}`).join(', ')}`
                                        })()}
                                      </div>
                                      <KeyValueEditor
                                        entries={Object.entries(toolConfig || {}).map(([k, v]) => [k, String(v)])}
                                        onChange={(entries) => {
                                          const next = { ...(assistant.tools || {}) }
                                          next[toolName] = Object.fromEntries(
                                            entries.map(([k, v]) => {
                                              const num = Number(v)
                                              return [k.trim(), Number.isFinite(num) ? num : v]
                                            })
                                          )
                                          updateAssistant(idx, { tools: next })
                                        }}
                                        disabled={!canWrite}
                                        $t={$t}
                                      />
                                    </div>
                                  )}
                                </div>
                              )
                            })}
                          </div>
                        ))
                      })()}
                      <div className="auto-tool-group">
                        <div className="tool-group-header">{$t('configEditor.autoToolsHeader')}</div>
                        <div className="auto-tool-badges">
                          <span className="assistant-badge auto-tool-badge">agent_todo</span>
                          <span className="assistant-badge auto-tool-badge">wait</span>
                          {assistant.agent_memory_enabled && (
                            <span className="assistant-badge auto-tool-badge">agent_memory</span>
                          )}
                          {assistant.subagents?.enabled && (
                            <>
                              <span className="assistant-badge auto-tool-badge subagent-auto-badge">spawn_subagent</span>
                              <span className="assistant-badge auto-tool-badge subagent-auto-badge">wait_subagent</span>
                              <span className="assistant-badge auto-tool-badge subagent-auto-badge">subagent_status</span>
                            </>
                          )}
                          {assistant.workspace?.enabled && (
                            <span className="assistant-badge auto-tool-badge workspace-auto-badge">agent_workspace</span>
                          )}
                        </div>
                      </div>
                      {canWrite && (() => {
                        const ownToolName = 'assistant__' + assistant.name.split('').map((c) => (/[a-zA-Z0-9]/.test(c) ? c : '_')).join('').toLowerCase()
                        return (
                          <ToolAddDropdown
                            availableEntries={availableToolEntries.filter((e) => !(assistant.tools || {})[e.name] && !['agent_memory', 'agent_todo', 'wait', 'spawn_subagent', 'wait_subagent', 'subagent_status', 'agent_workspace', 'mcp_servers'].includes(e.name) && e.name !== ownToolName)}
                            onSelect={(name) => {
                              updateAssistant(idx, { tools: { ...(assistant.tools || {}), [name]: {} } })
                            }}
                          />
                        )
                      })()}
                    </div>
                  </>)}

                  {aTab === 'workspace' && (<>
                    <div className="form-group wide workspace-config-section">
                      <label>{$t('configEditor.workspaceTitle')}</label>
                      <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={Boolean(assistant.workspace?.enabled)}
                          onChange={(e) =>
                            updateAssistant(idx, {
                              workspace: e.target.checked ? { enabled: true, scope: 'conversation' } : null,
                            })
                          }
                          disabled={!canWrite}
                        />
                        <span>{$t('configEditor.enableWorkspace')}</span>
                      </label>
                      <small className="field-hint" style={{ display: 'block', margin: '5px 0 12px' }}>
                        {$t('configEditor.workspaceDescription')}
                      </small>
                      {!assistant.agent_enabled && assistant.workspace?.enabled && (
                        <small className="field-hint" style={{ display: 'block', marginBottom: 8, color: 'var(--warning)' }}>
                          {$t('configEditor.workspaceAgentHint')}
                        </small>
                      )}
                      {assistant.workspace?.enabled && (
                        <>
                          <div className="form-group">
                            <label>{$t('configEditor.workspaceScope')}</label>
                            <select
                              value={assistant.workspace?.scope || 'conversation'}
                              onChange={(e) =>
                                updateAssistant(idx, {
                                  workspace: { ...(assistant.workspace || { enabled: true }), scope: e.target.value as 'assistant' | 'user' | 'conversation' },
                                })
                              }
                              disabled={!canWrite}
                            >
                              <option value="conversation">{$t('configEditor.scopeConversation')}</option>
                              <option value="user">{$t('configEditor.scopeUser')}</option>
                              <option value="assistant">{$t('configEditor.scopeAssistant')}</option>
                            </select>
                            <small className="field-hint" style={{ display: 'block', marginTop: 4 }}>
                              {$t(`configEditor.scopeHint_${assistant.workspace?.scope || 'conversation'}`)}
                            </small>
                            {(assistant.workspace?.scope || 'conversation') === 'assistant' && (
                              <small className="field-hint" style={{ display: 'block', marginTop: 4, color: 'var(--warning)' }}>
                                {$t('configEditor.scopeSharedWarning')}
                              </small>
                            )}
                          </div>
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', marginTop: 8 }}>
                            <input
                              type="checkbox"
                              checked={Boolean(assistant.workspace?.api_enabled)}
                              onChange={(e) =>
                                updateAssistant(idx, {
                                  workspace: { ...(assistant.workspace || { enabled: true }), api_enabled: e.target.checked },
                                })
                              }
                              disabled={!canWrite}
                            />
                            <span>{$t('configEditor.workspaceApiEnabled')}</span>
                          </label>
                          {assistant.workspace?.api_enabled && (
                            <small className="field-hint" style={{ display: 'block', marginTop: 4, color: 'var(--danger)' }}>
                              {$t('configEditor.workspaceApiWarning')}
                            </small>
                          )}
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', marginTop: 8 }}>
                            <input
                              type="checkbox"
                              checked={Boolean(assistant.workspace?.vision_read)}
                              onChange={(e) =>
                                updateAssistant(idx, {
                                  workspace: { ...(assistant.workspace || { enabled: true }), vision_read: e.target.checked },
                                })
                              }
                              disabled={!canWrite}
                            />
                            <span>{$t('configEditor.workspaceVisionRead')}</span>
                          </label>
                          {assistant.workspace?.vision_read && (
                            <small className="field-hint" style={{ display: 'block', marginTop: 4 }}>
                              {$t('configEditor.workspaceVisionReadHint')}
                            </small>
                          )}
                          <div className="subagents-config-grid" style={{ marginTop: 8 }}>
                            {([
                              ['max_file_mb', 'configEditor.workspaceMaxFileMb', 1, 500, 50],
                              ['max_total_mb', 'configEditor.workspaceMaxTotalMb', 10, 10000, 500],
                              ['max_files', 'configEditor.workspaceMaxFiles', 1, 100000, 5000],
                            ] as const).map(([field, label, min, max, fallback]) => (
                              <div className="form-group" key={field}>
                                <label>{$t(label)}</label>
                                <input
                                  type="number"
                                  min={min}
                                  max={max}
                                  value={assistant.workspace?.[field] ?? fallback}
                                  onChange={(e) => {
                                    const value = Number(e.target.value)
                                    updateAssistant(idx, {
                                      workspace: {
                                        ...(assistant.workspace || { enabled: true }),
                                        [field]: Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback,
                                      },
                                    })
                                  }}
                                  disabled={!canWrite}
                                />
                              </div>
                            ))}
                          </div>
                        </>
                      )}
                    </div>
                  </>)}

                  {aTab === 'agent' && (<>
                    {assistant.agent_enabled && (
                      <>
                        <div className="form-group">
                          <label>Max Iterations</label>
                          <input
                            type="number"
                            min={1}
                            max={50}
                            value={assistant.max_iterations ?? 10}
                            onChange={(e) =>
                              updateAssistant(idx, {
                                max_iterations:
                                  e.target.value === ''
                                    ? null
                                    : parseInt(e.target.value),
                              })
                            }
                            disabled={!canWrite}
                          />
                        </div>
                        <div className="form-group">
                          <label>Max Tool Calls</label>
                          <input
                            type="number"
                            min={1}
                            max={20}
                            value={assistant.max_tool_calls ?? 10}
                            onChange={(e) =>
                              updateAssistant(idx, {
                                max_tool_calls:
                                  e.target.value === ''
                                    ? null
                                    : parseInt(e.target.value),
                              })
                            }
                            disabled={!canWrite}
                          />
                        </div>
                        <div className="form-group">
                          <label>Max Parallel Tool Calls</label>
                          <input
                            type="number"
                            min={1}
                            max={20}
                            value={assistant.max_parallel_tool_calls ?? 5}
                            onChange={(e) =>
                              updateAssistant(idx, {
                                max_parallel_tool_calls:
                                  e.target.value === ''
                                    ? null
                                    : parseInt(e.target.value),
                              })
                            }
                            disabled={!canWrite}
                          />
                        </div>
                      </>
                    )}
                  </>)}

                  {aTab === 'model' && (<>
                    <div className="form-group">
                      <label>
                        <Tooltip {...fieldMeta.context_window} label="Context Window" $t={$t} />
                      </label>
                      <input
                        type="number"
                        value={assistant.context_window ?? ''}
                        onChange={(e) =>
                          updateAssistant(idx, {
                            context_window:
                              e.target.value === ''
                                ? null
                                : parseInt(e.target.value),
                          })
                        }
                        disabled={!canWrite}
                      />
                    </div>

                    <div className="form-group">
                      <label>
                        <Tooltip {...fieldMeta.max_output_tokens} label="Max Output Tokens" $t={$t} />
                      </label>
                      <input
                        type="number"
                        value={assistant.max_output_tokens ?? ''}
                        onChange={(e) =>
                          updateAssistant(idx, {
                            max_output_tokens:
                              e.target.value === ''
                                ? null
                                : parseInt(e.target.value),
                          })
                        }
                        disabled={!canWrite}
                      />
                    </div>

                    <div className="form-group">
                      <label>
                        <Tooltip {...fieldMeta.rag_reserved_tokens} label="RAG Reserved Tokens" $t={$t} />
                      </label>
                      <input
                        type="number"
                        value={assistant.rag_reserved_tokens ?? ''}
                        onChange={(e) =>
                          updateAssistant(idx, {
                            rag_reserved_tokens:
                              e.target.value === ''
                                ? null
                                : parseInt(e.target.value),
                          })
                        }
                        disabled={!canWrite}
                      />
                    </div>

                    <div className="form-group">
                      <label>
                        <Tooltip {...fieldMeta.min_recent_turns} label="Min Recent Turns" $t={$t} />
                      </label>
                      <input
                        type="number"
                        value={assistant.min_recent_turns ?? ''}
                        onChange={(e) =>
                          updateAssistant(idx, {
                            min_recent_turns:
                              e.target.value === ''
                                ? null
                                : parseInt(e.target.value),
                          })
                        }
                        disabled={!canWrite}
                      />
                    </div>
                  </>)}

                  {aTab === 'media' && (<>
                    <div className="form-group wide" style={{ marginTop: 8 }}>
                      <label
                        style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', userSelect: 'none' }}
                        onClick={() => toggleAudio(idx)}
                      >
                        {audioExpanded.has(idx) ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                        Audio / Sprache
                      </label>
                    </div>

                    {audioExpanded.has(idx) && (
                      <div className="form-grid" style={{ gridColumn: '1 / -1' }}>
                        <div className="form-group wide">
                          <small className="field-hint" style={{ display: 'block', marginBottom: 8 }}>
                            Diese Schalter sind Assistant-Freigaben. Globale Audio-Systeme müssen zusätzlich in der Runtime Config aktiv sein. Für TTS muss mindestens eine Piper-Stimme eingerichtet sein.
                          </small>
                        </div>
                        <div className="form-group">
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={assistant.audio_stt_enabled || false}
                              onChange={(e) => updateAssistant(idx, { audio_stt_enabled: e.target.checked })}
                              disabled={!canWrite}
                            />
                            <span>Spracheingabe aktivieren</span>
                          </label>
                        </div>
                        <div className="form-group">
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={assistant.audio_tts_enabled || false}
                              onChange={(e) => updateAssistant(idx, { audio_tts_enabled: e.target.checked })}
                              disabled={!canWrite}
                            />
                            <span>Sprachausgabe aktivieren</span>
                          </label>
                        </div>
                        <div className="form-group">
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={assistant.audio_phone_enabled || false}
                              onChange={(e) => updateAssistant(idx, { audio_phone_enabled: e.target.checked })}
                              disabled={!canWrite}
                            />
                            <span>Telefonat-Modus aktivieren</span>
                          </label>
                          <small className="field-hint" style={{ marginTop: 4, display: 'block' }}>
                            Benötigt aktivierte STT und TTS.
                          </small>
                        </div>
                        <div className="form-group">
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={assistant.audio_tts_auto_read || false}
                              onChange={(e) => updateAssistant(idx, { audio_tts_auto_read: e.target.checked })}
                              disabled={!canWrite}
                            />
                            <span>Antworten automatisch vorlesen</span>
                          </label>
                        </div>
                        <div className="form-group">
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={assistant.audio_tts_streaming_enabled !== false}
                              onChange={(e) => updateAssistant(idx, { audio_tts_streaming_enabled: e.target.checked })}
                              disabled={!canWrite}
                            />
                            <span>TTS-Streaming erlauben</span>
                          </label>
                        </div>
                        <div className="form-group">
                          <label>Stimme überschreiben</label>
                          <input
                            type="text"
                            value={assistant.audio_voice_id || ''}
                            onChange={(e) =>
                              updateAssistant(idx, {
                                audio_voice_id: e.target.value.trim() || null,
                              })
                            }
                            placeholder="Voice-ID (optional)"
                            disabled={!canWrite}
                          />
                          <small className="field-hint" style={{ marginTop: 4, display: 'block' }}>
                            Optional: eigene Voice-ID statt globaler Default.
                          </small>
                        </div>
                      </div>
                    )}

                    <div className="form-group wide" style={{ marginTop: 8 }}>
                      <label
                        style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', userSelect: 'none' }}
                        onClick={() => toggleEmotion(idx)}
                      >
                        {emotionExpanded.has(idx) ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                        Emotion Detection
                      </label>
                    </div>

                    {emotionExpanded.has(idx) && (
                      <div className="form-grid" style={{ gridColumn: '1 / -1' }}>
                        <div className="form-group wide">
                          <small className="field-hint" style={{ display: 'block', marginBottom: 8 }}>
                            Diese Schalter sind Assistant-Freigaben. Globale Emotion-Systeme müssen zusätzlich in der Runtime Config aktiv sein. Webcam-Auswertung erfordert zusätzlich ein explizites User-Opt-in im Chat.
                          </small>
                        </div>
                        <div className="form-group">
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={assistant.emotion_detection_enabled || false}
                              onChange={(e) => updateAssistant(idx, { emotion_detection_enabled: e.target.checked })}
                              disabled={!canWrite}
                            />
                            <span>Emotion Detection aktivieren</span>
                          </label>
                        </div>
                        <div className="form-group">
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={assistant.emotion_text_enabled || false}
                              onChange={(e) => updateAssistant(idx, { emotion_text_enabled: e.target.checked })}
                              disabled={!canWrite}
                            />
                            <span>Text-Emotionen verwenden</span>
                          </label>
                        </div>
                        <div className="form-group">
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={assistant.emotion_webcam_enabled || false}
                              onChange={(e) => updateAssistant(idx, { emotion_webcam_enabled: e.target.checked })}
                              disabled={!canWrite}
                            />
                            <span>Webcam-Emotionen erlauben</span>
                          </label>
                          <small className="field-hint" style={{ marginTop: 4, display: 'block' }}>
                            Benutzer muss zusätzlich im Chat einwilligen.
                          </small>
                        </div>
                        <div className="form-group">
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={assistant.emotion_prompt_steering_enabled || false}
                              onChange={(e) => updateAssistant(idx, { emotion_prompt_steering_enabled: e.target.checked })}
                              disabled={!canWrite}
                            />
                            <span>Prompt-Steering erlauben</span>
                          </label>
                        </div>
                        <div className="form-group">
                          <label className="toggle-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                            <input
                              type="checkbox"
                              checked={assistant.emotion_analytics_enabled || false}
                              onChange={(e) => updateAssistant(idx, { emotion_analytics_enabled: e.target.checked })}
                              disabled={!canWrite}
                            />
                            <span>Emotion Analytics speichern</span>
                          </label>
                        </div>
                      </div>
                    )}
                  </>)}

                  {aTab === 'basis' && (<>
                    <div className="form-group wide">
                      <label>System Instructions *</label>
                      <InlineMarkdownEditor
                        value={assistant.system_instructions}
                        onChange={(md) =>
                          updateAssistant(idx, {
                            system_instructions: md,
                          })
                        }
                        placeholder="# System Prompt..."
                        disabled={!canWrite}
                      />
                      <small className="field-hint">{$t('configEditor.systemInstructionsPlaceholders')}</small>
                    </div>

                    <div className="form-group wide">
                      <label>Extra Parameter <small>(optional)</small></label>
                      <KeyValueEditor
                        entries={Object.entries(assistant.extra_parameters || {}).map(([k, v]) => [k, String(v)])}
                        onChange={(entries) =>
                          updateAssistant(idx, {
                            extra_parameters: Object.fromEntries(
                              entries.map(([k, v]) => {
                                const num = Number(v)
                                return [k.trim(), Number.isFinite(num) ? num : v]
                              })
                            ) || null,
                          })
                        }
                        disabled={!canWrite}
                        $t={$t}
                      />
                    </div>

                    <div className="form-group wide">
                      <label>Extra Body (JSON) <small>(optional)</small></label>
                      <textarea
                        rows={4}
                        value={extraBodyInputs[ebKey] ?? (assistant.extra_body ? JSON.stringify(assistant.extra_body, null, 2) : '')}
                        onChange={(e) => {
                          const text = e.target.value
                          setExtraBodyInputs((prev) => ({ ...prev, [ebKey]: text }))
                          if (!text.trim()) {
                            setExtraBodyErrors((prev) => ({ ...prev, [ebKey]: '' }))
                            updateAssistant(idx, { extra_body: null })
                            return
                          }
                          try {
                            const parsed = JSON.parse(text)
                            setExtraBodyErrors((prev) => ({ ...prev, [ebKey]: '' }))
                            updateAssistant(idx, { extra_body: parsed })
                          } catch (err) {
                            setExtraBodyErrors((prev) => ({ ...prev, [ebKey]: (err as Error).message }))
                          }
                        }}
                        placeholder='{"chat_template_kwargs": {"enable_thinking": false}}'
                        disabled={!canWrite}
                        style={{
                          width: '100%',
                          fontFamily: 'monospace',
                          borderColor: extraBodyErrors[ebKey] ? '#dc2626' : undefined,
                        }}
                      />
                      {extraBodyErrors[ebKey] ? (
                        <small className="field-error" style={{ color: '#dc2626' }}>
                          {extraBodyErrors[ebKey]}
                        </small>
                      ) : (
                        <small className="field-hint">{$t('configEditor.extraBodyThinkingHint')}</small>
                      )}
                    </div>
                  </>)}
                  </div>
                </div>
              )}
            </div>
      )
    })}
  </div>

      <ConfirmModal
        open={deleteModal !== null}
        title="Assistant löschen"
        message={`Möchtest du den Assistant "${deleteModal?.name ?? ''}" wirklich löschen?`}
        onConfirm={() => {
          if (deleteModal !== null) {
            removeAssistant(deleteModal.idx)
            setDeleteModal(null)
          }
        }}
        onCancel={() => setDeleteModal(null)}
      />

      <VersionDiffModal
        open={diffModalOpen}
        onClose={() => setDiffModalOpen(false)}
        diffResult={diffResult}
        onRollback={async (versionId) => {
          await handleRollback(versionId)
          setDiffModalOpen(false)
        }}
      />

      <ConfirmModal
        open={rollbackTarget !== null}
        title={$t('common.confirm')}
        message={$t('configEditor.rollbackConfirm', { version: rollbackTarget ?? 0 })}
        onConfirm={() => {
          if (rollbackTarget !== null) doRollback(rollbackTarget)
          setRollbackTarget(null)
        }}
        onCancel={() => setRollbackTarget(null)}
      />

      <UnsavedChangesBar
        dirty={isDirty && canWrite}
        saving={saveStatus === 'saving'}
        onSave={handleSave}
        onDiscard={handleDiscard}
      />
    </div>
  )
}
