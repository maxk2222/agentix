import { useEffect, useRef, useState } from 'react'
import { Plus, X } from 'lucide-react'
import { providers as providersApi, type ProviderEntry, type ToolEntry } from '../../api/client'
import {
  configGet, configSet, isValidCidr,
  SECRET_SENTINEL, type ConfigField,
} from './catalog'
import { McpServersEditor } from './McpServersEditor'

/* ------------------------------------------------------------------ */
/* MultiSelect (moved from the old ToolsPage — collections picker)      */
/* ------------------------------------------------------------------ */
function MultiSelect({
  options,
  selected,
  onChange,
  disabled = false,
}: {
  options: string[]
  selected: string[]
  onChange: (values: string[]) => void
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
        {selected.length === 0 ? '—' : selected.join(', ')}
      </button>
      {open && (
        <div className="multi-select-dropdown">
          {options.length === 0 && (
            <span className="multi-select-empty">No collections available</span>
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
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* StringListEditor — chips-style editor for string_list / cidr_list    */
/* ------------------------------------------------------------------ */
function StringListEditor({
  value,
  onChange,
  placeholder,
  validate,
  disabled,
}: {
  value: string[]
  onChange: (v: string[]) => void
  placeholder?: string
  validate?: (v: string) => boolean
  disabled?: boolean
}) {
  const [draft, setDraft] = useState('')
  const draftValid = !validate || validate(draft)

  const add = () => {
    const v = draft.trim()
    if (!v || !draftValid) return
    onChange([...value, v])
    setDraft('')
  }

  return (
    <div className="string-list-editor">
      {value.map((item, i) => {
        const bad = validate ? !validate(item) : false
        return (
          <span key={`${item}-${i}`} className={`string-chip ${bad ? 'invalid' : ''}`}>
            {item}
            {!disabled && (
              <button
                type="button"
                className="chip-remove"
                onClick={() => onChange(value.filter((_, j) => j !== i))}
              >
                <X size={12} />
              </button>
            )}
          </span>
        )
      })}
      {!disabled && (
        <span className="string-list-add">
          <input
            type="text"
            value={draft}
            placeholder={placeholder}
            className={draft && !draftValid ? 'invalid' : ''}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                add()
              }
            }}
          />
          <button type="button" className="btn-icon" onClick={add} disabled={!draft.trim() || !draftValid}>
            <Plus size={14} />
          </button>
        </span>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* ProviderModelInput — text input with live model suggestions          */
/* ------------------------------------------------------------------ */
function ProviderModelInput({
  value,
  providerId,
  onChange,
  disabled,
  placeholder,
}: {
  value: string
  providerId: string
  onChange: (v: string) => void
  disabled: boolean
  placeholder?: string
}) {
  const [models, setModels] = useState<string[]>([])
  const pid = providerId.trim()
  // 'env-default' has no /check endpoint (synthetic row) — skip the fetch.
  const canFetch = pid !== '' && pid !== 'env-default'

  useEffect(() => {
    setModels([])
    if (!canFetch) return
    let cancelled = false
    providersApi.check(pid)
      .then((r) => { if (!cancelled) setModels(r.models || []) })
      .catch(() => { /* no providers:check permission or unreachable — plain input */ })
    return () => { cancelled = true }
  }, [pid, canFetch])

  const listId = `pm-${pid || 'default'}`
  return (
    <>
      <input
        type="text"
        list={models.length > 0 ? listId : undefined}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
      />
      {models.length > 0 && (
        <datalist id={listId}>
          {models.map((m) => <option key={m} value={m} />)}
        </datalist>
      )}
    </>
  )
}

/* ------------------------------------------------------------------ */
/* ConfigFieldRenderer — schema-driven settings form                    */
/* ------------------------------------------------------------------ */
export function ConfigFieldRenderer({
  fields,
  config,
  onConfigChange,
  canWrite,
  collections,
  collectionsLoading,
  providers,
  providersLoading,
  tool,
  $t,
}: {
  fields: ConfigField[]
  config: Record<string, unknown>
  onConfigChange: (next: Record<string, unknown>) => void
  canWrite: boolean
  collections: string[]
  collectionsLoading: boolean
  providers?: ProviderEntry[] | null
  providersLoading?: boolean
  tool: ToolEntry
  $t: (key: string) => string
}) {
  // Group consecutive fields sharing a `group` label into sections.
  const sections: { group?: string; fields: ConfigField[] }[] = []
  for (const f of fields) {
    const last = sections[sections.length - 1]
    if (last && last.group === f.group) last.fields.push(f)
    else sections.push({ group: f.group, fields: [f] })
  }

  const labelFor = (f: ConfigField) => {
    const i18nKey = `toolsCfg.${tool.name}.${f.key.split('.').pop()}`
    const translated = $t(i18nKey)
    return translated === i18nKey ? f.label : translated
  }

  const setVal = (key: string, value: unknown) =>
    onConfigChange(configSet(config, key, value))

  const renderField = (f: ConfigField) => {
    const raw = configGet(config, f.key)
    const disabled = !canWrite
    switch (f.type) {
      case 'toggle':
        return (
          <label className={`rc-toggle ${disabled ? 'locked' : ''}`}>
            <input
              type="checkbox"
              checked={raw == null ? Boolean(f.default) : Boolean(raw)}
              onChange={(e) => setVal(f.key, e.target.checked)}
              disabled={disabled}
            />
            <span className="rc-toggle-track"><span className="rc-toggle-knob" /></span>
          </label>
        )
      case 'number':
        return (
          <input
            type="number"
            value={raw == null ? (f.default as number | undefined) ?? '' : Number(raw)}
            min={f.min}
            max={f.max}
            onChange={(e) => setVal(f.key, e.target.value === '' ? null : Number(e.target.value))}
            disabled={disabled}
          />
        )
      case 'select':
        return (
          <select
            value={String(raw ?? f.default ?? '')}
            onChange={(e) => setVal(f.key, e.target.value)}
            disabled={disabled}
          >
            {(f.options ?? []).map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        )
      case 'password': {
        const isSet = raw === SECRET_SENTINEL
        return (
          <span className="secret-field">
            <input
              type="password"
              value={isSet ? '' : String(raw ?? '')}
              placeholder={isSet ? '•••••••• (configured — type to replace)' : (f.placeholder ?? '')}
              onChange={(e) => setVal(f.key, e.target.value)}
              onBlur={(e) => {
                // Empty-after-set means "keep" — restore the sentinel so an
                // accidental blur doesn't wipe the stored secret.
                if (isSet && e.target.value === '') setVal(f.key, SECRET_SENTINEL)
              }}
              disabled={disabled}
              autoComplete="new-password"
            />
            {isSet && !disabled && (
              <button
                type="button"
                className="btn-secondary btn-xs"
                title={$t('toolsCfg.clearSecret')}
                onClick={() => setVal(f.key, '')}
              >
                {$t('toolsCfg.clear')}
              </button>
            )}
          </span>
        )
      }
      case 'textarea': {
        const isSet = raw === SECRET_SENTINEL
        return (
          <>
            <textarea
              rows={4}
              className="code-textarea"
              value={isSet ? '' : String(raw ?? '')}
              placeholder={isSet ? '(configured — paste to replace)' : (f.placeholder ?? '')}
              onChange={(e) => setVal(f.key, e.target.value)}
              onBlur={(e) => {
                if (isSet && e.target.value === '') setVal(f.key, SECRET_SENTINEL)
              }}
              disabled={disabled}
            />
            {isSet && !disabled && (
              <button
                type="button"
                className="btn-secondary btn-xs"
                onClick={() => setVal(f.key, '')}
              >
                {$t('toolsCfg.clear')}
              </button>
            )}
          </>
        )
      }
      case 'json':
        return (
          <textarea
            rows={6}
            className="code-textarea"
            value={raw == null ? '' : JSON.stringify(raw, null, 2)}
            onChange={(e) => {
              try {
                setVal(f.key, JSON.parse(e.target.value))
              } catch { /* keep typing */ }
            }}
            disabled={disabled}
          />
        )
      case 'string_list':
      case 'cidr_list': {
        const list = Array.isArray(raw) ? raw.map(String) : []
        return (
          <StringListEditor
            value={list}
            onChange={(v) => setVal(f.key, v)}
            placeholder={f.placeholder}
            validate={f.type === 'cidr_list' ? isValidCidr : undefined}
            disabled={disabled}
          />
        )
      }
      case 'collections': {
        const list = Array.isArray(raw) ? raw.map(String) : []
        if (collectionsLoading) return <span className="text-muted">Loading collections…</span>
        return (
          <MultiSelect
            options={collections}
            selected={list}
            onChange={(v) => setVal(f.key, v)}
            disabled={disabled}
          />
        )
      }
      case 'provider': {
        if (providersLoading) return <span className="text-muted">Loading providers…</span>
        if (providers == null) {
          // providers:read missing or fetch failed — stay editable via text.
          return (
            <input
              type="text"
              value={String(raw ?? '')}
              placeholder={f.placeholder ?? 'provider id (empty = default)'}
              onChange={(e) => setVal(f.key, e.target.value)}
              disabled={disabled}
            />
          )
        }
        const current = String(raw ?? '')
        const options = providers.map((p) => ({
          value: p.id,
          label: `${p.name}${p.is_default ? ' (default)' : ''}${p.enabled ? '' : ' — disabled'}`,
        }))
        if (current && !options.some((o) => o.value === current)) {
          options.push({ value: current, label: `${current} (unknown)` })
        }
        return (
          <select
            value={current}
            onChange={(e) => setVal(f.key, e.target.value)}
            disabled={disabled}
          >
            <option value="">— default provider —</option>
            {options.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        )
      }
      case 'provider_model': {
        const providerKey = f.provider_key || 'provider_id'
        const pid = String(configGet(config, providerKey) ?? '')
        return (
          <ProviderModelInput
            value={String(raw ?? '')}
            providerId={pid}
            onChange={(v) => setVal(f.key, v)}
            disabled={disabled}
            placeholder={f.placeholder}
          />
        )
      }
      case 'color': {
        const v = String(raw ?? f.default ?? '#000000')
        return (
          <span className="color-field">
            <input
              type="color"
              value={/^#[0-9a-fA-F]{6}$/.test(v) ? v : '#000000'}
              onChange={(e) => setVal(f.key, e.target.value)}
              disabled={disabled}
            />
            <input
              type="text"
              value={v}
              onChange={(e) => setVal(f.key, e.target.value)}
              disabled={disabled}
              className="color-hex"
            />
          </span>
        )
      }
      case 'mcp_servers': {
        const servers = Array.isArray(raw) ? raw : []
        return (
          <McpServersEditor
            servers={servers as Record<string, unknown>[]}
            onChange={(v) => setVal(f.key, v)}
            canWrite={canWrite}
            $t={$t}
          />
        )
      }
      case 'text':
      default:
        return (
          <input
            type="text"
            value={String(raw ?? f.default ?? '')}
            placeholder={f.placeholder}
            onChange={(e) => setVal(f.key, e.target.value)}
            disabled={disabled}
          />
        )
    }
  }

  return (
    <>
      {sections.map((sec, si) => (
        <div key={si} className="cfg-section">
          {sec.group && <div className="cfg-section-title">{sec.group}</div>}
          <div className="form-grid">
            {sec.fields.map((f) => (
              <div
                key={f.key}
                className={`form-group ${['textarea', 'json', 'string_list', 'cidr_list', 'collections', 'mcp_servers'].includes(f.type) ? 'wide' : ''}`}
              >
                <label>
                  {labelFor(f)}
                  {f.required_for_use && <span className="required-star"> *</span>}
                </label>
                {f.help && <small className="field-help">{f.help}</small>}
                {renderField(f)}
              </div>
            ))}
          </div>
        </div>
      ))}
    </>
  )
}
