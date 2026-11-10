import type { ToolEntry } from '../../api/client'

/** Mirrors app/tool_config_schemas.py — the backend ships this shape per tool. */
export type ConfigFieldOption = { value: string; label: string }
export type ConfigField = {
  key: string
  type:
    | 'text' | 'password' | 'number' | 'select' | 'toggle' | 'textarea'
    | 'json' | 'string_list' | 'cidr_list' | 'collections' | 'color'
    | 'mcp_servers'
  label: string
  default?: unknown
  secret?: boolean
  required_for_use?: boolean
  options?: ConfigFieldOption[]
  min?: number
  max?: number
  placeholder?: string
  help?: string
  group?: string
}

export const SECRET_SENTINEL = '__agentix_secret_set__'

/** Category ids — labels live in i18n under tools.cat.<id>. */
export type ToolCategoryId =
  | 'knowledge' | 'web' | 'files' | 'comms' | 'internals' | 'mcp'
  | 'custom' | 'assistants' | 'other'

/** Same taxonomy as docs/tools.md. */
const NAME_CATEGORY: Record<string, ToolCategoryId> = {
  rag_search: 'knowledge',
  collection_manager: 'knowledge',
  agent_memory: 'knowledge',
  long_document_summarizer: 'knowledge',
  web_search: 'web',
  serpapi_search: 'web',
  openalex_search: 'web',
  arxiv_search: 'web',
  wikipedia_search: 'web',
  web_scraper: 'web',
  rss_feed_reader: 'web',
  code_execution: 'files',
  document_generator: 'files',
  agent_workspace: 'files',
  ssh_exec: 'files',
  email_sender: 'comms',
  email_reader: 'comms',
  google_calendar: 'comms',
  webhook_post: 'comms',
  weather_lookup: 'comms',
  agent_todo: 'internals',
  plan_generator: 'internals',
  wait: 'internals',
  spawn_subagent: 'internals',
  wait_subagent: 'internals',
  subagent_status: 'internals',
  nudge_subagent: 'internals',
  assistant_caller: 'internals',
  mcp_servers: 'mcp',
}

export const CATEGORY_ORDER: ToolCategoryId[] = [
  'knowledge', 'web', 'files', 'comms', 'internals', 'mcp', 'custom',
  'assistants', 'other',
]

export function categoryFor(tool: ToolEntry): ToolCategoryId {
  if (tool.kind === 'assistant' || tool.name.startsWith('assistant__')) {
    return 'assistants'
  }
  if (tool.kind === 'mcp' || tool.name.startsWith('mcp__')) return 'mcp'
  if (tool.kind === 'custom') return 'custom'
  return NAME_CATEGORY[tool.name] ?? 'other'
}

/** config[key] resolution with dotted-path support ("theme.accent"). */
export function configGet(config: Record<string, unknown> | undefined, key: string): unknown {
  if (!config) return undefined
  let cur: unknown = config
  for (const part of key.split('.')) {
    if (cur == null || typeof cur !== 'object') return undefined
    cur = (cur as Record<string, unknown>)[part]
  }
  return cur
}

export function configSet(
  config: Record<string, unknown>, key: string, value: unknown,
): Record<string, unknown> {
  const parts = key.split('.')
  if (parts.length === 1) return { ...config, [key]: value }
  const [head, ...rest] = parts
  const child = config[head]
  const base = child && typeof child === 'object' && !Array.isArray(child)
    ? (child as Record<string, unknown>)
    : {}
  return { ...config, [head]: configSet(base, rest.join('.'), value) }
}

/** A tool "needs config" when any required_for_use field is unset/empty. */
export function needsConfig(tool: ToolEntry): boolean {
  const schema = (tool.config_schema as ConfigField[] | null | undefined) ?? []
  return schema.some((f) => {
    if (!f.required_for_use) return false
    const v = configGet(tool.config, f.key)
    if (v === SECRET_SENTINEL) return false
    return v == null || v === '' || (Array.isArray(v) && v.length === 0)
  })
}

/** Count of configured (non-empty) schema fields — for the card meta line. */
export function configuredCount(tool: ToolEntry): { set: number; total: number } {
  const schema = (tool.config_schema as ConfigField[] | null | undefined) ?? []
  const set = schema.filter((f) => {
    const v = configGet(tool.config, f.key)
    if (v === SECRET_SENTINEL) return true
    return v != null && v !== '' && !(Array.isArray(v) && v.length === 0)
  }).length
  return { set, total: schema.length }
}

/** IPv4/IPv6 CIDR sanity check for the cidr_list editor (server re-validates). */
export function isValidCidr(raw: string): boolean {
  const s = raw.trim()
  if (!s) return false
  const [addr, prefix] = s.split('/')
  const v4 = /^(\d{1,3}\.){3}\d{1,3}$/.test(addr)
    && addr.split('.').every((o) => Number(o) <= 255)
  const v6 = addr.includes(':')
  if (!v4 && !v6) return false
  if (prefix === undefined) return true
  const n = Number(prefix)
  if (!Number.isInteger(n)) return false
  return v4 ? n >= 0 && n <= 32 : n >= 0 && n <= 128
}
