import { useEffect, useState, useCallback, useMemo } from 'react'
import { tools as toolsApi, invalidateCache, type ToolEntry } from '../api/client'
import { useAuth } from '../api/AuthContext'
import { useT } from '../i18n'
import { useAppPreferences } from '../ui/AppPreferences'
import { ToolCard } from './tools/ToolCard'
import { ToolSettingsDrawer } from './tools/ToolSettingsDrawer'
import { CATEGORY_ORDER, categoryFor, type ToolCategoryId } from './tools/catalog'
import SearchToolbar from './SearchToolbar'
import ConfirmModal from './ConfirmModal'
import { AlertCircle, Plus } from 'lucide-react'

const DEFAULT_TOOL: ToolEntry = {
  id: '',
  name: '',
  description: '',
  enabled: true,
  kind: 'custom',
  tool_schema: {},
  config: {},
  code: '',
  venv_config: {},
  runtime_mode: 'venv',
  created_at: null,
  updated_at: null,
}

const KIND_FILTERS = ['all', 'builtin', 'custom', 'mcp', 'assistant'] as const
type KindFilter = (typeof KIND_FILTERS)[number]

export default function ToolsPage() {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const { hasPermission } = useAuth()
  const canWrite = hasPermission('tools:write')

  const [tools, setTools] = useState<ToolEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [kindFilter, setKindFilter] = useState<KindFilter>('all')
  const [editing, setEditing] = useState<ToolEntry | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<ToolEntry | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await toolsApi.list()
      setTools(data)
      setError(null)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  /* ---- filtering + grouping ---- */

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    return tools.filter((t) => {
      if (kindFilter !== 'all' && t.kind !== kindFilter) return false
      if (!q) return true
      return (
        t.name.toLowerCase().includes(q) ||
        (t.description || '').toLowerCase().includes(q)
      )
    })
  }, [tools, search, kindFilter])

  const groups = useMemo(() => {
    const map = new Map<ToolCategoryId, ToolEntry[]>()
    for (const t of filtered) {
      const cat = categoryFor(t)
      if (!map.has(cat)) map.set(cat, [])
      map.get(cat)!.push(t)
    }
    return CATEGORY_ORDER
      .filter((c) => map.has(c))
      .map((c) => ({ id: c, tools: map.get(c)! }))
  }, [filtered])

  /* ---- actions ---- */

  const handleToggle = async (tool: ToolEntry) => {
    if (!canWrite) return
    try {
      if (tool.enabled) await toolsApi.disable(tool.id)
      else await toolsApi.enable(tool.id)
      invalidateCache('/admin/tools')
      load()
    } catch (err) {
      setError(String(err))
    }
  }

  const handleDelete = async () => {
    if (!canWrite || !deleteTarget) return
    try {
      await toolsApi.delete(deleteTarget.id)
      invalidateCache('/admin/tools')
      setEditing(null)
      setDeleteTarget(null)
      load()
    } catch (err) {
      setError(String(err))
      setDeleteTarget(null)
    }
  }

  const handleSave = async (draft: ToolEntry) => {
    if (!canWrite) return
    const payload = {
      name: draft.name,
      description: draft.description,
      enabled: draft.enabled,
      kind: draft.kind,
      tool_schema: draft.tool_schema,
      config: draft.config,
      code: draft.code || '',
      venv_config: draft.venv_config || {},
      runtime_mode: draft.runtime_mode || 'venv',
    }
    if (!draft.id) await toolsApi.create(payload)
    else await toolsApi.update(draft.id, payload)
    invalidateCache('/admin/tools')
    setEditing(null)
    load()
  }

  const handleDuplicate = (tool: ToolEntry) => {
    setEditing({
      ...tool,
      id: '',
      name: `${tool.name}-copy`,
      enabled: true,
      kind: tool.kind === 'builtin' ? 'custom' : tool.kind,
      created_at: null,
      updated_at: null,
    })
  }

  const handleCreate = () => setEditing({ ...DEFAULT_TOOL })

  if (loading) return <div className="loading">{$t('common.loading')}</div>

  return (
    <div>
      <div className="page-header">
        <h2>{$t('tools.title')}</h2>
        {canWrite && (
          <button className="btn-primary" onClick={handleCreate}>
            <Plus size={14} /> {$t('tools.create')}
          </button>
        )}
      </div>

      <SearchToolbar
        search={search}
        onSearch={setSearch}
        placeholder={$t('tools.search')}
        chips={KIND_FILTERS.map((k) => ({
          id: k,
          label: k === 'all' ? $t('tools.allKinds') : $t(`tools.${k}`) || k,
        }))}
        activeChip={kindFilter}
        onChip={(id) => setKindFilter(id as KindFilter)}
      />

      {error && (
        <div className="error-box">
          <AlertCircle size={14} /> {error}
        </div>
      )}

      {groups.length === 0 ? (
        <div className="text-muted">{$t('tools.empty')}</div>
      ) : (
        groups.map((g) => (
          <section key={g.id} className="card-group">
            <h3 className="card-group-title">{$t(`tools.cat.${g.id}`)}</h3>
            <div className="card-grid">
              {g.tools.map((tool) => (
                <ToolCard
                  key={tool.id || `new-${tool.name}`}
                  tool={tool}
                  canWrite={canWrite}
                  onEdit={() => setEditing(tool)}
                  onDuplicate={() => handleDuplicate(tool)}
                  onDelete={() => setDeleteTarget(tool)}
                  onToggle={() => handleToggle(tool)}
                  $t={$t}
                />
              ))}
            </div>
          </section>
        ))
      )}

      {editing && (
        <ToolSettingsDrawer
          tool={editing}
          canWrite={canWrite}
          onSave={handleSave}
          onClose={() => setEditing(null)}
          $t={$t}
        />
      )}

      <ConfirmModal
        open={deleteTarget !== null}
        title={$t('tools.delete')}
        message={deleteTarget ? `${$t('tools.delete')} "${deleteTarget.name}"?` : ''}
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  )
}
