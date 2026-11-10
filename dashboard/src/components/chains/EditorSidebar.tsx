import { useMemo, useState, type DragEvent } from 'react'
import type { Edge, Node } from '@xyflow/react'
import { AlertCircle, AlertTriangle, ExternalLink, Plus, Search, Trash2 } from 'lucide-react'
import type { AgentChainCreatePayload, AgentChainEntry, AssistantEntry, ToolEntry } from '../../api/client'
import {
  ADDABLE_NODE_TYPES,
  FALLBACK_NODE_ICON,
  NODE_ICON_MAP,
  TEMPLATE_INPUT_TYPES,
  nodeTypeOf,
} from './flowRules'
import type { ValidationIssue } from './flowValidation'

const CONDITION_OPS = ['contains', 'not_contains', 'equals', 'starts_with', 'ends_with', 'regex'] as const

export type NodePreset = { name: string; nodeType: string; data: Record<string, unknown> }

function AssistantSelect({
  value,
  assistants,
  agentsOnly,
  disabled,
  onChange,
}: {
  value: string
  assistants: AssistantEntry[]
  agentsOnly?: boolean
  disabled?: boolean
  onChange: (v: string) => void
}) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
      <option value="">—</option>
      {assistants
        .filter((a) => (agentsOnly ? a.agent_enabled === true : !a.agent_enabled))
        .map((a) => (
          <option key={a.name} value={a.name}>
            {a.name}
          </option>
        ))}
    </select>
  )
}

function ModelSelect({
  value,
  assistants,
  disabled,
  onChange,
}: {
  value: string
  assistants: AssistantEntry[]
  disabled?: boolean
  onChange: (v: string) => void
}) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
      <option value="">—</option>
      {assistants.map((a) => (
        <option key={a.name} value={a.name}>
          {a.name}
        </option>
      ))}
    </select>
  )
}

function RouterRoutesEditor({
  node,
  canWrite,
  updateNodeData,
  pruneRouteEdges,
  $t,
}: {
  node: Node
  canWrite: boolean
  updateNodeData: (key: string, value: unknown) => void
  pruneRouteEdges: (sourceId: string, handles: Set<string>) => void
  $t: (k: string) => string
}) {
  const routes = ((node.data?.routes as any[]) || []) as Array<{
    id: string | number
    name: string
    description?: string
  }>
  const fallback = String(node.data?.fallback_route || '')

  const setRoutes = (next: typeof routes) => updateNodeData('routes', next)

  const updateRoute = (idx: number, patch: Partial<(typeof routes)[number]>) => {
    setRoutes(routes.map((r, i) => (i === idx ? { ...r, ...patch } : r)))
  }

  const addRoute = () => {
    const nextId = String(Math.max(0, ...routes.map((r) => Number(r.id) || 0)) + 1)
    setRoutes([...routes, { id: nextId, name: `route_${nextId}`, description: '' }])
  }

  const removeRoute = (idx: number) => {
    const removed = routes[idx]
    setRoutes(routes.filter((_, i) => i !== idx))
    pruneRouteEdges(node.id, new Set([`route-${removed.id}`]))
    if (fallback === removed.name) updateNodeData('fallback_route', '')
  }

  return (
    <div className="route-editor">
      <label>{$t('agentChains.inspector.routes')}</label>
      {routes.map((r, i) => (
        <div key={r.id} className="route-editor-row">
          <input
            type="text"
            className="route-name"
            value={r.name}
            onChange={(e) => updateRoute(i, { name: e.target.value })}
            disabled={!canWrite}
            placeholder={$t('agentChains.inspector.routeName')}
          />
          <input
            type="text"
            className="route-desc"
            value={r.description || ''}
            onChange={(e) => updateRoute(i, { description: e.target.value })}
            disabled={!canWrite}
            placeholder={$t('agentChains.inspector.routeDesc')}
          />
          {canWrite && (
            <button
              type="button"
              className="route-del"
              onClick={() => removeRoute(i)}
              disabled={routes.length <= 1}
              title={$t('agentChains.inspector.routeDelete')}
            >
              <Trash2 size={12} />
            </button>
          )}
        </div>
      ))}
      {canWrite && (
        <button type="button" className="btn-secondary route-add" onClick={addRoute}>
          <Plus size={12} /> {$t('agentChains.inspector.routeAdd')}
        </button>
      )}
      <div>
        <label>{$t('agentChains.inspector.fallbackRoute')}</label>
        <select
          value={fallback}
          onChange={(e) => updateNodeData('fallback_route', e.target.value)}
          disabled={!canWrite}
        >
          <option value="">{$t('agentChains.inspector.fallbackFirst')}</option>
          {routes.map((r) => (
            <option key={r.id} value={r.name}>
              {r.name}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}

function SwitchCasesEditor({
  node,
  canWrite,
  updateNodeData,
  pruneRouteEdges,
  $t,
}: {
  node: Node
  canWrite: boolean
  updateNodeData: (key: string, value: unknown) => void
  pruneRouteEdges: (sourceId: string, handles: Set<string>) => void
  $t: (k: string) => string
}) {
  const cases = ((node.data?.cases as any[]) || []) as Array<{
    id: string | number
    name: string
    op: string
    value: string
  }>
  const setCases = (next: typeof cases) => updateNodeData('cases', next)
  const updateCase = (idx: number, patch: Partial<(typeof cases)[number]>) =>
    setCases(cases.map((c, i) => (i === idx ? { ...c, ...patch } : c)))
  const addCase = () => {
    const nextId = String(Math.max(0, ...cases.map((c) => Number(c.id) || 0)) + 1)
    setCases([...cases, { id: nextId, name: `case_${nextId}`, op: 'contains', value: '' }])
  }
  const removeCase = (idx: number) => {
    const removed = cases[idx]
    setCases(cases.filter((_, i) => i !== idx))
    pruneRouteEdges(node.id, new Set([`route-${removed.id}`]))
  }

  return (
    <div className="route-editor">
      <label>{$t('agentChains.inspector.cases')}</label>
      {cases.map((c, i) => (
        <div key={c.id} className="switch-case-row">
          <input
            type="text"
            className="route-name"
            value={c.name}
            onChange={(e) => updateCase(i, { name: e.target.value })}
            disabled={!canWrite}
            placeholder={$t('agentChains.inspector.caseName')}
          />
          <select
            className="switch-case-op"
            value={c.op || 'contains'}
            onChange={(e) => updateCase(i, { op: e.target.value })}
            disabled={!canWrite}
          >
            {CONDITION_OPS.map((op) => (
              <option key={op} value={op}>
                {$t(`agentChains.condOps.${op}`)}
              </option>
            ))}
          </select>
          <input
            type="text"
            className="route-desc"
            value={c.value ?? ''}
            onChange={(e) => updateCase(i, { value: e.target.value })}
            disabled={!canWrite}
            placeholder={$t('agentChains.inspector.caseValue')}
          />
          {canWrite && (
            <button
              type="button"
              className="route-del"
              onClick={() => removeCase(i)}
              disabled={cases.length <= 1}
              title={$t('agentChains.inspector.routeDelete')}
            >
              <Trash2 size={12} />
            </button>
          )}
        </div>
      ))}
      {canWrite && (
        <button type="button" className="btn-secondary route-add" onClick={addCase}>
          <Plus size={12} /> {$t('agentChains.inspector.caseAdd')}
        </button>
      )}
      <div className="field-help">{$t('agentChains.inspector.switchElseHelp')}</div>
    </div>
  )
}

function NodeInspector({
  node,
  assistants,
  chains,
  tools,
  canWrite,
  updateNodeData,
  pruneRouteEdges,
  onOpenChain,
  onSavePreset,
  $t,
}: {
  node: Node
  assistants: AssistantEntry[]
  chains: AgentChainEntry[]
  tools: ToolEntry[]
  canWrite: boolean
  updateNodeData: (key: string, value: unknown) => void
  pruneRouteEdges: (sourceId: string, handles: Set<string>) => void
  onOpenChain?: (chainId: string) => void
  onSavePreset?: () => void
  $t: (k: string) => string
}) {
  const nodeType = nodeTypeOf(node)
  const Icon = NODE_ICON_MAP[nodeType] || FALLBACK_NODE_ICON
  const colorVar = `var(--node-${nodeType})`

  return (
    <div className="node-inspector" style={{ border: `2px solid ${colorVar}` }}>
      <div className="node-inspector-head" style={{ background: colorVar }}>
        <Icon size={14} />
        {$t('agentChains.nodeInspector')}
      </div>
      <div className="node-inspector-body">
        <div>
          <label>{$t('agentChains.label')}</label>
          <input
            type="text"
            value={String(node.data?.label || '')}
            onChange={(e) => updateNodeData('label', e.target.value)}
            disabled={!canWrite}
          />
        </div>

        {(nodeType === 'assistant' || nodeType === 'agent') && (
          <div>
            <label>{$t('agentChains.assistant')}</label>
            <AssistantSelect
              value={String(node.data?.assistant_name || '')}
              assistants={assistants}
              agentsOnly={nodeType === 'agent'}
              disabled={!canWrite}
              onChange={(v) => updateNodeData('assistant_name', v)}
            />
          </div>
        )}

        {nodeType === 'executor' && (
          <>
            <div>
              <label>{$t('agentChains.inspector.targetKind')}</label>
              <select
                value={String(node.data?.target_kind || 'assistant')}
                onChange={(e) => updateNodeData('target_kind', e.target.value)}
                disabled={!canWrite}
              >
                <option value="assistant">{$t('agentChains.inspector.kindAssistant')}</option>
                <option value="agent">{$t('agentChains.inspector.kindAgent')}</option>
                <option value="chain">{$t('agentChains.inspector.kindChain')}</option>
              </select>
            </div>
            <div>
              <label>{$t('agentChains.inspector.target')}</label>
              {(node.data?.target_kind || 'assistant') === 'chain' ? (
                <div className="chain-target-row">
                  <select
                    value={String(node.data?.target || '')}
                    onChange={(e) => updateNodeData('target', e.target.value)}
                    disabled={!canWrite}
                  >
                    <option value="">—</option>
                    {chains.map((c) => (
                      <option key={c.id} value={c.name}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                  {typeof node.data?.target === 'string' && node.data.target && onOpenChain && (() => {
                    const target = chains.find((c) => c.name === node.data?.target || c.id === node.data?.target)
                    return target ? (
                      <button
                        type="button"
                        className="btn-secondary chain-open-btn"
                        title={$t('agentChains.inspector.openChain')}
                        onClick={() => onOpenChain(target.id)}
                      >
                        <ExternalLink size={12} />
                      </button>
                    ) : null
                  })()}
                </div>
              ) : (
                <AssistantSelect
                  value={String(node.data?.target || '')}
                  assistants={assistants}
                  agentsOnly={node.data?.target_kind === 'agent'}
                  disabled={!canWrite}
                  onChange={(v) => updateNodeData('target', v)}
                />
              )}
            </div>
          </>
        )}

        {nodeType === 'switch' && (
          <SwitchCasesEditor
            node={node}
            canWrite={canWrite}
            updateNodeData={updateNodeData}
            pruneRouteEdges={pruneRouteEdges}
            $t={$t}
          />
        )}

        {nodeType === 'delay' && (
          <div>
            <label>{$t('agentChains.inspector.delayMs')}</label>
            <input
              type="number"
              min={0}
              step={100}
              value={typeof node.data?.ms === 'number' ? node.data.ms : 0}
              onChange={(e) => updateNodeData('ms', Math.max(0, Number(e.target.value) || 0))}
              disabled={!canWrite}
            />
            <div className="field-help">{$t('agentChains.inspector.delayHelp')}</div>
          </div>
        )}

        {nodeType === 'tool' && (
          <>
            <div>
              <label>{$t('agentChains.inspector.toolName')}</label>
              <select
                value={String(node.data?.tool_name || '')}
                onChange={(e) => updateNodeData('tool_name', e.target.value)}
                disabled={!canWrite}
              >
                <option value="">—</option>
                {tools.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.id}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label>{$t('agentChains.inspector.toolParams')}</label>
              <textarea
                rows={4}
                value={
                  typeof node.data?.params === 'string'
                    ? node.data.params
                    : JSON.stringify(node.data?.params || {}, null, 2)
                }
                onChange={(e) => updateNodeData('params', e.target.value)}
                disabled={!canWrite}
                placeholder='{"query": "{{input}}"}'
              />
              <div className="field-help">{$t('agentChains.inspector.toolParamsHelp')}</div>
            </div>
            <div className="chain-flag" style={{ marginTop: 4 }}>
              <label>
                <input
                  type="checkbox"
                  checked={Boolean(node.data?.pass_input !== false)}
                  onChange={(e) => updateNodeData('pass_input', e.target.checked)}
                  disabled={!canWrite}
                />
                <span>{$t('agentChains.inspector.toolPassInput')}</span>
              </label>
            </div>
            <div className="chain-flag">
              <label>
                <input
                  type="checkbox"
                  checked={Boolean(node.data?.allow_side_effects)}
                  onChange={(e) => updateNodeData('allow_side_effects', e.target.checked)}
                  disabled={!canWrite}
                />
                <span>{$t('agentChains.inspector.toolSideEffects')}</span>
              </label>
            </div>
            <div>
              <label>{$t('agentChains.inspector.toolTimeout')}</label>
              <input
                type="number"
                min={1}
                value={typeof node.data?.timeout_s === 'number' ? node.data.timeout_s : 60}
                onChange={(e) => updateNodeData('timeout_s', Math.max(1, Number(e.target.value) || 60))}
                disabled={!canWrite}
              />
            </div>
          </>
        )}

        {nodeType === 'json' && (
          <>
            <div>
              <label>{$t('agentChains.inspector.promptTemplate')}</label>
              <textarea
                rows={3}
                value={String(node.data?.prompt_template || '')}
                onChange={(e) => updateNodeData('prompt_template', e.target.value)}
                disabled={!canWrite}
                placeholder="{{input}}"
              />
            </div>
            <div>
              <label>{$t('agentChains.inspector.jsonSchema')}</label>
              <textarea
                rows={4}
                value={
                  typeof node.data?.json_schema === 'string'
                    ? node.data.json_schema
                    : node.data?.json_schema
                      ? JSON.stringify(node.data.json_schema, null, 2)
                      : ''
                }
                onChange={(e) => updateNodeData('json_schema', e.target.value)}
                disabled={!canWrite}
                placeholder='{"type":"object","required":["answer"]}'
              />
              <div className="field-help">{$t('agentChains.inspector.jsonSchemaHelp')}</div>
            </div>
            <div>
              <label>{$t('agentChains.inspector.maxRetries')}</label>
              <input
                type="number"
                min={0}
                max={5}
                value={typeof node.data?.max_retries === 'number' ? node.data.max_retries : 1}
                onChange={(e) =>
                  updateNodeData('max_retries', Math.min(5, Math.max(0, Number(e.target.value) || 0)))
                }
                disabled={!canWrite}
              />
            </div>
          </>
        )}

        {(nodeType === 'router' || nodeType === 'llm' || nodeType === 'json') && (
          <div>
            <label>{nodeType === 'router' ? $t('agentChains.inspector.routerModel') : $t('agentChains.inspector.llmModel')}</label>
            <ModelSelect
              value={String(node.data?.assistant_name || '')}
              assistants={assistants}
              disabled={!canWrite}
              onChange={(v) => updateNodeData('assistant_name', v)}
            />
          </div>
        )}

        {nodeType === 'router' && (
          <RouterRoutesEditor
            node={node}
            canWrite={canWrite}
            updateNodeData={updateNodeData}
            pruneRouteEdges={pruneRouteEdges}
            $t={$t}
          />
        )}

        {nodeType === 'condition' && (
          <>
            <div>
              <label>{$t('agentChains.inspector.condOp')}</label>
              <select
                value={String((node.data?.rule as any)?.op || 'contains')}
                onChange={(e) =>
                  updateNodeData('rule', {
                    ...((node.data?.rule as object) || {}),
                    op: e.target.value,
                  })
                }
                disabled={!canWrite}
              >
                {CONDITION_OPS.map((op) => (
                  <option key={op} value={op}>
                    {$t(`agentChains.condOps.${op}`)}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label>{$t('agentChains.inspector.condValue')}</label>
              <input
                type="text"
                value={String((node.data?.rule as any)?.value ?? '')}
                onChange={(e) =>
                  updateNodeData('rule', {
                    ...((node.data?.rule as object) || {}),
                    value: e.target.value,
                  })
                }
                disabled={!canWrite}
              />
            </div>
          </>
        )}

        {nodeType === 'llm' && (
          <div>
            <label>{$t('agentChains.inspector.promptTemplate')}</label>
            <textarea
              rows={4}
              value={String(node.data?.prompt_template || '')}
              onChange={(e) => updateNodeData('prompt_template', e.target.value)}
              disabled={!canWrite}
              placeholder="{{input}}"
            />
            <div className="field-help">{$t('agentChains.inspector.promptTemplateHelp')}</div>
          </div>
        )}

        {nodeType === 'transform' && (
          <div>
            <label>{$t('agentChains.inspector.template')}</label>
            <textarea
              rows={4}
              value={String(node.data?.template || '')}
              onChange={(e) => updateNodeData('template', e.target.value)}
              disabled={!canWrite}
              placeholder="{{input}}"
            />
            <div className="field-help">{$t('agentChains.inspector.templateHelp')}</div>
          </div>
        )}

        {nodeType === 'note' && (
          <div>
            <label>{$t('agentChains.inspector.noteText')}</label>
            <textarea
              rows={5}
              value={String(node.data?.text || '')}
              onChange={(e) => updateNodeData('text', e.target.value)}
              disabled={!canWrite}
              placeholder={$t('agentChains.inspector.noteTextPlaceholder')}
            />
          </div>
        )}

        {nodeType === 'merge' && <div className="field-help">{$t('agentChains.inspector.mergeInfo')}</div>}
        {nodeType === 'parallel' && <div className="field-help">{$t('agentChains.inspector.parallelInfo')}</div>}

        {TEMPLATE_INPUT_TYPES.has(nodeType) && (
          <div>
            <label>{$t('agentChains.inspector.inputTemplate')}</label>
            <textarea
              rows={2}
              value={String(node.data?.input_template || '')}
              onChange={(e) => updateNodeData('input_template', e.target.value)}
              disabled={!canWrite}
              placeholder="{{input}}"
            />
            <div className="field-help">{$t('agentChains.inspector.inputTemplateHelp')}</div>
          </div>
        )}

        {!['input', 'output', 'note'].includes(nodeType) && (
          <div>
            <label>{$t('agentChains.inspector.mergeMode')}</label>
            <select
              value={String(node.data?.merge_mode || 'all')}
              onChange={(e) =>
                updateNodeData('merge_mode', e.target.value === 'all' ? undefined : e.target.value)
              }
              disabled={!canWrite}
            >
              <option value="all">{$t('agentChains.inspector.mergeModeAll')}</option>
              <option value="first">{$t('agentChains.inspector.mergeModeFirst')}</option>
            </select>
            <div className="field-help">{$t('agentChains.inspector.mergeModeHelp')}</div>
          </div>
        )}

        {['assistant', 'agent', 'executor', 'transform', 'router', 'llm', 'condition', 'switch', 'json', 'tool', 'delay'].includes(nodeType) && (
          <div>
            <label>{$t('agentChains.inspector.note')}</label>
            <textarea
              rows={2}
              value={String(node.data?.note || '')}
              onChange={(e) => updateNodeData('note', e.target.value)}
              disabled={!canWrite}
              placeholder={$t('agentChains.inspector.notePlaceholder')}
            />
          </div>
        )}

        {canWrite && onSavePreset && !['input', 'output'].includes(nodeType) && (
          <button type="button" className="btn-secondary preset-save-btn" onClick={onSavePreset}>
            <Plus size={12} /> {$t('agentChains.inspector.savePreset')}
          </button>
        )}
      </div>
    </div>
  )
}

function EdgeInspector({
  edge,
  canWrite,
  updateEdgeLabel,
  onDelete,
  $t,
}: {
  edge: Edge
  canWrite: boolean
  updateEdgeLabel: (label: string) => void
  onDelete: () => void
  $t: (k: string) => string
}) {
  return (
    <div className="node-inspector" style={{ border: '2px solid var(--info)' }}>
      <div className="node-inspector-head" style={{ background: 'var(--info)' }}>
        {$t('agentChains.inspector.edgeTitle')}
      </div>
      <div className="node-inspector-body">
        <div>
          <label>{$t('agentChains.inspector.edgeLabel')}</label>
          <input
            type="text"
            value={String(edge.label || '')}
            onChange={(e) => updateEdgeLabel(e.target.value)}
            disabled={!canWrite}
            placeholder={$t('agentChains.inspector.edgeLabelPlaceholder')}
          />
        </div>
        <div className="text-muted">
          {edge.source} → {edge.target}
        </div>
        {canWrite && (
          <button type="button" className="btn-secondary chain-edge-delete" onClick={onDelete}>
            <Trash2 size={13} /> {$t('agentChains.inspector.deleteEdge')}
          </button>
        )}
      </div>
    </div>
  )
}

export function EditorSidebar({
  draft,
  setDraft,
  assistants,
  chains,
  tools,
  canWrite,
  selectedNode,
  selectedEdge,
  updateNodeData,
  updateEdgeLabel,
  onDeleteEdge,
  pruneRouteEdges,
  issues,
  serverErrors,
  onFocusNode,
  onAddNode,
  onAddPreset,
  presets,
  onDeletePreset,
  onOpenChain,
  onSavePreset,
  $t,
}: {
  draft: AgentChainCreatePayload
  setDraft: (d: AgentChainCreatePayload) => void
  assistants: AssistantEntry[]
  chains: AgentChainEntry[]
  tools: ToolEntry[]
  canWrite: boolean
  selectedNode?: Node
  selectedEdge?: Edge
  updateNodeData: (key: string, value: unknown) => void
  updateEdgeLabel: (label: string) => void
  onDeleteEdge: () => void
  pruneRouteEdges: (sourceId: string, handles: Set<string>) => void
  issues: ValidationIssue[]
  serverErrors: string[]
  onFocusNode: (id: string) => void
  onAddNode: (type: string) => void
  onAddPreset: (preset: NodePreset) => void
  presets: NodePreset[]
  onDeletePreset: (name: string) => void
  onOpenChain?: (chainId: string) => void
  onSavePreset?: (node: Node) => void
  $t: (k: string) => string
}) {
  const [paletteQuery, setPaletteQuery] = useState('')

  const onPaletteDragStart = (e: DragEvent, type: string) => {
    e.dataTransfer.setData('application/chain-node', type)
    e.dataTransfer.effectAllowed = 'move'
  }
  const onPresetDragStart = (e: DragEvent, preset: NodePreset) => {
    e.dataTransfer.setData('application/chain-preset', JSON.stringify(preset))
    e.dataTransfer.effectAllowed = 'move'
  }

  const filteredTypes = useMemo(() => {
    const q = paletteQuery.trim().toLowerCase()
    if (!q) return [...ADDABLE_NODE_TYPES]
    return ADDABLE_NODE_TYPES.filter((t) => $t(`agentChains.palette.${t}`).toLowerCase().includes(q) || t.includes(q))
  }, [paletteQuery, $t])

  const filteredPresets = useMemo(() => {
    const q = paletteQuery.trim().toLowerCase()
    if (!q) return presets
    return presets.filter((p) => p.name.toLowerCase().includes(q))
  }, [paletteQuery, presets])

  return (
    <div className="chain-sidebar">
      {/* Chain Meta Card */}
      <div className="chain-meta-card">
        <input
          type="text"
          className="chain-name-input"
          value={draft.name}
          onChange={(e) => setDraft({ ...draft, name: e.target.value })}
          disabled={!canWrite}
          placeholder={$t('agentChains.title')}
        />
        <textarea
          rows={1}
          className="chain-desc-input"
          value={draft.description}
          onChange={(e) => setDraft({ ...draft, description: e.target.value })}
          disabled={!canWrite}
          placeholder={$t('agentChains.description')}
        />
        <div className="chain-flags">
          <label className="chain-flag">
            <input
              type="checkbox"
              checked={draft.enabled}
              onChange={(e) => setDraft({ ...draft, enabled: e.target.checked })}
              disabled={!canWrite}
            />
            <span>{$t('agentChains.enabled')}</span>
          </label>
          <label className="chain-flag">
            <input
              type="checkbox"
              checked={draft.api_accessible}
              onChange={(e) => setDraft({ ...draft, api_accessible: e.target.checked })}
              disabled={!canWrite}
            />
            <span>API</span>
          </label>
        </div>
      </div>

      {/* Node Palette */}
      <div>
        <label className="chain-palette-title">{$t('agentChains.paletteTitle')}</label>
        <div className="field-help chain-palette-hint">{$t('agentChains.paletteHint')}</div>
        <div className="palette-search">
          <Search size={12} />
          <input
            type="text"
            value={paletteQuery}
            onChange={(e) => setPaletteQuery(e.target.value)}
            placeholder={$t('agentChains.paletteSearch')}
          />
        </div>
        <div className="chain-palette-grid">
          {filteredTypes.map((type) => {
            const Icon = NODE_ICON_MAP[type] || FALLBACK_NODE_ICON
            return (
              <button
                key={type}
                type="button"
                className="chain-palette-btn"
                data-type={type}
                draggable={canWrite}
                onDragStart={(e) => onPaletteDragStart(e, type)}
                onClick={() => onAddNode(type)}
                disabled={!canWrite}
                title={$t(`agentChains.palette.${type}`)}
              >
                <Icon size={16} />
                <span>{$t(`agentChains.palette.${type}`)}</span>
              </button>
            )
          })}
        </div>
        {filteredPresets.length > 0 && (
          <div className="chain-preset-list">
            <label className="chain-palette-title">{$t('agentChains.presetsTitle')}</label>
            {filteredPresets.map((p) => {
              const Icon = NODE_ICON_MAP[p.nodeType] || FALLBACK_NODE_ICON
              return (
                <div key={p.name} className="chain-preset-row">
                  <button
                    type="button"
                    className="chain-palette-btn chain-preset-btn"
                    draggable={canWrite}
                    onDragStart={(e) => onPresetDragStart(e, p)}
                    onClick={() => onAddPreset(p)}
                    disabled={!canWrite}
                    title={p.name}
                  >
                    <Icon size={14} />
                    <span>{p.name}</span>
                  </button>
                  {canWrite && (
                    <button
                      type="button"
                      className="route-del"
                      onClick={() => onDeletePreset(p.name)}
                      title={$t('agentChains.presetsDelete')}
                    >
                      <Trash2 size={11} />
                    </button>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Inspector */}
      {selectedNode && (
        <NodeInspector
          node={selectedNode}
          assistants={assistants}
          chains={chains}
          tools={tools}
          canWrite={canWrite}
          updateNodeData={updateNodeData}
          pruneRouteEdges={pruneRouteEdges}
          onOpenChain={onOpenChain}
          onSavePreset={onSavePreset ? () => onSavePreset(selectedNode) : undefined}
          $t={$t}
        />
      )}
      {!selectedNode && selectedEdge && (
        <EdgeInspector
          edge={selectedEdge}
          canWrite={canWrite}
          updateEdgeLabel={updateEdgeLabel}
          onDelete={onDeleteEdge}
          $t={$t}
        />
      )}

      {/* Validation */}
      {(issues.length > 0 || serverErrors.length > 0) && (
        <div className="chain-validation">
          <div className="chain-validation-title">
            <AlertCircle size={12} />
            {$t('agentChains.validationErrors')}
          </div>
          <ul className="chain-issue-list">
            {issues.map((issue, i) => (
              <li key={`i-${i}`}>
                <button
                  type="button"
                  className={`chain-issue chain-issue--${issue.level}`}
                  onClick={() => issue.nodeId && onFocusNode(issue.nodeId)}
                  disabled={!issue.nodeId}
                  title={issue.nodeId ? $t('agentChains.validation.clickToFocus') : undefined}
                >
                  {issue.level === 'warning' ? <AlertTriangle size={11} /> : <AlertCircle size={11} />}
                  <span>{issue.message}</span>
                </button>
              </li>
            ))}
            {serverErrors.map((msg, i) => (
              <li key={`s-${i}`}>
                <span className="chain-issue chain-issue--error">
                  <AlertCircle size={11} />
                  <span>{msg}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
