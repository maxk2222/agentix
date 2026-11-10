import {
  createContext,
  useCallback,
  useContext,
  useMemo,
} from 'react'
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Panel,
  Position,
  SelectionMode,
  useReactFlow,
  type Connection,
  type Edge,
  type Node,
  type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { GitMerge } from 'lucide-react'
import type {
  AgentChainCreatePayload,
  AgentChainEntry,
  AgentChainRunEntry,
  AgentChainTestRunResponse,
  AssistantEntry,
  ToolEntry,
} from '../../api/client'
import { useT } from '../../i18n'
import { useAppPreferences } from '../../ui/AppPreferences'
import { ERROR_PORT_TYPES, FALLBACK_NODE_ICON, NODE_ICON_MAP, nodeTypeOf } from './flowRules'
import type { ValidationIssue } from './flowValidation'
import { layoutFlow } from './autoLayout'
import { EditorSidebar, type NodePreset } from './EditorSidebar'
import { EditorToolbar } from './EditorToolbar'
import { ChainTestPanel } from './ChainTestPanel'

/* ------------------------------------------------------------------ */
/* Warning context: node ids with validation issues (no data mutation)  */
/* ------------------------------------------------------------------ */

const ChainWarnContext = createContext<Map<string, string>>(new Map())

/* Per-node runtime info (test-run status + in-degree for merge badge). */
export type NodeRunStatus = { status: string; chosen_route?: string }
const ChainNodeInfoContext = createContext<{
  status: Map<string, NodeRunStatus>
  inDegree: Map<string, number>
}>({ status: new Map(), inDegree: new Map() })

/* ------------------------------------------------------------------ */
/* Custom Node Component                                                */
/* ------------------------------------------------------------------ */

type RouteDef = { id: string | number; name: string; description?: string; handle?: string }

function RouteRows({
  routes,
  nodeType,
  chosen,
}: {
  routes: RouteDef[]
  nodeType: string
  chosen?: string
}) {
  return (
    <div className="node-routes">
      {routes.map((r) => {
        const handleId = r.handle ?? (nodeType === 'condition' ? String(r.id) : `route-${r.id}`)
        const isChosen = chosen !== undefined && r.name === chosen
        const dimmed = chosen !== undefined && !isChosen
        return (
          <div
            key={r.id}
            className={`node-route${isChosen ? ' chosen' : ''}${dimmed ? ' dimmed' : ''}`}
            title={r.description || r.name}
          >
            <span className="node-route-label">{r.name}</span>
            <Handle
              type="source"
              id={handleId}
              position={Position.Right}
              className="node-route-handle"
              style={{ background: `var(--node-${nodeType})` }}
            />
          </div>
        )
      })}
    </div>
  )
}

/** Red fallback port — visible on every executable node type. */
function ErrorRouteRow({ fired }: { fired?: boolean }) {
  return (
    <div className="node-routes">
      <div
        className={`node-route node-errroute${fired ? ' chosen' : ''}`}
        title="on_error — fires when this node fails"
      >
        <span className="node-route-label">⚠ error</span>
        <Handle
          type="source"
          id="on_error"
          position={Position.Right}
          className="node-route-handle node-error-handle"
        />
      </div>
    </div>
  )
}

function ChainNodeComponent({ id, data, selected }: NodeProps) {
  const warnMsg = useContext(ChainWarnContext).get(id)
  const { status, inDegree } = useContext(ChainNodeInfoContext)
  const nodeType = (data as any)?.nodeType || 'executor'
  const Icon = NODE_ICON_MAP[nodeType] || FALLBACK_NODE_ICON
  const label = (data as any)?.label || nodeType
  const runStatus = status.get(id)
  const isMerge = (inDegree.get(id) || 0) > 1
  const className = [
    'chain-node',
    `chain-node--${nodeType}`,
    selected ? 'selected' : '',
    warnMsg ? 'warn' : '',
    runStatus ? `st-${runStatus.status}` : '',
  ].filter(Boolean).join(' ')
  const handleStyle = {
    background: `var(--node-${nodeType})`,
    border: '2px solid white',
    width: 10,
    height: 10,
  }

  /* Subtitle: type-specific summary line under the label. */
  let subtitle: string | undefined
  switch (nodeType) {
    case 'executor': {
      const kind = (data as any)?.target_kind || 'assistant'
      const target = (data as any)?.target || ''
      subtitle = target ? `${kind} · ${target}` : kind
      break
    }
    case 'assistant':
    case 'agent':
      subtitle = (data as any)?.assistant_name || undefined
      break
    case 'router':
    case 'llm':
      subtitle = (data as any)?.assistant_name || undefined
      break
    case 'condition': {
      const rule = (data as any)?.rule
      if (rule?.op) subtitle = `${rule.op} "${String(rule.value ?? '').slice(0, 24)}"`
      break
    }
    case 'switch': {
      const cases = ((data as any)?.cases as any[]) || []
      subtitle = `${cases.length} cases`
      break
    }
    case 'json':
      subtitle = (data as any)?.assistant_name ? `${(data as any).assistant_name} → json` : 'json'
      break
    case 'tool':
      subtitle = (data as any)?.tool_name || 'tool'
      break
    case 'delay': {
      const ms = (data as any)?.ms
      subtitle = typeof ms === 'number' ? `${ms} ms` : undefined
      break
    }
    default:
      break
  }

  const routes: RouteDef[] =
    nodeType === 'router'
      ? (((data as any)?.routes as RouteDef[]) || [])
      : nodeType === 'switch'
        ? [
            ...(((data as any)?.cases as RouteDef[]) || []),
            { id: 'else', name: 'else', handle: 'else' },
          ]
        : nodeType === 'condition'
          ? [
              { id: 'cond-true', name: 'true' },
              { id: 'cond-false', name: 'false' },
            ]
          : []

  const hasTarget = !['input', 'note'].includes(nodeType)
  const hasSource = !['output', 'note', 'router', 'condition', 'switch'].includes(nodeType)
  const hasErrorPort = ERROR_PORT_TYPES.has(nodeType)

  if (nodeType === 'note') {
    return (
      <div className={className}>
        <div className="node-note-icon"><Icon size={13} /></div>
        <div className="node-note-text">{(data as any)?.text || label}</div>
      </div>
    )
  }

  return (
    <div className={className}>
      {hasTarget && <Handle type="target" position={Position.Left} style={handleStyle} />}
      <div className="node-main">
        <div className="node-icon"><Icon size={14} /></div>
        <div className="node-text">
          <span className="node-label">{label}</span>
          {subtitle && <span className="node-sub" title={subtitle}>{subtitle}</span>}
        </div>
        {isMerge && <GitMerge size={11} className="node-badge-merge" />}
      </div>
      {routes.length > 0 && (
        <RouteRows routes={routes} nodeType={nodeType} chosen={runStatus?.chosen_route} />
      )}
      {hasErrorPort && (
        <ErrorRouteRow fired={runStatus?.status === 'error'} />
      )}
      {warnMsg && <span className="node-warn" title={warnMsg}>!</span>}
      {runStatus && <span className={`node-status node-status--${runStatus.status}`} />}
      {hasSource && <Handle type="source" position={Position.Right} style={handleStyle} />}
    </div>
  )
}

const nodeTypes = {
  chainNode: ChainNodeComponent,
}

/* ------------------------------------------------------------------ */
/* Inner editor (inside ReactFlowProvider)                              */
/* ------------------------------------------------------------------ */

export type ChainEditorProps = {
  draft: AgentChainCreatePayload
  setDraft: (d: AgentChainCreatePayload) => void
  nodes: Node[]
  edges: Edge[]
  onNodesChange: any
  onEdgesChange: any
  onConnect: (conn: Connection) => void
  onReconnect: (oldEdge: Edge, conn: Connection) => void
  onNodeDragStart: () => void
  onSelectionDragStart: () => void
  onNodeClick: (node: Node) => void
  onEdgeClick: (edge: Edge) => void
  onPaneClick: () => void
  isValidConnection: (conn: Edge | Connection) => boolean
  selectedNode?: Node
  selectedEdge?: Edge
  updateNodeData: (key: string, value: unknown) => void
  updateEdgeLabel: (label: string) => void
  onDeleteEdge: () => void
  pruneRouteEdges: (sourceId: string, handles: Set<string>) => void
  assistants: AssistantEntry[]
  chains: AgentChainEntry[]
  tools: ToolEntry[]
  presets: NodePreset[]
  onAddPresetNode: (preset: NodePreset, position?: { x: number; y: number }) => void
  onDeletePreset: (name: string) => void
  onSavePreset?: (node: Node) => void
  onOpenChain?: (chainId: string) => void
  addNodeAt: (type: string, position?: { x: number; y: number }) => void
  issues: ValidationIssue[]
  serverErrors: string[]
  onValidate: () => void
  onExport: () => void
  onClose: () => void
  canWrite: boolean
  isNew: boolean
  /* toolbar */
  canUndo: boolean
  canRedo: boolean
  hasSelection: boolean
  hasClipboard: boolean
  templateNames: string[]
  onUndo: () => void
  onRedo: () => void
  onCopy: () => void
  onPaste: () => void
  onDuplicate: () => void
  onApplyNodes: (nodes: Node[]) => void
  onApplyTemplate: (name: string) => void
  /* test panel */
  testOpen: boolean
  onToggleTest: () => void
  testRunning: boolean
  testResult: AgentChainTestRunResponse | null
  testError: string | null
  onRunTest: (input: string) => void
  onCancelTest: () => void
  /* run history */
  runsOpen?: boolean
  onToggleRuns?: () => void
  runs?: AgentChainRunEntry[]
  runsLoading?: boolean
  onReplayRun?: (run: AgentChainRunEntry) => void
}

function ChainEditorInner(props: ChainEditorProps) {
  const {
    draft,
    setDraft,
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    onReconnect,
    onNodeDragStart,
    onSelectionDragStart,
    onNodeClick,
    onEdgeClick,
    onPaneClick,
    isValidConnection,
    selectedNode,
    selectedEdge,
    updateNodeData,
    updateEdgeLabel,
    onDeleteEdge,
    pruneRouteEdges,
    assistants,
    chains,
    tools,
    presets,
    onAddPresetNode,
    onDeletePreset,
    onSavePreset,
    onOpenChain,
    addNodeAt,
    issues,
    serverErrors,
    onValidate,
    onExport,
    onClose,
    canWrite,
    isNew,
    canUndo,
    canRedo,
    hasSelection,
    hasClipboard,
    templateNames,
    onUndo,
    onRedo,
    onCopy,
    onPaste,
    onDuplicate,
    onApplyNodes,
    onApplyTemplate,
    testOpen,
    onToggleTest,
    testRunning,
    testResult,
    testError,
    onRunTest,
    onCancelTest,
    runsOpen,
    onToggleRuns,
    runs,
    runsLoading,
    onReplayRun,
  } = props

  const { language, theme } = useAppPreferences()
  const $t = useT(language)
  const { screenToFlowPosition, setCenter, fitView } = useReactFlow()

  const warnedIds = useMemo(() => {
    const map = new Map<string, string>()
    for (const i of issues) {
      if (!i.nodeId || i.level !== 'error') continue
      map.set(i.nodeId, map.has(i.nodeId) ? `${map.get(i.nodeId)}\n${i.message}` : i.message)
    }
    return map
  }, [issues])

  /* Runtime info for node rendering: test-run statuses + fan-in counts. */
  const nodeInfo = useMemo(() => {
    const statusMap = new Map<string, NodeRunStatus>()
    for (const [nid, st] of Object.entries(testResult?.node_status || {})) {
      statusMap.set(nid, st as NodeRunStatus)
    }
    const inDegree = new Map<string, number>()
    for (const e of edges) {
      inDegree.set(e.target, (inDegree.get(e.target) || 0) + 1)
    }
    return { status: statusMap, inDegree }
  }, [testResult, edges])

  /* Focus helper used by validation list + test panel. */
  const focusNode = useCallback(
    (id: string) => {
      const n = nodes.find((x) => x.id === id)
      if (!n) return
      const w = n.measured?.width ?? 170
      const h = n.measured?.height ?? 44
      setCenter(n.position.x + w / 2, n.position.y + h / 2, { zoom: 1.15, duration: 300 })
      onNodeClick(n)
    },
    [nodes, setCenter, onNodeClick],
  )

  /* Drag & drop from palette */
  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      if (!canWrite) return
      const pos = screenToFlowPosition({ x: e.clientX, y: e.clientY })
      const presetJson = e.dataTransfer.getData('application/chain-preset')
      if (presetJson) {
        try {
          onAddPresetNode(JSON.parse(presetJson) as NodePreset, pos)
        } catch {
          // malformed preset payload — ignore
        }
        return
      }
      const type = e.dataTransfer.getData('application/chain-node')
      if (!type) return
      addNodeAt(type, pos)
    },
    [addNodeAt, onAddPresetNode, screenToFlowPosition, canWrite],
  )
  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  /* Auto layout — computed here so we can fitView afterwards. */
  const runAutoLayout = useCallback(() => {
    onApplyNodes(layoutFlow(nodes, edges))
    window.requestAnimationFrame(() => fitView({ duration: 300, padding: 0.2 }))
  }, [nodes, edges, onApplyNodes, fitView])

  /* MiniMap colors resolved from theme CSS vars */
  const miniMapColors = useMemo(() => {
    const cs = getComputedStyle(document.documentElement)
    const g = (v: string) => cs.getPropertyValue(v).trim() || undefined
    return {
      byType: {
        input: g('--node-input'),
        executor: g('--node-executor'),
        router: g('--node-router'),
        condition: g('--node-condition'),
        switch: g('--node-switch'),
        llm: g('--node-llm'),
        tool: g('--node-tool'),
        json: g('--node-json'),
        delay: g('--node-delay'),
        note: g('--node-note'),
        assistant: g('--node-assistant'),
        agent: g('--node-agent'),
        transform: g('--node-transform'),
        merge: g('--node-merge'),
        parallel: g('--node-parallel'),
        output: g('--node-output'),
      } as Record<string, string | undefined>,
      bg: g('--card'),
      mask: g('--bg'),
      fallback: g('--primary') || '#888',
    }
  }, [theme])

  /* Error edges render red — derived, never persisted. */
  const styledEdges = useMemo(
    () =>
      edges.map((e) =>
        (e as any).sourceHandle === 'on_error'
          ? { ...e, className: 'chain-edge-error' }
          : e,
      ),
    [edges],
  )

  const onlyAnchors = useMemo(
    () => nodes.every((n) => nodeTypeOf(n) === 'input' || nodeTypeOf(n) === 'output'),
    [nodes],
  )

  return (
    <div className="chain-editor">
      <EditorSidebar
        draft={draft}
        setDraft={setDraft}
        assistants={assistants}
        canWrite={canWrite}
        selectedNode={selectedNode}
        selectedEdge={selectedEdge}
        updateNodeData={updateNodeData}
        updateEdgeLabel={updateEdgeLabel}
        onDeleteEdge={onDeleteEdge}
        pruneRouteEdges={pruneRouteEdges}
        chains={chains}
        tools={tools}
        presets={presets}
        onAddPreset={(preset) => onAddPresetNode(preset)}
        onDeletePreset={onDeletePreset}
        onSavePreset={onSavePreset}
        onOpenChain={onOpenChain}
        issues={issues}
        serverErrors={serverErrors}
        onFocusNode={focusNode}
        onAddNode={(type) => addNodeAt(type)}
        $t={$t}
      />

      <div className="chain-canvas" onDrop={onDrop} onDragOver={onDragOver}>
        <ChainWarnContext.Provider value={warnedIds}>
          <ChainNodeInfoContext.Provider value={nodeInfo}>
          <ReactFlow
            nodes={nodes}
            edges={styledEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onReconnect={onReconnect}
            onNodeClick={(_e, node) => onNodeClick(node)}
            onEdgeClick={(_e, edge) => onEdgeClick(edge)}
            onPaneClick={onPaneClick}
            onNodeDragStart={onNodeDragStart}
            onSelectionDragStart={onSelectionDragStart}
            isValidConnection={isValidConnection}
            nodeTypes={nodeTypes}
            colorMode={theme}
            selectionOnDrag={canWrite}
            panOnDrag={canWrite ? [1, 2] : true}
            selectionMode={SelectionMode.Partial}
            deleteKeyCode={canWrite ? ['Backspace', 'Delete'] : null}
            nodesDraggable={canWrite}
            nodesConnectable={canWrite}
            edgesReconnectable={canWrite}
            snapToGrid
            snapGrid={[16, 16]}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            defaultEdgeOptions={{
              type: 'smoothstep',
              markerEnd: { type: MarkerType.ArrowClosed },
            }}
            attributionPosition="bottom-left"
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
            <Controls />
            <MiniMap
              pannable
              zoomable
              bgColor={miniMapColors.bg}
              maskColor={miniMapColors.mask}
              nodeColor={(n: Node) => miniMapColors.byType[nodeTypeOf(n)] || miniMapColors.fallback}
            />
            <Panel position="top-center">
              <EditorToolbar
                $t={$t}
                canWrite={canWrite}
                isNew={isNew}
                canUndo={canUndo}
                canRedo={canRedo}
                hasSelection={hasSelection}
                hasClipboard={hasClipboard}
                templateNames={templateNames}
                testOpen={testOpen}
                onValidate={onValidate}
                onUndo={onUndo}
                onRedo={onRedo}
                onCopy={onCopy}
                onPaste={onPaste}
                onDuplicate={onDuplicate}
                onAutoLayout={runAutoLayout}
                onApplyTemplate={onApplyTemplate}
                onToggleTest={onToggleTest}
                runsOpen={runsOpen}
                onToggleRuns={onToggleRuns}
                onExport={onExport}
                onClose={onClose}
              />
            </Panel>
          </ReactFlow>
          </ChainNodeInfoContext.Provider>
        </ChainWarnContext.Provider>

        {onlyAnchors && canWrite && (
          <div className="chain-canvas-hint">{$t('agentChains.canvasHint')}</div>
        )}

        {testOpen && canWrite && (
          <ChainTestPanel
            nodes={nodes}
            running={testRunning}
            result={testResult}
            error={testError}
            onRun={onRunTest}
            onCancel={onCancelTest}
            onClose={onToggleTest}
            onFocusNode={focusNode}
            $t={$t}
          />
        )}

        {runsOpen && (
          <div className="chain-runs-panel">
            <div className="chain-runs-head">
              <span>{$t('agentChains.runs.title')}</span>
              <button type="button" className="chain-tool-btn" onClick={onToggleRuns} title={$t('agentChains.cancel')}>×</button>
            </div>
            {runsLoading ? (
              <div className="loading">{$t('common.loading')}</div>
            ) : !runs || runs.length === 0 ? (
              <div className="text-muted" style={{ padding: 10 }}>{$t('agentChains.runs.empty')}</div>
            ) : (
              <ul className="chain-runs-list">
                {runs.map((r) => (
                  <li key={r.id}>
                    <button
                      type="button"
                      className="chain-run-row"
                      onClick={() => onReplayRun?.(r)}
                      title={$t('agentChains.runs.replay')}
                    >
                      <span className={`node-status node-status--${r.status}`} />
                      <span className="chain-run-time">
                        {r.started_at ? new Date(r.started_at).toLocaleString() : '—'}
                      </span>
                      <span className="chain-run-meta">
                        {r.source} · {r.duration_ms}ms
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Public shell — provides the ReactFlow context                        */
/* ------------------------------------------------------------------ */

export function ChainEditor(props: ChainEditorProps) {
  return (
    <ReactFlowProvider>
      <ChainEditorInner {...props} />
    </ReactFlowProvider>
  )
}
