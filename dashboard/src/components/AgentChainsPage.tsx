import { useEffect, useState, useCallback, useRef, useMemo } from 'react'
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  reconnectEdge,
  MarkerType,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from '@xyflow/react'
import {
  agentChains,
  config,
  tools as toolsApi,
  type AgentChainEntry,
  type AgentChainCreatePayload,
  type AgentChainRunEntry,
  type AgentChainTestRunResponse,
  type AssistantEntry,
  type ToolEntry,
} from '../api/client'
import { useAuth } from '../api/AuthContext'
import UnsavedChangesBar, { useUnsavedGuard } from './UnsavedChangesBar'
import { useData } from '../api/DataContext'
import { useT } from '../i18n'
import { useAppPreferences } from '../ui/AppPreferences'
import { ChainCard } from './chains/ChainCard'
import { ChainEditor } from './chains/ChainEditor'
import {
  ADDABLE_NODE_TYPES,
  ANCHOR_TYPES,
  defaultDataFor,
  isValidConnection as isValidConn,
  nodeTypeOf,
  sanitizeFlow,
} from './chains/flowRules'
import { validateFlowLocal, type ValidationIssue } from './chains/flowValidation'
import { useChainHistory } from './chains/useChainHistory'
import { useChainShortcuts } from './chains/useChainShortcuts'
import SearchToolbar from './SearchToolbar'
import ConfirmModal from './ConfirmModal'
import PromptModal from './PromptModal'
import { type NodePreset } from './chains/EditorSidebar'
import {
  Plus,
  AlertCircle,
  Workflow,
} from 'lucide-react'

/* ------------------------------------------------------------------ */
/* Templates                                                            */
/* ------------------------------------------------------------------ */

function createEmptyFlow() {
  return {
    nodes: [
      {
        id: 'input-1',
        type: 'chainNode',
        position: { x: 50, y: 200 },
        data: { label: 'Input', nodeType: 'input' },
        deletable: false,
      },
      {
        id: 'output-1',
        type: 'chainNode',
        position: { x: 650, y: 200 },
        data: { label: 'Output', nodeType: 'output' },
        deletable: false,
      },
    ],
    edges: [],
  }
}

const n = (
  id: string,
  nodeType: string,
  x: number,
  y: number,
  label: string,
  extra: Record<string, unknown> = {},
) => ({
  id,
  type: 'chainNode',
  position: { x, y },
  data: { label, nodeType, ...extra },
  ...(nodeType === 'input' || nodeType === 'output' ? { deletable: false } : {}),
})

const e = (id: string, source: string, target: string, sourceHandle?: string) =>
  sourceHandle ? { id, source, target, sourceHandle } : { id, source, target }

const CHAIN_TEMPLATES: Record<string, AgentChainCreatePayload> = {
  'Sequential Pipeline': {
    name: 'Sequential Pipeline',
    description: 'Input -> Executor -> Output',
    enabled: true,
    api_accessible: true,
    execution_options: {},
    flow_definition: {
      nodes: [
        n('input-1', 'input', 50, 200, 'Input'),
        n('executor-1', 'executor', 300, 200, 'Executor', {
          target_kind: 'assistant',
          target: '',
        }),
        n('output-1', 'output', 550, 200, 'Output'),
      ],
      edges: [e('e1', 'input-1', 'executor-1'), e('e2', 'executor-1', 'output-1')],
    },
  },
  'Router Dispatch': {
    name: 'Router Dispatch',
    description: 'Input -> Router -> two branches -> Output',
    enabled: true,
    api_accessible: true,
    execution_options: {},
    flow_definition: {
      nodes: [
        n('input-1', 'input', 40, 220, 'Input'),
        n('router-1', 'router', 280, 220, 'Decider', {
          assistant_name: '',
          routes: [
            { id: '1', name: 'path_a', description: '' },
            { id: '2', name: 'path_b', description: '' },
          ],
          fallback_route: '',
        }),
        n('exec-a', 'executor', 560, 130, 'Branch A', { target_kind: 'assistant', target: '' }),
        n('exec-b', 'executor', 560, 320, 'Branch B', { target_kind: 'assistant', target: '' }),
        n('output-1', 'output', 830, 220, 'Output'),
      ],
      edges: [
        e('e1', 'input-1', 'router-1'),
        e('e2', 'router-1', 'exec-a', 'route-1'),
        e('e3', 'router-1', 'exec-b', 'route-2'),
        e('e4', 'exec-a', 'output-1'),
        e('e5', 'exec-b', 'output-1'),
      ],
    },
  },
  'Parallel Experts': {
    name: 'Parallel Experts',
    description: 'Input -> two experts in parallel -> auto-merge -> Output',
    enabled: true,
    api_accessible: true,
    execution_options: {},
    flow_definition: {
      nodes: [
        n('input-1', 'input', 20, 200, 'Input'),
        n('expert-a', 'executor', 280, 110, 'Expert A', { target_kind: 'agent', target: '' }),
        n('expert-b', 'executor', 280, 290, 'Expert B', { target_kind: 'agent', target: '' }),
        n('output-1', 'output', 560, 200, 'Output'),
      ],
      edges: [
        e('e1', 'input-1', 'expert-a'),
        e('e2', 'input-1', 'expert-b'),
        e('e3', 'expert-a', 'output-1'),
        e('e4', 'expert-b', 'output-1'),
      ],
    },
  },
  'Condition Gate': {
    name: 'Condition Gate',
    description: 'Input -> Condition -> true/false branches -> Output',
    enabled: true,
    api_accessible: true,
    execution_options: {},
    flow_definition: {
      nodes: [
        n('input-1', 'input', 40, 220, 'Input'),
        n('condition-1', 'condition', 280, 220, 'Gate', {
          rule: { op: 'contains', value: '' },
        }),
        n('exec-t', 'executor', 560, 130, 'On True', { target_kind: 'assistant', target: '' }),
        n('exec-f', 'executor', 560, 320, 'On False', { target_kind: 'assistant', target: '' }),
        n('output-1', 'output', 830, 220, 'Output'),
      ],
      edges: [
        e('e1', 'input-1', 'condition-1'),
        e('e2', 'condition-1', 'exec-t', 'cond-true'),
        e('e3', 'condition-1', 'exec-f', 'cond-false'),
        e('e4', 'exec-t', 'output-1'),
        e('e5', 'exec-f', 'output-1'),
      ],
    },
  },
  'Transform & Assist': {
    name: 'Transform & Assist',
    description: 'Input -> Transform (template) -> Executor -> Output',
    enabled: true,
    api_accessible: true,
    execution_options: {},
    flow_definition: {
      nodes: [
        n('input-1', 'input', 50, 200, 'Input'),
        n('transform-1', 'transform', 290, 200, 'Prompt', {
          template: 'Please answer the following request:\n\n{{input}}',
        }),
        n('executor-1', 'executor', 540, 200, 'Executor', {
          target_kind: 'assistant',
          target: '',
        }),
        n('output-1', 'output', 790, 200, 'Output'),
      ],
      edges: [
        e('e1', 'input-1', 'transform-1'),
        e('e2', 'transform-1', 'executor-1'),
        e('e3', 'executor-1', 'output-1'),
      ],
    },
  },
}

const DEFAULT_CHAIN: AgentChainCreatePayload = {
  name: '',
  description: '',
  enabled: true,
  api_accessible: true,
  flow_definition: createEmptyFlow(),
  execution_options: {},
}

const chainToPayload = (chain: AgentChainEntry): AgentChainCreatePayload => ({
  name: chain.name,
  description: chain.description,
  enabled: chain.enabled,
  api_accessible: chain.api_accessible,
  flow_definition: chain.flow_definition,
  execution_options: chain.execution_options,
})

const normalizeAnchors = (nodeList: any[]) =>
  nodeList.map((nd: any) => {
    /* Legacy flows may store the semantic type on `type` (e.g. 'assistant')
       instead of data.nodeType — backfill so every node renders through the
       chainNode component. 'chainNode' itself is the ReactFlow type, not a
       semantic type. */
    const nt = nd.data?.nodeType || (nd.type !== 'chainNode' ? nd.type : '') || ''
    const out = { ...nd, type: 'chainNode', data: { ...(nd.data || {}), nodeType: nt } }
    if (nt === 'input' || nt === 'output') {
      return { ...out, draggable: true, selectable: true, deletable: false }
    }
    return out
  })

/* Stored edges may predate the smoothstep/arrow styling — normalize so old
   chains look identical to freshly drawn edges. */
const normalizeEdges = (edgeList: any[]) =>
  edgeList.map((ed: any) => ({
    ...ed,
    type: ed.type || 'smoothstep',
    markerEnd: ed.markerEnd || { type: MarkerType.ArrowClosed },
  }))

/* ------------------------------------------------------------------ */
/* Node presets (per-user, localStorage)                                */
/* ------------------------------------------------------------------ */

const PRESETS_KEY = 'agentix.chain.nodePresets.v1'

function loadPresets(): NodePreset[] {
  try {
    const raw = localStorage.getItem(PRESETS_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function storePresets(presets: NodePreset[]) {
  try {
    localStorage.setItem(PRESETS_KEY, JSON.stringify(presets))
  } catch {
    // quota / privacy mode — presets stay session-only
  }
}

/* ------------------------------------------------------------------ */
/* Component                                                            */
/* ------------------------------------------------------------------ */

export default function AgentChainsPage() {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const { hasPermission } = useAuth()
  const { getChains, invalidateChains } = useData()
  const canWrite = hasPermission('agent_chains:write')

  const [chains, setChains] = useState<AgentChainEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')

  const [editingId, setEditingId] = useState<string | null>(null)
  const [showEditor, setShowEditor] = useState(false)
  const [draft, setDraft] = useState<AgentChainCreatePayload>(DEFAULT_CHAIN)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null)

  const [nodes, setNodes] = useState<Node[]>([])
  const [edges, setEdges] = useState<Edge[]>([])
  const [issues, setIssues] = useState<ValidationIssue[]>([])
  const [serverErrors, setServerErrors] = useState<string[]>([])

  const [assistants, setAssistants] = useState<AssistantEntry[]>([])
  const [toolList, setToolList] = useState<ToolEntry[]>([])
  const [presets, setPresets] = useState<NodePreset[]>(loadPresets)
  const [presetNode, setPresetNode] = useState<Node | null>(null)
  const [saving, setSaving] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<AgentChainEntry | null>(null)
  const baselineRef = useRef<string>('')

  /* Clipboard */
  const clipboardRef = useRef<{ nodes: Node[]; edges: Edge[] } | null>(null)
  const [hasClipboard, setHasClipboard] = useState(false)

  /* Test run */
  const [testOpen, setTestOpen] = useState(false)
  const [testRunning, setTestRunning] = useState(false)
  const [testResult, setTestResult] = useState<AgentChainTestRunResponse | null>(null)
  const [testError, setTestError] = useState<string | null>(null)
  const testAbortRef = useRef<AbortController | null>(null)

  /* Debounce ref for inspector-driven edits (one undo step per burst) */
  const editBurstRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  /* Monotonic counter to keep generated node/edge ids collision-free even
     when multiple mutations land in the same millisecond. */
  const idSeqRef = useRef(0)

  /* -------------------------- history -------------------------- */

  const history = useChainHistory(setNodes, setEdges)
  history.bind(nodes, edges)

  /** One undo step per continuous typing burst (400 ms quiet period). */
  function pushDebounced() {
    if (editBurstRef.current) {
      clearTimeout(editBurstRef.current)
      editBurstRef.current = setTimeout(() => (editBurstRef.current = null), 400)
      return
    }
    history.push()
    editBurstRef.current = setTimeout(() => (editBurstRef.current = null), 400)
  }

  /* -------------------------- data loading -------------------------- */

  useEffect(() => {
    loadChains()
    loadAssistants()
    loadTools()
    return () => testAbortRef.current?.abort()
  }, [])

  async function loadTools() {
    try {
      setToolList(await toolsApi.list())
    } catch {
      // tools page permission may differ — palette falls back to free text
    }
  }

  async function loadChains() {
    setLoading(true)
    try {
      const data = await getChains()
      setChains(data)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  async function loadAssistants() {
    try {
      const data = await config.getInstructions()
      setAssistants(data.assistants)
    } catch {
      // ignore
    }
  }

  /* -------------------------- editor lifecycle -------------------------- */

  function openEditor(chain?: AgentChainEntry, template?: AgentChainCreatePayload) {
    let newDraft: AgentChainCreatePayload
    let newNodes: any[]
    let newEdges: any[]

    if (chain) {
      setEditingId(chain.id)
      newDraft = chainToPayload(chain)
      const flow = chain.flow_definition as any
      newNodes = normalizeAnchors(flow.nodes || [])
      newEdges = normalizeEdges(flow.edges || [])
    } else if (template) {
      setEditingId(null)
      newDraft = { ...template }
      const flow = template.flow_definition as any
      newNodes = normalizeAnchors(flow.nodes || [])
      newEdges = normalizeEdges(flow.edges || [])
    } else {
      setEditingId(null)
      newDraft = DEFAULT_CHAIN
      const empty = createEmptyFlow()
      newNodes = empty.nodes
      newEdges = empty.edges
    }

    setDraft(newDraft)
    setNodes(newNodes)
    setEdges(newEdges)
    baselineRef.current = JSON.stringify({
      draft: newDraft,
      flow: sanitizeFlow(newNodes, newEdges),
    })
    history.reset()
    setShowEditor(true)
    setSelectedNodeId(null)
    setSelectedEdgeId(null)
    setIssues([])
    setServerErrors([])
    setTestOpen(false)
    setTestResult(null)
    setTestError(null)
    lastFlowSigRef.current = ''
    setError(null)
  }

  /* -------------------------- dirty tracking -------------------------- */

  const flowSnapshot = () => ({ draft, flow: sanitizeFlow(nodes, edges) })

  const isDirty =
    showEditor &&
    baselineRef.current !== '' &&
    JSON.stringify(flowSnapshot()) !== baselineRef.current

  useUnsavedGuard(isDirty, $t('common.leaveWarning'))

  function currentSnapshotDirty() {
    return (
      baselineRef.current !== '' &&
      JSON.stringify(flowSnapshot()) !== baselineRef.current
    )
  }

  function handleDiscard() {
    if (!baselineRef.current) return
    const snap = JSON.parse(baselineRef.current)
    setDraft(snap.draft)
    setNodes(normalizeAnchors(snap.flow.nodes))
    setEdges(normalizeEdges(snap.flow.edges))
    history.reset()
    setSelectedNodeId(null)
    setSelectedEdgeId(null)
  }

  /** Pending action that needs discard confirmation (close / load template). */
  const [discardAction, setDiscardAction] = useState<(() => void) | null>(null)

  function closeEditor() {
    if (currentSnapshotDirty()) {
      setDiscardAction(() => doCloseEditor)
      return
    }
    doCloseEditor()
  }

  function doCloseEditor() {
    testAbortRef.current?.abort()
    baselineRef.current = ''
    setShowEditor(false)
    setEditingId(null)
    setDraft(DEFAULT_CHAIN)
    setNodes([])
    setEdges([])
    setSelectedNodeId(null)
    setSelectedEdgeId(null)
    setIssues([])
    setServerErrors([])
    setTestOpen(false)
    setTestResult(null)
    setTestError(null)
    lastFlowSigRef.current = ''
    history.reset()
  }

  /* -------------------------- save / delete / etc -------------------------- */

  async function handleSave() {
    setError(null)
    setSaving(true)
    const payload = { ...draft, flow_definition: flowSnapshot().flow }

    try {
      if (editingId) {
        await agentChains.update(editingId, payload)
      } else {
        await agentChains.create(payload)
      }
      baselineRef.current = ''
      invalidateChains()
      await loadChains()
      closeEditor()
    } catch (err) {
      setError(String(err))
    } finally {
      setSaving(false)
    }
  }

  async function handleDelete() {
    if (!deleteTarget) return
    try {
      await agentChains.delete(deleteTarget.id)
      invalidateChains()
      setDeleteTarget(null)
      await loadChains()
    } catch (err) {
      setError(String(err))
      setDeleteTarget(null)
    }
  }

  async function handleToggle(chain: AgentChainEntry) {
    if (!canWrite) return
    try {
      await agentChains.update(chain.id, {
        ...chainToPayload(chain),
        enabled: !chain.enabled,
      })
      invalidateChains()
      await loadChains()
    } catch (err) {
      setError(String(err))
    }
  }

  async function handleDuplicate(chain: AgentChainEntry) {
    if (!canWrite) return
    try {
      await agentChains.create({
        ...chainToPayload(chain),
        name: `${chain.name} (${$t('configEditor.copySuffix')})`,
      })
      invalidateChains()
      await loadChains()
    } catch (err) {
      setError(String(err))
    }
  }

  async function exportChain(id: string, name: string) {
    try {
      const data = await agentChains.export(id)
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `chain_${name.replace(/\s+/g, '_')}.json`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(String(err))
    }
  }

  async function handleImport(file: File) {
    try {
      const text = await file.text()
      const data = JSON.parse(text) as AgentChainCreatePayload
      await agentChains.import(data)
      invalidateChains()
      await loadChains()
    } catch (err) {
      setError(String(err))
    }
  }

  async function handleValidate() {
    const flow = flowSnapshot().flow
    try {
      const result = await agentChains.validate(flow as Record<string, unknown>, draft.execution_options)
      setServerErrors(result.valid ? [] : result.errors)
    } catch (err) {
      setServerErrors([String(err)])
    }
  }

  /* -------------------------- test run -------------------------- */

  async function handleTestRun(input: string) {
    testAbortRef.current?.abort()
    const ac = new AbortController()
    testAbortRef.current = ac
    setTestRunning(true)
    setTestError(null)
    setTestResult(null)
    try {
      const stream = agentChains.testRunStream(
        flowSnapshot().flow as unknown as Record<string, unknown>,
        input,
        ac.signal,
        draft.execution_options,
      )
      for await (const ev of stream) {
        if (ac.signal.aborted) return
        if (ev.type === 'node_start') {
          setTestResult((prev) => ({
            answer: prev?.answer ?? '',
            reasoning: prev?.reasoning ?? '',
            node_outputs: prev?.node_outputs ?? {},
            node_status: {
              ...(prev?.node_status || {}),
              [ev.node_id]: { status: 'running' },
            },
            tool_calls: prev?.tool_calls ?? [],
          }))
        } else if (ev.type === 'node_end') {
          setTestResult((prev) => ({
            answer: prev?.answer ?? '',
            reasoning: prev?.reasoning ?? '',
            node_outputs: {
              ...(prev?.node_outputs || {}),
              ...(ev.preview ? { [ev.node_id]: ev.preview } : {}),
            },
            node_status: {
              ...(prev?.node_status || {}),
              [ev.node_id]: {
                status: ev.status,
                ...(ev.chosen_route ? { chosen_route: ev.chosen_route } : {}),
                ...(ev.error ? { error: ev.error } : {}),
              },
            },
            tool_calls: prev?.tool_calls ?? [],
          }))
        } else if (ev.type === 'run_result') {
          setTestResult(ev.result)
        } else if (ev.type === 'run_error') {
          setTestError(ev.error)
        }
      }
    } catch (err) {
      if (!ac.signal.aborted) {
        setTestError(err instanceof Error ? err.message : String(err))
      }
    } finally {
      if (!ac.signal.aborted) setTestRunning(false)
    }
  }

  function handleCancelTest() {
    testAbortRef.current?.abort()
    setTestRunning(false)
  }

  /* -------------------------- run history -------------------------- */

  const [runsOpen, setRunsOpen] = useState(false)
  const [runs, setRuns] = useState<AgentChainRunEntry[]>([])
  const [runsLoading, setRunsLoading] = useState(false)

  async function loadRuns() {
    if (!editingId) return
    setRunsLoading(true)
    try {
      setRuns(await agentChains.runs(editingId, 50))
    } catch (err) {
      setError(String(err))
    } finally {
      setRunsLoading(false)
    }
  }

  function openRuns() {
    setRunsOpen(true)
    void loadRuns()
  }

  /** Replay a recorded run inside the test panel (statuses + outputs). */
  function replayRun(run: AgentChainRunEntry) {
    setTestResult({
      answer: run.answer,
      reasoning: '',
      node_outputs: run.node_outputs || {},
      node_status: run.node_status || {},
      tool_calls: [],
    })
    setTestError(null)
    setTestOpen(true)
    setRunsOpen(false)
  }

  /* -------------------------- presets -------------------------- */

  function handleAddPresetNode(preset: NodePreset, position?: { x: number; y: number }) {
    if (!canWrite) return
    const id = `${preset.nodeType}-${Date.now()}-${++idSeqRef.current}`
    history.push()
    const { label, ...rest } = preset.data as Record<string, unknown>
    setNodes((prev) => [
      ...prev.map((nd) => (nd.selected ? { ...nd, selected: false } : nd)),
      {
        id,
        type: 'chainNode',
        position: position ?? { x: 250 + Math.random() * 200, y: 120 + Math.random() * 200 },
        data: { nodeType: preset.nodeType, label: preset.name, ...rest },
        selected: true,
      },
    ])
    setSelectedNodeId(id)
    setSelectedEdgeId(null)
  }

  function handleDeletePreset(name: string) {
    setPresets((prev) => {
      const next = prev.filter((p) => p.name !== name)
      storePresets(next)
      return next
    })
  }

  function handleSavePresetConfirm(name: string) {
    const node = presetNode
    setPresetNode(null)
    if (!node || !name.trim()) return
    const { label, ...rest } = (node.data || {}) as Record<string, unknown>
    const preset: NodePreset = {
      name: name.trim(),
      nodeType: nodeTypeOf(node),
      data: { label: name.trim(), ...rest },
    }
    setPresets((prev) => {
      const next = [...prev.filter((p) => p.name !== preset.name), preset]
      storePresets(next)
      return next
    })
  }

  /* -------------------------- sub-chain drill-down -------------------------- */

  function handleOpenChain(chainId: string) {
    const target = chains.find((c) => c.id === chainId)
    if (!target) return
    if (currentSnapshotDirty()) {
      setDiscardAction(() => () => openEditor(target))
      return
    }
    openEditor(target)
  }

  /* -------------------------- graph mutations -------------------------- */

  function onNodesChange(changes: NodeChange[]) {
    if (changes.some((c) => c.type === 'remove')) history.push()
    for (const c of changes) {
      if (c.type === 'remove' && c.id === selectedNodeId) setSelectedNodeId(null)
    }
    setNodes((prev) => {
      const next = applyNodeChanges(changes, prev)
      return next.map((nd: any) => {
        const nt = nd.data?.nodeType
        if (nt === 'input' || nt === 'output') {
          return { ...nd, draggable: true, selectable: true, deletable: false }
        }
        return nd
      })
    })
  }

  function onEdgesChange(changes: EdgeChange[]) {
    if (changes.some((c) => c.type === 'remove')) history.push()
    for (const c of changes) {
      if (c.type === 'remove' && c.id === selectedEdgeId) setSelectedEdgeId(null)
    }
    setEdges((prev) => applyEdgeChanges(changes, prev))
  }

  const onConnect = useCallback(
    (conn: Connection) => {
      history.push()
      setEdges((eds: any) =>
        addEdge(
          { ...conn, type: 'smoothstep', markerEnd: { type: MarkerType.ArrowClosed } },
          eds,
        ),
      )
    },
    [history.push],
  )

  const onReconnect = useCallback(
    (oldEdge: Edge, conn: Connection) => {
      if (!isValidConn(nodes, edges, conn)) return
      history.push()
      setEdges((eds) => reconnectEdge(oldEdge, conn, eds))
    },
    [nodes, edges, history.push],
  )

  const checkConnection = useCallback(
    (conn: Edge | Connection) => isValidConn(nodes, edges, conn),
    [nodes, edges],
  )

  function addNodeAt(type: string, position?: { x: number; y: number }) {
    if (!canWrite) return
    if (!(ADDABLE_NODE_TYPES as readonly string[]).includes(type)) return
    const id = `${type}-${Date.now()}-${++idSeqRef.current}`
    const label = $t(`agentChains.palette.${type}`)
    history.push()
    const newNode: Node = {
      id,
      type: 'chainNode',
      position: position ?? {
        x: 250 + Math.random() * 200,
        y: 120 + Math.random() * 200,
      },
      data: defaultDataFor(type, label),
      selected: true,
    }
    setNodes((prev: any) => [
      ...prev.map((nd: Node) => (nd.selected ? { ...nd, selected: false } : nd)),
      newNode,
    ])
    setSelectedNodeId(id)
    setSelectedEdgeId(null)
  }

  function handleApplyNodes(nextNodes: Node[]) {
    history.push()
    setNodes(nextNodes)
  }

  function handleApplyTemplate(name: string) {
    const template = CHAIN_TEMPLATES[name]
    if (!template) return
    if (currentSnapshotDirty()) {
      setDiscardAction(() => () => applyTemplate(name))
      return
    }
    applyTemplate(name)
  }

  function applyTemplate(name: string) {
    const template = CHAIN_TEMPLATES[name]
    if (!template) return
    history.push()
    const flow = template.flow_definition as any
    setNodes(normalizeAnchors(flow.nodes || []))
    setEdges(normalizeEdges(flow.edges || []))
    if (!draft.name.trim()) setDraft({ ...draft, name: template.name })
    setSelectedNodeId(null)
    setSelectedEdgeId(null)
  }

  /* -------------------------- selection -------------------------- */

  function onNodeClick(node: Node) {
    setSelectedNodeId(node.id)
    setSelectedEdgeId(null)
  }

  function onEdgeClick(edge: Edge) {
    setSelectedEdgeId(edge.id)
    setSelectedNodeId(null)
  }

  function onPaneClick() {
    setSelectedNodeId(null)
    setSelectedEdgeId(null)
  }

  const selectedNode = nodes.find((nd) => nd.id === selectedNodeId)
  const selectedEdge = edges.find((ed) => ed.id === selectedEdgeId)
  const hasSelection = nodes.some((nd) => nd.selected)

  function selectAll() {
    setNodes((prev) => prev.map((nd) => ({ ...nd, selected: true })))
    setEdges((prev) => prev.map((ed) => ({ ...ed, selected: true })))
  }

  function deselectAll() {
    setSelectedNodeId(null)
    setSelectedEdgeId(null)
    setNodes((prev) => prev.map((nd) => (nd.selected ? { ...nd, selected: false } : nd)))
    setEdges((prev) => prev.map((ed) => (ed.selected ? { ...ed, selected: false } : ed)))
  }

  /* -------------------------- node/edge updates -------------------------- */

  function updateNodeData(key: string, value: unknown) {
    if (!selectedNodeId || !canWrite) return
    pushDebounced()
    setNodes((prev: any) =>
      prev.map((nd: any) => {
        if (nd.id !== selectedNodeId) return nd
        return { ...nd, data: { ...nd.data, [key]: value } }
      }),
    )
  }

  /** Drop edges leaving `sourceId` through handles that no longer exist
      (e.g. a router route was deleted in the inspector). */
  function pruneRouteEdges(sourceId: string, removedHandles: Set<string>) {
    if (!canWrite || !removedHandles.size) return
    history.push()
    setEdges((prev) =>
      prev.filter(
        (ed) =>
          !(
            ed.source === sourceId &&
            ed.sourceHandle &&
            removedHandles.has(ed.sourceHandle)
          ),
      ),
    )
  }

  function updateEdgeLabel(label: string) {
    if (!selectedEdgeId || !canWrite) return
    pushDebounced()
    setEdges((prev) =>
      prev.map((ed) => (ed.id === selectedEdgeId ? { ...ed, label: label || undefined } : ed)),
    )
  }

  function onDeleteEdge() {
    if (!selectedEdgeId || !canWrite) return
    history.push()
    setEdges((prev) => prev.filter((ed) => ed.id !== selectedEdgeId))
    setSelectedEdgeId(null)
  }

  /* -------------------------- clipboard -------------------------- */

  function cloneSelection() {
    const sel = nodes.filter((nd) => nd.selected && !ANCHOR_TYPES.has(nodeTypeOf(nd)))
    const selIds = new Set(sel.map((nd) => nd.id))
    const internal = edges.filter((ed) => selIds.has(ed.source) && selIds.has(ed.target))
    return {
      nodes: JSON.parse(JSON.stringify(sel)) as Node[],
      edges: JSON.parse(JSON.stringify(internal)) as Edge[],
    }
  }

  function pasteGraph(clip: { nodes: Node[]; edges: Edge[] }) {
    if (!clip.nodes.length) return
    history.push()
    const ts = `${Date.now()}-${++idSeqRef.current}`
    const idMap = new Map<string, string>()
    const newNodes = clip.nodes.map((nd, i) => {
      const nid = `${nodeTypeOf(nd) || 'node'}-${ts}-${i}`
      idMap.set(nd.id, nid)
      return {
        ...nd,
        id: nid,
        selected: true,
        position: { x: nd.position.x + 40, y: nd.position.y + 40 },
        data: { ...nd.data },
      }
    })
    const newEdges = clip.edges.map((ed, i) => ({
      ...ed,
      id: `e-${ts}-${i}`,
      source: idMap.get(ed.source)!,
      target: idMap.get(ed.target)!,
      selected: false,
    }))
    setNodes((prev) => [...prev.map((nd) => ({ ...nd, selected: false })), ...newNodes])
    setEdges((prev) => [...prev, ...newEdges])
    setSelectedNodeId(null)
    setSelectedEdgeId(null)
  }

  function handleCopy() {
    const clip = cloneSelection()
    if (!clip.nodes.length) return
    clipboardRef.current = clip
    setHasClipboard(true)
  }

  function handlePaste() {
    if (!canWrite || !clipboardRef.current) return
    pasteGraph(clipboardRef.current)
  }

  function handleDuplicateNodes() {
    if (!canWrite) return
    pasteGraph(cloneSelection())
  }

  /* -------------------------- shortcuts -------------------------- */

  useChainShortcuts(showEditor, {
    undo: history.undo,
    redo: history.redo,
    copy: handleCopy,
    paste: handlePaste,
    duplicate: handleDuplicateNodes,
    selectAll,
    deselect: deselectAll,
    hasSelection: () => hasSelection,
  })

  /* -------------------------- live validation -------------------------- */

  /* Only re-validate/clear server errors when the graph content actually
     changed — selection/dimension changes keep the panel stable. */
  const lastFlowSigRef = useRef('')
  useEffect(() => {
    if (!showEditor) return
    const t = setTimeout(() => {
      const sig = JSON.stringify(sanitizeFlow(nodes, edges))
      if (sig !== lastFlowSigRef.current) {
        lastFlowSigRef.current = sig
        setIssues(validateFlowLocal(nodes, edges))
        setServerErrors([])
        // Status rings/skips from an earlier run describe the old graph —
        // drop them as soon as the flow is edited.
        setTestResult(null)
      }
    }, 300)
    return () => clearTimeout(t)
  }, [nodes, edges, showEditor])

  /* -------------------------- list view -------------------------- */

  const filteredChains = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return chains
    return chains.filter(
      (c) =>
        c.name.toLowerCase().includes(q) ||
        (c.description || '').toLowerCase().includes(q),
    )
  }, [chains, search])

  return (
    <div className="chains-page">
      {/* Header */}
      <div className="chains-header">
        <h2>
          <Workflow size={20} />
          {$t('nav.agentChains')}
        </h2>
        <div className="chains-header-actions">
          {canWrite && (
            <>
              <label className="btn-secondary" style={{ cursor: 'pointer' }}>
                <input
                  type="file"
                  accept=".json"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) handleImport(file)
                    e.target.value = ''
                  }}
                />
                {$t('agentChains.import')}
              </label>
              <button className="btn-primary" onClick={() => openEditor()}>
                <Plus size={14} /> {$t('agentChains.new')}
              </button>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="error-banner" style={{ margin: '12px 24px' }}>
          <AlertCircle size={14} /> {error}
        </div>
      )}

      {/* Editor or List */}
      {showEditor ? (
        <ChainEditor
          draft={draft}
          setDraft={setDraft}
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onReconnect={onReconnect}
          onNodeDragStart={history.push}
          onSelectionDragStart={history.push}
          onNodeClick={onNodeClick}
          onEdgeClick={onEdgeClick}
          onPaneClick={onPaneClick}
          isValidConnection={checkConnection}
          selectedNode={selectedNode}
          selectedEdge={selectedEdge}
          updateNodeData={updateNodeData}
          updateEdgeLabel={updateEdgeLabel}
          onDeleteEdge={onDeleteEdge}
          pruneRouteEdges={pruneRouteEdges}
          assistants={assistants}
          chains={chains.filter((c) => c.id !== editingId)}
          tools={toolList}
          presets={presets}
          onAddPresetNode={handleAddPresetNode}
          onDeletePreset={handleDeletePreset}
          onSavePreset={(node) => setPresetNode(node)}
          onOpenChain={handleOpenChain}
          addNodeAt={addNodeAt}
          issues={issues}
          serverErrors={serverErrors}
          onValidate={handleValidate}
          onExport={() => editingId && exportChain(editingId, draft.name)}
          onClose={closeEditor}
          canWrite={canWrite}
          isNew={!editingId}
          canUndo={history.canUndo}
          canRedo={history.canRedo}
          hasSelection={hasSelection}
          hasClipboard={hasClipboard}
          templateNames={Object.keys(CHAIN_TEMPLATES)}
          onUndo={history.undo}
          onRedo={history.redo}
          onCopy={handleCopy}
          onPaste={handlePaste}
          onDuplicate={handleDuplicateNodes}
          onApplyNodes={handleApplyNodes}
          onApplyTemplate={handleApplyTemplate}
          testOpen={testOpen}
          onToggleTest={() => setTestOpen((v) => !v)}
          testRunning={testRunning}
          testResult={testResult}
          testError={testError}
          onRunTest={handleTestRun}
          onCancelTest={handleCancelTest}
          runsOpen={runsOpen}
          onToggleRuns={() => (runsOpen ? setRunsOpen(false) : openRuns())}
          runs={runs}
          runsLoading={runsLoading}
          onReplayRun={replayRun}
        />
      ) : (
        <div style={{ padding: 24, overflowY: 'auto' }}>
          <SearchToolbar
            search={search}
            onSearch={setSearch}
            placeholder={$t('agentChains.search')}
          />

          {loading ? (
            <div className="loading">{$t('common.loading')}</div>
          ) : chains.length === 0 ? (
            <div className="chain-empty">
              <div className="chain-empty-text">{$t('agentChains.noChains')}</div>
              {canWrite && (
                <div className="chain-templates">
                  <div className="chain-templates-title">{$t('agentChains.templates')}:</div>
                  {Object.entries(CHAIN_TEMPLATES).map(([name, template]) => (
                    <button
                      key={name}
                      className="btn-secondary chain-template-btn"
                      onClick={() => openEditor(undefined, template)}
                    >
                      <Plus size={14} /> {name}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : filteredChains.length === 0 ? (
            <div className="text-muted">{$t('agentChains.noChains')}</div>
          ) : (
            <div className="card-grid">
              {filteredChains.map((chain) => (
                <ChainCard
                  key={chain.id}
                  chain={chain}
                  canWrite={canWrite}
                  onEdit={() => openEditor(chain)}
                  onDuplicate={() => handleDuplicate(chain)}
                  onDelete={() => setDeleteTarget(chain)}
                  onExport={() => exportChain(chain.id, chain.name)}
                  onToggle={() => handleToggle(chain)}
                  $t={$t}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {showEditor && (
        <UnsavedChangesBar
          dirty={isDirty && canWrite}
          saving={saving}
          onSave={handleSave}
          onDiscard={handleDiscard}
        />
      )}

      <ConfirmModal
        open={deleteTarget !== null}
        title={$t('common.delete')}
        message={deleteTarget ? `${$t('common.delete')} "${deleteTarget.name}"?` : ''}
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />

      <ConfirmModal
        open={discardAction !== null}
        title={$t('common.discard')}
        message={$t('common.discardConfirm')}
        confirmText={$t('common.discard')}
        onConfirm={() => {
          const action = discardAction
          setDiscardAction(null)
          action?.()
        }}
        onCancel={() => setDiscardAction(null)}
      />

      <PromptModal
        open={presetNode !== null}
        title={$t('agentChains.presets.nameTitle')}
        message={$t('agentChains.presets.namePrompt')}
        initialValue={presetNode ? String(presetNode.data?.label || '') : ''}
        placeholder={$t('agentChains.presets.namePlaceholder')}
        onSubmit={handleSavePresetConfirm}
        onCancel={() => setPresetNode(null)}
      />
    </div>
  )
}
