import type { Connection, Edge, Node } from '@xyflow/react'
import {
  Braces,
  FileInput,
  FileOutput,
  Bot,
  BrainCircuit,
  GitMerge,
  GitFork,
  ListFilter,
  ListChecks,
  Split,
  Sparkles,
  Play,
  StickyNote,
  Timer,
  FileJson,
  Wrench,
} from 'lucide-react'

/* Node types that can be added from the palette (input/output are anchors).
   Legacy structural nodes (assistant/agent/merge/parallel) stay renderable in
   existing flows but are no longer offered — executor covers assistants and
   agents, fan-out/merge are pure topology. */
export const ADDABLE_NODE_TYPES = [
  'executor',
  'router',
  'condition',
  'switch',
  'llm',
  'tool',
  'json',
  'transform',
  'delay',
  'note',
] as const

export type AddableNodeType = (typeof ADDABLE_NODE_TYPES)[number]

export const NODE_ICON_MAP: Record<string, any> = {
  input: FileInput,
  executor: Play,
  router: GitFork,
  condition: ListFilter,
  switch: ListChecks,
  llm: Sparkles,
  tool: Wrench,
  json: FileJson,
  transform: Braces,
  delay: Timer,
  note: StickyNote,
  assistant: Bot,
  agent: BrainCircuit,
  merge: GitMerge,
  parallel: Split,
  output: FileOutput,
}

export const FALLBACK_NODE_ICON = Sparkles

/** Node types the user is not allowed to copy/paste (anchors stay unique). */
export const ANCHOR_TYPES = new Set(['input', 'output'])

/** Types that decide which outgoing port stays live. */
export const DECISION_TYPES = new Set(['router', 'condition', 'switch'])

/** sourceHandle id of the error port. */
export const ERROR_HANDLE = 'on_error'
/** sourceHandle id of the switch fallback port. */
export const ELSE_HANDLE = 'else'

/** Types that may carry an `on_error` output port. */
export const ERROR_PORT_TYPES = new Set([
  'executor',
  'router',
  'condition',
  'switch',
  'llm',
  'transform',
  'tool',
  'delay',
  'json',
  'merge',
  'parallel',
  'assistant',
  'agent',
])

/** Types where `data.input_template` may override the merged input. */
export const TEMPLATE_INPUT_TYPES = new Set([
  'executor',
  'router',
  'condition',
  'switch',
  'llm',
  'transform',
  'tool',
  'delay',
  'json',
  'assistant',
  'agent',
])

export function nodeTypeOf(node: Node | undefined | null): string {
  return (node?.data?.nodeType as string) || (node?.type as string) || ''
}

/** Route handle ids: `route-<route.id>` for routers, `cond-true|false` for conditions. */
export function routeHandle(routeId: string | number): string {
  return `route-${routeId}`
}

/** Default `data` payload for a freshly added node. */
export function defaultDataFor(type: string, label: string): Record<string, unknown> {
  switch (type) {
    case 'assistant':
    case 'agent':
      return { label, nodeType: type, assistant_name: '' }
    case 'executor':
      return { label, nodeType: type, target_kind: 'assistant', target: '' }
    case 'router':
      return {
        label,
        nodeType: type,
        assistant_name: '',
        routes: [
          { id: '1', name: 'route_a', description: '' },
          { id: '2', name: 'route_b', description: '' },
        ],
        fallback_route: '',
      }
    case 'condition':
      return { label, nodeType: type, rule: { op: 'contains', value: '' } }
    case 'switch':
      return {
        label,
        nodeType: type,
        cases: [
          { id: '1', name: 'case_a', op: 'contains', value: '' },
          { id: '2', name: 'case_b', op: 'contains', value: '' },
        ],
      }
    case 'llm':
      return { label, nodeType: type, assistant_name: '', prompt_template: '{{input}}' }
    case 'json':
      return {
        label,
        nodeType: type,
        assistant_name: '',
        prompt_template: '{{input}}',
        json_schema: null,
        max_retries: 1,
      }
    case 'tool':
      return { label, nodeType: type, tool_name: '', params: {}, timeout_s: 60 }
    case 'delay':
      return { label, nodeType: type, ms: 1000 }
    case 'transform':
      return { label, nodeType: type, template: '{{input}}' }
    case 'note':
      return { label, nodeType: type, text: '' }
    default:
      return { label, nodeType: type }
  }
}

/**
 * Strip volatile ReactFlow internals before the flow is persisted,
 * compared for dirty state or sent to the test-run endpoint.
 */
export function sanitizeFlow(nodes: Node[], edges: Edge[]) {
  return {
    nodes: nodes.map((n: any) => {
      const out: any = { id: n.id, type: n.type, position: n.position, data: n.data }
      // Only persist non-default flags — true is the ReactFlow default and
      // normalizeAnchors restores them on load anyway.
      if (n.deletable === false) out.deletable = false
      if (n.draggable === false) out.draggable = false
      if (n.selectable === false) out.selectable = false
      return out
    }),
    edges: edges.map((e: any) => {
      const out: any = { id: e.id, source: e.source, target: e.target }
      if (e.type) out.type = e.type
      if (e.label) out.label = e.label
      if (e.animated) out.animated = true
      if (e.markerEnd) out.markerEnd = e.markerEnd
      if (e.sourceHandle) out.sourceHandle = e.sourceHandle
      if (e.targetHandle) out.targetHandle = e.targetHandle
      return out
    }),
  }
}

/** True if following `edges` from `startId` can reach `targetId` (cycle probe). */
function reaches(edges: Edge[], startId: string, targetId: string): boolean {
  const adjacency = new Map<string, string[]>()
  for (const e of edges) {
    const list = adjacency.get(e.source) || []
    list.push(e.target)
    adjacency.set(e.source, list)
  }
  const seen = new Set<string>()
  const queue = [startId]
  while (queue.length) {
    const cur = queue.pop()!
    if (cur === targetId) return true
    if (seen.has(cur)) continue
    seen.add(cur)
    for (const next of adjacency.get(cur) || []) queue.push(next)
  }
  return false
}

/**
 * ReactFlow `isValidConnection` — blocks connections that the backend
 * validator would reject anyway (self-loops, edges into input, edges out
 * of output, duplicates, and anything that creates a cycle).
 */
export function isValidConnection(nodes: Node[], edges: Edge[], conn: Edge | Connection): boolean {
  const { source, target } = conn
  if (!source || !target || source === target) return false

  const sourceNode = nodes.find((n) => n.id === source)
  const targetNode = nodes.find((n) => n.id === target)
  if (!sourceNode || !targetNode) return false

  if (nodeTypeOf(targetNode) === 'input') return false
  if (nodeTypeOf(sourceNode) === 'output') return false

  // Note nodes are annotations — they carry no handles.
  if (nodeTypeOf(sourceNode) === 'note' || nodeTypeOf(targetNode) === 'note') return false

  // on_error edges are contingencies — only from executable types, never
  // into input/output anchors.
  if ((conn as Edge).sourceHandle === ERROR_HANDLE) {
    if (!ERROR_PORT_TYPES.has(nodeTypeOf(sourceNode))) return false
    if (nodeTypeOf(targetNode) === 'output') return false
  }

  // Same source→target is allowed via different route ports (that merge
  // rejoins downstream); identical (source, target, handle) is a duplicate.
  if (
    edges.some(
      (e) =>
        e.source === source &&
        e.target === target &&
        (e.sourceHandle ?? null) === (conn.sourceHandle ?? null),
    )
  )
    return false

  // Adding source→target must not create a cycle, i.e. there must be no
  // existing path from target back to source.
  if (reaches(edges, target, source)) return false

  return true
}
