import { useCallback, useRef, useState } from 'react'
import type { Edge, Node } from '@xyflow/react'

export type FlowSnapshot = { nodes: Node[]; edges: Edge[] }

const HISTORY_LIMIT = 100

function clone<T>(v: T): T {
  return JSON.parse(JSON.stringify(v)) as T
}

/**
 * Undo/redo stack for the {nodes, edges} pair. `push()` snapshots the
 * CURRENT state and must be called *before* applying a mutation, so undo
 * restores the pre-mutation graph.
 */
export function useChainHistory(
  setNodes: (n: Node[]) => void,
  setEdges: (e: Edge[]) => void,
) {
  const past = useRef<FlowSnapshot[]>([])
  const future = useRef<FlowSnapshot[]>([])
  const live = useRef<FlowSnapshot>({ nodes: [], edges: [] })
  const [depths, setDepths] = useState({ past: 0, future: 0 })

  const bind = useCallback((nodes: Node[], edges: Edge[]) => {
    live.current = { nodes, edges }
  }, [])

  const push = useCallback(() => {
    // Skip no-op pushes: ReactFlow can fire onNodeDragStart AND
    // onSelectionDragStart for one gesture — storing the same pre-mutation
    // state twice would produce a phantom undo step.
    const last = past.current[past.current.length - 1]
    if (last && JSON.stringify(last) === JSON.stringify(live.current)) return
    past.current.push(clone(live.current))
    if (past.current.length > HISTORY_LIMIT) past.current.shift()
    future.current = []
    setDepths({ past: past.current.length, future: 0 })
  }, [])

  const undo = useCallback(() => {
    const snap = past.current.pop()
    if (!snap) return
    future.current.push(clone(live.current))
    setDepths({ past: past.current.length, future: future.current.length })
    setNodes(snap.nodes)
    setEdges(snap.edges)
    live.current = snap
  }, [setNodes, setEdges])

  const redo = useCallback(() => {
    const snap = future.current.pop()
    if (!snap) return
    past.current.push(clone(live.current))
    setDepths({ past: past.current.length, future: future.current.length })
    setNodes(snap.nodes)
    setEdges(snap.edges)
    live.current = snap
  }, [setNodes, setEdges])

  const reset = useCallback(() => {
    past.current = []
    future.current = []
    live.current = { nodes: [], edges: [] }
    setDepths({ past: 0, future: 0 })
  }, [])

  return {
    bind,
    push,
    undo,
    redo,
    reset,
    canUndo: depths.past > 0,
    canRedo: depths.future > 0,
  }
}
