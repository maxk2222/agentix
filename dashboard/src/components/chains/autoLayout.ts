import type { Edge, Node } from '@xyflow/react'
import { nodeTypeOf } from './flowRules'

const LAYER_SPACING = 260
const NODE_SPACING = 120

/**
 * Layered DAG layout, dependency-free.
 * 1. Each node gets a layer = longest path distance from the input node.
 * 2. Within a layer nodes are ordered by the barycenter of their
 *    predecessors' positions (reduces edge crossings).
 * 3. x = layer index, y = centered stack inside the layer.
 */
export function layoutFlow(nodes: Node[], edges: Edge[]): Node[] {
  if (!nodes.length) return nodes

  const inputNode =
    nodes.find((n) => nodeTypeOf(n) === 'input') ?? nodes[0]

  // Predecessors / successors maps (self-loops and unknown ids filtered)
  const ids = new Set(nodes.map((n) => n.id))
  const preds = new Map<string, string[]>()
  const succs = new Map<string, string[]>()
  for (const id of ids) {
    preds.set(id, [])
    succs.set(id, [])
  }
  for (const e of edges) {
    if (!ids.has(e.source) || !ids.has(e.target)) continue
    preds.get(e.target)!.push(e.source)
    succs.get(e.source)!.push(e.target)
  }

  // Longest-path layering (memoized DFS; cycles break by depth guard)
  const layer = new Map<string, number>()
  const assign = (id: string, depth: number): number => {
    const cached = layer.get(id)
    if (cached !== undefined) return cached
    if (depth > ids.size) return 0 // safety against accidental cycles
    let best = 0
    for (const p of preds.get(id) || []) {
      best = Math.max(best, assign(p, depth + 1) + 1)
    }
    layer.set(id, best)
    return best
  }
  layer.set(inputNode.id, 0)
  for (const n of nodes) assign(n.id, 0)

  // Nodes not reachable from input → push them into a trailing layer
  const reachable = new Set<string>()
  {
    const q = [inputNode.id]
    while (q.length) {
      const c = q.pop()!
      if (reachable.has(c)) continue
      reachable.add(c)
      for (const s of succs.get(c) || []) q.push(s)
    }
  }
  const maxLayer = Math.max(0, ...layer.values())
  for (const n of nodes) {
    if (!reachable.has(n.id)) layer.set(n.id, maxLayer + 1)
  }

  // Group by layer
  const layers = new Map<number, Node[]>()
  for (const n of nodes) {
    const l = layer.get(n.id) ?? maxLayer + 1
    const list = layers.get(l) || []
    list.push(n)
    layers.set(l, list)
  }
  const sortedLayers = [...layers.keys()].sort((a, b) => a - b)

  // Order inside layers by predecessor barycenter (single sweep)
  const yPos = new Map<string, number>()
  const placed = new Map<number, Node[]>()
  for (const l of sortedLayers) {
    const layerNodes = layers.get(l)!
    const ordered = [...layerNodes].sort((a, b) => {
      const bary = (n: Node) => {
        const ps = (preds.get(n.id) || []).map((p) => yPos.get(p)).filter((v): v is number => v !== undefined)
        if (!ps.length) return layerNodes.indexOf(n)
        return ps.reduce((s, v) => s + v, 0) / ps.length
      }
      return bary(a) - bary(b)
    })
    placed.set(l, ordered)
    ordered.forEach((n, i) => yPos.set(n.id, i * NODE_SPACING))
  }

  // Assign final positions (layers centered vertically)
  const result = nodes.map((n) => {
    const l = layer.get(n.id) ?? maxLayer + 1
    const ordered = placed.get(l)!
    const idx = ordered.findIndex((x) => x.id === n.id)
    const centeredY = idx * NODE_SPACING - ((ordered.length - 1) * NODE_SPACING) / 2
    return {
      ...n,
      position: { x: l * LAYER_SPACING, y: centeredY + 200 },
    }
  })

  return result
}
