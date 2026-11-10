import { useMemo, useState, useCallback, useEffect, useRef } from 'react'
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  useReactFlow,
  type Node,
  type Edge,
  applyNodeChanges,
  applyEdgeChanges,
  type NodeChange,
  type EdgeChange,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { X, FileText, File, Presentation, Sheet, Type } from 'lucide-react'

export type OKFConcept = {
  okf_concept_id: string
  okf_type: string
  okf_title: string
  source: string
  collection: string
  description: string
  date_downloaded: string
}

const TYPE_COLORS: Record<string, string> = {
  document: '#3b82f6',
  text: '#22c55e',
  spreadsheet: '#f97316',
  presentation: '#a855f7',
}

const TYPE_ICONS: Record<string, typeof FileText> = {
  document: File,
  text: Type,
  spreadsheet: Sheet,
  presentation: Presentation,
}

function typeColor(type: string): string {
  return TYPE_COLORS[type] || '#6b7280'
}

function typeIcon(type: string): typeof FileText {
  return TYPE_ICONS[type] || FileText
}

const STOP_WORDS = new Set([
  'der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'aber', 'ist', 'sind',
  'war', 'waren', 'sein', 'mit', 'von', 'zu', 'zur', 'zum', 'auf', 'in', 'im',
  'an', 'bei', 'für', 'nach', 'vor', 'über', 'unter', 'durch', 'ohne', 'nicht',
  'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'with', 'from',
  'to', 'of', 'in', 'on', 'at', 'by', 'for', 'about', 'this', 'that', 'these',
  'those', 'a', 'an', 'it', 'its', 'as', 'if', 'then', 'than', 'also', 'not',
])

function tokenize(text: string): Set<string> {
  return new Set(
    text.toLowerCase()
      .split(/[\s\-_.,;:!?/()\[\]{}"'`]+/)
      .filter(w => w.length > 2 && !STOP_WORDS.has(w))
  )
}

function conceptTokens(concept: OKFConcept): Set<string> {
  return tokenize(`${concept.okf_title} ${concept.description || ''}`)
}

function keywordOverlap(a: Set<string>, b: Set<string>): number {
  let count = 0
  for (const t of a) {
    if (b.has(t)) count++
  }
  return count
}

interface AdjEntry {
  index: number
  overlap: number
}

function filterRelevantConcepts(
  concepts: OKFConcept[],
  retrievedDocIds: Set<string>,
  maxNodes = 30,
): { indices: number[]; adjacency: Map<number, AdjEntry[]> } {
  if (concepts.length === 0) return { indices: [], adjacency: new Map() }

  const allTokens = concepts.map(c => conceptTokens(c))
  const retrievedIndices = new Set<number>()
  for (let i = 0; i < concepts.length; i++) {
    const cid = concepts[i].okf_concept_id || ''
    const src = concepts[i].source || ''
    for (const rid of retrievedDocIds) {
      if (rid && (cid === rid || src === rid || cid.endsWith(rid) || rid.endsWith(cid) || src.endsWith(rid) || rid.endsWith(src) || cid.includes(rid) || rid.includes(cid) || src.includes(rid) || rid.includes(src))) {
        retrievedIndices.add(i)
        break
      }
    }
  }

  if (retrievedIndices.size === 0) {
    const limited = concepts.slice(0, maxNodes)
    const adj = new Map<number, AdjEntry[]>()
    for (let i = 0; i < limited.length; i++) {
      adj.set(i, [])
      for (let j = i + 1; j < limited.length; j++) {
        const ov = keywordOverlap(allTokens[i], allTokens[j])
        if (ov >= 2) {
          adj.get(i)!.push({ index: j, overlap: ov })
          if (!adj.has(j)) adj.set(j, [])
          adj.get(j)!.push({ index: i, overlap: ov })
        }
      }
    }
    return { indices: limited.map((_, i) => i), adjacency: adj }
  }

  const adjacency = new Map<number, AdjEntry[]>()
  const relevanceScore = new Map<number, number>()

  for (const ri of retrievedIndices) {
    relevanceScore.set(ri, 1000)
  }

  for (let i = 0; i < concepts.length; i++) {
    if (retrievedIndices.has(i)) continue
    let bestOverlap = 0
    let bestRetrieved = -1
    for (const ri of retrievedIndices) {
      const ov = keywordOverlap(allTokens[i], allTokens[ri])
      if (ov > bestOverlap) {
        bestOverlap = ov
        bestRetrieved = ri
      }
    }
    if (bestOverlap >= 2) {
      relevanceScore.set(i, bestOverlap)
      if (!adjacency.has(bestRetrieved)) adjacency.set(bestRetrieved, [])
      adjacency.get(bestRetrieved)!.push({ index: i, overlap: bestOverlap })
      if (!adjacency.has(i)) adjacency.set(i, [])
      adjacency.get(i)!.push({ index: bestRetrieved, overlap: bestOverlap })
    }
  }

  for (const ri of retrievedIndices) {
    if (!adjacency.has(ri)) adjacency.set(ri, [])
  }

  const sorted = [...relevanceScore.entries()].sort((a, b) => b[1] - a[1])
  const selected = new Set<number>(sorted.slice(0, maxNodes).map(([i]) => i))

  const filteredAdj = new Map<number, AdjEntry[]>()
  for (const [i, neighbors] of adjacency) {
    if (selected.has(i)) {
      filteredAdj.set(i, neighbors.filter(n => selected.has(n.index)))
    }
  }

  return { indices: [...selected], adjacency: filteredAdj }
}

function buildEdges(
  indices: number[],
  adjacency: Map<number, AdjEntry[]>,
  idMap: Map<number, string>,
): Edge[] {
  const edges: Edge[] = []
  const seen = new Set<string>()

  for (const i of indices) {
    const neighbors = adjacency.get(i) || []
    for (const n of neighbors) {
      const key = i < n.index ? `${i}-${n.index}` : `${n.index}-${i}`
      if (seen.has(key)) continue
      seen.add(key)
      const opacity = Math.min(0.85, 0.1 + n.overlap * 0.1)
      const width = Math.min(4, 0.5 + n.overlap * 0.5)
      const shouldAnimate = n.overlap >= 4
      const animDuration = shouldAnimate ? `${Math.max(0.3, 2 - n.overlap * 0.15)}s` : undefined
      edges.push({
        id: `kw-${key}`,
        source: idMap.get(i)!,
        target: idMap.get(n.index)!,
        style: { stroke: '#6366f1', strokeWidth: width, opacity },
        type: 'default',
        animated: shouldAnimate,
        data: { animDuration },
      })
    }
  }

  return edges
}

function forceDirectedLayout(
  indices: number[],
  adjacency: Map<number, AdjEntry[]>,
  idMap: Map<number, string>,
  width: number,
  height: number,
): Node[] {
  const n = indices.length
  if (n === 0) return []

  const positions = new Map<number, { x: number; y: number }>()
  const centerX = width / 2
  const centerY = height / 2

  const minDim = Math.min(width, height)
  for (let k = 0; k < n; k++) {
    const angle = (k / n) * 2 * Math.PI
    const r = minDim * 0.35 + Math.random() * 30
    positions.set(indices[k], {
      x: centerX + r * Math.cos(angle),
      y: centerY + r * Math.sin(angle),
    })
  }

  const iterations = 100
  const kSpring = 0.05
  const kRepel = 20000 * (width * height) / (1200 * 800)
  const kCenter = 30 / minDim
  const damping = 0.85
  const minDist = Math.max(80, minDim / 8)
  const maxRadius = minDim * 0.48

  for (let iter = 0; iter < iterations; iter++) {
    const forces = new Map<number, { fx: number; fy: number }>()
    for (const i of indices) forces.set(i, { fx: 0, fy: 0 })

    for (let a = 0; a < n; a++) {
      for (let b = a + 1; b < n; b++) {
        const ia = indices[a]
        const ib = indices[b]
        const pa = positions.get(ia)!
        const pb = positions.get(ib)!
        const dx = pa.x - pb.x
        const dy = pa.y - pb.y
        const dist = Math.sqrt(dx * dx + dy * dy) || 1
        const repel = kRepel / (dist * dist)
        forces.get(ia)!.fx += (dx / dist) * repel
        forces.get(ia)!.fy += (dy / dist) * repel
        forces.get(ib)!.fx -= (dx / dist) * repel
        forces.get(ib)!.fy -= (dy / dist) * repel
      }
    }

    for (const [i, neighbors] of adjacency) {
      const pi = positions.get(i)
      if (!pi) continue
      for (const n of neighbors) {
        const pn = positions.get(n.index)
        if (!pn) continue
        const dx = pn.x - pi.x
        const dy = pn.y - pi.y
        const dist = Math.sqrt(dx * dx + dy * dy) || 1
        const idealDist = 120 - n.overlap * 15
        const spring = (dist - idealDist) * kSpring
        forces.get(i)!.fx += (dx / dist) * spring
        forces.get(i)!.fy += (dy / dist) * spring
        if (forces.has(n.index)) {
          forces.get(n.index)!.fx -= (dx / dist) * spring
          forces.get(n.index)!.fy -= (dy / dist) * spring
        }
      }
    }

    for (const i of indices) {
      const p = positions.get(i)!
      const f = forces.get(i)!
      f.fx += (centerX - p.x) * kCenter
      f.fy += (centerY - p.y) * kCenter
    }

    for (const i of indices) {
      const p = positions.get(i)!
      const f = forces.get(i)!
      p.x += f.fx * damping
      p.y += f.fy * damping
    }

    for (let a = 0; a < n; a++) {
      for (let b = a + 1; b < n; b++) {
        const ia = indices[a]
        const ib = indices[b]
        const pa = positions.get(ia)!
        const pb = positions.get(ib)!
        const dx = pa.x - pb.x
        const dy = pa.y - pb.y
        const dist = Math.sqrt(dx * dx + dy * dy) || 1
        if (dist < minDist) {
          const push = (minDist - dist) / 2
          pa.x += (dx / dist) * push
          pa.y += (dy / dist) * push
          pb.x -= (dx / dist) * push
          pb.y -= (dy / dist) * push
        }
      }
    }

    for (const i of indices) {
      const p = positions.get(i)!
      const dx = p.x - centerX
      const dy = p.y - centerY
      const dist = Math.sqrt(dx * dx + dy * dy)
      if (dist > maxRadius) {
        p.x = centerX + (dx / dist) * maxRadius
        p.y = centerY + (dy / dist) * maxRadius
      }
    }
  }

  return indices.map((origIdx, i) => ({
    id: idMap.get(origIdx)!,
    type: 'okfConcept',
    position: positions.get(origIdx)!,
    data: { concept: null as any, index: i, origIndex: origIdx },
  }))
}

// (layout is now handled by forceDirectedLayout above)

function OKFConceptNode({ data }: { data: { concept: OKFConcept; index: number } }) {
  const { concept } = data
  const color = typeColor(concept.okf_type)
  const Icon = typeIcon(concept.okf_type)
  const isRetrieved = (data as any).isRetrieved

  return (
    <div
      style={{
        padding: '4px 6px',
        borderRadius: '6px',
        border: `${isRetrieved ? 5 : 2}px solid ${color}`,
        background: isRetrieved ? `${color}50` : '#fff',
        fontSize: '9px',
        maxWidth: isRetrieved ? '100px' : '90px',
        boxShadow: isRetrieved ? `0 0 16px ${color}, 0 0 6px ${color}` : '0 1px 3px rgba(0,0,0,0.1)',
        outline: isRetrieved ? `2px solid ${color}` : 'none',
        outlineOffset: '1px',
      }}
    >
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '2px' }}>
        <Icon size={8} color={color} />
        <span style={{ fontWeight: 600, color, fontSize: '7px', textTransform: 'uppercase' }}>
          {isRetrieved ? '★ ' : ''}{concept.okf_type}
        </span>
        {isRetrieved && (
          <span style={{
            fontSize: '7px',
            background: color,
            color: '#fff',
            padding: '1px 4px',
            borderRadius: '3px',
            marginLeft: 'auto',
          }}>
            RAG
          </span>
        )}
      </div>
      <div style={{
        fontWeight: isRetrieved ? 700 : 500,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        color: '#1f2937',
      }}>
        {concept.okf_title}
      </div>
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
    </div>
  )
}

const nodeTypes = { okfConcept: OKFConceptNode }

function OKFGraphInner({
  concepts,
  retrievedDocIds,
  loading,
  error,
}: {
  concepts: OKFConcept[]
  retrievedDocIds: Set<string>
  loading: boolean
  error: string | null
}) {
  const [nodes, setNodes] = useState<Node[]>([])
  const [edges, setEdges] = useState<Edge[]>([])
  const [selectedConcept, setSelectedConcept] = useState<OKFConcept | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerSize, setContainerSize] = useState({ width: 1200, height: 800 })
  const { fitView } = useReactFlow()

  useEffect(() => {
    if (!containerRef.current) return
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        if (width > 0 && height > 0) {
          setContainerSize({ width: Math.round(width), height: Math.round(height) })
        }
      }
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [])

  const isRetrievedConcept = useCallback((concept: OKFConcept) => {
    const cid = concept.okf_concept_id || ''
    const src = concept.source || ''
    for (const rid of retrievedDocIds) {
      if (rid && (cid === rid || src === rid || cid.endsWith(rid) || rid.endsWith(cid) || src.endsWith(rid) || rid.endsWith(src) || cid.includes(rid) || rid.includes(cid) || src.includes(rid) || rid.includes(src))) {
        return true
      }
    }
    return false
  }, [retrievedDocIds])

  const { computedNodes, computedEdges } = useMemo(() => {
    console.log('[OKFGraphView] retrievedDocIds:', [...retrievedDocIds])
    console.log('[OKFGraphView] concepts sources:', concepts.map(c => ({ id: c.okf_concept_id, src: c.source })))
    const { indices, adjacency } = filterRelevantConcepts(concepts, retrievedDocIds)
    const idMap = new Map<number, string>()
    for (let k = 0; k < indices.length; k++) {
      idMap.set(indices[k], `node-${k}`)
    }
    const effWidth = containerSize.width
    const effHeight = containerSize.height
    const laidOut = forceDirectedLayout(indices, adjacency, idMap, effWidth, effHeight)
    const nodes = laidOut.map(node => {
      const origIdx = (node.data as any).origIndex as number
      const concept = concepts[origIdx]
      return {
        ...node,
        data: {
          ...node.data,
          concept,
          isRetrieved: isRetrievedConcept(concept),
        },
      }
    })
    const edges = buildEdges(indices, adjacency, idMap)
    return { computedNodes: nodes, computedEdges: edges }
  }, [concepts, retrievedDocIds, isRetrievedConcept, containerSize])

  useEffect(() => {
    setNodes(computedNodes)
    setEdges(computedEdges)
    if (computedNodes.length > 0) {
      const timer = setTimeout(() => fitView({ padding: 0.1 }), 100)
      return () => clearTimeout(timer)
    }
  }, [computedNodes, computedEdges, fitView])

  useEffect(() => {
    if (!selectedNodeId) {
      setNodes(nds => nds.map(n => ({ ...n, selected: false, style: { ...n.style, opacity: 1 } })))
      setEdges(computedEdges.map(e => {
        const baseWidth = (e.style as any)?.strokeWidth || 2
        const baseOpacity = (e.style as any)?.opacity || 0.5
        return {
          ...e,
          animated: baseWidth > 2.5,
          style: { ...(e.style || {}), opacity: baseOpacity, zIndex: 0 },
          selected: false,
        }
      }))
      return
    }
    const connectedNodeIds = new Set<string>([selectedNodeId])
    const connectedEdgeIds = new Set<string>()
    for (const e of computedEdges) {
      if (e.source === selectedNodeId) {
        connectedNodeIds.add(e.target)
        connectedEdgeIds.add(e.id)
      } else if (e.target === selectedNodeId) {
        connectedNodeIds.add(e.source)
        connectedEdgeIds.add(e.id)
      }
    }
    setNodes(nds => nds.map(n => ({
      ...n,
      selected: n.id === selectedNodeId,
      style: { ...n.style, opacity: connectedNodeIds.has(n.id) ? 1 : 0.25 },
    })))
    setEdges(computedEdges.map(e => {
      const isConn = connectedEdgeIds.has(e.id)
      const baseWidth = (e.style as any)?.strokeWidth || 2
      if (isConn) {
        return {
          ...e,
          animated: true,
          style: { ...(e.style || {}), stroke: '#f59e0b', strokeWidth: Math.max(baseWidth, 4), opacity: 1, zIndex: 10 },
          selected: true,
        }
      }
      return {
        ...e,
        animated: false,
        style: { ...(e.style || {}), opacity: 0.08, zIndex: 0 },
        selected: false,
      }
    }))
  }, [selectedNodeId, computedEdges])

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setNodes(nds => applyNodeChanges(changes, nds)),
    []
  )
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => setEdges(eds => applyEdgeChanges(changes, eds)),
    []
  )

  const onNodeClick = useCallback((_: any, node: Node) => {
    const concept = node.data.concept as OKFConcept
    setSelectedConcept(concept)
    setSelectedNodeId(node.id)
  }, [])

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null)
    setSelectedConcept(null)
  }, [])

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '300px', color: '#6b7280' }}>
        <div style={{ textAlign: 'center' }}>
          <div className="typing-indicator" style={{ marginBottom: '8px' }} />
          <span>Loading graph …</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '300px', color: '#ef4444' }}>
        <span>{error}</span>
      </div>
    )
  }

  if (concepts.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '300px', color: '#6b7280' }}>
        <span>No OKF concepts found</span>
      </div>
    )
  }

  return (
    <div ref={containerRef} style={{ position: 'relative', width: '100%', height: '100%', minHeight: '400px' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.1 }}
        attributionPosition="bottom-left"
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
        <Controls />
      </ReactFlow>

      {/* Legend */}
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        background: 'rgba(255,255,255,0.9)',
        border: '1px solid #e5e7eb',
        borderRadius: '6px',
        padding: '8px 10px',
        fontSize: '10px',
        zIndex: 5,
      }}>
        <div style={{ fontWeight: 600, marginBottom: '4px' }}>OKF Types</div>
        {Object.entries(TYPE_COLORS).map(([type, color]) => (
          <div key={type} style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '2px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: color }} />
            <span style={{ textTransform: 'capitalize' }}>{type}</span>
          </div>
        ))}
        <div style={{ marginTop: '6px', fontWeight: 600, marginBottom: '4px' }}>Edges</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ width: '20px', height: '2px', background: '#6366f1' }} />
          <span>Keyword similarity</span>
        </div>
      </div>

      {/* Retrieved debug badge */}
      {retrievedDocIds.size > 0 && (
        <div style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          background: 'rgba(99,102,241,0.1)',
          border: '1px solid #6366f1',
          borderRadius: '4px',
          padding: '2px 6px',
          fontSize: '9px',
          color: '#6366f1',
          zIndex: 5,
          pointerEvents: 'none',
        }}>
          Retrieved: {retrievedDocIds.size} | Matched: {nodes.filter(n => (n.data as any).isRetrieved).length}
        </div>
      )}

      {/* Detail panel */}
      {selectedConcept && (
        <div style={{
          position: 'absolute',
          bottom: '10px',
          right: '10px',
          maxWidth: '320px',
          maxHeight: '60%',
          overflowY: 'auto',
          background: '#fff',
          border: '1px solid #e5e7eb',
          borderRadius: '8px',
          padding: '12px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
          zIndex: 5,
          wordBreak: 'break-word',
          overflowWrap: 'anywhere',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
            <span style={{
              fontSize: '9px',
              fontWeight: 600,
              textTransform: 'uppercase',
              color: typeColor(selectedConcept.okf_type),
            }}>
              {selectedConcept.okf_type}
            </span>
            <button
              onClick={() => setSelectedConcept(null)}
              style={{ border: 'none', background: 'none', cursor: 'pointer', padding: '0' }}
            >
              <X size={14} color="#6b7280" />
            </button>
          </div>
          <div style={{ fontWeight: 600, fontSize: '13px', marginBottom: '6px', color: '#1f2937' }}>
            {selectedConcept.okf_title}
          </div>
          {selectedConcept.description && (
            <div style={{ fontSize: '11px', color: '#4b5563', marginBottom: '6px', lineHeight: 1.4 }}>
              {selectedConcept.description}
            </div>
          )}
          <div style={{ fontSize: '10px', color: '#6b7280' }}>
            <div><strong>Source:</strong> {selectedConcept.source}</div>
            <div><strong>Collection:</strong> {selectedConcept.collection}</div>
            {selectedConcept.date_downloaded && (
              <div><strong>Downloaded:</strong> {new Date(selectedConcept.date_downloaded).toLocaleDateString("de-DE")}</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export function OKFGraphView(props: {
  concepts: OKFConcept[]
  retrievedDocIds: Set<string>
  loading: boolean
  error: string | null
}) {
  return (
    <ReactFlowProvider>
      <OKFGraphInner {...props} />
    </ReactFlowProvider>
  )
}
