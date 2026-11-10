import type { Edge, Node } from '@xyflow/react'
import { nodeTypeOf, ERROR_HANDLE, ELSE_HANDLE, ERROR_PORT_TYPES, TEMPLATE_INPUT_TYPES } from './flowRules'

export type ValidationIssue = {
  level: 'error' | 'warning'
  message: string
  nodeId?: string
}

const VALID_TYPES = new Set([
  'input', 'output',
  // legacy — still supported
  'assistant', 'agent', 'transform', 'merge', 'parallel',
  // V2 types
  'executor', 'router', 'condition', 'llm', 'note',
  // V3 types
  'switch', 'delay', 'json', 'tool',
])

const RULE_OPS = new Set(['contains', 'not_contains', 'equals', 'starts_with', 'ends_with', 'regex'])
const NODE_REF_RE = /\{\{\s*node\.([A-Za-z0-9_.:-]+)\s*\}\}/g

/**
 * Mirror of app/agent_chains_validation.py::validate_flow — keep rules in
 * sync; the backend stays the source of truth on save. Adds client-side
 * 'warning' issues the backend does not report.
 */
export function validateFlowLocal(nodes: Node[], edges: Edge[]): ValidationIssue[] {
  const issues: ValidationIssue[] = []
  const err = (message: string, nodeId?: string) => issues.push({ level: 'error', message, nodeId })
  const warn = (message: string, nodeId?: string) => issues.push({ level: 'warning', message, nodeId })

  if (!nodes.length) err('Flow must contain at least one node')
  if (!edges.length) err('Flow must contain at least one edge')

  const nodeIds = new Set<string>()
  const inputIds: string[] = []
  const outputIds: string[] = []

  for (const node of nodes) {
    if (!node.id) {
      err('A node is missing its id')
      continue
    }
    if (nodeIds.has(node.id)) {
      err(`Duplicate node id: ${node.id}`, node.id)
      continue
    }
    nodeIds.add(node.id)

    const t = nodeTypeOf(node)
    if (!t) {
      err(`Node ${node.id} is missing 'type'`, node.id)
      continue
    }
    if (!VALID_TYPES.has(t)) {
      err(`Node ${node.id} has invalid type '${t}'`, node.id)
      continue
    }
    if (t === 'input') inputIds.push(node.id)
    else if (t === 'output') outputIds.push(node.id)
    else if (t === 'assistant' || t === 'agent') {
      if (!node.data?.assistant_name) {
        err(`Assistant node ${node.id} is missing 'assistant_name'`, node.id)
      }
    } else if (t === 'executor') {
      const kind = (node.data?.target_kind as string) || 'assistant'
      if (!['assistant', 'agent', 'chain'].includes(kind)) {
        err(`Executor node ${node.id} has invalid target_kind '${kind}'`, node.id)
      }
      if (!node.data?.target && !node.data?.assistant_name) {
        err(`Executor node ${node.id} is missing 'target'`, node.id)
      }
    } else if (t === 'router') {
      if (!node.data?.assistant_name) {
        err(`Router node ${node.id} is missing 'assistant_name' (LLM)`, node.id)
      }
      const routes = (node.data?.routes as any[]) || []
      if (!routes.length) {
        err(`Router node ${node.id} needs at least one route`, node.id)
      } else {
        const ids = routes.map((r) => String(r?.id ?? '')).filter(Boolean)
        const names = routes.map((r) => String(r?.name ?? '').trim()).filter(Boolean)
        if (routes.some((r) => r?.id === undefined || r?.id === null || String(r.id) === '')) {
          err(`Router node ${node.id} has a route without an 'id'`, node.id)
        }
        if (new Set(ids).size !== ids.length) {
          err(`Router node ${node.id} has duplicate route ids`, node.id)
        }
        if (routes.some((r) => !String(r?.name ?? '').trim())) {
          err(`Router node ${node.id} has a route without a name`, node.id)
        }
        if (new Set(names).size !== names.length) {
          err(`Router node ${node.id} has duplicate route names`, node.id)
        }
      }
    } else if (t === 'condition') {
      const rule = node.data?.rule as any
      if (!rule?.op || rule?.value === undefined) {
        err(`Condition node ${node.id} is missing rule.op/rule.value`, node.id)
      }
    } else if (t === 'llm') {
      if (!node.data?.assistant_name) {
        err(`LLM node ${node.id} is missing 'assistant_name' (model)`, node.id)
      }
      if (!node.data?.prompt_template) {
        err(`LLM node ${node.id} is missing 'prompt_template'`, node.id)
      }
    } else if (t === 'switch') {
      const cases = (node.data?.cases as any[]) || []
      if (!cases.length) {
        err(`Switch node ${node.id} needs at least one case`, node.id)
      } else {
        const ids = cases.map((c) => String(c?.id ?? '')).filter(Boolean)
        if (cases.some((c) => c?.id === undefined || c?.id === null || String(c.id) === '')) {
          err(`Switch node ${node.id} has a case without an 'id'`, node.id)
        }
        if (new Set(ids).size !== ids.length) {
          err(`Switch node ${node.id} has duplicate case ids`, node.id)
        }
        for (const c of cases) {
          if (c?.op && !RULE_OPS.has(String(c.op))) {
            err(`Switch node ${node.id} case '${c.id}' has invalid op '${c.op}'`, node.id)
          }
        }
      }
    } else if (t === 'delay') {
      const ms = node.data?.ms
      if (ms === undefined || typeof ms !== 'number' || ms < 0) {
        err(`Delay node ${node.id} needs a non-negative 'ms' value`, node.id)
      }
    } else if (t === 'json') {
      if (!node.data?.assistant_name) {
        err(`JSON node ${node.id} is missing 'assistant_name' (model)`, node.id)
      }
      const schema = node.data?.json_schema
      if (typeof schema === 'string' && schema.trim()) {
        try { JSON.parse(schema) } catch {
          err(`JSON node ${node.id} has invalid 'json_schema' (not parseable)`, node.id)
        }
      } else if (schema !== undefined && schema !== null && typeof schema !== 'object') {
        err(`JSON node ${node.id} has invalid 'json_schema' (must be object)`, node.id)
      }
    } else if (t === 'tool') {
      if (!node.data?.tool_name) {
        err(`Tool node ${node.id} is missing 'tool_name'`, node.id)
      }
      const params = node.data?.params
      if (typeof params === 'string' && params.trim()) {
        try { JSON.parse(params) } catch {
          err(`Tool node ${node.id} has invalid 'params' (not parseable JSON)`, node.id)
        }
      } else if (params !== undefined && params !== null && typeof params !== 'object') {
        err(`Tool node ${node.id} 'params' must be an object`, node.id)
      }
    }
  }

  const validPairs: Array<[string, string]> = []
  for (const [idx, edge] of edges.entries()) {
    if (!edge.source) {
      err(`Edge at index ${idx} is missing 'source'`)
      continue
    }
    if (!edge.target) {
      err(`Edge at index ${idx} is missing 'target'`)
      continue
    }
    if (!nodeIds.has(edge.source)) {
      err(`Edge references unknown source node: ${edge.source}`)
      continue
    }
    if (!nodeIds.has(edge.target)) {
      err(`Edge references unknown target node: ${edge.target}`)
      continue
    }
    if (edge.source === edge.target) {
      err(`Self-loop detected on node ${edge.source}`, edge.source)
      continue
    }
    const srcNode = nodes.find((n) => n.id === edge.source)
    const tgtNode = nodes.find((n) => n.id === edge.target)
    const srcType = nodeTypeOf(srcNode)
    const tgtType = nodeTypeOf(tgtNode)
    if (srcType === 'note' || tgtType === 'note') {
      err('Note nodes cannot be connected by edges', srcType === 'note' ? edge.source : edge.target)
      continue
    }
    const handle = String(edge.sourceHandle || '')
    if (srcType === 'router') {
      const routes = (srcNode?.data?.routes as any[]) || []
      const routeIds = new Set(routes.map((r) => `route-${r?.id}`))
      if (handle !== ERROR_HANDLE && !routeIds.has(handle)) {
        err(`Router edge ${edge.source}→${edge.target} needs a route port sourceHandle`, edge.source)
        continue
      }
    } else if (srcType === 'switch') {
      const cases = (srcNode?.data?.cases as any[]) || []
      const caseIds = new Set(cases.map((c) => `route-${c?.id}`))
      if (!caseIds.has(handle) && handle !== ERROR_HANDLE && handle !== ELSE_HANDLE) {
        err(`Switch edge ${edge.source}→${edge.target} needs a case port or 'else' sourceHandle`, edge.source)
        continue
      }
    } else if (srcType === 'condition') {
      if (handle !== 'cond-true' && handle !== 'cond-false' && handle !== ERROR_HANDLE) {
        err(`Condition edge ${edge.source}→${edge.target} needs sourceHandle 'cond-true' or 'cond-false'`, edge.source)
        continue
      }
    } else if (handle === ERROR_HANDLE && !ERROR_PORT_TYPES.has(srcType)) {
      err(`on_error edge not allowed from node type '${srcType}' (${edge.source})`, edge.source)
      continue
    }
    if (handle === ERROR_HANDLE && (tgtType === 'input' || tgtType === 'output')) {
      err(`on_error edge cannot target an '${tgtType}' node (${edge.source}→${edge.target})`, edge.source)
      continue
    }
    validPairs.push([edge.source, edge.target])
  }

  if (inputIds.length === 0) err("Flow must contain exactly one 'input' node (found 0)")
  else if (inputIds.length > 1) err(`Flow must contain exactly one 'input' node (found ${inputIds.length})`)

  if (outputIds.length === 0) err("Flow must contain at least one 'output' node")

  if (inputIds.length && !edges.some((e) => e.source === inputIds[0])) {
    err('Input node must have at least one outgoing edge', inputIds[0])
  }
  if (outputIds.length && !edges.some((e) => outputIds.includes(e.target))) {
    err('Output node must have at least one incoming edge')
  }

  const adjacency = new Map<string, string[]>()
  for (const id of nodeIds) adjacency.set(id, [])
  for (const [s, t] of validPairs) adjacency.get(s)!.push(t)

  // Cycle detection (mirrors backend DFS)
  const visited = new Set<string>()
  const stack = new Set<string>()
  const hasCycle = (id: string): boolean => {
    visited.add(id)
    stack.add(id)
    for (const next of adjacency.get(id) || []) {
      if (!visited.has(next)) {
        if (hasCycle(next)) return true
      } else if (stack.has(next)) {
        return true
      }
    }
    stack.delete(id)
    return false
  }
  for (const id of nodeIds) {
    if (!visited.has(id) && hasCycle(id)) {
      err('Flow contains a cycle (cycles are not allowed in V1)', id)
      break
    }
  }

  // Reachability from input
  if (inputIds.length && validPairs.length) {
    const reachable = new Set<string>()
    const queue = [inputIds[0]]
    while (queue.length) {
      const cur = queue.shift()!
      if (reachable.has(cur)) continue
      reachable.add(cur)
      for (const n of adjacency.get(cur) || []) {
        if (!reachable.has(n)) queue.push(n)
      }
    }
    const noteIds = new Set(nodes.filter((n) => nodeTypeOf(n) === 'note').map((n) => n.id))
    const unreachable = [...nodeIds].filter((id) => !reachable.has(id) && !noteIds.has(id))
    if (unreachable.length) {
      err(`Nodes unreachable from input: ${unreachable.sort().join(', ')}`, unreachable[0])
    }
  }

  /* ----------------------- client-side warnings ----------------------- */

  if (outputIds.length > 1) {
    for (const id of outputIds.slice(1)) {
      warn('Multiple output nodes — only the first one is used by the runner', id)
    }
  }

  for (const node of nodes) {
    const t = nodeTypeOf(node)
    if (t === 'transform' && !String(node.data?.template ?? '').includes('{{input}}')) {
      warn(`Transform node ${node.id} has no {{input}} placeholder — acts as pass-through`, node.id)
    }
    if (t === 'merge') {
      const inDegree = edges.filter((e) => e.target === node.id).length
      if (inDegree < 2) warn(`Merge node ${node.id} has fewer than 2 inputs`, node.id)
    }
    if (t === 'router') {
      const routes = (node.data?.routes as any[]) || []
      for (const r of routes) {
        if (String(r?.name ?? '').trim() && !String(r?.description ?? '').trim()) {
          warn(`Router node ${node.id}: route '${r.name}' has no description — the LLM decides on the name alone`, node.id)
        }
      }
    }
  }

  /* ---------- template refs: {{node.X}} must point at ancestors ---------- */

  const ancestorsOf = (id: string): Set<string> => {
    const seen = new Set<string>()
    const stack = validPairs.filter(([, t]) => t === id).map(([s]) => s)
    while (stack.length) {
      const cur = stack.pop()!
      if (seen.has(cur)) continue
      seen.add(cur)
      for (const [s, t] of validPairs) if (t === cur) stack.push(s)
    }
    return seen
  }
  const checkRefs = (nodeId: string, field: string, value: unknown) => {
    const anc = ancestorsOf(nodeId)
    for (const m of String(value).matchAll(NODE_REF_RE)) {
      let ref = m[1]
      if (ref.endsWith('.route')) ref = ref.slice(0, -'.route'.length)
      if (ref === nodeId) {
        err(`Node ${nodeId}: '${field}' references itself`, nodeId)
      } else if (!nodeIds.has(ref)) {
        err(`Node ${nodeId}: '${field}' references unknown node '${ref}'`, nodeId)
      } else if (!anc.has(ref)) {
        err(`Node ${nodeId}: '${field}' references '${ref}' which is not an upstream ancestor`, nodeId)
      }
    }
  }
  for (const node of nodes) {
    const t = nodeTypeOf(node)
    const data = (node.data || {}) as Record<string, any>
    if (data.input_template && !TEMPLATE_INPUT_TYPES.has(t)) {
      err(`Node ${node.id}: 'input_template' is not supported on type '${t}'`, node.id)
    }
    for (const field of ['input_template', 'prompt_template', 'template'] as const) {
      if (data[field]) checkRefs(node.id, field, data[field])
    }
    if (data.params && typeof data.params === 'object') {
      checkRefs(node.id, 'params', JSON.stringify(data.params))
    } else if (typeof data.params === 'string' && data.params.trim()) {
      checkRefs(node.id, 'params', data.params)
    }
  }

  return issues
}
