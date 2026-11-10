import type { SubagentEvent, SubagentMeta } from '../api/client'
import { INTERNAL_TOOL_NAMES, type ToolCallLive } from './ChatElements'
import { mergeSubagents } from './SubagentTags'

export interface ToolUsageStat {
  name: string
  count: number
  running: number
}

/* Aggregate tool usage across the root agent and its whole subagent tree.

   Pure recompute over the live accumulators — memoized by callers per render,
   so stream replays/resume snapshots can never double-count.

   Sources:
   - root:      msg.toolCalls (minus internal coordination tools; `running`
                placeholders count as in-flight)
   - subagents: each merged SubagentMeta's `activity[]` — one entry per
                completed call. `nested_tool_calls` mirrors activity 1:1
                (subagents.py appends both together) so it must NOT be
                counted again — it is only used to recurse into
                `metadata.subagent` children (depth >= 2).
   - in-flight: `activity_item` (transient per-event field) counts only while
                the subagent's status is 'running'. */
export function aggregateToolUsage(
  toolCalls?: ToolCallLive[],
  subagentEvents?: SubagentEvent[],
): ToolUsageStat[] {
  const acc = new Map<string, { count: number; running: number }>()
  const bump = (name: string, running: boolean) => {
    if (!name || INTERNAL_TOOL_NAMES.has(name)) return
    const stat = acc.get(name) || { count: 0, running: 0 }
    stat.count += 1
    if (running) stat.running += 1
    acc.set(name, stat)
  }

  for (const call of toolCalls || []) {
    bump(call.tool_name, !!call.running)
  }

  const walk = (meta: SubagentMeta) => {
    for (const item of meta.activity || []) {
      bump(item.tool_name, item.status === 'running')
    }
    const inFlight = meta.status === 'running' ? meta.activity_item : null
    if (inFlight) bump(inFlight.tool_name, true)
    for (const call of meta.nested_tool_calls || []) {
      const child = call.metadata?.subagent
      if (child) walk(child)
    }
  }
  for (const meta of mergeSubagents(subagentEvents || [], toolCalls || [])) {
    walk(meta)
  }

  return [...acc.entries()].map(([name, stat]) => ({ name, ...stat }))
}
