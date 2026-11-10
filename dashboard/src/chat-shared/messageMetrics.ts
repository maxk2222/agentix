import type { ResponseMetrics } from '../components/ResponseMetrics'

/* Rebuild the metrics badge from persisted retrieval_info after a reload.
   New turns carry a complete `_msg_metrics` payload (same dict as the SSE
   trailer); older turns only have `_llm_usage_metadata` +
   `_final_answer_usage`, from which the badge is derived as a fallback. */
export function metricsFromRetrievalInfo(info: any): ResponseMetrics | undefined {
  if (!info) return undefined
  const persisted = info._msg_metrics
  if (persisted && typeof persisted === 'object') {
    return {
      prompt_tokens: persisted.prompt_tokens ?? null,
      completion_tokens: persisted.completion_tokens ?? null,
      answer_tokens: persisted.answer_tokens ?? null,
      total_tokens: persisted.total_tokens ?? null,
      reasoning_tokens: persisted.reasoning_tokens ?? null,
      cache_tokens: persisted.cache_tokens ?? null,
      tool_result_tokens: persisted.tool_result_tokens ?? null,
      rag_context_tokens: persisted.rag_context_tokens ?? info.rag_context_tokens ?? null,
      observer_tokens: persisted.observer_tokens ?? null,
      compression_tokens: persisted.compression_tokens ?? null,
      subagent_tokens: persisted.subagent_tokens ?? null,
      final_prompt_tokens: persisted.final_prompt_tokens ?? null,
      final_completion_tokens: persisted.final_completion_tokens ?? null,
      final_reasoning_tokens: persisted.final_reasoning_tokens ?? null,
      final_answer_ms: persisted.final_answer_ms ?? null,
      response_time_ms: persisted.response_time_ms ?? null,
      time_to_first_token_ms: persisted.time_to_first_token_ms ?? null,
      time_to_first_answer_token_ms: persisted.time_to_first_answer_token_ms ?? null,
      time_to_first_reasoning_token_ms: persisted.time_to_first_reasoning_token_ms ?? null,
      tokens_per_second: persisted.tokens_per_second ?? null,
      agent_iterations: persisted.agent_iterations ?? null,
      llm_calls: persisted.llm_calls ?? null,
      subagent_count: persisted.subagent_count ?? null,
      per_call: Array.isArray(persisted.per_call) ? persisted.per_call : null,
      estimated: persisted.estimated ?? null,
    }
  }
  // Legacy fallback — derive from the aggregated usage metadata. TTFT/answer-
  // start timings were never persisted on old turns, so they stay null.
  const usage = info._llm_usage_metadata
  const finalUsage = info._final_answer_usage
  const auxList = Array.isArray(info._compression_usage) ? info._compression_usage : []
  if (!usage && !finalUsage && auxList.length === 0) return undefined
  let auxPrompt = 0, auxCompletion = 0, auxTotal = 0, auxReasoning = 0, auxCache = 0
  for (const u of auxList) {
    auxPrompt += u?.prompt_tokens ?? 0
    auxCompletion += u?.completion_tokens ?? 0
    auxTotal += u?.total_tokens ?? 0
    auxReasoning += u?.reasoning_tokens ?? 0
    auxCache += u?.cache_tokens ?? 0
  }
  const faCompletion = finalUsage?.completion_tokens
  const faMs = finalUsage?._response_time_ms
  const tps = faCompletion && faMs ? faCompletion / (faMs / 1000) : undefined
  const hasAux = auxList.length > 0
  const prompt = (usage?.prompt_tokens ?? null) != null || hasAux ? (usage?.prompt_tokens ?? 0) + auxPrompt : null
  const completion = (usage?.completion_tokens ?? null) != null || hasAux ? (usage?.completion_tokens ?? 0) + auxCompletion : null
  const reasoning = (usage?.reasoning_tokens ?? null) != null || hasAux ? (usage?.reasoning_tokens ?? 0) + auxReasoning : null
  const total = (usage?.total_tokens ?? null) != null || hasAux ? (usage?.total_tokens ?? 0) + auxTotal : null
  const cache = (usage?.cache_tokens ?? null) != null || auxCache ? (usage?.cache_tokens ?? 0) + auxCache : null
  const iterations = usage?.iterations ?? null
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    answer_tokens: completion != null ? Math.max(completion - (reasoning ?? 0), 0) : null,
    total_tokens: total,
    reasoning_tokens: reasoning,
    cache_tokens: cache,
    tool_result_tokens: usage?.tool_result_tokens ?? null,
    rag_context_tokens: info.rag_context_tokens ?? null,
    observer_tokens: usage?.observer_tokens ?? null,
    subagent_tokens: usage?.subagent_tokens ?? null,
    compression_tokens: (usage?.compression_tokens ?? 0) + auxTotal || null,
    final_prompt_tokens: finalUsage?.prompt_tokens ?? null,
    final_completion_tokens: finalUsage?.completion_tokens ?? null,
    final_reasoning_tokens: finalUsage?.reasoning_tokens ?? null,
    final_answer_ms: faMs ?? null,
    response_time_ms: null,
    time_to_first_token_ms: null,
    time_to_first_answer_token_ms: null,
    tokens_per_second: tps ?? null,
    agent_iterations: iterations,
    llm_calls: (iterations ?? 0) + (usage?.compression_calls ?? 0) + (usage?.observer_calls ?? 0) + auxList.length || null,
    per_call: Array.isArray(usage?.per_call) ? usage.per_call : null,
    estimated: (usage?.estimated_iterations ?? 0) > 0 ? true : null,
  }
}
