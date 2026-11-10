import { useState } from 'react'
import type { Node } from '@xyflow/react'
import { ChevronDown, ChevronRight, Loader, Play, X } from 'lucide-react'
import type { AgentChainTestRunResponse } from '../../api/client'

export function ChainTestPanel({
  nodes,
  running,
  result,
  error,
  onRun,
  onCancel,
  onClose,
  onFocusNode,
  $t,
}: {
  nodes: Node[]
  running: boolean
  result: AgentChainTestRunResponse | null
  error: string | null
  onRun: (input: string) => void
  onCancel: () => void
  onClose: () => void
  onFocusNode: (id: string) => void
  $t: (k: string) => string
}) {
  const [input, setInput] = useState('')
  const [openNodes, setOpenNodes] = useState<Set<string>>(new Set())
  const [showReasoning, setShowReasoning] = useState(false)

  const labelFor = (id: string) => {
    const n = nodes.find((x) => x.id === id)
    const data = n?.data as any
    return data?.label || data?.assistant_name || data?.target || id
  }

  const nodeOutputs = result?.node_outputs ? Object.entries(result.node_outputs) : []
  const nodeStatus = result?.node_status || {}
  const skippedIds = Object.entries(nodeStatus)
    .filter(([, st]) => st?.status === 'skipped')
    .map(([id]) => id)
  const statusOf = (id: string) => nodeStatus[id]?.status as string | undefined
  const chosenOf = (id: string) => nodeStatus[id]?.chosen_route
  const toolCalls = (result?.tool_calls as any[] | undefined) || []

  const toggleNode = (id: string) => {
    setOpenNodes((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return (
    <div className="chain-test-panel">
      <div className="chain-test-head">
        <span>{$t('agentChains.test.title')}</span>
        <button type="button" className="chain-tool-btn" onClick={onClose} title={$t('common.close')}>
          <X size={13} />
        </button>
      </div>

      <textarea
        rows={3}
        className="chain-test-input"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder={$t('agentChains.test.placeholder')}
        disabled={running}
      />
      <div className="chain-test-actions">
        {running ? (
          <button type="button" className="btn-secondary" onClick={onCancel}>
            <Loader size={13} className="spin" /> {$t('agentChains.test.cancel')}
          </button>
        ) : (
          <button
            type="button"
            className="btn-primary"
            onClick={() => onRun(input)}
            disabled={!input.trim()}
          >
            <Play size={13} /> {$t('agentChains.test.run')}
          </button>
        )}
      </div>

      {running && (
        <div className="chain-test-status text-muted">
          <Loader size={13} className="spin" /> {$t('agentChains.test.running')}
        </div>
      )}

      {error && <div className="error-banner chain-test-error">{error}</div>}

      {result && (
        <div className="chain-test-results">
          <div className="chain-test-section-title">{$t('agentChains.test.answer')}</div>
          <div className="chain-test-answer">{result.answer || `(${$t('agentChains.test.empty')})`}</div>

          {result.reasoning && (
            <div className="chain-test-block">
              <button type="button" className="chain-test-toggle" onClick={() => setShowReasoning((v) => !v)}>
                {showReasoning ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                {$t('agentChains.test.reasoning')}
              </button>
              {showReasoning && <pre className="chain-test-pre">{result.reasoning}</pre>}
            </div>
          )}

          {nodeOutputs.length > 0 && (
            <div className="chain-test-block">
              <div className="chain-test-section-title">{$t('agentChains.test.nodeOutputs')}</div>
              {nodeOutputs.map(([id, content]) => (
                <div key={id} className="chain-node-output-item">
                  <button type="button" className="chain-test-toggle" onClick={() => toggleNode(id)}>
                    {openNodes.has(id) ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                    {statusOf(id) && (
                      <span className={`chain-status-dot chain-status-dot--${statusOf(id)}`} />
                    )}
                    <span
                      className="chain-node-output-label"
                      onClick={(e) => {
                        e.stopPropagation()
                        onFocusNode(id)
                      }}
                      title={$t('agentChains.validation.clickToFocus')}
                    >
                      {labelFor(id)}
                    </span>
                    {chosenOf(id) && (
                      <span className="badge info chain-route-badge">→ {chosenOf(id)}</span>
                    )}
                  </button>
                  {openNodes.has(id) && (
                    <>
                      {nodeStatus[id]?.error && (
                        <div className="error-banner chain-node-error">{nodeStatus[id]?.error}</div>
                      )}
                      <pre className="chain-test-pre">{content || '—'}</pre>
                    </>
                  )}
                </div>
              ))}
            </div>
          )}

          {skippedIds.length > 0 && (
            <div className="chain-test-block">
              <div className="chain-test-section-title">{$t('agentChains.test.skipped')}</div>
              <div className="chain-test-skipped">
                {skippedIds.map((id) => (
                  <button
                    key={id}
                    type="button"
                    className="chain-skipped-chip"
                    onClick={() => onFocusNode(id)}
                    title={$t('agentChains.validation.clickToFocus')}
                  >
                    {labelFor(id)}
                  </button>
                ))}
              </div>
            </div>
          )}

          {toolCalls.length > 0 && (
            <div className="chain-test-block">
              <div className="chain-test-section-title">{$t('agentChains.test.toolCalls')}</div>
              <div className="chain-test-toolcalls">
                {toolCalls.slice(0, 12).map((tc: any, i) => (
                  <span key={i} className="badge info">
                    {tc?.name || tc?.tool || `#${i + 1}`}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
