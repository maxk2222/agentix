import { useState } from 'react'
import { ChevronDown, ChevronUp, Plus, Trash2 } from 'lucide-react'

type McpServer = Record<string, unknown>

const strList = (v: unknown): string[] =>
  Array.isArray(v) ? v.map(String) : []

const kvToText = (v: unknown): string => {
  if (!v || typeof v !== 'object' || Array.isArray(v)) return ''
  return Object.entries(v as Record<string, unknown>)
    .map(([k, val]) => `${k}=${String(val)}`)
    .join('\n')
}

const textToKv = (text: string): Record<string, string> => {
  const out: Record<string, string> = {}
  for (const line of text.split('\n')) {
    const idx = line.indexOf('=')
    if (idx > 0) out[line.slice(0, idx).trim()] = line.slice(idx + 1).trim()
  }
  return out
}

const NEW_SERVER: McpServer = {
  name: '',
  transport: 'stdio',
  command: '',
  args: [],
  env: {},
  url: '',
  headers: {},
  enabled: true,
  tool_filter: [],
  deny: [],
  call_timeout: 30,
}

/** Structured editor for mcp_servers.config.servers — writes the same dict
 *  shape app/mcp_client.py's McpServerSpec consumes. */
export function McpServersEditor({
  servers,
  onChange,
  canWrite,
  $t,
}: {
  servers: McpServer[]
  onChange: (v: McpServer[]) => void
  canWrite: boolean
  $t: (key: string) => string
}) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set())

  const update = (idx: number, patch: Partial<McpServer>) => {
    onChange(servers.map((s, i) => (i === idx ? { ...s, ...patch } : s)))
  }

  const toggleExpand = (idx: number) => {
    setExpanded((prev) => {
      const s = new Set(prev)
      if (s.has(idx)) s.delete(idx)
      else s.add(idx)
      return s
    })
  }

  return (
    <div className="mcp-servers-editor">
      {servers.map((srv, idx) => {
        const isStdio = String(srv.transport ?? 'stdio') === 'stdio'
        const open = expanded.has(idx)
        return (
          <div key={idx} className="mcp-server-card">
            <div className="mcp-server-head" onClick={() => toggleExpand(idx)}>
              <label
                className={`rc-toggle sm ${!canWrite ? 'locked' : ''}`}
                onClick={(e) => e.stopPropagation()}
              >
                <input
                  type="checkbox"
                  checked={Boolean(srv.enabled ?? true)}
                  onChange={(e) => update(idx, { enabled: e.target.checked })}
                  disabled={!canWrite}
                />
                <span className="rc-toggle-track"><span className="rc-toggle-knob" /></span>
              </label>
              <span className="mcp-server-name">
                {String(srv.name || $t('tools.mcpUnnamed'))}
              </span>
              <span className="badge mcp">{String(srv.transport ?? 'stdio')}</span>
              {canWrite && (
                <button
                  type="button"
                  className="btn-icon danger"
                  onClick={(e) => {
                    e.stopPropagation()
                    onChange(servers.filter((_, i) => i !== idx))
                  }}
                >
                  <Trash2 size={14} />
                </button>
              )}
              {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </div>

            {open && (
              <div className="mcp-server-body form-grid">
                <div className="form-group">
                  <label>{$t('tools.mcpName')} *</label>
                  <input
                    type="text"
                    value={String(srv.name ?? '')}
                    onChange={(e) => update(idx, { name: e.target.value })}
                    disabled={!canWrite}
                  />
                </div>
                <div className="form-group">
                  <label>{$t('tools.mcpTransport')}</label>
                  <select
                    value={String(srv.transport ?? 'stdio')}
                    onChange={(e) => update(idx, { transport: e.target.value })}
                    disabled={!canWrite}
                  >
                    <option value="stdio">stdio — local command</option>
                    <option value="http">http — streamable HTTP</option>
                  </select>
                </div>

                {isStdio ? (
                  <>
                    <div className="form-group">
                      <label>{$t('tools.mcpCommand')}</label>
                      <input
                        type="text"
                        value={String(srv.command ?? '')}
                        placeholder="npx"
                        onChange={(e) => update(idx, { command: e.target.value })}
                        disabled={!canWrite}
                      />
                    </div>
                    <div className="form-group">
                      <label>{$t('tools.mcpArgs')}</label>
                      <input
                        type="text"
                        value={strList(srv.args).join(' ')}
                        placeholder="-y @modelcontextprotocol/server-filesystem /path"
                        onChange={(e) =>
                          update(idx, { args: e.target.value.split(/\s+/).filter(Boolean) })
                        }
                        disabled={!canWrite}
                      />
                    </div>
                    <div className="form-group wide">
                      <label>{$t('tools.mcpEnv')}</label>
                      <small className="field-help">KEY=VALUE — {$t('tools.mcpKvHint')}</small>
                      <textarea
                        rows={3}
                        className="code-textarea"
                        value={kvToText(srv.env)}
                        onChange={(e) => update(idx, { env: textToKv(e.target.value) })}
                        disabled={!canWrite}
                      />
                    </div>
                  </>
                ) : (
                  <>
                    <div className="form-group wide">
                      <label>{$t('tools.mcpUrl')}</label>
                      <input
                        type="text"
                        value={String(srv.url ?? '')}
                        placeholder="https://mcp.example.org/mcp"
                        onChange={(e) => update(idx, { url: e.target.value })}
                        disabled={!canWrite}
                      />
                    </div>
                    <div className="form-group wide">
                      <label>{$t('tools.mcpHeaders')}</label>
                      <small className="field-help">KEY=VALUE — {$t('tools.mcpKvHint')}</small>
                      <textarea
                        rows={3}
                        className="code-textarea"
                        value={kvToText(srv.headers)}
                        onChange={(e) => update(idx, { headers: textToKv(e.target.value) })}
                        disabled={!canWrite}
                      />
                    </div>
                  </>
                )}

                <div className="form-group">
                  <label>{$t('tools.mcpToolFilter')}</label>
                  <small className="field-help">{$t('tools.mcpToolFilterHelp')}</small>
                  <input
                    type="text"
                    value={strList(srv.tool_filter).join(', ')}
                    placeholder="read_file, write_file"
                    onChange={(e) =>
                      update(idx, {
                        tool_filter: e.target.value.split(',').map((s) => s.trim()).filter(Boolean),
                      })
                    }
                    disabled={!canWrite}
                  />
                </div>
                <div className="form-group">
                  <label>{$t('tools.mcpDeny')}</label>
                  <small className="field-help">{$t('tools.mcpDenyHelp')}</small>
                  <input
                    type="text"
                    value={strList(srv.deny).join(', ')}
                    placeholder="delete_*"
                    onChange={(e) =>
                      update(idx, {
                        deny: e.target.value.split(',').map((s) => s.trim()).filter(Boolean),
                      })
                    }
                    disabled={!canWrite}
                  />
                </div>
                <div className="form-group">
                  <label>{$t('tools.mcpCallTimeout')}</label>
                  <input
                    type="number"
                    min={5}
                    max={600}
                    value={Number(srv.call_timeout ?? 30)}
                    onChange={(e) => update(idx, { call_timeout: Number(e.target.value) })}
                    disabled={!canWrite}
                  />
                </div>
              </div>
            )}
          </div>
        )
      })}

      {canWrite && (
        <button
          type="button"
          className="btn-secondary"
          onClick={() => {
            onChange([...servers, { ...NEW_SERVER }])
            setExpanded((prev) => new Set(prev).add(servers.length))
          }}
        >
          <Plus size={14} /> {$t('tools.mcpAddServer')}
        </button>
      )}
    </div>
  )
}
