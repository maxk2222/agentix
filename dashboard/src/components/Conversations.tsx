import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { agentMemory, conversations, type ConversationMessage, type ConversationSummary, type ConversationUser } from '../api/client'
import { useData } from '../api/DataContext'
import { ArrowDown, ArrowUp, ArrowUpDown, Bot, Brain, BookOpen, Check, Copy, Download, Eye, Info, KeyRound, MessageSquareOff, RefreshCw, Save, Search, Smile, Trash2, User, UserX, X } from 'lucide-react'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'
import ConfirmModal from './ConfirmModal'
import AlertModal from './AlertModal'

function toInputDateValue(date: Date): string {
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000)
  return local.toISOString().slice(0, 16)
}

function fromInputDateValue(value: string): string | undefined {
  if (!value) return undefined
  return new Date(value).toISOString()
}

export default function Conversations() {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const [searchParams, setSearchParams] = useSearchParams()

  function formatDateTime(value: string): string {
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) return value
    return date.toLocaleString(language === 'de' ? 'de-DE' : 'en-GB', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  }

  type DetailState = {
    conversationId: string
    row?: ConversationSummary
    loading: boolean
    error: string
    messages: ConversationMessage[]
  }

  type MetadataState = {
    conversationId: string
    loading: boolean
    error: string
    data: ConversationSummary | null
  }

  const initialTo = useMemo(() => new Date(), [])
  const initialFrom = useMemo(() => new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), [])

  const [dateFrom, setDateFrom] = useState(toInputDateValue(initialFrom))
  const [dateTo, setDateTo] = useState(toInputDateValue(initialTo))
  const [searchInput, setSearchInput] = useState('')
  const [query, setQuery] = useState('')
  const [userFilter, setUserFilter] = useState('')
  const [model, setModel] = useState('')
  const [minUserMessages, setMinUserMessages] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(50)
  const [convUsers, setConvUsers] = useState<ConversationUser[]>([])
  const [anonCount, setAnonCount] = useState(0)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const copyTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const { getModels } = useData()
  const [models, setModels] = useState<string[]>([])
  const [rows, setRows] = useState<ConversationSummary[]>([])
  const [total, setTotal] = useState(0)
  const [totalPages, setTotalPages] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const [detail, setDetail] = useState<DetailState | null>(null)
  const [metadata, setMetadata] = useState<MetadataState | null>(null)

  const [exportFormat, setExportFormat] = useState<'json' | 'csv' | 'txt'>('json')
  const [exporting, setExporting] = useState(false)

  const [sortBy, setSortBy] = useState('last_message_at')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')

  const [detailSortOrder, setDetailSortOrder] = useState<'asc' | 'desc'>('asc')

  type MemoryEntry = { assistant_name: string; memory: string; updated_at: string | null }
  type MemoryModalState = {
    conversationId: string
    loading: boolean
    error: string
    memories: MemoryEntry[]
  }

  const [memoryMap, setMemoryMap] = useState<Set<string>>(new Set())
  const [memoryModal, setMemoryModal] = useState<MemoryModalState | null>(null)
  const [editingMemory, setEditingMemory] = useState<{ assistant_name: string; memory: string } | null>(null)
  const [notice, setNotice] = useState<{ message: string; danger?: boolean } | null>(null)
  const [memDeleteTarget, setMemDeleteTarget] = useState<{ conversationId: string; assistantName: string } | null>(null)
  const [selectedRAGInfo, setSelectedRAGInfo] = useState<{
    status: string
    chunks_retrieved: number
    rag_context_tokens: number
    rag_context: Array<{ content: string; metadata?: Record<string, unknown> }>
  } | null>(null)

  useEffect(() => {
    let cancelled = false
    async function loadModels() {
      try {
        const list = await getModels()
        if (!cancelled) setModels(list)
      } catch {
        if (!cancelled) setModels([])
      }
    }
    loadModels()
    return () => {
      cancelled = true
    }
  }, [])

  // Conversation owners for the user filter dropdown
  useEffect(() => {
    let cancelled = false
    conversations.listUsers().then((res) => {
      if (!cancelled) {
        setConvUsers(res.users)
        setAnonCount(res.anonymous_count)
      }
    }).catch(() => {
      if (!cancelled) {
        setConvUsers([])
        setAnonCount(0)
      }
    })
    return () => {
      cancelled = true
    }
  }, [])

  // Debounce the title/id search input
  useEffect(() => {
    const timer = setTimeout(() => {
      setQuery(searchInput.trim())
      setPage(1)
    }, 300)
    return () => clearTimeout(timer)
  }, [searchInput])

  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true)
      setError('')
      try {
        const result = await conversations.list({
          date_from: fromInputDateValue(dateFrom),
          date_to: fromInputDateValue(dateTo),
          model: model || undefined,
          min_user_messages: minUserMessages,
          user_id: userFilter || undefined,
          query: query || undefined,
          page,
          page_size: pageSize,
          sort_by: sortBy,
          sort_order: sortOrder,
        })
        if (!cancelled) {
          setRows(result.items)
          setTotal(result.total)
          setTotalPages(result.total_pages)
          // Check which conversations have agent memory
          const ids = result.items.map((r) => r.conversation_id)
          if (ids.length > 0) {
            agentMemory.check(ids).then((check) => {
              if (!cancelled) setMemoryMap(new Set(check.conversation_ids_with_memory))
            }).catch(() => {
              if (!cancelled) setMemoryMap(new Set())
            })
          }
        }
      } catch (e: any) {
        if (!cancelled) setError(String(e.message || e))
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [dateFrom, dateTo, query, userFilter, model, minUserMessages, page, pageSize, sortBy, sortOrder])

  function toggleSort(column: string) {
    setPage(1)
    if (sortBy === column) {
      setSortOrder((o) => (o === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortBy(column)
      setSortOrder(column === 'title' || column === 'user' || column === 'model_name' ? 'asc' : 'desc')
    }
  }

  function sortIcon(column: string) {
    if (sortBy !== column) return <ArrowUpDown size={12} className="sort-icon sort-icon-idle" />
    return sortOrder === 'asc' ? <ArrowUp size={12} className="sort-icon" /> : <ArrowDown size={12} className="sort-icon" />
  }

  function copyConversationId(id: string) {
    navigator.clipboard.writeText(id).then(() => {
      setCopiedId(id)
      if (copyTimer.current) clearTimeout(copyTimer.current)
      copyTimer.current = setTimeout(() => setCopiedId(null), 1500)
    }).catch(() => {})
  }

  function userInitials(row: ConversationSummary): string {
    const source = row.display_name || row.username || ''
    return source
      .split(/[\s@._-]+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part[0]?.toUpperCase() || '')
      .join('') || '·'
  }

  const activeFilterCount =
    (query ? 1 : 0) +
    (userFilter ? 1 : 0) +
    (model ? 1 : 0) +
    (minUserMessages > 0 ? 1 : 0) +
    (dateFrom !== toInputDateValue(initialFrom) || dateTo !== toInputDateValue(initialTo) ? 1 : 0)

  function resetFilters() {
    setSearchInput('')
    setQuery('')
    setUserFilter('')
    setModel('')
    setMinUserMessages(0)
    setDateFrom(toInputDateValue(initialFrom))
    setDateTo(toInputDateValue(initialTo))
    setSortBy('last_message_at')
    setSortOrder('desc')
    setPage(1)
  }

  function pageNumbers(): (number | '…')[] {
    const items: (number | '…')[] = []
    const windowStart = Math.max(2, page - 2)
    const windowEnd = Math.min(totalPages - 1, page + 2)
    items.push(1)
    if (windowStart > 2) items.push('…')
    for (let p = windowStart; p <= windowEnd; p++) items.push(p)
    if (windowEnd < totalPages - 1) items.push('…')
    if (totalPages > 1) items.push(totalPages)
    return items
  }

  async function openDetail(row: ConversationSummary) {
    const conversationId = row.conversation_id
    setDetail({ conversationId, row, loading: true, error: '', messages: [] })
    try {
      const response = await conversations.detail(conversationId)
      const normalizedMessages = response.messages.map((msg) => {
        if (!msg.emotion || !msg.emotion.phases || Array.isArray(msg.emotion.phases)) {
          return msg
        }
        const phases = msg.emotion.phases as Record<string, any>
        return {
          ...msg,
          emotion: {
            ...msg.emotion,
            phases: Object.entries(phases).map(([phase, data]) => ({
              phase,
              primary_emotion: data.primary_emotion,
              confidence: data.confidence,
              intensity: data.intensity,
              source: data.source || `phase__${phase}`,
              timestamp_ms: data.timestamp_ms,
              sampled_frames: data.sampled_frames,
              valid_face_frames: data.valid_face_frames,
              face_detection_rate: data.face_detection_rate,
              window_ms: data.window_ms,
            })),
          },
        }
      })
      setDetail({ conversationId, row, loading: false, error: '', messages: normalizedMessages })
    } catch (e: any) {
      setDetail({ conversationId, row, loading: false, error: String(e.message || e), messages: [] })
    }
  }

  // Deep-link: /conversations?conversation_id=<id> opens the detail modal
  // directly (used by overview live-activity cards and conversation links).
  useEffect(() => {
    const cid = searchParams.get('conversation_id')
    if (!cid) return
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev)
      next.delete('conversation_id')
      return next
    }, { replace: true })
    let cancelled = false
    conversations.list({ conversation_id: cid, page_size: 1 })
      .then((r) => {
        if (cancelled) return
        openDetail(r.items[0] ?? ({ conversation_id: cid } as ConversationSummary))
      })
      .catch(() => {
        if (!cancelled) openDetail({ conversation_id: cid } as ConversationSummary)
      })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams])

  async function openMetadata(conversationId: string) {
    setMetadata({ conversationId, loading: true, error: '', data: null })
    try {
      const response = await conversations.metadata(conversationId)
      setMetadata({ conversationId, loading: false, error: '', data: response })
    } catch (e: any) {
      setMetadata({ conversationId, loading: false, error: String(e.message || e), data: null })
    }
  }

  async function openMemoryModal(conversationId: string) {
    setMemoryModal({ conversationId, loading: true, error: '', memories: [] })
    setEditingMemory(null)
    try {
      const response = await agentMemory.get(conversationId)
      setMemoryModal({ conversationId, loading: false, error: '', memories: response.memories })
    } catch (e: any) {
      setMemoryModal({ conversationId, loading: false, error: String(e.message || e), memories: [] })
    }
  }

  async function handleSaveMemory(conversationId: string, assistantName: string, memory: string) {
    try {
      await agentMemory.set(conversationId, assistantName, memory)
      setMemoryModal((prev) => {
        if (!prev) return prev
        return {
          ...prev,
          memories: prev.memories.map((m) =>
            m.assistant_name === assistantName ? { ...m, memory, updated_at: new Date().toISOString() } : m
          ),
        }
      })
      setEditingMemory(null)
      setMemoryMap((prev) => new Set(prev).add(conversationId))
    } catch (e: any) {
      setNotice({ danger: true, message: 'Failed to save memory: ' + String(e.message || e) })
    }
  }

  async function handleDeleteMemory(conversationId: string, assistantName: string) {
    try {
      await agentMemory.delete(conversationId, assistantName)
      setMemoryModal((prev) => {
        if (!prev) return prev
        const next = prev.memories.filter((m) => m.assistant_name !== assistantName)
        return { ...prev, memories: next }
      })
      if (memoryModal && memoryModal.memories.filter((m) => m.assistant_name !== assistantName).length === 0) {
        setMemoryMap((prev) => {
          const next = new Set(prev)
          next.delete(conversationId)
          return next
        })
      }
    } catch (e: any) {
      setNotice({ danger: true, message: 'Failed to delete memory: ' + String(e.message || e) })
    }
  }

  async function handleExport() {
    const parsedFrom = fromInputDateValue(dateFrom)
    const parsedTo = fromInputDateValue(dateTo)
    if (!parsedFrom || !parsedTo) {
      setError($t('conversations.exportDateError'))
      return
    }

    setExporting(true)
    setError('')
    try {
      const blob = await conversations.export({
        date_from: parsedFrom,
        date_to: parsedTo,
        model: model || undefined,
        min_user_messages: minUserMessages,
        user_id: userFilter || undefined,
        query: query || undefined,
        format: exportFormat,
      })

      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')
      link.href = url
      link.download = `conversations-${stamp}.${exportFormat}`
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (e: any) {
      setError(String(e.message || e))
    } finally {
      setExporting(false)
    }
  }

  const canPageBack = page > 1
  const canPageForward = page < totalPages

  return (
    <div>
      <div className="page-header">
        <h2>{$t('conversations.title')}</h2>
        <div className="filter-bar conversations-export-bar">
          <select value={exportFormat} onChange={(e) => setExportFormat(e.target.value as 'json' | 'csv' | 'txt')}>
            <option value="json">JSON</option>
            <option value="csv">CSV</option>
            <option value="txt">TXT</option>
          </select>
          <button className="btn-primary" onClick={handleExport} disabled={exporting}>
            <Download size={16} />
            {exporting ? $t('common.loading') : $t('common.download')}
          </button>
        </div>
      </div>

      <div className="rc-card filter-card">
        <div className="filter-grid">
          <div className="filter-field filter-field-search">
            <label>{$t('conversations.search')}</label>
            <div className="filter-search-wrap">
              <Search size={14} className="filter-search-icon" />
              <input
                type="text"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder={$t('conversations.searchPlaceholder')}
              />
            </div>
          </div>
          <div className="filter-field">
            <label>{$t('conversations.owner')}</label>
            <select
              value={userFilter}
              onChange={(e) => {
                setUserFilter(e.target.value)
                setPage(1)
              }}
            >
              <option value="">{$t('conversations.allUsers')}</option>
              {convUsers.map((u) => (
                <option key={u.user_id} value={u.user_id}>
                  {u.display_name || u.username || u.user_id} ({u.conversation_count})
                </option>
              ))}
              <option value="anonymous" disabled={anonCount === 0}>
                {$t('conversations.anonUser')}{anonCount > 0 ? ` (${anonCount})` : ''}
              </option>
            </select>
          </div>
          <div className="filter-field">
            <label>{$t('overview.model')}</label>
            <select
              value={model}
              onChange={(e) => {
                setModel(e.target.value)
                setPage(1)
              }}
            >
              <option value="">{$t('common.all')} {$t('overview.model')}</option>
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div className="filter-field">
            <label>{$t('conversations.dateFrom')}</label>
            <input
              type="datetime-local"
              value={dateFrom}
              onChange={(e) => {
                setDateFrom(e.target.value)
                setPage(1)
              }}
            />
          </div>
          <div className="filter-field">
            <label>{$t('conversations.dateTo')}</label>
            <input
              type="datetime-local"
              value={dateTo}
              onChange={(e) => {
                setDateTo(e.target.value)
                setPage(1)
              }}
            />
          </div>
          <div className="filter-field">
            <label>{$t('conversations.minMessages')}</label>
            <input
              type="number"
              min={0}
              value={minUserMessages}
              onChange={(e) => {
                const value = Number(e.target.value)
                setMinUserMessages(Number.isFinite(value) && value >= 0 ? value : 0)
                setPage(1)
              }}
            />
          </div>
          <div className="filter-field filter-actions">
            {activeFilterCount > 0 && (
              <span className="filter-active-badge">
                {activeFilterCount} {$t('conversations.activeFilters')}
              </span>
            )}
            <button className="btn-secondary" onClick={resetFilters} disabled={activeFilterCount === 0}>
              {$t('conversations.resetFilters')}
            </button>
            <button className="btn-secondary btn-icon" onClick={() => setPage(1)} title={$t('common.refresh')}>
              <RefreshCw size={14} />
            </button>
          </div>
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}
      {loading && <div className="loading">{$t('common.loading')}</div>}

      {!loading && (
        <>
          <div className="conversations-topline">
            <span>{total.toLocaleString()} {$t('conversations.title').toLowerCase()}</span>
            <span>{$t('common.page')} {page} {$t('common.of')} {totalPages}</span>
          </div>

          <div className="table-card">
            <table className="data-table modern">
              <thead>
                <tr>
                  <th className="sortable-th" onClick={() => toggleSort('title')}>
                    {$t('conversations.convTitle')} {sortIcon('title')}
                  </th>
                  <th className="sortable-th" onClick={() => toggleSort('user')}>
                    {$t('conversations.owner')} {sortIcon('user')}
                  </th>
                  <th className="sortable-th" onClick={() => toggleSort('model_name')}>
                    {$t('overview.model')} {sortIcon('model_name')}
                  </th>
                  <th className="sortable-th" onClick={() => toggleSort('message_count')}>
                    {$t('conversations.messages')} {sortIcon('message_count')}
                  </th>
                  <th className="sortable-th" onClick={() => toggleSort('first_message_at')}>
                    {$t('conversations.firstMessage')} {sortIcon('first_message_at')}
                  </th>
                  <th className="sortable-th" onClick={() => toggleSort('last_message_at')}>
                    {$t('conversations.lastMessage')} {sortIcon('last_message_at')}
                  </th>
                  <th>{$t('conversations.actions')}</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const displayTitle = row.title || row.conv_model_name || null
                  const owner = row.display_name || row.username
                  return (
                    <tr key={row.conversation_id}>
                      <td className="conv-title-cell">
                        <button className="conv-title-link" onClick={() => openDetail(row)} title={displayTitle || row.conversation_id}>
                          {displayTitle || $t('conversations.untitled')}
                        </button>
                        <div className="conv-id-sub">
                          <span title={row.conversation_id}>{row.conversation_id.slice(0, 14)}…</span>
                          <button
                            className="btn-icon conv-copy-btn"
                            onClick={() => copyConversationId(row.conversation_id)}
                            title={$t('conversations.copyId')}
                          >
                            {copiedId === row.conversation_id ? <Check size={12} /> : <Copy size={12} />}
                          </button>
                        </div>
                      </td>
                      <td>
                        {owner ? (
                          <div className="conv-user">
                            <span className="conv-avatar">{userInitials(row)}</span>
                            <div className="conv-user-names">
                              <span className="conv-user-name">{owner}</span>
                              {row.username && row.display_name !== row.username && (
                                <span className="conv-user-sub">@{row.username}</span>
                              )}
                            </div>
                          </div>
                        ) : (
                          <span className="conv-user anon" title={row.conversation_id}>
                            {row.used_api_key ? <KeyRound size={13} /> : <UserX size={13} />}
                            {row.used_api_key ? $t('conversations.apiKey') : $t('conversations.anonUser')}
                          </span>
                        )}
                      </td>
                      <td>{row.model_name || '—'}</td>
                      <td>
                        <span className="conv-msg-count">{row.message_count}</span>
                        <span className="msg-count-sub">{row.user_message_count}u · {row.assistant_message_count}a</span>
                      </td>
                      <td>{formatDateTime(row.first_message_at)}</td>
                      <td>{formatDateTime(row.last_message_at)}</td>
                      <td>
                        <div className="conv-actions">
                          <button className="btn-icon" onClick={() => openDetail(row)} title={$t('conversations.viewDetails')}>
                            <Eye size={16} />
                          </button>
                          <button className="btn-icon" onClick={() => openMetadata(row.conversation_id)} title={$t('conversations.conversationDetails')}>
                            <Info size={16} />
                          </button>
                          {memoryMap.has(row.conversation_id) && (
                            <button
                              className="btn-icon"
                              title="Agent memory available"
                              onClick={() => openMemoryModal(row.conversation_id)}
                              style={{ color: 'var(--accent, #2563eb)' }}
                            >
                              <Brain size={16} />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            {rows.length === 0 && (
              <div className="conv-empty-state">
                <MessageSquareOff size={32} />
                <p>{$t('common.empty')}</p>
              </div>
            )}
          </div>

          <div className="conversations-pagination">
            <button className="btn-secondary" disabled={!canPageBack} onClick={() => setPage((p) => Math.max(1, p - 1))}>
              {$t('common.previous')}
            </button>
            <div className="pagination-pages">
              {pageNumbers().map((p, i) =>
                p === '…' ? (
                  <span key={`ellipsis-${i}`} className="pagination-ellipsis">…</span>
                ) : (
                  <button
                    key={p}
                    className={`page-num ${p === page ? 'active' : ''}`}
                    onClick={() => setPage(p)}
                  >
                    {p}
                  </button>
                )
              )}
            </div>
            <button className="btn-secondary" disabled={!canPageForward} onClick={() => setPage((p) => p + 1)}>
              {$t('common.next')}
            </button>
            <select
              className="page-size-select"
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value))
                setPage(1)
              }}
            >
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>
        </>
      )}

      {detail && (
        <div className="modal-overlay" onClick={() => setDetail(null)}>
          <div className="modal-box preview-modal conversation-modal" onClick={(e) => e.stopPropagation()}>
            <div className="preview-header">
              <div>
                <h4>{detail.row?.title || detail.row?.conv_model_name || $t('conversations.untitled')}</h4>
                <span>{detail.conversationId}</span>
                {detail.row && (
                  <div className="conv-detail-chips">
                    {detail.row.display_name || detail.row.username ? (
                      <span className="conv-chip">
                        <User size={12} /> {detail.row.display_name || detail.row.username}
                        {detail.row.username && detail.row.display_name !== detail.row.username ? ` (@${detail.row.username})` : ''}
                      </span>
                    ) : (
                      <span className="conv-chip conv-chip-anon">
                        <UserX size={12} /> {detail.row.used_api_key ? $t('conversations.apiKey') : $t('conversations.anonUser')}
                      </span>
                    )}
                    {detail.row.model_name && (
                      <span className="conv-chip">
                        <Bot size={12} /> {detail.row.model_name}
                      </span>
                    )}
                    <span className="conv-chip">
                      {detail.row.message_count} {$t('conversations.messages').toLowerCase()}
                    </span>
                  </div>
                )}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <select
                  value={detailSortOrder}
                  onChange={(e) => setDetailSortOrder(e.target.value as 'asc' | 'desc')}
                >
                  <option value="asc">{$t('common.ascending')}</option>
                  <option value="desc">{$t('common.descending')}</option>
                </select>
                <button className="btn-icon" onClick={() => setDetail(null)} title={$t('common.close')}>
                  <X size={18} />
                </button>
              </div>
            </div>
            <div className="preview-content conversation-content">
              {detail.loading && <div className="loading">{$t('common.loading')}</div>}
              {detail.error && <div className="error-box">{detail.error}</div>}
              {!detail.loading && !detail.error && detail.messages.length === 0 && (
                <div className="loading">{$t('common.empty')}</div>
              )}
              {!detail.loading && !detail.error && [...detail.messages].sort((a, b) => {
                const ta = new Date(a.created_at).getTime()
                const tb = new Date(b.created_at).getTime()
                return detailSortOrder === 'asc' ? ta - tb : tb - ta
              }).map((msg: ConversationMessage) => (
                <div className="conversation-turn" key={msg.id}>
                  <div className="conversation-meta">
                    <strong>{formatDateTime(msg.created_at)}</strong>
                    <span>{msg.model_name || '—'}</span>
                    {msg.emotion?.fused_emotion && (
                      <span
                        className={`emotion-badge em-${['angry', 'frustrated', 'happy', 'sad'].includes(msg.emotion.fused_emotion.primary_emotion || '') ? msg.emotion.fused_emotion.primary_emotion : 'neutral'}`}
                        title={`Source: ${msg.emotion.fused_emotion.source}, Confidence: ${msg.emotion.fused_emotion.confidence}, Intensity: ${msg.emotion.fused_emotion.intensity}`}
                      >
                        <Smile size={12} />
                        {msg.emotion.fused_emotion.primary_emotion} ({msg.emotion.fused_emotion.intensity})
                      </span>
                    )}
                  </div>
                  {msg.user_content && (
                    <div className="conversation-bubble user">
                      <div className="conversation-role">{$t('conversations.user')}</div>
                      <div className="conversation-text markdown-body">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.user_content}</ReactMarkdown>
                      </div>
                    </div>
                  )}
                  {msg.emotion && (
                    <details className="conversation-emotion-history">
                      <summary>
                        Emotion History
                        {msg.emotion.fused_emotion && (
                          <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--text-secondary)' }}>
                            — {msg.emotion.fused_emotion.primary_emotion} ({msg.emotion.fused_emotion.confidence}, {msg.emotion.fused_emotion.intensity})
                          </span>
                        )}
                      </summary>
                      <div className="conversation-emotion-history-body">
                        {msg.emotion.fused_emotion && (
                          <div className="emotion-history-row fused">
                            <strong>Fused:</strong>{' '}
                            {msg.emotion.fused_emotion.primary_emotion} — confidence: {msg.emotion.fused_emotion.confidence}, intensity: {msg.emotion.fused_emotion.intensity}, source: {msg.emotion.fused_emotion.source}
                          </div>
                        )}
                        {msg.emotion.text_emotion && (
                          <div className="emotion-history-row">
                            <strong>Text:</strong>{' '}
                            {msg.emotion.text_emotion.primary_emotion} — confidence: {msg.emotion.text_emotion.confidence}, intensity: {msg.emotion.text_emotion.intensity}
                          </div>
                        )}
                        {msg.emotion.visual_pre_send && (
                          <div className="emotion-history-row">
                            <strong>Webcam:</strong>{' '}
                            {msg.emotion.visual_pre_send.primary_emotion} — confidence: {msg.emotion.visual_pre_send.confidence}, intensity: {msg.emotion.visual_pre_send.intensity}
                            {msg.emotion.visual_pre_send.valid_face_frames !== undefined && (
                              <span> (faces: {msg.emotion.visual_pre_send.valid_face_frames}/{msg.emotion.visual_pre_send.sampled_frames}, rate: {Math.round((msg.emotion.visual_pre_send.face_detection_rate || 0) * 100)}%)</span>
                            )}
                          </div>
                        )}
                        {msg.emotion.audio_prosody && (
                          <div className="emotion-history-row">
                            <strong>Audio Prosody:</strong>{' '}
                            {msg.emotion.audio_prosody.primary_emotion} — confidence: {msg.emotion.audio_prosody.confidence}, intensity: {msg.emotion.audio_prosody.intensity}
                          </div>
                        )}
                        {msg.emotion.phases && msg.emotion.phases.length > 0 && (
                          <div className="emotion-history-row">
                            <strong>Phases:</strong>{' '}
                            <div className="emotion-timeline-list">
                              {msg.emotion.phases.map((p, idx) => (
                                <div key={idx} className="emotion-timeline-item">
                                  {p.phase}: {p.primary_emotion} ({p.confidence}, {p.intensity})
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </details>
                  )}
                  {msg.tool_calls && msg.tool_calls.length > 0 && (
                    <details className="conversation-toolcalls">
                      <summary>Tool Calls ({msg.tool_calls.length})</summary>
                      <div className="conversation-toolcalls-body">
                        {msg.tool_calls.map((tc, i) => (
                          <div key={i} className={`toolcall-row ${tc.ok ? 'ok' : 'err'}`}>
                            <span className="toolcall-status">{tc.ok ? '✓' : '✗'}</span>
                            <span className={`toolcall-name ${tc.kind === 'subagent' ? 'subagent-toolcall-name' : ''}`}>
                              {tc.kind === 'subagent' ? `🤖 ${tc.metadata?.subagent?.label || tc.tool_name}` : tc.tool_name}
                            </span>
                            <pre className="toolcall-args">{JSON.stringify(tc.args, null, 2)}</pre>
                            {typeof tc.duration_ms === 'number' && (
                              <span className="toolcall-duration">{tc.duration_ms}ms</span>
                            )}
                            {tc.error && <span className="toolcall-error">{tc.error}</span>}
                            {tc.kind === 'subagent' && tc.metadata?.subagent && (
                              <details className="conversation-subagent-detail">
                                <summary>{tc.metadata.subagent.status} · {tc.metadata.subagent.iterations || 0} iterations</summary>
                                <strong>Task</strong>
                                <pre>{tc.metadata.subagent.task}</pre>
                                {tc.metadata.subagent.answer && (
                                  <>
                                    <strong>Output</strong>
                                    <pre>{tc.metadata.subagent.answer}</pre>
                                  </>
                                )}
                                {tc.metadata.subagent.activity?.map((activity, activityIndex) => (
                                  <div key={activityIndex} className="conversation-subagent-activity">
                                    {activity.tool_name} · {activity.status}
                                    {activity.args_preview ? ` · ${activity.args_preview}` : ''}
                                  </div>
                                ))}
                              </details>
                            )}
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                  {(() => {
                    const trace = (msg.retrieval_info as Record<string, any> | undefined)?._agent_trace
                    const toolCalls = trace?.tool_calls as Array<{
                      tool_name: string
                      args: Record<string, unknown>
                      output?: string
                    }> | undefined
                    if (!toolCalls) return null
                    const codeCalls = toolCalls.filter(
                      (tc) => tc.tool_name === 'code_execution' && tc.args?.code
                    )
                    if (codeCalls.length === 0) return null
                    return (
                      <details className="conversation-terminal">
                        <summary>Code Sandbox ({codeCalls.length})</summary>
                        <div className="conversation-terminal-body">
                          {codeCalls.map((tc, i) => (
                            <div key={i} className="conversation-terminal-event">
                              <div className="conversation-terminal-label">
                                <span className="conversation-terminal-prompt">{'>'}</span> python sandbox
                              </div>
                              <pre className="conversation-terminal-code">{String(tc.args.code)}</pre>
                              {tc.output && (
                                <pre className="conversation-terminal-output">{tc.output}</pre>
                              )}
                            </div>
                          ))}
                        </div>
                      </details>
                    )
                  })()}
                  {msg.assistant_reasoning && (
                    <details className="conversation-reasoning">
                      <summary>Reasoning</summary>
                      <div className="conversation-reasoning-body">{msg.assistant_reasoning}</div>
                    </details>
                  )}
                  {msg.assistant_content && (
                    <div className="conversation-bubble assistant">
                      <div className="conversation-role">{$t('conversations.assistant')}</div>
                      <div className="conversation-text markdown-body">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.assistant_content}</ReactMarkdown>
                      </div>
                    </div>
                  )}
                  {(() => {
                    const info = msg.retrieval_info as {
                      status?: string
                      chunks_retrieved?: number
                      rag_context_tokens?: number
                      rag_context?: Array<{ content: string; metadata?: Record<string, unknown> }>
                    } | undefined
                    if (!info) return null
                    const shouldShow = info.status === 'error' || (info.status === 'ok' && (info.chunks_retrieved || 0) > 0)
                    if (!shouldShow) return null
                    const isOk = info.status === 'ok'
                    const label = isOk
                      ? `RAG \u00b7 ${info.chunks_retrieved || 0} Docs \u00b7 ${(info.rag_context_tokens || 0).toLocaleString()} Tokens`
                      : `RAG \u00b7 ${info.status}`
                    return (
                      <div style={{ marginTop: 4 }}>
                        <span
                          className={`chat-rag-badge ${isOk ? 'ok' : 'err'}`}
                          onClick={() => setSelectedRAGInfo({
                            status: info.status || 'unknown',
                            chunks_retrieved: info.chunks_retrieved || 0,
                            rag_context_tokens: info.rag_context_tokens || 0,
                            rag_context: (info.rag_context || []) as Array<{ content: string; metadata?: Record<string, unknown> }>,
                          })}
                          style={{ cursor: 'pointer' }}
                        >
                          <BookOpen size={12} /> {label}
                        </span>
                      </div>
                    )
                  })()}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {metadata && (
        <div className="modal-overlay" onClick={() => setMetadata(null)}>
          <div className="modal-box" onClick={(e) => e.stopPropagation()}>
            <h4>{$t('conversations.conversationDetails')}</h4>
            <p className="metadata-id">{metadata.conversationId}</p>
            {metadata.loading && <div className="loading">{$t('common.loading')}</div>}
            {metadata.error && <div className="error-box">{metadata.error}</div>}
            {metadata.data && (
              <div className="metadata-grid">
                {metadata.data.title && (
                  <div><strong>{$t('conversations.convTitle')}:</strong> {metadata.data.title}</div>
                )}
                <div>
                  <strong>{$t('conversations.owner')}:</strong>{' '}
                  {metadata.data.display_name || metadata.data.username
                    ? `${metadata.data.display_name || metadata.data.username}${metadata.data.username && metadata.data.display_name !== metadata.data.username ? ` (@${metadata.data.username})` : ''}`
                    : metadata.data.used_api_key
                      ? $t('conversations.apiKey')
                      : $t('conversations.anonUser')}
                </div>
                <div><strong>{$t('conversations.messages')}:</strong> {metadata.data.message_count}</div>
                <div><strong>{$t('conversations.userMessages')}:</strong> {metadata.data.user_message_count}</div>
                <div><strong>{$t('conversations.assistantMessages')}:</strong> {metadata.data.assistant_message_count}</div>
                <div><strong>{$t('conversations.retrievalRows')}:</strong> {metadata.data.retrieval_rows}</div>
                <div><strong>{$t('overview.model')}:</strong> {metadata.data.model_name || '—'}</div>
                <div><strong>{$t('conversations.firstMessage')}:</strong> {formatDateTime(metadata.data.first_message_at)}</div>
                <div><strong>{$t('conversations.lastMessage')}:</strong> {formatDateTime(metadata.data.last_message_at)}</div>
                {metadata.data.chat_metrics && (
                  <>
                    <div><strong>{$t('overview.totalTokens')}:</strong> {metadata.data.chat_metrics.total_tokens.toLocaleString()}</div>
                    <div><strong>{$t('overview.avgPromptTokens')}:</strong> {metadata.data.chat_metrics.avg_prompt_tokens_per_second.toFixed(1)}</div>
                    <div><strong>{$t('overview.avgCompletionTokens')}:</strong> {metadata.data.chat_metrics.avg_completion_tokens_per_second.toFixed(1)}</div>
                    <div><strong>{$t('overview.ragTokens')}:</strong> {metadata.data.chat_metrics.total_rag_context_tokens.toLocaleString()}</div>
                    <div><strong>{$t('overview.reasoningTokens')}:</strong> {metadata.data.chat_metrics.total_reasoning_tokens.toLocaleString()}</div>
                    <div><strong>{$t('overview.avgResponseTime')}:</strong> {metadata.data.chat_metrics.avg_response_time_ms.toFixed(0)} ms</div>
                    <div><strong>{$t('overview.avgTTFT')}:</strong> {metadata.data.chat_metrics.avg_time_to_first_token_ms.toFixed(0)} ms</div>
                  </>
                )}
                {metadata.data.vectorstore_metrics && (
                  <>
                    <div><strong>{$t('overview.avgRetrievalTime')}:</strong> {metadata.data.vectorstore_metrics.avg_retrieval_time_ms.toFixed(1)} ms</div>
                    <div><strong>{$t('overview.totalChunks')}:</strong> {metadata.data.vectorstore_metrics.total_chunks}</div>
                    <div><strong>{$t('overview.ragTokens')}:</strong> {metadata.data.vectorstore_metrics.total_rag_context_tokens.toLocaleString()}</div>
                    <div><strong>{$t('models.errorRate')}:</strong> {(metadata.data.vectorstore_metrics.error_rate * 100).toFixed(1)} %</div>
                  </>
                )}
                {metadata.data.emotion_summary && (
                  <>
                    <div><strong>Emotion (dominant):</strong> {metadata.data.emotion_summary.dominant}</div>
                    <div><strong>Emotion (text):</strong> {metadata.data.emotion_summary.text_count}</div>
                    <div><strong>Emotion (webcam):</strong> {metadata.data.emotion_summary.webcam_count}</div>
                  </>
                )}
              </div>
            )}
            <div className="modal-actions">
              <button className="btn-secondary" onClick={() => setMetadata(null)}>
                {$t('common.close')}
              </button>
            </div>
          </div>
        </div>
      )}

      {selectedRAGInfo && (
        <div className="modal-overlay" onClick={() => setSelectedRAGInfo(null)}>
          <div className="modal-box" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '640px', width: '90vw' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h4 style={{ margin: 0 }}>RAG Context</h4>
              <button className="btn-icon" onClick={() => setSelectedRAGInfo(null)} title={$t('common.close')}>
                <X size={18} />
              </button>
            </div>
            <div className="chat-rag-overlay-meta" style={{ marginBottom: '12px' }}>
              Status: {selectedRAGInfo.status} \u00b7 Documents: {selectedRAGInfo.chunks_retrieved} \u00b7 Tokens: {(selectedRAGInfo.rag_context_tokens || 0).toLocaleString()}
            </div>
            {selectedRAGInfo.rag_context.length > 0 && (
              <div style={{ maxHeight: '60vh', overflowY: 'auto' }}>
                {selectedRAGInfo.rag_context.map((doc, idx) => (
                  <div key={idx} className="chat-rag-overlay-doc">
                    <div className="chat-rag-overlay-doc-title">Document {idx + 1}</div>
                    <pre className="chat-rag-overlay-doc-content">{doc.content}</pre>
                    {doc.metadata && Object.keys(doc.metadata).length > 0 && (
                      <pre className="chat-rag-overlay-doc-meta">{JSON.stringify(doc.metadata, null, 2)}</pre>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {memoryModal && (
        <div className="modal-overlay" onClick={() => { setMemoryModal(null); setEditingMemory(null) }}>
          <div className="modal-box preview-modal conversation-modal" onClick={(e) => e.stopPropagation()}>
            <div className="preview-header">
              <div>
                <h4>Agent Memory</h4>
                <span>{memoryModal.conversationId}</span>
              </div>
              <button className="btn-icon" onClick={() => { setMemoryModal(null); setEditingMemory(null) }} title={$t('common.close')}>
                <X size={18} />
              </button>
            </div>
            <div className="preview-content conversation-content">
              {memoryModal.loading && <div className="loading">{$t('common.loading')}</div>}
              {memoryModal.error && <div className="error-box">{memoryModal.error}</div>}
              {!memoryModal.loading && !memoryModal.error && memoryModal.memories.length === 0 && (
                <div className="loading">No agent memory for this conversation.</div>
              )}
              {!memoryModal.loading && !memoryModal.error && memoryModal.memories.map((mem) => (
                <div key={mem.assistant_name} style={{ marginBottom: 20, borderBottom: '1px solid var(--border)', paddingBottom: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <strong style={{ fontSize: 14 }}>{mem.assistant_name}</strong>
                    <div style={{ display: 'flex', gap: 6 }}>
                      {editingMemory?.assistant_name !== mem.assistant_name && (
                        <button
                          className="btn-secondary btn-small"
                          onClick={() => setEditingMemory({ assistant_name: mem.assistant_name, memory: mem.memory })}
                        >
                          Edit
                        </button>
                      )}
                      <button
                        className="btn-danger btn-small"
                        onClick={() => setMemDeleteTarget({ conversationId: memoryModal.conversationId, assistantName: mem.assistant_name })}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                  {editingMemory?.assistant_name === mem.assistant_name ? (
                    <>
                      <textarea
                        value={editingMemory.memory}
                        onChange={(e) => setEditingMemory({ ...editingMemory, memory: e.target.value })}
                        style={{ width: '100%', minHeight: 120, fontFamily: 'inherit', fontSize: 13, padding: 8 }}
                      />
                      <div style={{ display: 'flex', gap: 8, marginTop: 8, justifyContent: 'flex-end' }}>
                        <button className="btn-secondary btn-small" onClick={() => setEditingMemory(null)}>
                          Cancel
                        </button>
                        <button
                          className="btn-primary btn-small"
                          onClick={() => handleSaveMemory(memoryModal.conversationId, mem.assistant_name, editingMemory.memory)}
                        >
                          <Save size={14} /> Save
                        </button>
                      </div>
                    </>
                  ) : (
                    <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 13, background: 'var(--surface)', padding: 10, borderRadius: 6, margin: 0 }}>
                      {mem.memory}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      <ConfirmModal
        open={memDeleteTarget !== null}
        title={$t('common.delete')}
        message={memDeleteTarget ? `Delete agent memory for ${memDeleteTarget.assistantName}?` : ''}
        onConfirm={() => {
          const target = memDeleteTarget
          setMemDeleteTarget(null)
          if (target) void handleDeleteMemory(target.conversationId, target.assistantName)
        }}
        onCancel={() => setMemDeleteTarget(null)}
      />
      <AlertModal
        open={notice !== null}
        title={notice?.danger ? 'Error' : 'Info'}
        message={notice?.message || ''}
        danger={notice?.danger}
        onClose={() => setNotice(null)}
      />
    </div>
  )
}
