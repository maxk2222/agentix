/* Folder browser for an assistant's agent workspace (workspace.enabled).
   Shared between dashboard chat and public chat — auth: dashboard session
   cookie, chat-user session or ?pct= public token for download links. */
import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowUp, Download, Eye, FileText, Folder, FolderPlus, Loader2, RefreshCw, Trash2, Upload, X } from 'lucide-react'
import ConfirmModal from '../components/ConfirmModal'
import PromptModal from '../components/PromptModal'

export interface WsEntry {
  name: string
  type?: string
  kind?: string
  size?: number
  updated_at?: number
}

interface Preview {
  name: string
  rendered: boolean
  page_count: number
  pages_rendered?: number[]
  images?: string[]
  outline?: { title: string; lines: string[] }[] | null
}

interface Props {
  apiUrl: string
  assistant: string
  conversationId?: string
  accessToken?: string
  onClose: () => void
}

export function WorkspaceBrowser({ apiUrl, assistant, conversationId, accessToken, onClose }: Props) {
  const [cwd, setCwd] = useState('')
  const [entries, setEntries] = useState<WsEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [preview, setPreview] = useState<Preview | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [pageIdx, setPageIdx] = useState(0)
  const [uploading, setUploading] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const params = useCallback(() => {
    const q = new URLSearchParams()
    if (conversationId) q.set('conversation_id', conversationId)
    if (accessToken) q.set('pct', accessToken)
    return q.toString() ? `&${q}` : ''
  }, [conversationId, accessToken])

  const load = useCallback(async (dir: string) => {
    setLoading(true)
    setError('')
    try {
      const res = await fetch(
        `${apiUrl}/chat/assistants/${encodeURIComponent(assistant)}/workspace/files?path=${encodeURIComponent(dir || '.')}${params()}`,
        { credentials: 'include' },
      )
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setEntries(data.entries || [])
      setCwd(dir)
    } catch (e: any) {
      setError(e?.message || 'Failed to load')
    } finally {
      setLoading(false)
    }
  }, [apiUrl, assistant, params])

  useEffect(() => { load('') }, [load])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (preview) setPreview(null)
        else onClose()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [preview, onClose])

  const openPreview = async (name: string) => {
    const rel = cwd ? `${cwd}/${name}` : name
    setPreviewLoading(true)
    setPageIdx(0)
    try {
      const res = await fetch(
        `${apiUrl}/chat/assistants/${encodeURIComponent(assistant)}/workspace/file/preview?path=${encodeURIComponent(rel)}${params()}`,
        { credentials: 'include' },
      )
      if (!res.ok) throw new Error(await res.text())
      setPreview(await res.json())
    } catch (e: any) {
      setError(e?.message || 'Preview failed')
    } finally {
      setPreviewLoading(false)
    }
  }

  const downloadUrl = (name: string) => {
    const rel = cwd ? `${cwd}/${name}` : name
    const q = new URLSearchParams({ path: rel })
    if (conversationId) q.set('conversation_id', conversationId)
    if (accessToken) q.set('pct', accessToken)
    return `${apiUrl}/chat/assistants/${encodeURIComponent(assistant)}/workspace/file?${q}`
  }

  const upload = async (files: FileList | null) => {
    if (!files?.length) return
    setUploading(true)
    setError('')
    try {
      for (const f of Array.from(files)) {
        const fd = new FormData()
        fd.append('file', f)
        const res = await fetch(
          `${apiUrl}/chat/assistants/${encodeURIComponent(assistant)}/workspace/file?path=${encodeURIComponent(cwd || '.')}${params()}`,
          { method: 'POST', body: fd, credentials: 'include' },
        )
        if (!res.ok) throw new Error(await res.text())
      }
      await load(cwd)
    } catch (e: any) {
      setError(e?.message || 'Upload failed')
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)
  const [mkdirOpen, setMkdirOpen] = useState(false)

  const deleteEntry = async (name: string) => {
    const rel = cwd ? `${cwd}/${name}` : name
    setError('')
    try {
      const res = await fetch(
        `${apiUrl}/chat/assistants/${encodeURIComponent(assistant)}/workspace/file?path=${encodeURIComponent(rel)}${params()}`,
        { method: 'DELETE', credentials: 'include' },
      )
      if (!res.ok) throw new Error(await res.text())
      if (preview?.name === name) setPreview(null)
      await load(cwd)
    } catch (e: any) {
      setError(e?.message || 'Delete failed')
    }
  }

  const mkdir = async (name: string) => {
    const rel = cwd ? `${cwd}/${name.trim()}` : name.trim()
    setError('')
    try {
      const res = await fetch(
        `${apiUrl}/chat/assistants/${encodeURIComponent(assistant)}/workspace/mkdir?path=${encodeURIComponent(rel)}${params()}`,
        { method: 'POST', credentials: 'include' },
      )
      if (!res.ok) throw new Error(await res.text())
      await load(cwd)
    } catch (e: any) {
      setError(e?.message || 'Create folder failed')
    }
  }

  const isDir = (e: WsEntry) => e.type === 'dir' || e.kind === 'dir' || e.size == null
  const canPreview = (name: string) => /\.(txt|md|json|log|py|js|ts|tsx|css|html|xml|svg|ya?ml|toml|ini|cfg|sh|sql|env|rst|diff|patch|png|jpe?g|gif|webp|bmp|tiff?|pdf|docx|pptx|xlsx|csv)$/i.test(name)

  return (
    <div className="ws-browser">
      <div className="ws-browser-head">
        <div className="ws-browser-title">
          <Folder size={15} />
          <strong>Workspace</strong>
          <span className="ws-browser-assistant">{assistant}</span>
        </div>
        <button className="ws-icon-btn" title="Close" onClick={onClose}><X size={16} /></button>
      </div>
      <div className="ws-browser-toolbar">
        <button
          className="ws-icon-btn"
          title="Up"
          disabled={!cwd}
          onClick={() => load(cwd.split('/').slice(0, -1).join('/'))}
        >
          <ArrowUp size={14} />
        </button>
        <span className="ws-browser-path" title={`/${cwd}`}>/{cwd}</span>
        <button className="ws-icon-btn" title="Refresh" disabled={loading} onClick={() => load(cwd)}>
          <RefreshCw size={14} className={loading ? 'spin' : undefined} />
        </button>
        <button className="ws-icon-btn" title="Upload" disabled={uploading} onClick={() => fileRef.current?.click()}>
          {uploading ? <Loader2 size={14} className="spin" /> : <Upload size={14} />}
        </button>
        <button className="ws-icon-btn" title="New folder" onClick={() => setMkdirOpen(true)}>
          <FolderPlus size={14} />
        </button>
        <input ref={fileRef} type="file" multiple hidden onChange={(e) => upload(e.target.files)} />
      </div>
      {error && <div className="ws-browser-error">{error}</div>}
      {loading ? <div className="ws-browser-hint">Loading…</div> : (
        <ul className="ws-browser-list">
          {cwd && (
            <li>
              <button className="ws-entry" onClick={() => load(cwd.split('/').slice(0, -1).join('/'))}>
                <Folder size={14} className="ws-icon" /> <span className="ws-name">..</span>
              </button>
            </li>
          )}
          {entries.filter(e => isDir(e)).map((e) => (
            <li key={e.name} className="ws-file">
              <button className="ws-entry ws-file-name" onClick={() => load(cwd ? `${cwd}/${e.name}` : e.name)}>
                <Folder size={14} className="ws-icon" /> <span className="ws-name">{e.name}</span>
              </button>
              <button className="ws-icon-btn" title="Delete" onClick={() => setDeleteTarget(e.name)}><Trash2 size={14} /></button>
            </li>
          ))}
          {entries.filter(e => !isDir(e)).map((e) => (
            <li key={e.name} className="ws-file">
              {canPreview(e.name) ? (
                <button className="ws-entry ws-file-name" onClick={() => openPreview(e.name)} title="Preview">
                  <FileText size={14} className="ws-icon" /> <span className="ws-name">{e.name}</span>
                </button>
              ) : (
                <span className="ws-entry ws-file-name">
                  <FileText size={14} className="ws-icon" /> <span className="ws-name">{e.name}</span>
                </span>
              )}
              {e.size != null && <span className="ws-size">{Math.ceil(e.size / 1024)} KB</span>}
              {canPreview(e.name) && (
                <button className="ws-icon-btn" title="Preview" onClick={() => openPreview(e.name)}><Eye size={14} /></button>
              )}
              <a className="ws-icon-btn" title="Download" href={downloadUrl(e.name)} download={e.name}><Download size={14} /></a>
              <button className="ws-icon-btn" title="Delete" onClick={() => setDeleteTarget(e.name)}><Trash2 size={14} /></button>
            </li>
          ))}
          {!entries.length && !cwd && <li className="ws-browser-hint">Empty workspace</li>}
        </ul>
      )}
      {previewLoading && <div className="ws-browser-hint">Rendering preview…</div>}
      {preview && (
        <div className="ws-preview">
          <div className="ws-preview-head">
            <span>{preview.name}</span>
            <button className="ws-icon-btn" onClick={() => setPreview(null)}><X size={14} /></button>
          </div>
          {preview.rendered && preview.images?.length ? (
            <>
              <img className="ws-preview-img" src={preview.images[pageIdx]} alt={`page ${pageIdx + 1}`} />
              {preview.page_count > 1 && (
                <div className="ws-preview-pager">
                  <button className="ws-icon-btn" disabled={pageIdx === 0} onClick={() => setPageIdx(pageIdx - 1)}>‹</button>
                  <span>{pageIdx + 1} / {preview.images.length}{preview.page_count > preview.images.length ? ` (of ${preview.page_count} pages)` : ''}</span>
                  <button className="ws-icon-btn" disabled={pageIdx >= preview.images.length - 1} onClick={() => setPageIdx(pageIdx + 1)}>›</button>
                </div>
              )}
            </>
          ) : preview.outline?.length ? (
            <pre className="ws-preview-text">{preview.outline[0].lines.join('\n')}</pre>
          ) : (
            <div className="ws-browser-hint">No preview available</div>
          )}
        </div>
      )}
      <ConfirmModal
        open={deleteTarget !== null}
        title="Delete"
        message={`Delete "${cwd ? `${cwd}/${deleteTarget}` : deleteTarget}"?`}
        confirmText="Delete"
        cancelText="Cancel"
        onConfirm={() => {
          const name = deleteTarget
          setDeleteTarget(null)
          if (name) deleteEntry(name)
        }}
        onCancel={() => setDeleteTarget(null)}
      />
      <PromptModal
        open={mkdirOpen}
        title="New folder"
        placeholder="Folder name"
        confirmText="Create"
        cancelText="Cancel"
        onSubmit={(v) => {
          setMkdirOpen(false)
          mkdir(v)
        }}
        onCancel={() => setMkdirOpen(false)}
      />
    </div>
  )
}
