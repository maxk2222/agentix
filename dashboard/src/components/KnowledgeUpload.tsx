import { useEffect, useState, useCallback, useRef, useMemo } from 'react'
import { useData } from '../api/DataContext'
import { knowledge, invalidateCache, type CollectionInfo, type WebDAVSource } from '../api/client'
import ConfirmModal from './ConfirmModal'
import WebDAVSourceModal, { type WebDAVSourceFormData } from './WebDAVSourceModal'
import {
  Trash2,
  FileText,
  FolderOpen,
  Plus,
  Upload,
  ChevronDown,
  ChevronUp,
  MoreVertical,
  Pencil,
  X,
  ExternalLink,
  Save,
  Code,
  Eye,
  FilePlus,
  RefreshCw,
  Globe,
} from 'lucide-react'
import {
  MDXEditor,
  headingsPlugin,
  listsPlugin,
  quotePlugin,
  thematicBreakPlugin,
  markdownShortcutPlugin,
  linkPlugin,
  tablePlugin,
  codeBlockPlugin,
  toolbarPlugin,
  UndoRedo,
  BoldItalicUnderlineToggles,
  InsertCodeBlock,
  InsertThematicBreak,
  ListsToggle,
  BlockTypeSelect,
  CreateLink,
} from '@mdxeditor/editor'
import '@mdxeditor/editor/style.css'
import { useAuth } from '../api/AuthContext'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

function countFiles(col: CollectionInfo): number {
  let count = col.files.length + (col.db_files?.length ?? 0)
  for (const sub of col.subfolders) {
    count += countFiles(sub)
  }
  return count
}

type PreviewFile = {
  collection: string
  filename: string
}

function fileExtension(filename: string): string {
  const index = filename.lastIndexOf('.')
  return index >= 0 ? filename.slice(index + 1).toLowerCase() : ''
}

function formatDate(dateStr: string | undefined, lang: string): string {
  if (!dateStr) return ''
  try {
    const d = new Date(dateStr)
    return d.toLocaleString(lang, {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return ''
  }
}

/* ------------------------------------------------------------------ */
/* PromptModal                                                        */
/* ------------------------------------------------------------------ */
function PromptModal({
  open,
  title,
  initialValue,
  onConfirm,
  onCancel,
}: {
  open: boolean
  title: string
  initialValue?: string
  onConfirm: (value: string) => void
  onCancel: () => void
}) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const [value, setValue] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setValue(initialValue ?? '')
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [open, initialValue])

  if (!open) return null

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const v = value.trim()
    if (v) onConfirm(v)
  }

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-box" onClick={(e) => e.stopPropagation()}>
        <h4>{title}</h4>
        <form onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={$t('knowledge.enterName')}
          />
          <div className="modal-actions">
            <button type="button" className="btn-secondary" onClick={onCancel}>
              {$t('common.cancel')}
            </button>
            <button type="submit" className="btn-primary">
              {$t('common.apply')}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function PreviewModal({
  file,
  onClose,
}: {
  file: PreviewFile | null
  onClose: () => void
}) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const [textContent, setTextContent] = useState('')
  const [textError, setTextError] = useState<string | null>(null)
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [previewError, setPreviewError] = useState<string | null>(null)

  useEffect(() => {
    if (!file) return
    const ext = fileExtension(file.filename)
    const url = knowledge.previewUrl(file.collection, file.filename)

    setTextContent('')
    setTextError(null)
    setPreviewError(null)
    setPdfUrl(null)

    if (ext === 'pdf') {
      let activeUrl: string | null = null
      fetch(url, { credentials: 'include' })
        .then(async (res) => {
          if (!res.ok) throw new Error(await res.text())
          return res.blob()
        })
        .then((blob) => {
          activeUrl = window.URL.createObjectURL(blob)
          setPdfUrl(activeUrl)
        })
        .catch((err) => setPreviewError(String(err)))
      return () => {
        if (activeUrl) window.URL.revokeObjectURL(activeUrl)
      }
    }

    if (ext !== 'txt') return

    fetch(url, { credentials: 'include' })
      .then(async (res) => {
        if (!res.ok) throw new Error(await res.text())
        return res.text()
      })
      .then(setTextContent)
      .catch((err) => setTextError(String(err)))
  }, [file])

  if (!file) return null

  const ext = fileExtension(file.filename)
  const url = knowledge.previewUrl(file.collection, file.filename)
  const isWord = ext === 'doc' || ext === 'docx'
  const isPdf = ext === 'pdf'

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box preview-modal" onClick={(e) => e.stopPropagation()}>
        <div className="preview-header">
          <div>
            <h4>{file.filename}</h4>
            <span>{file.collection}</span>
          </div>
          <div className="preview-header-actions">
            <a className="btn-secondary" href={url} target="_blank" rel="noreferrer">
              <ExternalLink size={14} /> {$t('knowledge.open')}
            </a>
            <button className="btn-icon" onClick={onClose} title={$t('common.close')}>
              <X size={18} />
            </button>
          </div>
        </div>

        <div className="preview-content">
          {isPdf && (
            <>
              {previewError ? (
                <div className="preview-placeholder">
                  <FileText size={42} />
                  <h4>PDF-Vorschau konnte nicht geladen werden</h4>
                  <p>{previewError}</p>
                  <div className="preview-actions">
                    <a className="btn-primary" href={url} target="_blank" rel="noreferrer">
                      <ExternalLink size={16} /> {$t('knowledge.openFile')}
                    </a>
                  </div>
                </div>
              ) : (
                <iframe
                  src={pdfUrl ?? url}
                  title={file.filename}
                  className="preview-frame"
                />
              )}
            </>
          )}
          {ext === 'txt' && (
            <pre className="preview-text">
              {textError ? textError : textContent || $t('knowledge.loadingShort')}
            </pre>
          )}
          {isWord && (
            <div className="preview-placeholder">
              <FileText size={42} />
              <h4>{$t('knowledge.wordPreviewUnavailable')}</h4>
              <p>
                {$t('knowledge.wordPreviewDesc')}
              </p>
              <a className="btn-primary" href={url} target="_blank" rel="noreferrer">
                <ExternalLink size={16} /> {$t('knowledge.openDocument')}
              </a>
            </div>
          )}
          {!['pdf', 'txt', 'doc', 'docx'].includes(ext) && (
            <div className="preview-placeholder">
              <FileText size={42} />
              <h4>{$t('knowledge.noPreview')}</h4>
              <a className="btn-primary" href={url} target="_blank" rel="noreferrer">
                <ExternalLink size={16} /> {$t('knowledge.openFile')}
              </a>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* CollectionNode                                                     */
/* ------------------------------------------------------------------ */
function CollectionNode({
  col,
  parentPath,
  depth,
  onUpload,
  onDeleteFile,
  onDeleteCollection,
  onRenameCollection,
  onRebuildCollection,
  onPreviewFile,
  onEditFile,
  onCreateFile,
  defaultExpanded = false,
  disabled = false,
  t,
  lang,
}: {
  col: CollectionInfo
  parentPath: string
  depth: number
  onUpload: (path: string, files: File[]) => void
  onDeleteFile: (path: string, filename: string) => void
  onDeleteCollection: (path: string) => void
  onRenameCollection: (path: string) => void
  onRebuildCollection: (path: string) => void
  onPreviewFile: (path: string, filename: string) => void
  onEditFile: (path: string, filename: string) => void
  onCreateFile: (path: string) => void
  defaultExpanded?: boolean
  disabled?: boolean
  t: (key: string) => string
  lang: string
}) {
  const fullPath = parentPath ? `${parentPath}/${col.name}` : col.name
  const [expanded, setExpanded] = useState(defaultExpanded)
  const [dragOver, setDragOver] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const fileCount = countFiles(col)

  useEffect(() => {
    if (!menuOpen) return
    const close = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [menuOpen])

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    if (disabled) return
    setDragOver(false)
    const files = Array.from(e.dataTransfer.files || [])
    if (files.length > 0) onUpload(fullPath, files)
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return
    const files = Array.from(e.target.files || [])
    if (files.length > 0) onUpload(fullPath, files)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <div className="assistant-card">
      <div className="assistant-header" onClick={() => setExpanded((v) => !v)}>
        <div className="assistant-title">
          <FolderOpen size={16} />
          <span className="assistant-name">{col.name}</span>
          {col.db_only && (
            <span className="status-badge db-only">{t('knowledge.dbOnlyShort')}</span>
          )}
          <span className="assistant-model">
            {fileCount} {fileCount !== 1 ? t('knowledge.filesCountPlural') : t('knowledge.filesCount')}
          </span>
          {col.updated_at && (
            <span className="status-badge updated" title={col.updated_at}>
              {formatDate(col.updated_at, lang)}
            </span>
          )}
          {typeof col.vector_doc_count === 'number' && col.vector_doc_count > 0 && (
            <span className="status-badge vectors" title={`${col.vector_doc_count} vector docs`}>
              {col.vector_doc_count}
            </span>
          )}
        </div>
        <div className="assistant-actions">
          <div className="dropdown-wrap" ref={menuRef}>
            <button
              className="btn-icon"
              onClick={(e) => {
                e.stopPropagation()
                if (!disabled) onCreateFile(fullPath)
              }}
              disabled={disabled}
              title={t('knowledge.newFile')}
            >
              <FilePlus size={14} />
            </button>
            <button
              className="btn-icon"
              onClick={(e) => {
                e.stopPropagation()
                if (!disabled) setMenuOpen((v) => !v)
              }}
              disabled={disabled}
              title={t('common.actions')}
            >
              <MoreVertical size={14} />
            </button>
            {menuOpen && !disabled && (
              <div className="dropdown-menu">
                <button
                  className="dropdown-item"
                  onClick={(e) => {
                    e.stopPropagation()
                    setMenuOpen(false)
                    onRenameCollection(fullPath)
                  }}
                >
                  <Pencil size={14} /> {t('knowledge.rename')}
                </button>
                <button
                  className="dropdown-item"
                  onClick={(e) => {
                    e.stopPropagation()
                    setMenuOpen(false)
                    onRebuildCollection(fullPath)
                  }}
                >
                  <RefreshCw size={14} /> Rebuild Index
                </button>
                <button
                  className="dropdown-item danger"
                  onClick={(e) => {
                    e.stopPropagation()
                    setMenuOpen(false)
                    onDeleteCollection(fullPath)
                  }}
                >
                  <Trash2 size={14} /> {t('common.delete')}
                </button>
              </div>
            )}
          </div>
          {expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </div>
      </div>

      {expanded && (
        <div className="assistant-body">
          <div
            className={`upload-dropzone ${dragOver ? 'drag-over' : ''}`}
            onDragOver={(e) => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => !disabled && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt,.doc,.docx,.md,.odt,.xls,.xlsx,.ppt,.pptx,.csv,.png,.jpg,.jpeg,.gif,.bmp,.tiff,.webp"
              style={{ display: 'none' }}
              onChange={handleFileSelect}
            />
            <Upload size={24} />
            <span className="dropzone-text">
              {dragOver ? t('knowledge.dragDrop') : t('knowledge.uploadFiles')}
            </span>
            <span className="dropzone-hint">{t('knowledge.dropHint')}</span>
          </div>

          {(col.files.length > 0 || (col.db_files?.length ?? 0) > 0) && (
            <ul className="file-list">
              {col.files.map((f) => {
                const isIndexed = col.indexed_files?.includes(f)
                return (
                  <li
                  key={f}
                  className="file-item clickable"
                  onClick={() => {
                    const ext = fileExtension(f)
                    if (ext === 'md' || ext === 'txt') {
                      onEditFile(fullPath, f)
                    } else {
                      onPreviewFile(fullPath, f)
                    }
                  }}
                >
                  <FileText size={14} />
                  <span className="file-name">{f}</span>
                  {isIndexed && (
                    <span className="status-badge indexed">{t('knowledge.filePlusDb')}</span>
                  )}
                  {col.file_updated_at?.[f] && (
                    <span className="status-badge updated" title={col.file_updated_at[f]}>
                      {formatDate(col.file_updated_at[f], lang)}
                    </span>
                  )}
                  {typeof col.file_vector_doc_count?.[f] === 'number' && (col.file_vector_doc_count[f] ?? 0) > 0 && (
                    <span className="status-badge vectors" title={`${col.file_vector_doc_count[f]} vector docs`}>
                      {col.file_vector_doc_count[f]}
                    </span>
                  )}
                  {(fileExtension(f) === 'md' || fileExtension(f) === 'txt') && (
                    <button
                      className="btn-icon"
                      onClick={(e) => {
                        e.stopPropagation()
                        onEditFile(fullPath, f)
                      }}
                      disabled={disabled}
                      title={t('common.edit')}
                    >
                      <Pencil size={14} />
                    </button>
                  )}
                  <button
                    className="btn-icon"
                    onClick={(e) => {
                      e.stopPropagation()
                      onDeleteFile(fullPath, f)
                    }}
                    disabled={disabled}
                    title={t('common.delete')}
                  >
                    <Trash2 size={14} />
                  </button>
                </li>
                )
              })}
              {col.db_files?.map((f) => (
                <li key={`db-${f}`} className="file-item db-only">
                  <FileText size={14} />
                  <span className="file-name">{f}</span>
                  <span className="status-badge db-only">{t('knowledge.dbOnly')}</span>
                  {col.file_updated_at?.[f] && (
                    <span className="status-badge updated" title={col.file_updated_at[f]}>
                      {formatDate(col.file_updated_at[f], lang)}
                    </span>
                  )}
                  {typeof col.file_vector_doc_count?.[f] === 'number' && (col.file_vector_doc_count[f] ?? 0) > 0 && (
                    <span className="status-badge vectors" title={`${col.file_vector_doc_count[f]} vector docs`}>
                      {col.file_vector_doc_count[f]}
                    </span>
                  )}
                  <button
                    className="btn-icon"
                    onClick={() => onDeleteFile(fullPath, f)}
                    disabled={disabled}
                    title={t('common.delete')}
                  >
                    <Trash2 size={14} />
                  </button>
                </li>
              ))}
            </ul>
          )}

          {col.subfolders.length > 0 && (
            <div className="subfolder-list">
              {col.subfolders.map((sub) => (
                <CollectionNode
                  key={sub.name}
                  col={sub}
                  parentPath={fullPath}
                  depth={depth + 1}
                  onUpload={onUpload}
                  onDeleteFile={onDeleteFile}
                  onDeleteCollection={onDeleteCollection}
                  onRenameCollection={onRenameCollection}
                  onRebuildCollection={onRebuildCollection}
                  onPreviewFile={onPreviewFile}
                  onEditFile={onEditFile}
                  onCreateFile={onCreateFile}
                  defaultExpanded={false}
                  disabled={disabled}
                  t={t}
                  lang={lang}
                />
              ))}
            </div>
          )}

          {col.files.length === 0 && col.subfolders.length === 0 && (
            <p className="empty-files">{t('common.empty')}</p>
          )}
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* TextEditorModal                                                    */
/* ------------------------------------------------------------------ */
type EditorFile = {
  collection: string
  filename: string
  isNew: boolean
  isTextFile: boolean
}

function TextEditorModal({
  file,
  onClose,
  onSave,
  canWrite = true,
}: {
  file: EditorFile | null
  onClose: () => void
  onSave: (collection: string, filename: string, content: string, isNew: boolean) => Promise<void>
  canWrite?: boolean
}) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const [content, setContent] = useState('')
  const [mode, setMode] = useState<'rich' | 'code'>('rich')
  const [saving, setSaving] = useState(false)
  const [loading, setLoading] = useState(false)
  const editorRef = useRef<any>(null)
  const isTextFile = file?.isTextFile ?? false

  useEffect(() => {
    if (!file) return
    setMode(isTextFile ? 'rich' : 'code')
    if (file.isNew) {
      setContent('')
      editorRef.current?.setMarkdown('')
      return
    }
    setLoading(true)
    knowledge
      .readText(file.collection, file.filename)
      .then((res) => {
        setContent(res.content)
        editorRef.current?.setMarkdown(res.content)
      })
      .catch(() => {
        setContent('')
        editorRef.current?.setMarkdown('')
      })
      .finally(() => setLoading(false))
  }, [file, isTextFile])

  if (!file) return null

  const handleSave = async () => {
    setSaving(true)
    try {
      await onSave(file.collection, file.filename, content, file.isNew)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box editor-modal" onClick={(e) => e.stopPropagation()}>
        <div className="editor-header">
          <h4>{file.filename}</h4>
          <div className="editor-header-actions">
            {isTextFile && (
              <div className="editor-mode-toggle">
                <button
                  className={mode === 'rich' ? 'active' : ''}
                  onClick={() => setMode('rich')}
                  title="Rich Text"
                >
                  <Eye size={14} />
                </button>
                <button
                  className={mode === 'code' ? 'active' : ''}
                  onClick={() => setMode('code')}
                  title="Markdown"
                >
                  <Code size={14} />
                </button>
              </div>
            )}
            {canWrite && (
              <button
                className="btn-primary"
                onClick={handleSave}
                disabled={saving}
              >
                <Save size={14} /> {saving ? $t('knowledge.saving') : $t('knowledge.save')}
              </button>
            )}
            <button className="btn-icon" onClick={onClose} title={$t('common.close')}>
              <X size={18} />
            </button>
          </div>
        </div>

        <div className="editor-body">
          {loading ? (
            <div className="loading">{$t('knowledge.loadingShort')}</div>
          ) : isTextFile && mode === 'rich' ? (
            <MDXEditor
              ref={editorRef}
              markdown={content}
              onChange={(md) => setContent(md)}
              plugins={[
                headingsPlugin(),
                listsPlugin(),
                quotePlugin(),
                thematicBreakPlugin(),
                markdownShortcutPlugin(),
                linkPlugin(),
                tablePlugin(),
                codeBlockPlugin(),
                toolbarPlugin({
                  toolbarContents: () => (
                    <>
                      <UndoRedo />
                      <BlockTypeSelect />
                      <BoldItalicUnderlineToggles />
                      <ListsToggle />
                      <CreateLink />
                      <InsertCodeBlock />
                      <InsertThematicBreak />
                    </>
                  ),
                }),
              ]}
            />
          ) : (
            <textarea
              className="editor-textarea"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              spellCheck={false}
            />
          )}
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* KnowledgeUpload                                                      */
/* ------------------------------------------------------------------ */
export default function KnowledgeUpload() {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const { hasPermission } = useAuth()
  const canWrite = hasPermission('knowledge:write')
  const canDelete = hasPermission('knowledge:delete')

  const [collections, setCollections] = useState<CollectionInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [modalOpen, setModalOpen] = useState(false)
  const [modalTitle, setModalTitle] = useState('')
  const [modalInitial, setModalInitial] = useState('')
  const [modalCallback, setModalCallback] = useState<(val: string) => void>(() => {})
  const [previewFile, setPreviewFile] = useState<PreviewFile | null>(null)
  const [deleteModal, setDeleteModal] = useState<{ path: string; name: string } | null>(null)
  const [deleteFileModal, setDeleteFileModal] = useState<{ open: boolean; path: string; filename: string } | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)
  const [editorFile, setEditorFile] = useState<EditorFile | null>(null)
  const [editorStatus, setEditorStatus] = useState<string | null>(null)
  const [createFileModal, setCreateFileModal] = useState<{
    open: boolean
    path: string
    name: string
    ext: 'md' | 'txt'
  }>({ open: false, path: '', name: '', ext: 'md' })

  const [webdavSources, setWebDAVSources] = useState<WebDAVSource[]>([])
  const [webdavLoading, setWebDAVLoading] = useState(false)
  const [webdavModalOpen, setWebDAVModalOpen] = useState(false)
  const [webdavEditSource, setWebDAVEditSource] = useState<WebDAVSource | null>(null)
  const [webdavDeleteModal, setWebDAVDeleteModal] = useState<WebDAVSource | null>(null)
  const [webdavDeleting, setWebdavDeleting] = useState(false)
  const [webdavDeleteError, setWebdavDeleteError] = useState<string | null>(null)
  const [webdavSyncing, setWebDAVSyncing] = useState<Record<string, boolean>>({})
  const [webdavProgress, setWebDAVProgress] = useState<Record<string, { phase: string; current: number; total: number; message?: string }>>({})
  const [webdavStats, setWebDAVStats] = useState<Record<string, { file_count: number; vector_doc_count: number }>>({})
  const webdavPollRef = useRef<Record<string, ReturnType<typeof setInterval>>>({})
  const [webdavMenuOpen, setWebDAVMenuOpen] = useState<string | null>(null)
  const webdavMenuRefs = useRef<Record<string, HTMLDivElement | null>>({})

  const webdavEditInitial = useMemo<Partial<WebDAVSourceFormData> | undefined>(() => {
    if (!webdavEditSource) return undefined
    return {
      name: webdavEditSource.name,
      webdav_url: webdavEditSource.webdav_url,
      remote_path: webdavEditSource.remote_path,
      username: webdavEditSource.username,
      password: '',
      enabled: webdavEditSource.enabled,
      convert_to_okf: webdavEditSource.convert_to_okf,
    }
  }, [webdavEditSource])

  useEffect(() => {
    if (!webdavMenuOpen) return
    const close = (e: MouseEvent) => {
      const el = webdavMenuRefs.current[webdavMenuOpen]
      if (el && !el.contains(e.target as Node)) {
        setWebDAVMenuOpen(null)
      }
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [webdavMenuOpen])

  const { getCollections, refreshCollections, invalidateInstructions } = useData()

  const load = useCallback(() => {
    setLoading(true)
    getCollections()
      .then((res) => {
        setCollections(res)
        setError(null)
        // Load vector counts asynchronously after structure is rendered
        Promise.all(
          res.map((col) =>
            knowledge
              .getCollectionStats(col.name)
              .then((stats) => ({ name: col.name, stats }))
              .catch(() => ({ name: col.name, stats: null }))
          )
        ).then((results) => {
          setCollections((prev) =>
            prev.map((col) => {
              const r = results.find((x) => x.name === col.name)
              if (r?.stats) {
                return {
                  ...col,
                  vector_doc_count: r.stats.vector_doc_count,
                  file_vector_doc_count: r.stats.file_vector_doc_count,
                }
              }
              return col
            })
          )
        })
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false))
  }, [getCollections])

  const reload = useCallback(() => {
    setLoading(true)
    refreshCollections()
      .then((res) => {
        setCollections(res)
        setError(null)
        Promise.all(
          res.map((col) =>
            knowledge
              .getCollectionStats(col.name)
              .then((stats) => ({ name: col.name, stats }))
              .catch(() => ({ name: col.name, stats: null }))
          )
        ).then((results) => {
          setCollections((prev) =>
            prev.map((col) => {
              const r = results.find((x) => x.name === col.name)
              if (r?.stats) {
                return {
                  ...col,
                  vector_doc_count: r.stats.vector_doc_count,
                  file_vector_doc_count: r.stats.file_vector_doc_count,
                }
              }
              return col
            })
          )
        })
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false))
  }, [refreshCollections])

  const startWebDAVPoll = useCallback((name: string) => {
    if (webdavPollRef.current[name]) return
    webdavPollRef.current[name] = setInterval(async () => {
      try {
        const status = await knowledge.getWebDAVStatus(name)
        const progress = status.sync_progress
        if (progress) {
          setWebDAVProgress((s) => ({ ...s, [name]: progress }))
        }
        if (!progress || ['done', 'error', 'cancelled'].includes(progress.phase)) {
          const interval = webdavPollRef.current[name]
          if (interval) clearInterval(interval)
          delete webdavPollRef.current[name]
          setWebDAVSyncing((s) => ({ ...s, [name]: false }))
          setWebDAVProgress((s) => {
            const next = { ...s }
            delete next[name]
            return next
          })
          loadWebDAVRef.current()
        }
      } catch {
        // ignore polling errors
      }
    }, 1000)
  }, [])

  const loadWebDAV = useCallback(() => {
    setWebDAVLoading(true)
    knowledge.listWebDAVSources()
      .then((res) => {
        setWebDAVSources(res)
        // Fetch file count and vector doc count for each source
        Promise.all(
          res.map(async (src) => {
            try {
              const [status, stats] = await Promise.all([
                knowledge.getWebDAVStatus(src.name),
                knowledge.getCollectionStats(src.name),
              ])
              return {
                name: src.name,
                file_count: status.file_count,
                vector_doc_count: stats.vector_doc_count,
                sync_progress: status.sync_progress,
              }
            } catch {
              return { name: src.name, file_count: 0, vector_doc_count: 0, sync_progress: undefined }
            }
          })
        ).then((results) => {
          const map: Record<string, { file_count: number; vector_doc_count: number }> = {}
          for (const r of results) {
            map[r.name] = { file_count: r.file_count, vector_doc_count: r.vector_doc_count }
            // Restore progress UI for syncs still running server-side (e.g. after a
            // page switch while a sync request was in flight).
            const progress = r.sync_progress
            if (progress && !['done', 'error', 'cancelled'].includes(progress.phase)) {
              setWebDAVSyncing((s) => ({ ...s, [r.name]: true }))
              setWebDAVProgress((s) => ({ ...s, [r.name]: progress }))
              startWebDAVPoll(r.name)
            }
          }
          setWebDAVStats(map)
        })
      })
      .catch((err) => {
        console.error('Failed to load WebDAV sources:', err)
      })
      .finally(() => setWebDAVLoading(false))
  }, [startWebDAVPoll])

  const loadWebDAVRef = useRef(loadWebDAV)
  loadWebDAVRef.current = loadWebDAV

  useEffect(() => {
    load()
    loadWebDAV()
  }, [load, loadWebDAV])

  // Stop all status polling when leaving the page.
  useEffect(() => () => {
    for (const interval of Object.values(webdavPollRef.current)) {
      clearInterval(interval)
    }
    webdavPollRef.current = {}
  }, [])

  const handleUpload = useCallback(
    async (path: string, files: File[]) => {
      if (uploading) return
      setUploading(true)
      setUploadStatus('Upload läuft …')
      try {
        const res = await knowledge.upload(path, files)
        invalidateCache('/admin/knowledge')
        reload()

        let statusMsg = `${res.filenames.length} Datei${res.filenames.length !== 1 ? 'en' : ''} hochgeladen`
        if (res.indexing) {
          if (res.indexing.error) {
            statusMsg += ` • Indexing-Fehler: ${res.indexing.error}`
          } else if (res.indexing.initialized) {
            statusMsg += ` • Collection initialisiert (${res.indexing.chunks} Chunks)`
          } else if (res.indexing.sources) {
            const updated = res.indexing.sources.filter((s) => s.action === 'updated').length
            const skipped = res.indexing.sources.filter((s) => s.action === 'skipped').length
            if (updated > 0 || skipped > 0) {
              statusMsg += ` • ${updated} aktualisiert, ${skipped} übersprungen`
            } else {
              statusMsg += ' • Verarbeitet'
            }
          }
        }
        setUploadStatus(statusMsg)
      } catch (err) {
        setError(String(err))
        setUploadStatus(null)
      } finally {
        setUploading(false)
        setTimeout(() => setUploadStatus(null), 5000)
      }
    },
    [load]
  )

  const handleDeleteFile = useCallback(
    (path: string, filename: string) => {
      setDeleteFileModal({ open: true, path, filename })
    },
    []
  )

  const confirmDeleteFile = useCallback(
    async () => {
      if (!deleteFileModal) return
      try {
        await knowledge.delete(deleteFileModal.path, deleteFileModal.filename)
        invalidateCache('/admin/knowledge')
        reload()
      } catch (err) {
        setError(String(err))
      } finally {
        setDeleteFileModal(null)
      }
    },
    [deleteFileModal, reload]
  )

  const handleDeleteCollection = useCallback(
    (path: string) => {
      setDeleteModal({ path, name: path })
    },
    []
  )

  const handleRenameCollection = useCallback(
    async (path: string) => {
      const currentName = path.split('/').pop() || path
      setModalTitle($t('knowledge.rename'))
      setModalInitial(currentName)
      setModalCallback(() => async (newName: string) => {
        setModalOpen(false)
        try {
          await knowledge.renameCollection(path, newName)
        invalidateCache('/admin/knowledge')
          reload()
        } catch (err) {
          setError(String(err))
        }
      })
      setModalOpen(true)
    },
    [reload]
  )

  const openCreateModal = useCallback(() => {
    setModalTitle($t('knowledge.createCollection'))
    setModalInitial('')
    setModalCallback(() => async (name: string) => {
      setModalOpen(false)
      try {
        await knowledge.createCollection(name)
        invalidateCache('/admin/knowledge')
        reload()
      } catch (err) {
        setError(String(err))
      }
    })
    setModalOpen(true)
  }, [reload])

  const handleRebuildCollection = useCallback(
    async (path: string) => {
      try {
        setUploadStatus('Index wird neu aufgebaut …')
        const res = await knowledge.rebuildCollection(path)
        invalidateCache('/admin/knowledge')
        reload()
        let statusMsg = `Index für "${res.collection}" neu aufgebaut`
        if (res.indexing) {
          if (typeof res.indexing === 'object' && 'error' in res.indexing) {
            statusMsg += ` • Fehler: ${(res.indexing as any).error}`
          } else if (typeof res.indexing === 'object' && 'sources' in res.indexing) {
            const idx = res.indexing as any
            const updated = idx.sources?.filter((s: any) => s.action === 'updated').length ?? 0
            const skipped = idx.sources?.filter((s: any) => s.action === 'skipped').length ?? 0
            if (updated > 0 || skipped > 0) {
              statusMsg += ` • ${updated} aktualisiert, ${skipped} übersprungen`
            } else {
              statusMsg += ' • Verarbeitet'
            }
          }
        }
        setUploadStatus(statusMsg)
        setTimeout(() => setUploadStatus(null), 5000)
      } catch (err) {
        setError(String(err))
      }
    },
    [reload]
  )

  const handlePreviewFile = useCallback((path: string, filename: string) => {
    setPreviewFile({ collection: path, filename })
  }, [])

  const handleEditFile = useCallback((path: string, filename: string) => {
    const ext = fileExtension(filename)
    setEditorFile({
      collection: path,
      filename,
      isNew: false,
      isTextFile: ext === 'md' || ext === 'txt',
    })
  }, [])

  const handleCreateFile = useCallback((path: string) => {
    setCreateFileModal({ open: true, path, name: '', ext: 'md' })
  }, [])

  const handleSaveFile = useCallback(
    async (collection: string, filename: string, content: string, isNew: boolean) => {
      try {
        const res = await knowledge.saveText(collection, filename, content)
        invalidateCache('/admin/knowledge')
        reload()

        let statusMsg = isNew ? 'Datei erstellt' : 'Datei gespeichert'
        if (res.indexing) {
          if (res.indexing.error) {
            statusMsg += ` • Indexing-Fehler: ${res.indexing.error}`
          } else if (res.indexing.initialized) {
            statusMsg += ` • Collection initialisiert (${res.indexing.chunks} Chunks)`
          } else if (res.indexing.sources) {
            const updated = res.indexing.sources.filter((s) => s.action === 'updated').length
            const skipped = res.indexing.sources.filter((s) => s.action === 'skipped').length
            if (updated > 0 || skipped > 0) {
              statusMsg += ` • ${updated} aktualisiert, ${skipped} übersprungen`
            } else {
              statusMsg += ' • Verarbeitet'
            }
          }
        }
        setEditorStatus(statusMsg)
        setEditorFile(null)
        setTimeout(() => setEditorStatus(null), 5000)
      } catch (err) {
        setError(String(err))
      }
    },
    [load]
  )

  const handleWebDAVCreate = useCallback(
    async (data: WebDAVSourceFormData) => {
      try {
        await knowledge.createWebDAVSource(data)
        setWebDAVModalOpen(false)
        loadWebDAV()
      } catch (err: any) {
        setError(err.message || String(err))
      }
    },
    [loadWebDAV]
  )

  const handleWebDAVUpdate = useCallback(
    async (data: WebDAVSourceFormData) => {
      if (!webdavEditSource) return
      try {
        const payload: Record<string, any> = {
          webdav_url: data.webdav_url,
          remote_path: data.remote_path,
          username: data.username,
          enabled: data.enabled,
          convert_to_okf: data.convert_to_okf,
        }
        if (data.password) payload.password = data.password
        await knowledge.updateWebDAVSource(webdavEditSource.name, payload)
        setWebDAVModalOpen(false)
        setWebDAVEditSource(null)
        loadWebDAV()
      } catch (err: any) {
        setError(err.message || String(err))
      }
    },
    [loadWebDAV, webdavEditSource]
  )

  const handleWebDAVSync = useCallback(
    async (name: string) => {
      setWebDAVSyncing((s) => ({ ...s, [name]: true }))
      setWebDAVProgress((s) => ({ ...s, [name]: { phase: 'listing', current: 0, total: 0, message: 'Scanning...' } }))
      // The poll owns the syncing/progress lifecycle: it stops itself once the
      // backend reports a terminal phase (or none at all).
      startWebDAVPoll(name)

      try {
        const res = await knowledge.syncWebDAVSource(name)
        if (res.success) {
          setUploadStatus(`${name}: ${res.downloaded ?? 0} heruntergeladen, ${res.skipped ?? 0} übersprungen`)
          setTimeout(() => setUploadStatus(null), 5000)
        } else {
          setError(`Sync-Fehler: ${res.error ?? 'Unbekannter Fehler'}`)
        }
        loadWebDAV()
      } catch (err: any) {
        setError(err.message || String(err))
      }
    },
    [loadWebDAV, startWebDAVPoll]
  )

  if (loading) return <div className="loading">{$t('common.loading')}</div>

  const totalFiles = collections.reduce((sum, col) => sum + countFiles(col), 0)

  return (
    <div className="config-section">
      <div className="page-header">
        <h2>{$t('knowledge.title')}</h2>
      </div>
      {error && <div className="error-box">{error}</div>}
      {uploadStatus && <div className="success-box">{uploadStatus}</div>}
      {editorStatus && <div className="success-box">{editorStatus}</div>}

      <div className="collections-header">
        <h3>{$t('knowledge.collections')} ({collections.length})</h3>
        <button className="btn-secondary" onClick={openCreateModal}>
          <Plus size={16} /> {$t('knowledge.newCollection')}
        </button>
      </div>

      {collections.length === 0 && (
        <div className="empty-state">
          <FolderOpen size={32} />
          <p>{$t('common.empty')}</p>
        </div>
      )}
      {collections.map((col) => (
        <CollectionNode
          key={col.name}
          col={col}
          parentPath=""
          depth={0}
          onUpload={handleUpload}
          onDeleteFile={handleDeleteFile}
          onDeleteCollection={handleDeleteCollection}
          onRenameCollection={handleRenameCollection}
          onRebuildCollection={handleRebuildCollection}
          onPreviewFile={handlePreviewFile}
          onEditFile={handleEditFile}
          onCreateFile={handleCreateFile}
          defaultExpanded={false}
          disabled={uploading || !canWrite}
          t={$t}
          lang={language}
        />
      ))}

      <div className="total-files">{totalFiles} {$t('knowledge.totalFiles')}</div>

      {/* WebDAV Sources Section */}
      <div className="collections-header" style={{ marginTop: '32px' }}>
        <h3>{$t('knowledge.externalSources')}</h3>
        <button
          className="btn-secondary"
          onClick={() => {
            setWebDAVEditSource(null)
            setWebDAVModalOpen(true)
          }}
          disabled={!canWrite}
        >
          <Plus size={16} /> {$t('knowledge.addWebDAV')}
        </button>
      </div>

      {webdavSources.length === 0 && !webdavLoading && (
        <div className="empty-state">
          <Globe size={32} />
          <p>{$t('common.empty')}</p>
        </div>
      )}

      {webdavSources.map((src) => (
        <div key={src.name} className="assistant-card" style={{ position: 'relative' }}>
          <div className="assistant-header">
            <div className="assistant-title">
              <Globe size={16} />
              <span className="assistant-name">{src.name}</span>
              <span className="status-badge db-only">{$t('knowledge.webdavBadge')}</span>
              {src.convert_to_okf && (
                <span className="status-badge indexed">OKF</span>
              )}
              <span className="assistant-model" style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                {src.webdav_url}
              </span>
              {(() => { const stats = webdavStats[src.name]; const fc = stats?.file_count ?? 0; return fc > 0 ? (
                <span className="assistant-model">
                  {fc} {fc !== 1 ? $t('knowledge.filesCountPlural') : $t('knowledge.filesCount')}
                </span>
              ) : null; })()}
              {(() => { const stats = webdavStats[src.name]; const vc = stats?.vector_doc_count ?? 0; return vc > 0 ? (
                <span className="status-badge vectors" title={`${vc} vector docs`}>
                  {vc}
                </span>
              ) : null; })()}
              {src.last_sync_at ? (
                <span className="status-badge updated" title={src.last_sync_at}>
                  {formatDate(src.last_sync_at, language)}
                </span>
              ) : (
                <span className="status-badge pending">{$t('knowledge.neverSynced')}</span>
              )}
            </div>
            <div className="assistant-actions">
              {webdavSyncing[src.name] ? (
                <button
                  className="btn-icon"
                  onClick={() => knowledge.cancelWebDAVSync(src.name)}
                  disabled={!canWrite}
                  title={$t('knowledge.cancelSync')}
                  style={{ color: 'var(--danger, #ef4444)' }}
                >
                  <X size={14} />
                </button>
              ) : (
                <button
                  className="btn-icon"
                  onClick={() => handleWebDAVSync(src.name)}
                  disabled={!canWrite}
                  title={$t('knowledge.syncNow')}
                >
                  <RefreshCw size={14} />
                </button>
              )}
              <div className="dropdown-wrap" ref={(el) => { webdavMenuRefs.current[src.name] = el }} style={{ position: 'relative' }}>
                <button
                  className="btn-icon"
                  onClick={(e) => {
                    e.stopPropagation()
                    if (!canWrite) return
                    setWebDAVMenuOpen((open) => (open === src.name ? null : src.name))
                  }}
                  disabled={!canWrite}
                  title={$t('common.actions')}
                >
                  <MoreVertical size={14} />
                </button>
                {webdavMenuOpen === src.name && canWrite && (
                  <div className="dropdown-menu" style={{ right: 0, minWidth: '160px' }}>
                    <button
                      className="dropdown-item"
                      onClick={() => {
                        setWebDAVMenuOpen(null)
                        setWebDAVEditSource(src)
                        setWebDAVModalOpen(true)
                      }}
                    >
                      <Pencil size={14} /> {$t('knowledge.edit')}
                    </button>
                    <button
                      className="dropdown-item"
                      onClick={() => {
                        setWebDAVMenuOpen(null)
                        setWebDAVDeleteModal(src)
                      }}
                      disabled={!canDelete}
                    >
                      <Trash2 size={14} /> {$t('common.delete')}
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
          {webdavSyncing[src.name] && webdavProgress[src.name] && (
            <div className="webdav-progress-container">
              <div className="webdav-progress-bar">
                <div
                  className={`webdav-progress-fill ${webdavProgress[src.name].total === 0 ? 'indeterminate' : ''}`}
                  style={webdavProgress[src.name].total > 0 ? {
                    width: `${Math.round((webdavProgress[src.name].current / webdavProgress[src.name].total) * 100)}%`,
                  } : undefined}
                />
              </div>
              <span className="webdav-progress-label">
                {webdavProgress[src.name].message || webdavProgress[src.name].phase}
                {webdavProgress[src.name].total > 0 && ` ${webdavProgress[src.name].current}/${webdavProgress[src.name].total}`}
              </span>
            </div>
          )}
        </div>
      ))}

      <WebDAVSourceModal
        open={webdavModalOpen}
        initial={webdavEditInitial}
        onConfirm={webdavEditSource ? handleWebDAVUpdate : handleWebDAVCreate}
        onCancel={() => {
          setWebDAVModalOpen(false)
          setWebDAVEditSource(null)
        }}
      />

      <ConfirmModal
        open={webdavDeleteModal !== null}
        title={$t('knowledge.deleteWebDAV')}
        message={`${$t('knowledge.deleteWebDAVPrefix')} "${webdavDeleteModal?.name ?? ''}" ${$t('knowledge.deleteWebDAVSuffix')}${webdavDeleteError ? `\n\n⚠️ ${webdavDeleteError}` : ''}`}
        onConfirm={async () => {
          const target = webdavDeleteModal
          if (!target) return
          setWebdavDeleting(true)
          setWebdavDeleteError(null)
          try {
            await knowledge.deleteWebDAVSource(target.name)
            setWebDAVDeleteModal(null)
            loadWebDAV()
          } catch (err: any) {
            console.error('WebDAV delete failed:', err)
            setWebdavDeleteError(err.message || String(err))
          } finally {
            setWebdavDeleting(false)
          }
        }}
        onCancel={() => {
          setWebDAVDeleteModal(null)
          setWebdavDeleteError(null)
        }}
        confirmText={webdavDeleting ? '...' : undefined}
        confirmDisabled={webdavDeleting}
      />

      <PromptModal
        open={modalOpen}
        title={modalTitle}
        initialValue={modalInitial}
        onConfirm={(val) => modalCallback(val)}
        onCancel={() => setModalOpen(false)}
      />
      <PreviewModal file={previewFile} onClose={() => setPreviewFile(null)} />
      <TextEditorModal
        file={editorFile}
        onClose={() => setEditorFile(null)}
        onSave={handleSaveFile}
        canWrite={canWrite}
      />

      {createFileModal.open && (
        <div className="modal-overlay" onClick={() => setCreateFileModal((s) => ({ ...s, open: false }))}>
          <div className="modal-box" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '420px' }}>
            <div className="modal-header">
              <h4>{$t('knowledge.newCollection')}</h4>
              <button className="btn-icon" onClick={() => setCreateFileModal((s) => ({ ...s, open: false }))}>
                <X size={18} />
              </button>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: 500, marginBottom: '4px', color: 'var(--text-secondary)' }}>
                  Dateiname
                </label>
                <input
                  type="text"
                  value={createFileModal.name}
                  onChange={(e) => setCreateFileModal((s) => ({ ...s, name: e.target.value }))}
                  placeholder="Name eingeben"
                  autoFocus
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    borderRadius: '8px',
                    border: '1px solid var(--border)',
                    fontSize: '14px',
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      const name = createFileModal.name.trim()
                      if (!name) return
                      let filename = name
                      const ext = fileExtension(filename)
                      if (ext !== 'md' && ext !== 'txt') {
                        filename += '.md'
                      }
                      setCreateFileModal((s) => ({ ...s, open: false }))
                      setEditorFile({
                        collection: createFileModal.path,
                        filename,
                        isNew: true,
                        isTextFile: true,
                      })
                    }
                  }}
                />
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', margin: '4px 0 0' }}>
                  Ohne Endung wird automatisch .md ergänzt
                </p>
              </div>
              <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '14px', cursor: 'pointer' }}>
                  <input
                    type="radio"
                    name="ext"
                    checked={createFileModal.ext === 'md'}
                    onChange={() => setCreateFileModal((s) => ({ ...s, ext: 'md' }))}
                  />
                  .md (Markdown)
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '14px', cursor: 'pointer' }}>
                  <input
                    type="radio"
                    name="ext"
                    checked={createFileModal.ext === 'txt'}
                    onChange={() => setCreateFileModal((s) => ({ ...s, ext: 'txt' }))}
                  />
                  .txt (Text)
                </label>
              </div>
            </div>
            <div className="modal-footer" style={{ marginTop: '20px' }}>
              <button
                className="btn-secondary"
                onClick={() => setCreateFileModal((s) => ({ ...s, open: false }))}
              >
                Abbrechen
              </button>
              <button
                className="btn-primary"
                onClick={() => {
                  const name = createFileModal.name.trim()
                  if (!name) return
                  let filename = name
                  const ext = fileExtension(filename)
                  if (ext !== 'md' && ext !== 'txt') {
                    filename += `.${createFileModal.ext}`
                  }
                  setCreateFileModal((s) => ({ ...s, open: false }))
                  setEditorFile({
                    collection: createFileModal.path,
                    filename,
                    isNew: true,
                    isTextFile: true,
                  })
                }}
                disabled={!createFileModal.name.trim()}
              >
                {$t('common.add')}
              </button>
            </div>
          </div>
        </div>
      )}

      <ConfirmModal
        open={deleteFileModal?.open ?? false}
        title={$t('knowledge.deleteFile')}
        message={`${$t('knowledge.deleteFilePrefix')} "${deleteFileModal?.filename ?? ''}" ${$t('knowledge.deleteFileSuffix')}`}
        onConfirm={confirmDeleteFile}
        onCancel={() => setDeleteFileModal(null)}
      />
      <ConfirmModal
        open={deleteModal !== null}
        title={$t('knowledge.deleteCollection')}
        message={`${$t('knowledge.deleteCollectionPrefix')} "${deleteModal?.name ?? ''}" ${$t('knowledge.deleteCollectionSuffix')}`}
        onConfirm={async () => {
          if (deleteModal !== null) {
            try {
              await knowledge.deleteCollection(deleteModal.path)
              invalidateCache('/admin/knowledge')
              invalidateCache('/admin/config/instructions')
              invalidateInstructions()
              reload()
            } catch (err) {
              setError(String(err))
            } finally {
              setDeleteModal(null)
            }
          }
        }}
        onCancel={() => setDeleteModal(null)}
      />
    </div>
  )
}
