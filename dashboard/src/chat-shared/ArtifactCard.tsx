import { memo, useCallback, useEffect, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Download, Eye, EyeOff, File, FileImage, FileSpreadsheet, FileText, Presentation } from 'lucide-react'
import type { ArtifactFile } from './ChatElements'

export interface ArtifactLabels {
  download: string
  preview: string
  hidePreview: string
  downloadFailed: string
  previewFailed: string
  exportAs: string
  pageOf: (current: number, total: number) => string
}

interface PreviewOutlinePage {
  title: string
  lines: string[]
}

interface PreviewResponse {
  rendered: boolean
  kind: string
  filename: string
  size: number
  page_count?: number
  outline?: PreviewOutlinePage[]
}

function formatSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

const EXPORT_FORMATS: Record<string, string[]> = {
  docx: ['pdf', 'odt'],
  pptx: ['pdf', 'odp'],
  xlsx: ['pdf', 'ods'],
  csv: ['xlsx'],
}

const IMAGE_EXTS = new Set(['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tif', 'tiff'])
const TEXT_EXTS = new Set([
  'txt', 'md', 'markdown', 'json', 'log', 'py', 'js', 'ts', 'jsx', 'tsx', 'css',
  'scss', 'html', 'htm', 'xml', 'svg', 'yaml', 'yml', 'toml', 'ini', 'cfg', 'sh',
  'sql', 'env', 'rst', 'diff', 'patch',
])
const DOC_EXTS = new Set(['docx', 'pptx', 'pdf', 'xlsx', 'csv'])

export const ArtifactCard = memo(function ArtifactCard({
  file,
  apiUrl,
  labels,
  accessToken,
}: {
  file: ArtifactFile
  apiUrl: string
  labels: ArtifactLabels
  accessToken?: string | null
}) {
  const [expanded, setExpanded] = useState(false)
  const [preview, setPreview] = useState<PreviewResponse | null>(null)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const base = `${apiUrl}/chat/files/${encodeURIComponent(file.conversation_id)}/${encodeURIComponent(file.name)}`
  // <a href>/<img src> cannot send headers — the public-chat access token is
  // appended as ?pct= so the backend can apply public-chat availability rules.
  const tokenSuffix = accessToken ? `?pct=${encodeURIComponent(accessToken)}` : ''
  // Previewability is keyed off the filename extension — artifacts stored
  // with the generic "file" kind (code_execution, workspace offers) get the
  // same treatment as explicitly-typed documents.
  const ext = file.name.includes('.') ? (file.name.split('.').pop() || '').toLowerCase() : ''
  const exportFormats = EXPORT_FORMATS[ext] || []
  // Version bump after an in-place regeneration: busts page-image caches and
  // triggers a preview refetch below.
  const version = file.updated_at ?? file.size
  const expandedRef = useRef(expanded)
  expandedRef.current = expanded

  const loadPreview = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${base}/preview${tokenSuffix}`, { credentials: 'include' })
      if (!res.ok) throw new Error(String(res.status))
      setPreview(await res.json())
      setPage(1)
    } catch {
      setError(labels.previewFailed)
    } finally {
      setLoading(false)
    }
  }, [base, tokenSuffix, labels.previewFailed])

  const togglePreview = useCallback(() => {
    if (expanded) {
      setExpanded(false)
      return
    }
    setExpanded(true)
    if (preview === null) void loadPreview()
  }, [expanded, preview, loadPreview])

  // Reset + refetch when the artifact was regenerated under the same name.
  useEffect(() => {
    setPreview(null)
    setPage(1)
    setError(null)
    if (expandedRef.current) void loadPreview()
  }, [file.name, version, loadPreview])

  const isImage = IMAGE_EXTS.has(ext)
  const kind = DOC_EXTS.has(ext) || isImage || TEXT_EXTS.has(ext) ? ext : 'file'
  const Icon = isImage ? FileImage
    : kind === 'pptx' ? Presentation
    : (kind === 'xlsx' || kind === 'csv') ? FileSpreadsheet
    : kind === 'file' ? File : FileText
  const previewable = kind !== 'file'
  const totalPages = preview?.page_count ?? 0
  // The bare fallback outline is just the filename with zero lines — the card
  // header already shows that, so it isn't a useful preview.
  const outline = preview?.outline
  const hasUsefulOutline = !!outline?.length &&
    !(outline.length === 1 && outline[0].lines.length === 0)

  return (
    <div className="chat-artifact-card">
      <div className="chat-artifact-head">
        <span className={`chat-artifact-icon chat-artifact-icon-${kind}`}>
          <Icon size={16} />
        </span>
        <div className="chat-artifact-meta">
          <span className="chat-artifact-name" title={file.name}>{file.name}</span>
          <span className="chat-artifact-sub">{kind.toUpperCase()} · {formatSize(file.size)}</span>
        </div>
        <div className="chat-artifact-actions">
          {previewable && (
            <button type="button" className="chat-artifact-btn" onClick={togglePreview}
                    title={expanded ? labels.hidePreview : labels.preview}>
              {expanded ? <EyeOff size={13} /> : <Eye size={13} />}
            </button>
          )}
          <a className="chat-artifact-btn" href={`${base}${tokenSuffix}`} download={file.name} title={labels.download}>
            <Download size={13} />
          </a>
          {exportFormats.map((fmt) => (
            <a key={fmt} className="chat-artifact-btn chat-artifact-export"
               href={`${base}/export/${fmt}${tokenSuffix}`} download title={`${labels.exportAs} ${fmt.toUpperCase()}`}>
              {fmt.toUpperCase()}
            </a>
          ))}
        </div>
      </div>
      {expanded && (
        <div className="chat-artifact-preview">
          {loading && <div className="chat-artifact-preview-status">…</div>}
          {error && <div className="chat-artifact-preview-status">{error}</div>}
          {preview?.rendered && totalPages > 0 && (
            <>
              <div className="chat-artifact-page-nav">
                <button type="button" className="chat-artifact-btn" disabled={page <= 1}
                        onClick={() => setPage((p) => Math.max(1, p - 1))}>
                  <ChevronLeft size={13} />
                </button>
                <span>{labels.pageOf(page, totalPages)}</span>
                <button type="button" className="chat-artifact-btn" disabled={page >= totalPages}
                        onClick={() => setPage((p) => Math.min(totalPages, p + 1))}>
                  <ChevronRight size={13} />
                </button>
              </div>
              <img
                className="chat-artifact-page"
                src={`${base}/preview/page/${page}${tokenSuffix}${tokenSuffix ? '&' : '?'}v=${version}`}
                alt={`${file.name} — ${labels.pageOf(page, totalPages)}`}
                loading="lazy"
              />
            </>
          )}
          {preview && !preview.rendered && !hasUsefulOutline && (
            <div className="chat-artifact-preview-status">{labels.previewFailed}</div>
          )}
          {preview && !preview.rendered && hasUsefulOutline && outline && (
            <div className="chat-artifact-outline">
              {outline.map((section, i) => (
                <div key={i} className="chat-artifact-outline-page">
                  <div className="chat-artifact-outline-title">{section.title}</div>
                  {section.lines.slice(0, 12).map((line, j) => (
                    <div key={j} className="chat-artifact-outline-line">{line}</div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
})
