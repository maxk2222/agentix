import { useEffect, useState } from 'react'
import { FileText, Loader2, X, AlertCircle, Paperclip } from 'lucide-react'
import { IMG_EXTS, type PendingFile, type AttachmentMeta } from './uploads'

export interface UploadTrayLabels {
  uploading: string
  failed: string
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/* Pending uploads shown above the input: image thumbnails + document chips. */
export function UploadTray({
  pendingImages,
  pendingFiles,
  onRemoveImage,
  onRemoveFile,
  labels,
}: {
  pendingImages: string[]
  pendingFiles: PendingFile[]
  onRemoveImage: (index: number) => void
  onRemoveFile: (localId: string) => void
  labels: UploadTrayLabels
}) {
  if (pendingImages.length === 0 && pendingFiles.length === 0) return null
  return (
    <div style={{ display: 'flex', gap: '8px', padding: '4px 8px', flexWrap: 'wrap' }}>
      {pendingImages.map((img, idx) => (
        <div key={`img-${idx}`} style={{ position: 'relative', width: '48px', height: '48px' }}>
          <img src={img} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '6px' }} />
          <button
            onClick={() => onRemoveImage(idx)}
            style={{ position: 'absolute', top: '-4px', right: '-4px', width: '18px', height: '18px', borderRadius: '50%', background: '#dc3545', color: '#fff', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '11px' }}
          >
            <X size={12} />
          </button>
        </div>
      ))}
      {pendingFiles.map((f) => (
        <div
          key={f.localId}
          title={f.status === 'error' ? (f.error || labels.failed) : f.name}
          style={{
            display: 'flex', alignItems: 'center', gap: '6px', position: 'relative',
            padding: '6px 10px', borderRadius: '8px', fontSize: '12px',
            background: f.status === 'error' ? 'rgba(220,53,69,0.12)' : 'var(--bg-secondary, rgba(128,128,128,0.12))',
            color: f.status === 'error' ? '#dc3545' : 'var(--text-primary, inherit)',
            maxWidth: '220px',
          }}
        >
          {f.status === 'uploading'
            ? <Loader2 size={14} className="upload-spin" style={{ animation: 'spin 1s linear infinite' }} />
            : f.status === 'error'
              ? <AlertCircle size={14} />
              : <FileText size={14} />}
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {f.status === 'uploading' ? labels.uploading : f.name}
          </span>
          <span style={{ opacity: 0.6, flexShrink: 0 }}>{formatSize(f.size)}</span>
          <button
            onClick={() => onRemoveFile(f.localId)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', padding: '0', display: 'flex' }}
          >
            <X size={12} />
          </button>
        </div>
      ))}
    </div>
  )
}

/* Fetch the original attachment with auth headers and hand out an object URL
   — <img> tags cannot send headers, so a plain src= won't authenticate. */
function useAttachmentObjectUrl(
  apiUrl: string | undefined,
  conversationId: string | undefined,
  attachmentId: string | undefined,
  headers?: Record<string, string>,
): string | null {
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    if (!apiUrl || !conversationId || !attachmentId) return
    let objectUrl: string | null = null
    let cancelled = false
    fetch(
      `${apiUrl}/chat/attachments/${encodeURIComponent(conversationId)}/${encodeURIComponent(attachmentId)}/file`,
      { credentials: 'include', headers: headers || {} },
    )
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.blob()
      })
      .then((blob) => {
        if (cancelled) return
        objectUrl = URL.createObjectURL(blob)
        setUrl(objectUrl)
      })
      .catch(() => {})
    return () => {
      cancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
    // headers object identity changes per render — serialize for the dep.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiUrl, conversationId, attachmentId, JSON.stringify(headers || {})])
  return url
}

const IMAGE_EXT_SET = new Set(IMG_EXTS)

function AttachmentThumb({
  apiUrl,
  conversationId,
  attachment,
  headers,
}: {
  apiUrl?: string
  conversationId?: string
  attachment: AttachmentMeta
  headers?: Record<string, string>
}) {
  const url = useAttachmentObjectUrl(apiUrl, conversationId, attachment.id, headers)
  if (!url) return <AttachmentChip attachment={attachment} />
  return (
    <a href={url} target="_blank" rel="noreferrer" title={attachment.name}>
      <img
        src={url}
        alt={attachment.name}
        style={{
          display: 'block', maxWidth: '200px', maxHeight: '140px',
          borderRadius: '8px', objectFit: 'cover', cursor: 'zoom-in',
        }}
      />
    </a>
  )
}

function AttachmentChip({ attachment: a }: { attachment: AttachmentMeta }) {
  return (
    <span
      title={a.name}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: '4px',
        padding: '2px 8px', borderRadius: '6px', fontSize: '11px',
        background: 'rgba(128,128,128,0.15)', maxWidth: '200px',
      }}
    >
      <Paperclip size={11} />
      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{a.name}</span>
    </span>
  )
}

/* Chips rendered inside a sent message bubble for its attachments. Image
   attachments with a server id render as clickable thumbnails. */
export function AttachmentChips({
  attachments,
  conversationId,
  apiUrl,
  headers,
}: {
  attachments?: AttachmentMeta[]
  conversationId?: string
  apiUrl?: string
  headers?: Record<string, string>
}) {
  if (!attachments || attachments.length === 0) return null
  return (
    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '4px' }}>
      {attachments.map((a, i) => {
        const isImage = !!a.ext && IMAGE_EXT_SET.has(a.ext.toLowerCase())
        if (isImage && a.id) {
          return (
            <AttachmentThumb
              key={`${a.id}-${i}`}
              apiUrl={apiUrl}
              conversationId={conversationId}
              attachment={a}
              headers={headers}
            />
          )
        }
        return <AttachmentChip key={`${a.name}-${i}`} attachment={a} />
      })}
    </div>
  )
}
