/* Shared chat upload helpers — images go to the multimodal `images` field,
   documents are uploaded to /chat/attachments and injected server-side as
   extracted text on every turn. */

export const DOC_EXTS = [
  '.docx', '.xlsx', '.pptx', '.pdf', '.odt', '.csv', '.txt', '.md',
  '.doc', '.xls', '.ppt',
]

export const IMG_EXTS = [
  '.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff',
]

export const ACCEPT_STRING = [
  'image/png', 'image/jpeg', 'image/webp', 'image/gif',
  ...IMG_EXTS,
  ...DOC_EXTS,
].join(',')

export interface PendingFile {
  localId: string
  attachmentId?: string
  name: string
  size: number
  ext: string
  status: 'uploading' | 'ready' | 'error'
  error?: string
}

export interface AttachmentMeta {
  name: string
  ext?: string
  /** Server attachment id — present once the upload is stored; needed to
      build the /chat/attachments/{conv}/{id}/file URL for previews. */
  id?: string
  size?: number
}

export function isImageFile(file: File): boolean {
  // MIME type is authoritative, but drops from some file managers/browsers
  // arrive with an empty `type` — fall back to the extension so those files
  // are not silently discarded.
  if (file.type.startsWith('image/')) return true
  if (!file.name.includes('.')) return false
  const ext = `.${(file.name.split('.').pop() || '').toLowerCase()}`
  return IMG_EXTS.includes(ext)
}

export function isDocFile(file: File): boolean {
  const ext = `.${(file.name.split('.').pop() || '').toLowerCase()}`
  return DOC_EXTS.includes(ext)
}

export interface UploadMeta {
  id: string
  name: string
  ext: string
  size: number
  char_count: number
  truncated: boolean
}

export async function uploadChatAttachment(
  apiUrl: string,
  conversationId: string,
  file: File,
  headers?: Record<string, string>,
  model?: string,
): Promise<UploadMeta> {
  const form = new FormData()
  form.append('file', file)
  form.append('conversation_id', conversationId)
  if (model) form.append('model', model)
  const res = await fetch(`${apiUrl}/chat/attachments`, {
    method: 'POST',
    credentials: 'include',
    headers: headers || {},
    body: form,
  })
  const data = await res.json().catch(() => null)
  if (!res.ok) {
    throw new Error(data?.detail || `Upload failed (${res.status})`)
  }
  return data as UploadMeta
}

export async function deleteChatAttachment(
  apiUrl: string,
  conversationId: string,
  attachmentId: string,
  headers?: Record<string, string>,
): Promise<void> {
  await fetch(
    `${apiUrl}/chat/attachments/${encodeURIComponent(conversationId)}/${encodeURIComponent(attachmentId)}`,
    { method: 'DELETE', credentials: 'include', headers: headers || {} },
  ).catch(() => {})
}

/** Delete a conversation and all its server-side files (artifacts + uploads). */
export async function deleteChatConversation(
  apiUrl: string,
  conversationId: string,
  headers?: Record<string, string>,
): Promise<void> {
  await fetch(
    `${apiUrl}/chat/conversations/${encodeURIComponent(conversationId)}`,
    { method: 'DELETE', credentials: 'include', headers: headers || {} },
  ).catch(() => {})
}
