import { memo, useState } from 'react'
import { CornerDownRight, GripVertical, Pencil, Play, SendHorizontal, X } from 'lucide-react'

export interface QueuedChatMessage {
  id: string
  text: string
  images?: string[]
  audioEmotions?: { audio_prosody?: unknown }
}

export interface QueuedMessagesLabels {
  title: string
  sendNow: string
  inject?: string
  edit: string
  remove: string
  resume: string
  imageMessage: string
}

interface Props {
  items: QueuedChatMessage[]
  paused: boolean
  labels: QueuedMessagesLabels
  onSendNow: (item: QueuedChatMessage) => void
  onInject?: (item: QueuedChatMessage) => void
  onEdit: (item: QueuedChatMessage) => void
  onDelete: (item: QueuedChatMessage) => void
  onReorder: (fromIndex: number, toIndex: number) => void
  onResume: () => void
  hidden?: boolean
}

/* Compact chip row for messages enqueued while a run is streaming.
   In-memory only — lives purely in page state. */
export const QueuedMessagesBar = memo(function QueuedMessagesBar({
  items,
  paused,
  labels,
  onSendNow,
  onInject,
  onEdit,
  onDelete,
  onReorder,
  onResume,
  hidden,
}: Props) {
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const [overIndex, setOverIndex] = useState<number | null>(null)

  if (hidden || items.length === 0) return null

  return (
    <div className="chat-queued-bar" role="list" aria-label={labels.title}>
      {items.map((item, idx) => (
        <div
          key={item.id}
          role="listitem"
          className={`chat-queued-chip${overIndex === idx && dragIndex !== idx ? ' drag-over' : ''}${dragIndex === idx ? ' dragging' : ''}`}
          draggable
          onDragStart={(e) => {
            setDragIndex(idx)
            e.dataTransfer.effectAllowed = 'move'
            e.dataTransfer.setData('text/plain', item.id)
          }}
          onDragOver={(e) => {
            e.preventDefault()
            e.dataTransfer.dropEffect = 'move'
            if (overIndex !== idx) setOverIndex(idx)
          }}
          onDragLeave={() => {
            if (overIndex === idx) setOverIndex(null)
          }}
          onDrop={(e) => {
            e.preventDefault()
            if (dragIndex !== null && dragIndex !== idx) onReorder(dragIndex, idx)
            setDragIndex(null)
            setOverIndex(null)
          }}
          onDragEnd={() => {
            setDragIndex(null)
            setOverIndex(null)
          }}
        >
          <GripVertical size={12} className="chat-queued-grip" aria-hidden />
          <span className="chat-queued-chip-text" title={item.text || labels.imageMessage}>
            {item.text || labels.imageMessage}
          </span>
          {item.images && item.images.length > 0 && (
            <span className="chat-queued-chip-thumbs">
              {item.images.slice(0, 3).map((src, i) => (
                <img key={i} src={src} alt="" className="chat-queued-chip-thumb" />
              ))}
              {item.images.length > 3 && (
                <span className="chat-queued-chip-count">+{item.images.length - 3}</span>
              )}
            </span>
          )}
          <button
            type="button"
            className="chat-queued-chip-btn send"
            title={labels.sendNow}
            onClick={() => onSendNow(item)}
          >
            <SendHorizontal size={11} />
          </button>
          {onInject && labels.inject && (
            <button
              type="button"
              className="chat-queued-chip-btn inject"
              title={labels.inject}
              onClick={() => onInject(item)}
            >
              <CornerDownRight size={11} />
            </button>
          )}
          <button
            type="button"
            className="chat-queued-chip-btn"
            title={labels.edit}
            onClick={() => onEdit(item)}
          >
            <Pencil size={11} />
          </button>
          <button
            type="button"
            className="chat-queued-chip-btn"
            title={labels.remove}
            onClick={() => onDelete(item)}
          >
            <X size={11} />
          </button>
        </div>
      ))}
      {paused && (
        <button type="button" className="chat-queued-resume" onClick={onResume} title={labels.resume}>
          <Play size={11} /> {labels.resume}
        </button>
      )}
    </div>
  )
})
