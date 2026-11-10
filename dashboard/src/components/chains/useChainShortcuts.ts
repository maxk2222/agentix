import { useEffect } from 'react'

export type ChainShortcutHandlers = {
  undo: () => void
  redo: () => void
  copy: () => void
  paste: () => void
  duplicate: () => void
  selectAll: () => void
  deselect: () => void
  /** Whether copy/duplicate currently have a node selection to act on. */
  hasSelection: () => boolean
}

function isEditableTarget(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null
  if (!t) return false
  const tag = t.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable
}

/** Global keyboard shortcuts for the chain editor (active while `enabled`). */
export function useChainShortcuts(enabled: boolean, handlers: ChainShortcutHandlers) {
  useEffect(() => {
    if (!enabled) return

    const onKeyDown = (e: KeyboardEvent) => {
      if (isEditableTarget(e)) return
      const mod = e.ctrlKey || e.metaKey
      const key = e.key.toLowerCase()

      if (mod && key === 'z') {
        e.preventDefault()
        if (e.shiftKey) handlers.redo()
        else handlers.undo()
      } else if (mod && key === 'y') {
        e.preventDefault()
        handlers.redo()
      } else if (mod && key === 'c') {
        if (!handlers.hasSelection()) return // let normal text copy through
        handlers.copy()
      } else if (mod && key === 'v') {
        e.preventDefault()
        handlers.paste()
      } else if (mod && key === 'd') {
        if (!handlers.hasSelection()) return
        e.preventDefault() // suppress browser bookmark dialog
        handlers.duplicate()
      } else if (mod && key === 'a') {
        e.preventDefault()
        handlers.selectAll()
      } else if (e.key === 'Escape') {
        handlers.deselect()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
    // Handlers are re-created per render; the effect re-binds cheaply.
  })
}
