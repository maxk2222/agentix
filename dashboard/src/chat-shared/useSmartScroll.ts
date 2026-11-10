import { useRef, useCallback, useEffect } from 'react'

export function useSmartScroll<T extends HTMLElement>(deps: React.DependencyList = []) {
  const containerRef = useRef<T | null>(null)
  const endRef = useRef<T | null>(null)
  const userScrolledUpRef = useRef(false)

  const scrollToBottom = useCallback(() => {
    if (!userScrolledUpRef.current && endRef.current) {
      endRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [])

  useEffect(() => {
    scrollToBottom()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const onScroll = () => {
      const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100
      userScrolledUpRef.current = !nearBottom
    }
    container.addEventListener('scroll', onScroll)
    return () => container.removeEventListener('scroll', onScroll)
  }, [])

  return {
    containerRef,
    endRef,
    userScrolledUpRef,
    scrollToBottom,
  }
}
