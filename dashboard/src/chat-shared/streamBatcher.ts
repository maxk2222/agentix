/* Coalesces high-frequency streaming updates into ~150ms React state flushes.
   `schedule(flush)` stores the newest flush closure; the timer runs it once
   per window. `flushNow` runs it immediately — use for terminal events,
   stream end, errors and cancellations so no trailing deltas are lost. */
export class StreamBatcher {
  private timer: ReturnType<typeof setTimeout> | null = null
  private pending: (() => void) | null = null

  constructor(private readonly intervalMs = 150) {}

  schedule(flush: () => void) {
    this.pending = flush
    if (this.timer != null) return
    this.timer = setTimeout(() => {
      this.timer = null
      const fn = this.pending
      this.pending = null
      fn?.()
    }, this.intervalMs)
  }

  flushNow() {
    if (this.timer != null) {
      clearTimeout(this.timer)
      this.timer = null
    }
    const fn = this.pending
    this.pending = null
    fn?.()
  }

  cancel() {
    if (this.timer != null) {
      clearTimeout(this.timer)
      this.timer = null
    }
    this.pending = null
  }
}
