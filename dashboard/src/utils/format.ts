/** Auto-scaled duration: <1s→ms, <1min→s, <1h→min, ≥1h→h. One decimal <10. */
export function fmtDuration(ms: number | null | undefined, lang: 'de' | 'en' = 'en'): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return '—'
  if (ms < 1000) return `${Math.round(ms)} ms`
  const loc = (v: number) => {
    const s = v >= 10 ? v.toFixed(0) : v.toFixed(1)
    return lang === 'de' ? s.replace('.', ',') : s
  }
  if (ms < 60_000) return `${loc(ms / 1000)} s`
  if (ms < 3_600_000) return `${loc(ms / 60_000)} min`
  return `${loc(ms / 3_600_000)} h`
}
