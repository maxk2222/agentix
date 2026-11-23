import { useEffect, useMemo, useState } from 'react'
import { CalendarClock, CheckCircle2 } from 'lucide-react'
import { CronExpressionParser } from 'cron-parser'

/* ------------------------------------------------------------------ */
/* Types & constants                                                  */
/* ------------------------------------------------------------------ */

type Freq = 'minutes' | 'hourly' | 'daily' | 'weekly' | 'monthly' | 'custom'

type CronDraft = {
  freq: Freq
  everyN: number // minutes mode
  minute: number // hourly mode (minute of the hour)
  time: string // "HH:MM" — daily/weekly/monthly
  weekdays: number[] // cron day-of-week, 0=Sun..6=Sat
  dayOfMonth: number // 1..31
}

export const BROWSER_TZ =
  Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'

const COMMON_TIMEZONES = [
  BROWSER_TZ,
  'UTC',
  'Europe/Berlin',
  'Europe/London',
  'Europe/Paris',
  'Europe/Madrid',
  'America/New_York',
  'America/Chicago',
  'America/Los_Angeles',
  'Asia/Tokyo',
  'Asia/Shanghai',
  'Australia/Sydney',
]

/* UI order Mon..Sun; values are cron day-of-week numbers (0=Sun). */
const WEEKDAY_ORDER = [1, 2, 3, 4, 5, 6, 0]
const WEEKDAY_KEYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
const DOW_TO_KEY = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']

const MINUTE_PRESETS = [1, 2, 5, 10, 15, 20, 30]

/* ------------------------------------------------------------------ */
/* Parse: cron expression -> structured draft (null => custom mode)   */
/* ------------------------------------------------------------------ */

function numList(field: string): number[] | null {
  if (!/^\d+(,\d+)*$/.test(field)) return null
  return field.split(',').map(Number)
}

function timeStr(hour: number, minute: number): string {
  return `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`
}

function parseCron(expr: string): CronDraft | null {
  const f = expr.trim().split(/\s+/)
  if (f.length !== 5) return null
  const [min, hour, dom, mon, dow] = f
  if (mon !== '*') return null

  if (dom === '*' && dow === '*') {
    if (hour === '*') {
      const everyMin = min === '*' ? '1' : min.match(/^\*\/(\d+)$/)?.[1]
      if (everyMin) {
        const n = parseInt(everyMin)
        if (n >= 1 && n <= 59)
          return { freq: 'minutes', everyN: n, minute: 0, time: '09:00', weekdays: [1], dayOfMonth: 1 }
      }
      const m = numList(min)
      if (m && m.length === 1 && m[0] <= 59)
        return { freq: 'hourly', everyN: 15, minute: m[0], time: '09:00', weekdays: [1], dayOfMonth: 1 }
      return null
    }
    const hm = numList(hour)
    const mm = numList(min)
    if (hm && mm && hm.length === 1 && mm.length === 1 && hm[0] <= 23 && mm[0] <= 59)
      return { freq: 'daily', everyN: 15, minute: 0, time: timeStr(hm[0], mm[0]), weekdays: [1], dayOfMonth: 1 }
    return null
  }

  if (dom === '*' && dow !== '*') {
    const hm = numList(hour)
    const mm = numList(min)
    const days = numList(dow)
    if (!hm || !mm || !days || hm.length !== 1 || mm.length !== 1) return null
    if (hm[0] > 23 || mm[0] > 59) return null
    const normalized = [...new Set(days.map((d) => (d === 7 ? 0 : d)))]
    if (normalized.some((d) => d < 0 || d > 6)) return null
    return { freq: 'weekly', everyN: 15, minute: 0, time: timeStr(hm[0], mm[0]), weekdays: normalized, dayOfMonth: 1 }
  }

  if (dom !== '*' && dow === '*') {
    const hm = numList(hour)
    const mm = numList(min)
    const d = numList(dom)
    if (!hm || !mm || !d || hm.length !== 1 || mm.length !== 1 || d.length !== 1) return null
    if (hm[0] > 23 || mm[0] > 59 || d[0] < 1 || d[0] > 31) return null
    return { freq: 'monthly', everyN: 15, minute: 0, time: timeStr(hm[0], mm[0]), weekdays: [1], dayOfMonth: d[0] }
  }

  return null
}

/* ------------------------------------------------------------------ */
/* Emit: structured draft -> cron expression                          */
/* ------------------------------------------------------------------ */

function buildCron(d: CronDraft): string {
  const [h, m] = d.time.split(':').map(Number)
  switch (d.freq) {
    case 'minutes':
      return d.everyN <= 1 ? '* * * * *' : `*/${d.everyN} * * * *`
    case 'hourly':
      return `${d.minute} * * * *`
    case 'daily':
      return `${m} ${h} * * *`
    case 'weekly': {
      const days = [...new Set(d.weekdays)].sort((a, b) => a - b)
      return `${m} ${h} * * ${days.join(',')}`
    }
    case 'monthly':
      return `${m} ${h} ${d.dayOfMonth} * *`
    default:
      return '* * * * *'
  }
}

/* ------------------------------------------------------------------ */
/* Next executions preview                                            */
/* ------------------------------------------------------------------ */

function nextRuns(expr: string, tz: string, count: number): Date[] | null {
  try {
    const it = CronExpressionParser.parse(expr, {
      tz,
      currentDate: new Date(),
    })
    const out: Date[] = []
    for (let i = 0; i < count; i++) out.push(it.next().toDate())
    return out
  } catch {
    return null
  }
}

function fmtRun(date: Date, tz: string): string {
  return date.toLocaleString('de-DE', {
    timeZone: tz,
    weekday: 'short',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/* ------------------------------------------------------------------ */
/* Component                                                          */
/* ------------------------------------------------------------------ */

export function CronBuilder({
  value,
  onChange,
  disabled,
  $t,
}: {
  value: Record<string, unknown>
  onChange: (config: Record<string, unknown>) => void
  disabled?: boolean
  $t: (key: string, params?: Record<string, string | number>) => string
}) {
  const expr = String(value.cron ?? '* * * * *')
  const tz = String(value.timezone ?? 'UTC')

  /* Structured view of the current expression; null => expression is a
     custom pattern the builder cannot represent. */
  const parsed = useMemo(() => parseCron(expr), [expr])

  /* Custom mode is sticky: while editing a raw expression, intermediate
     values may momentarily parse as a standard pattern — we must not flip
     the UI out from under the user's cursor. Only an explicit click on a
     frequency button leaves custom mode. */
  const [customMode, setCustomMode] = useState(() => parsed === null)
  const freq: Freq = customMode ? 'custom' : (parsed?.freq ?? 'custom')

  /* Keep the last parsed draft so switching modes preserves settings. */
  const [lastDraft, setLastDraft] = useState<CronDraft>(
    () =>
      parsed ?? {
        freq: 'daily',
        everyN: 15,
        minute: 0,
        time: '09:00',
        weekdays: [1],
        dayOfMonth: 1,
      },
  )
  useEffect(() => {
    if (parsed) setLastDraft(parsed)
  }, [parsed])

  const draft: CronDraft = useMemo(
    () => (parsed ? { ...parsed, ...carryOver(parsed, lastDraft) } : lastDraft),
    [parsed, lastDraft],
  )

  const emit = (next: Partial<CronDraft>) => {
    const merged = { ...draft, ...next }
    setLastDraft(merged)
    onChange({ ...value, cron: buildCron(merged), timezone: tz })
  }

  const setTz = (nextTz: string) => onChange({ ...value, timezone: nextTz })

  const runs = useMemo(() => nextRuns(expr, tz, 3), [expr, tz])
  const summary = useMemo(() => describe(draft, freq, $t), [draft, freq, $t])
  const weeklyEmpty = freq === 'weekly' && draft.weekdays.length === 0
  const monthDayWarn = freq === 'monthly' && draft.dayOfMonth > 28
  const invalid = runs === null

  const freqButtons: { id: Freq; label: string }[] = [
    { id: 'minutes', label: $t('triggers.cronMinutes') },
    { id: 'hourly', label: $t('triggers.cronHourly') },
    { id: 'daily', label: $t('triggers.cronDaily') },
    { id: 'weekly', label: $t('triggers.cronWeekly') },
    { id: 'monthly', label: $t('triggers.cronMonthly') },
    { id: 'custom', label: $t('triggers.cronCustom') },
  ]

  const minuteOptions = MINUTE_PRESETS.includes(draft.everyN)
    ? MINUTE_PRESETS
    : [...MINUTE_PRESETS, draft.everyN].sort((a, b) => a - b)

  return (
    <div className="cron-builder">
      <div className="cron-modes">
        {freqButtons.map((b) => (
          <button
            key={b.id}
            type="button"
            className={`cron-mode-btn ${freq === b.id ? 'active' : ''}`}
            disabled={disabled}
            onClick={() => {
              if (b.id === 'custom') {
                setCustomMode(true)
                return
              }
              setCustomMode(false)
              emit({ freq: b.id })
            }}
          >
            {b.label}
          </button>
        ))}
      </div>

      {freq === 'minutes' && (
        <div className="cron-row">
          <span>{$t('triggers.cronEvery')}</span>
          <select
            value={draft.everyN}
            disabled={disabled}
            onChange={(e) => emit({ everyN: parseInt(e.target.value) })}
          >
            {minuteOptions.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
          <span>{$t('triggers.cronMinutesUnit')}</span>
        </div>
      )}

      {freq === 'hourly' && (
        <div className="cron-row">
          <span>{$t('triggers.cronAtMinute')}</span>
          <select
            value={draft.minute}
            disabled={disabled}
            onChange={(e) => emit({ minute: parseInt(e.target.value) })}
          >
            {Array.from({ length: 60 }, (_, i) => (
              <option key={i} value={i}>
                :{String(i).padStart(2, '0')}
              </option>
            ))}
          </select>
        </div>
      )}

      {(freq === 'daily' || freq === 'weekly' || freq === 'monthly') && (
        <div className="cron-row">
          <span>{$t('triggers.cronTime')}</span>
          <input
            type="time"
            value={draft.time}
            disabled={disabled}
            onChange={(e) => emit({ time: e.target.value || '09:00' })}
          />
        </div>
      )}

      {freq === 'weekly' && (
        <div className="cron-field">
          <span className="cron-label">{$t('triggers.cronWeekdays')}</span>
          <div className="cron-weekdays">
            {WEEKDAY_ORDER.map((dow, i) => {
              const active = draft.weekdays.includes(dow)
              const locked = active && draft.weekdays.length === 1
              return (
                <button
                  key={dow}
                  type="button"
                  className={`cron-chip ${active ? 'active' : ''}`}
                  disabled={disabled || locked}
                  title={$t(`triggers.cronWDLong.${DOW_TO_KEY[dow]}`)}
                  onClick={() => {
                    if (active && draft.weekdays.length === 1) return
                    const next = active
                      ? draft.weekdays.filter((d) => d !== dow)
                      : [...draft.weekdays, dow]
                    emit({ weekdays: next })
                  }}
                >
                  {$t(`triggers.cronWD.${WEEKDAY_KEYS[i]}`)}
                </button>
              )
            })}
          </div>
          {weeklyEmpty && (
            <small className="field-help" style={{ color: 'var(--warning)' }}>
              {$t('triggers.cronWeekdayPickOne')}
            </small>
          )}
        </div>
      )}

      {freq === 'monthly' && (
        <div className="cron-row">
          <span>{$t('triggers.cronDayOfMonth')}</span>
          <input
            type="number"
            min={1}
            max={31}
            value={draft.dayOfMonth}
            disabled={disabled}
            onChange={(e) => {
              const v = parseInt(e.target.value)
              if (!Number.isNaN(v)) emit({ dayOfMonth: Math.min(31, Math.max(1, v)) })
            }}
          />
        </div>
      )}
      {monthDayWarn && (
        <small className="field-help" style={{ color: 'var(--warning)' }}>
          {$t('triggers.cronMonthDayHint')}
        </small>
      )}

      {freq === 'custom' && (
        <div className="cron-field">
          <span className="cron-label">{$t('triggers.cronExpression')}</span>
          <input
            type="text"
            value={expr}
            disabled={disabled}
            placeholder="0 9 * * *"
            onChange={(e) => onChange({ ...value, cron: e.target.value })}
          />
          <small className="field-help">{$t('triggers.cronCustomHint')}</small>
        </div>
      )}

      <div className="cron-row">
        <span>{$t('triggers.cronTimezone')}</span>
        <select value={tz} disabled={disabled} onChange={(e) => setTz(e.target.value)}>
          {[...new Set([tz, ...COMMON_TIMEZONES])].map((zone) => (
            <option key={zone} value={zone}>
              {zone === BROWSER_TZ ? `${zone} (${$t('triggers.cronTzLocal')})` : zone}
            </option>
          ))}
        </select>
      </div>

      <div className="cron-preview">
        <div className="cron-summary">
          {invalid ? (
            <span className="text-danger">{$t('triggers.cronInvalid')}</span>
          ) : (
            <>
              <CheckCircle2 size={14} />
              <span>{summary}</span>
              <code className="cron-expr">{expr}</code>
            </>
          )}
        </div>
        {runs && (
          <div className="cron-nextruns">
            <span className="cron-label">
              {$t('triggers.cronNextRuns')} · {tz}
            </span>
            {runs.map((r, i) => (
              <span key={i} className="cron-next-item">
                <CalendarClock size={12} />
                {fmtRun(r, tz)}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

/* Carry time/minute-ish fields between modes so switching keeps settings. */
function carryOver(parsed: CronDraft, prev: CronDraft): Partial<CronDraft> {
  return {
    everyN: parsed.freq === 'minutes' ? parsed.everyN : prev.everyN,
    minute: parsed.freq === 'hourly' ? parsed.minute : prev.minute,
    time:
      parsed.freq === 'daily' || parsed.freq === 'weekly' || parsed.freq === 'monthly'
        ? parsed.time
        : prev.time,
    weekdays: parsed.freq === 'weekly' ? parsed.weekdays : prev.weekdays,
    dayOfMonth: parsed.freq === 'monthly' ? parsed.dayOfMonth : prev.dayOfMonth,
  }
}

function describe(
  d: CronDraft,
  freq: Freq,
  $t: (key: string, params?: Record<string, string | number>) => string,
): string {
  switch (freq) {
    case 'minutes':
      return $t('triggers.cronSumMinutes', { n: d.everyN })
    case 'hourly':
      return $t('triggers.cronSumHourly', { m: String(d.minute).padStart(2, '0') })
    case 'daily':
      return $t('triggers.cronSumDaily', { t: d.time })
    case 'weekly': {
      if (d.weekdays.length === 7) return $t('triggers.cronSumDaily', { t: d.time })
      const days = [...d.weekdays]
        .sort((a, b) => WEEKDAY_ORDER.indexOf(a) - WEEKDAY_ORDER.indexOf(b))
        .map((dow) => $t(`triggers.cronWDLong.${DOW_TO_KEY[dow]}`))
        .join(', ')
      return $t('triggers.cronSumWeekly', { days, t: d.time })
    }
    case 'monthly':
      return $t('triggers.cronSumMonthly', { d: d.dayOfMonth, t: d.time })
    default:
      return $t('triggers.cronSumCustom')
  }
}
