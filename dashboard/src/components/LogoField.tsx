import { useRef, useState } from 'react'
import { Upload, X } from 'lucide-react'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'

const MAX_IMAGE_BYTES = 1024 * 1024

/**
 * Shared logo editor: live preview + file upload (stored as data-URL) + URL
 * input + clear button. The `hint` prop can flag inherited defaults
 * ("empty = global logo").
 */
export default function LogoField({
  label,
  value,
  onChange,
  hint,
  previewHeight = 40,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  hint?: string
  previewHeight?: number
}) {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const fileRef = useRef<HTMLInputElement>(null)
  const [error, setError] = useState('')

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    setError('')
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setError($t('branding.logoNotImage'))
      return
    }
    if (file.size > MAX_IMAGE_BYTES) {
      setError($t('branding.logoTooLarge'))
      return
    }
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === 'string') onChange(reader.result)
    }
    reader.onerror = () => setError($t('branding.logoReadError'))
    reader.readAsDataURL(file)
  }

  return (
    <div className="logo-field">
      <label className="logo-field-label">{label}</label>
      <div className="logo-field-row">
        <div className="logo-field-preview" style={{ height: previewHeight + 8 }}>
          {value ? (
            <img
              src={value}
              alt={label}
              style={{ maxHeight: previewHeight, maxWidth: 160, objectFit: 'contain', display: 'block' }}
              onError={(e) => ((e.target as HTMLImageElement).style.opacity = '0.3')}
            />
          ) : (
            <span className="logo-field-empty">{$t('branding.noLogo')}</span>
          )}
        </div>
        <div className="logo-field-actions">
          <button type="button" className="btn btn-secondary" onClick={() => fileRef.current?.click()}>
            <Upload size={14} /> {$t('branding.uploadLogo')}
          </button>
          {value && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => { setError(''); onChange('') }}
              title={$t('common.delete')}
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={handleFile}
      />
      <input
        type="text"
        className="logo-field-url"
        value={value}
        onChange={(e) => { setError(''); onChange(e.target.value) }}
        placeholder="https://…"
      />
      {hint && <div className="logo-field-hint">{hint}</div>}
      {error && <div className="logo-field-error">{error}</div>}
    </div>
  )
}
