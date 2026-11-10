import { useEffect, useRef, useState, useCallback } from 'react'
import {
  Upload, Trash2, Play, CheckCircle, AlertCircle,
  Volume2, Settings, Mic, Speaker, X, ChevronDown, ChevronUp,
  FileAudio, GripVertical,
} from 'lucide-react'
import { useAuth } from '../api/AuthContext'
import UnsavedChangesBar, { useUnsavedGuard } from './UnsavedChangesBar'
import ConfirmModal from './ConfirmModal'
import { useT } from '../i18n'
import { useAppPreferences } from '../ui/AppPreferences'
import { audioApi, type VoiceEntry, type AudioSettings } from '../api/client'

type VoiceType = 'piper' | 'kokoro'

interface AudioHealth {
  stt_ready: boolean
  tts_ready: boolean
  active_voice: string | null
  voices_count: number
  piper_found: boolean
  piper_path: string
  kokoro_ready: boolean
  tts_provider: string
}

export default function AudioPage() {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const { hasPermission } = useAuth()
  const canWrite = hasPermission('instructions:write')
  const canWriteConfig = hasPermission('runtime_config:write')

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<AudioHealth | null>(null)
  const [piperVoices, setPiperVoices] = useState<VoiceEntry[]>([])
  const [kokoroVoices, setKokoroVoices] = useState<VoiceEntry[]>([])
  const [settings, setSettings] = useState<AudioSettings>({})
  const [settingsExpanded, setSettingsExpanded] = useState(false)
  const [savingSettings, setSavingSettings] = useState(false)
  const settingsBaselineRef = useRef<string>(JSON.stringify({}))
  const [dropOver, setDropOver] = useState<VoiceType | null>(null)
  const [uploading, setUploading] = useState(false)
  const [showDownloadModal, setShowDownloadModal] = useState(false)
  const [downloadForm, setDownloadForm] = useState({ voice_id: '', onnx_url: '', config_url: '' })
  const [downloading, setDownloading] = useState(false)
  const [playingVoice, setPlayingVoice] = useState<string | null>(null)
  const [testingVoice, setTestingVoice] = useState<string | null>(null)
  const [voiceDeleteTarget, setVoiceDeleteTarget] = useState<{ id: string; kind: 'piper' | 'kokoro' } | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const piperInputRef = useRef<HTMLInputElement>(null)
  const kokoroInputRef = useRef<HTMLInputElement>(null)

  const fetchAll = useCallback(async () => {
    setError(null)
    try {
      const [healthRes, piperRes, kokoroRes, settingsRes] = await Promise.all([
        fetch('/audio/health', { credentials: 'include' }).then((r) => (r.ok ? r.json() : null)).catch(() => null),
        audioApi.listPiperVoices().catch(() => ({ voices: [] })),
        audioApi.listKokoroVoices().catch(() => ({ voices: [] })),
        audioApi.getSettings().catch(() => ({} as AudioSettings)),
      ])
      setHealth(healthRes)
      setPiperVoices(piperRes.voices)
      setKokoroVoices(kokoroRes.voices)
      setSettings(settingsRes)
      settingsBaselineRef.current = JSON.stringify(settingsRes)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchAll()
  }, [fetchAll])

  const handleSettingChange = (key: keyof AudioSettings, value: string | number | null) => {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  const settingsDirty = JSON.stringify(settings) !== settingsBaselineRef.current

  useUnsavedGuard(settingsDirty && canWriteConfig, $t('common.leaveWarning'))

  const handleSaveSettings = async () => {
    if (!canWriteConfig) return
    setSavingSettings(true)
    setError(null)
    try {
      await audioApi.saveSettings(settings)
      await fetchAll()
    } catch (err) {
      setError(String(err))
    } finally {
      setSavingSettings(false)
    }
  }

  const handleDiscardSettings = () => {
    setSettings(JSON.parse(settingsBaselineRef.current))
  }

  const processFiles = async (files: FileList | null, type: VoiceType) => {
    if (!files || files.length < 2) {
      setError(type === 'piper'
        ? 'Piper: Beide Dateien (ONNX + .onnx.json) erforderlich'
        : 'Kokoro: Beide Dateien (ONNX + .npz) erforderlich')
      return
    }
    const fileArray = Array.from(files)
    if (type === 'piper') {
      const onnx = fileArray.find((f) => f.name.endsWith('.onnx') && !f.name.endsWith('.onnx.json'))
      const config = fileArray.find((f) => f.name.endsWith('.onnx.json'))
      if (!onnx || !config) { setError('Piper: .onnx und .onnx.json Dateien erforderlich'); return }
      setUploading(true)
      try {
        await audioApi.uploadPiperVoice(onnx, config)
        await fetchAll()
      } catch (err) { setError(String(err)) } finally { setUploading(false) }
    } else {
      const onnx = fileArray.find((f) => f.name.endsWith('.onnx'))
      const npz = fileArray.find((f) => f.name.endsWith('.npz'))
      if (!onnx || !npz) { setError('Kokoro: .onnx und .npz Dateien erforderlich'); return }
      setUploading(true)
      try {
        await audioApi.uploadKokoroVoice(onnx, npz)
        await fetchAll()
      } catch (err) { setError(String(err)) } finally { setUploading(false) }
    }
  }

  const handleDragOver = (e: React.DragEvent, type: VoiceType) => { e.preventDefault(); setDropOver(type) }
  const handleDragLeave = (e: React.DragEvent) => { e.preventDefault(); setDropOver(null) }
  const handleDrop = (e: React.DragEvent, type: VoiceType) => { e.preventDefault(); setDropOver(null); processFiles(e.dataTransfer.files, type) }
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>, type: VoiceType) => { processFiles(e.target.files, type); e.target.value = '' }

  const handleActivatePiper = async (id: string) => { try { await audioApi.activatePiperVoice(id); await fetchAll() } catch (err) { setError(String(err)) } }
  const handleDeletePiper = async (id: string) => { setVoiceDeleteTarget({ id, kind: 'piper' }) }
  const handleTestPiper = async (id: string) => { setTestingVoice(id); setError(null); try { const blob = await audioApi.testPiperVoice(id, 'Hallo, das ist ein Test.'); const url = URL.createObjectURL(blob); setPlayingVoice(`piper:${id}`); if (audioRef.current) { audioRef.current.src = url; audioRef.current.play() } } catch (err) { setError(String(err)) } finally { setTestingVoice(null) } }

  const handleActivateKokoro = async (id: string) => { try { await audioApi.activateKokoroVoice(id); await fetchAll() } catch (err) { setError(String(err)) } }
  const handleDeleteKokoro = async (id: string) => { setVoiceDeleteTarget({ id, kind: 'kokoro' }) }
  const handleTestKokoro = async (id: string) => { setTestingVoice(id); setError(null); try { const blob = await audioApi.testKokoroVoice(id, 'Hallo, das ist ein Test.'); const url = URL.createObjectURL(blob); setPlayingVoice(`kokoro:${id}`); if (audioRef.current) { audioRef.current.src = url; audioRef.current.play() } } catch (err) { setError(String(err)) } finally { setTestingVoice(null) } }

  const handleDownloadPiper = async () => {
    if (!downloadForm.voice_id || !downloadForm.onnx_url || !downloadForm.config_url) { setError('Alle Felder sind erforderlich'); return }
    setDownloading(true); setError(null)
    try {
      await audioApi.downloadPiperVoice(downloadForm.voice_id, downloadForm.onnx_url, downloadForm.config_url)
      await fetchAll(); setShowDownloadModal(false); setDownloadForm({ voice_id: '', onnx_url: '', config_url: '' })
    } catch (err) { setError(String(err)) } finally { setDownloading(false) }
  }

  const renderDropZone = (type: VoiceType) => {
    const isPiper = type === 'piper'
    const active = dropOver === type
    return (
      <div
        className={`vm-dropzone ${active ? 'active' : ''}`}
        onDragOver={(e) => handleDragOver(e, type)}
        onDragLeave={handleDragLeave}
        onDrop={(e) => handleDrop(e, type)}
        onClick={() => (isPiper ? piperInputRef : kokoroInputRef).current?.click()}
      >
        <Upload size={28} />
        <div className="vm-dropzone-title">{isPiper ? $t('voiceManagement.dropFilesPiper') : $t('voiceManagement.dropFilesKokoro')}</div>
        <div className="vm-dropzone-hint">{isPiper ? $t('voiceManagement.dropHintPiper') : $t('voiceManagement.dropHintKokoro')}</div>
        <div className="vm-dropzone-or">{$t('voiceManagement.orSelect')}</div>
        <input ref={isPiper ? piperInputRef : kokoroInputRef} type="file" multiple style={{ display: 'none' }} onChange={(e) => handleFileSelect(e, type)} />
      </div>
    )
  }

  const renderVoiceCard = (voice: VoiceEntry, type: VoiceType) => {
    const isPiper = type === 'piper'
    const isPlaying = playingVoice === `${type}:${voice.id}`
    const isTesting = testingVoice === voice.id
    return (
      <div key={voice.id} className={`vm-voice-card ${voice.active ? 'active' : ''}`}>
        <div className="vm-voice-info">
          <GripVertical size={14} className="vm-grip" />
          <FileAudio size={16} />
          <div>
            <div className="vm-voice-name">{voice.id}</div>
            <small className="vm-voice-meta">
              {isPiper ? (voice.has_config ? 'Config vorhanden' : 'Config fehlt') : (voice.has_voices ? 'Voices vorhanden' : 'Voices fehlen')}
            </small>
          </div>
          {voice.active && <span className="vm-active-badge">{$t('voiceManagement.active')}</span>}
        </div>
        <div className="vm-voice-actions">
          {!voice.active && canWrite && (
            <button className="btn-secondary btn-small" onClick={() => (isPiper ? handleActivatePiper : handleActivateKokoro)(voice.id)}>
              {$t('voiceManagement.activate')}
            </button>
          )}
          <button className={`btn-secondary btn-small ${isPlaying ? 'playing' : ''}`} onClick={() => (isPiper ? handleTestPiper : handleTestKokoro)(voice.id)} disabled={isTesting}>
            {isTesting ? $t('voiceManagement.testing') : <><Play size={12} /> {$t('voiceManagement.test')}</>}
          </button>
          {canWrite && (
            <button className="btn-icon danger" onClick={() => (isPiper ? handleDeletePiper : handleDeleteKokoro)(voice.id)} title={$t('voiceManagement.delete')}>
              <Trash2 size={14} />
            </button>
          )}
        </div>
      </div>
    )
  }

  if (loading) return <div className="loading">{$t('common.loading')}</div>

  return (
    <div className="vm-page">
      <div className="page-header">
        <h2>{$t('voiceManagement.title')}</h2>
      </div>

      {error && (
        <div className="error-box vm-error" style={{ marginBottom: 16 }}>
          <AlertCircle size={16} />
          <span>{error}</span>
          <button className="btn-text" onClick={() => setError(null)}><X size={14} /></button>
        </div>
      )}

      {/* Settings */}
      <div className="config-section">
        <button className="vm-section-toggle" onClick={() => setSettingsExpanded((v) => !v)}>
          <Settings size={16} />
          <span>{$t('voiceManagement.settings')}</span>
          {settingsExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
        {settingsExpanded && (
          <div className="vm-settings-body">
            <div className="vm-settings-grid">
              <div className="form-group">
                <label>{$t('voiceManagement.ttsProvider')}</label>
                <select value={settings.audio_tts_provider || ''} onChange={(e) => handleSettingChange('audio_tts_provider', e.target.value)} disabled={!canWriteConfig}>
                  <option value="piper">Piper</option>
                  <option value="kokoro">Kokoro</option>
                </select>
              </div>
              <div className="form-group">
                <label>{$t('voiceManagement.sttProvider')}</label>
                <select value={settings.audio_stt_provider || ''} onChange={(e) => handleSettingChange('audio_stt_provider', e.target.value)} disabled={!canWriteConfig}>
                  <option value="faster_whisper">faster-whisper</option>
                  <option value="distil_whisper">distil-whisper</option>
                </select>
              </div>
              <div className="form-group">
                <label>{$t('voiceManagement.sttModel')}</label>
                <select value={settings.audio_stt_model || ''} onChange={(e) => handleSettingChange('audio_stt_model', e.target.value)} disabled={!canWriteConfig}>
                  <option value="tiny">tiny</option>
                  <option value="base">base</option>
                  <option value="small">small</option>
                  <option value="medium">medium</option>
                  <option value="large">large</option>
                </select>
              </div>
              <div className="form-group">
                <label>{$t('voiceManagement.sttLanguage')}</label>
                <input type="text" value={settings.audio_stt_language || ''} onChange={(e) => handleSettingChange('audio_stt_language', e.target.value)} disabled={!canWriteConfig} />
              </div>
              <div className="form-group">
                <label>Min. SNR (dB) – Audio Emotion</label>
                <input
                  type="number"
                  min={0}
                  max={30}
                  step={1}
                  value={settings.audio_min_snr_db ?? 10}
                  onChange={(e) => {
                    const val = e.target.value === '' ? null : parseFloat(e.target.value)
                    handleSettingChange('audio_min_snr_db', val)
                  }}
                  disabled={!canWriteConfig}
                />
              </div>
              <div className="form-group">
                <label>Noise Gate Min</label>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.001}
                  value={settings.audio_noise_gate_min ?? 0.001}
                  onChange={(e) => {
                    const val = e.target.value === '' ? null : parseFloat(e.target.value)
                    handleSettingChange('audio_noise_gate_min', val)
                  }}
                  disabled={!canWriteConfig}
                />
              </div>
              <div className="form-group">
                <label>Noise Gate Max</label>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.001}
                  value={settings.audio_noise_gate_max ?? 0.5}
                  onChange={(e) => {
                    const val = e.target.value === '' ? null : parseFloat(e.target.value)
                    handleSettingChange('audio_noise_gate_max', val)
                  }}
                  disabled={!canWriteConfig}
                />
              </div>
              <div className="form-group">
                <label>{$t('voiceManagement.defaultVoice')}</label>
                <select value={settings.audio_tts_default_voice || ''} onChange={(e) => handleSettingChange('audio_tts_default_voice', e.target.value)} disabled={!canWriteConfig}>
                  <option value="">—</option>
                  {piperVoices.map((v) => (
                    <option key={v.id} value={v.id}>{v.id}</option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label>{$t('voiceManagement.kokoroVoice')}</label>
                <select value={settings.audio_tts_kokoro_voice || ''} onChange={(e) => handleSettingChange('audio_tts_kokoro_voice', e.target.value)} disabled={!canWriteConfig}>
                  <option value="">—</option>
                  {kokoroVoices.map((v) => (
                    <option key={v.id} value={v.id}>{v.id}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Status */}
      <div className="config-section">
        <h3 style={{ marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Speaker size={16} /> {$t('voiceManagement.status')}
        </h3>
        <div className="vm-status-grid">
          <div className="vm-status-item">
            {health?.tts_ready ? <CheckCircle size={16} className="vm-status-ok" /> : <AlertCircle size={16} className="vm-status-err" />}
            <span>TTS: {health?.tts_ready ? 'Bereit' : 'Nicht bereit'}</span>
          </div>
          <div className="vm-status-item">
            {health?.stt_ready ? <CheckCircle size={16} className="vm-status-ok" /> : <AlertCircle size={16} className="vm-status-err" />}
            <span>STT: {health?.stt_ready ? 'Bereit' : 'Nicht bereit'}</span>
          </div>
          <div className="vm-status-item">
            <Speaker size={16} />
            <span>Provider: {health?.tts_provider || '—'}</span>
          </div>
          <div className="vm-status-item">
            <Volume2 size={16} />
            <span>Piper: {health?.piper_found ? 'Gefunden' : 'Nicht gefunden'}</span>
          </div>
          <div className="vm-status-item">
            <Mic size={16} />
            <span>Kokoro: {health?.kokoro_ready ? 'Bereit' : 'Nicht bereit'}</span>
          </div>
        </div>
      </div>

      {/* Upload */}
      {canWrite && (
        <div className="config-section">
          <h3 style={{ marginBottom: 12 }}>{$t('voiceManagement.upload')}</h3>
          <div className="vm-upload-grid">
            {renderDropZone('piper')}
            {renderDropZone('kokoro')}
          </div>
          {uploading && <div className="loading" style={{ marginTop: 8 }}>{$t('common.uploading')}</div>}
        </div>
      )}

      {/* Voices */}
      <div className="config-section">
        <h3 style={{ marginBottom: 12 }}>{$t('voiceManagement.piperVoices')}</h3>
        {piperVoices.length === 0 ? (
          <div className="text-muted">{$t('voiceManagement.noVoices')}</div>
        ) : (
          <div className="vm-voice-list">{piperVoices.map((v) => renderVoiceCard(v, 'piper'))}</div>
        )}
      </div>

      <div className="config-section">
        <h3 style={{ marginBottom: 12 }}>{$t('voiceManagement.kokoroVoices')}</h3>
        {kokoroVoices.length === 0 ? (
          <div className="text-muted">{$t('voiceManagement.noVoices')}</div>
        ) : (
          <div className="vm-voice-list">{kokoroVoices.map((v) => renderVoiceCard(v, 'kokoro'))}</div>
        )}
      </div>

      {/* Hidden audio player */}
      <audio ref={audioRef} style={{ display: 'none' }} onEnded={() => setPlayingVoice(null)} onPause={() => setPlayingVoice(null)} />

      {/* Download modal */}
      {showDownloadModal && (
        <div className="modal-overlay" onClick={() => setShowDownloadModal(false)}>
          <div className="modal-box" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 480 }}>
            <h3 style={{ marginBottom: 16 }}>{$t('voiceManagement.downloadUrl')}</h3>
            <div className="form-group">
              <label>{$t('voiceManagement.voiceId')}</label>
              <input type="text" value={downloadForm.voice_id} onChange={(e) => setDownloadForm((p) => ({ ...p, voice_id: e.target.value.trim() }))} placeholder="z. B. de_DE-thorsten-medium" />
            </div>
            <div className="form-group">
              <label>{$t('voiceManagement.onnxUrl')}</label>
              <input type="text" value={downloadForm.onnx_url} onChange={(e) => setDownloadForm((p) => ({ ...p, onnx_url: e.target.value.trim() }))} placeholder="https://..." />
            </div>
            <div className="form-group">
              <label>{$t('voiceManagement.configUrl')}</label>
              <input type="text" value={downloadForm.config_url} onChange={(e) => setDownloadForm((p) => ({ ...p, config_url: e.target.value.trim() }))} placeholder="https://..." />
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 8 }}>
              <button className="btn-secondary" onClick={() => setShowDownloadModal(false)}>{$t('voiceManagement.cancel')}</button>
              <button className="btn-primary" onClick={handleDownloadPiper} disabled={downloading}>
                {downloading ? $t('common.uploading') : $t('voiceManagement.downloadUrl')}
              </button>
            </div>
          </div>
        </div>
      )}

      <UnsavedChangesBar
        dirty={settingsDirty && canWriteConfig}
        saving={savingSettings}
        onSave={handleSaveSettings}
        onDiscard={handleDiscardSettings}
      />
      <ConfirmModal
        open={voiceDeleteTarget !== null}
        title={$t('common.delete')}
        message={
          voiceDeleteTarget
            ? `${voiceDeleteTarget.kind === 'piper' ? 'Piper-Stimme' : 'Kokoro-Stimme'} "${voiceDeleteTarget.id}" wirklich löschen?`
            : ''
        }
        onConfirm={async () => {
          const target = voiceDeleteTarget
          setVoiceDeleteTarget(null)
          if (!target) return
          try {
            if (target.kind === 'piper') await audioApi.deletePiperVoice(target.id)
            else await audioApi.deleteKokoroVoice(target.id)
            if (playingVoice?.startsWith(`${target.kind}:`)) setPlayingVoice(null)
            await fetchAll()
          } catch (err) {
            setError(String(err))
          }
        }}
        onCancel={() => setVoiceDeleteTarget(null)}
      />
    </div>
  )
}
