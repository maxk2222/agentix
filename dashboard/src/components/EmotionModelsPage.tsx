import { useEffect, useRef, useState, useCallback } from 'react'
import {
  Upload, Trash2, CheckCircle, AlertCircle,
  Save, X, Download, BrainCircuit, ScanFace, Ear, MessageSquareText,
  SlidersHorizontal, ChevronUp, ChevronDown,
  Gauge, Shield, RotateCcw, Sparkles, TrendingUp,
} from 'lucide-react'
import { useAuth } from '../api/AuthContext'
import UnsavedChangesBar, { useUnsavedGuard } from './UnsavedChangesBar'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'
import { emotionApi, runtimeConfig, type EmotionModelEntry, type EmotionModelsResponse } from '../api/client'

type ModelType = 'face_detector' | 'expression' | 'audio_emotion' | 'text_emotion'

const MODEL_TYPE_META: Record<ModelType, { label: string; icon: typeof ScanFace; desc: string }> = {
  face_detector: { label: 'Face Detector', icon: ScanFace, desc: 'Detects faces in webcam frames (YOLO/RetinaFace ONNX)' },
  expression: { label: 'Expression Classifier', icon: BrainCircuit, desc: 'Classifies facial expressions into emotions (FER ONNX)' },
  audio_emotion: { label: 'Audio Emotion', icon: Ear, desc: 'Classifies emotion from audio prosody (Speech Emotion ONNX)' },
  text_emotion: { label: 'Text Emotion', icon: MessageSquareText, desc: 'German GoEmotions ONNX text classifier' },
}

type FusionConfig = {
  source_weights: Record<string, number>
  min_confidence: Record<string, number>
}

const DEFAULT_FUSION: FusionConfig = {
  source_weights: {
    text: 1.0,
    visual_pre: 1.0,
    audio_prosody: 0.9,
    phase_typing: 0.8,
    phase_waiting_for_response: 0.6,
    phase_reading: 0.4,
  },
  min_confidence: {
    text: 0.55,
    visual_pre: 0.55,
    audio_prosody: 0.55,
    phase: 0.55,
  },
}

export default function EmotionModelsPage() {
  const { hasPermission } = useAuth()
  const { language } = useAppPreferences()
  const $t = useT(language)
  const canWrite = hasPermission('instructions:write')

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<EmotionModelsResponse | null>(null)
  const [uploading, setUploading] = useState<ModelType | null>(null)
  const [downloadForm, setDownloadForm] = useState<Record<ModelType, { url: string; open: boolean }>>({
    face_detector: { url: '', open: false },
    expression: { url: '', open: false },
    audio_emotion: { url: '', open: false },
    text_emotion: { url: '', open: false },
  })
  const [fusionConfig, setFusionConfig] = useState<FusionConfig>(DEFAULT_FUSION)
  const [fusionOpen, setFusionOpen] = useState(false)
  const [fusionSaving, setFusionSaving] = useState(false)
  const fusionBaselineRef = useRef<string>(JSON.stringify(DEFAULT_FUSION))

  const inputRefs: Record<ModelType, React.RefObject<HTMLInputElement>> = {
    face_detector: useRef<HTMLInputElement>(null),
    expression: useRef<HTMLInputElement>(null),
    audio_emotion: useRef<HTMLInputElement>(null),
    text_emotion: useRef<HTMLInputElement>(null),
  }

  const fetchAll = useCallback(async () => {
    setError(null)
    try {
      const [modelsRes, runtimeRes] = await Promise.all([
        emotionApi.listModels(),
        runtimeConfig.get().catch(() => null),
      ])
      setData(modelsRes)
      if (runtimeRes?.effective) {
        const weights = runtimeRes.effective['emotion_fusion_source_weights']?.value
        const minConf = runtimeRes.effective['emotion_fusion_min_confidence']?.value
        if (weights || minConf) {
          const next = {
            source_weights: typeof weights === 'object' && weights ? { ...DEFAULT_FUSION.source_weights, ...weights } : DEFAULT_FUSION.source_weights,
            min_confidence: typeof minConf === 'object' && minConf ? { ...DEFAULT_FUSION.min_confidence, ...minConf } : DEFAULT_FUSION.min_confidence,
          }
          setFusionConfig(next)
          fusionBaselineRef.current = JSON.stringify(next)
        }
      }
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchAll()
  }, [fetchAll])

  const handleUpload = async (modelType: ModelType, file: File) => {
    if (!canWrite) return
    setUploading(modelType)
    setError(null)
    try {
      await emotionApi.uploadModel(modelType, file)
      await fetchAll()
    } catch (err) {
      setError(String(err))
    } finally {
      setUploading(null)
    }
  }

  const handleActivate = async (modelType: ModelType, filename: string) => {
    if (!canWrite) return
    setError(null)
    try {
      await emotionApi.activateModel(modelType, filename)
      await fetchAll()
    } catch (err) {
      setError(String(err))
    }
  }

  const handleDelete = async (modelType: ModelType, filename: string) => {
    if (!canWrite) return
    setError(null)
    try {
      await emotionApi.deleteModel(modelType, filename)
      await fetchAll()
    } catch (err) {
      setError(String(err))
    }
  }

  const handleDownload = async (modelType: ModelType) => {
    if (!canWrite) return
    const url = downloadForm[modelType].url.trim()
    if (!url) return
    setError(null)
    try {
      await emotionApi.downloadModel(modelType, url)
      setDownloadForm((prev) => ({ ...prev, [modelType]: { url: '', open: false } }))
      await fetchAll()
    } catch (err) {
      setError(String(err))
    }
  }

  const formatBytes = (b: number) => {
    if (b < 1024) return `${b} B`
    if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
    return `${(b / (1024 * 1024)).toFixed(1)} MB`
  }

  const fusionDirty = JSON.stringify(fusionConfig) !== fusionBaselineRef.current

  useUnsavedGuard(fusionDirty && canWrite, $t('common.leaveWarning'))

  const handleSaveFusion = async () => {
    if (!canWrite) return
    setFusionSaving(true)
    setError(null)
    try {
      await runtimeConfig.save({
        emotion_fusion_source_weights: fusionConfig.source_weights,
        emotion_fusion_min_confidence: fusionConfig.min_confidence,
      })
      await runtimeConfig.apply()
      fusionBaselineRef.current = JSON.stringify(fusionConfig)
    } catch (err) {
      setError(String(err))
    } finally {
      setFusionSaving(false)
    }
  }

  const handleDiscardFusion = () => {
    setFusionConfig(JSON.parse(fusionBaselineRef.current))
  }

  const updateWeight = (key: string, value: string) => {
    const num = parseFloat(value)
    if (Number.isNaN(num)) return
    setFusionConfig((prev) => ({
      ...prev,
      source_weights: { ...prev.source_weights, [key]: num },
    }))
  }

  const updateMinConf = (key: string, value: string) => {
    const num = parseFloat(value)
    if (Number.isNaN(num)) return
    setFusionConfig((prev) => ({
      ...prev,
      min_confidence: { ...prev.min_confidence, [key]: num },
    }))
  }

  const handleResetFusion = () => {
    setFusionConfig(DEFAULT_FUSION)
  }

  const computePreview = () => {
    const w = fusionConfig.source_weights
    const scores: Record<string, number> = {}
    const items = [
      { emo: 'angry', conf: 0.6, source: 'text' },
      { emo: 'sad', conf: 0.7, source: 'visual_pre' },
    ]
    items.forEach((it) => {
      const weight = w[it.source] ?? 1.0
      scores[it.emo] = (scores[it.emo] ?? 0) + it.conf * weight
    })
    const entries = Object.entries(scores).sort((a, b) => b[1] - a[1])
    const winner = entries[0]
    return { entries, winner }
  }

  if (loading) {
    return (
      <div className="em-page">
        <div className="em-loading">Loading...</div>
      </div>
    )
  }

  return (
    <div className="em-page">
      <div className="em-header">
        <h2>Emotion Models</h2>
        <p className="em-subtitle">Upload and manage ONNX emotion detection models</p>
      </div>

      {error && (
        <div className="em-alert em-alert-error">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {/* Fusion Config Panel */}
      <div className="em-card em-fusion-card">
        <div className="em-card-header em-fusion-header" onClick={() => setFusionOpen(!fusionOpen)} style={{ cursor: 'pointer' }}>
          <div className="em-fusion-header-icon">
            <SlidersHorizontal size={20} />
          </div>
          <div>
            <h3>Emotion Fusion Settings</h3>
            <p className="em-card-desc">Configure source weights and confidence thresholds for multimodal emotion fusion</p>
          </div>
          <span className="nav-chevron">{fusionOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</span>
        </div>
        {fusionOpen && (
          <div className="em-fusion-body">
            <div className="em-fusion-columns">
              {/* Source Weights */}
              <div className="em-fusion-section">
                <div className="em-fusion-section-header">
                  <Gauge size={18} />
                  <div>
                    <h4>Source Weights</h4>
                    <span className="em-fusion-badge">{Object.keys(fusionConfig.source_weights).length} sources</span>
                  </div>
                </div>
                <div className="em-fusion-sliders">
                  {Object.entries(fusionConfig.source_weights).map(([key, value]) => {
                    const pct = Math.min((value / 2) * 100, 100)
                    return (
                      <div key={key} className="em-fusion-slider-row">
                        <div className="em-fusion-slider-label">
                          <span>{key.replace(/_/g, ' ')}</span>
                          <span className="em-fusion-value">{value.toFixed(1)}</span>
                        </div>
                        <div className="em-fusion-slider-track">
                          <div className="em-fusion-bar" style={{ width: `${pct}%` }} />
                          {canWrite && (
                            <input
                              type="range"
                              min="0"
                              max="2"
                              step="0.1"
                              value={value}
                              onChange={(e) => updateWeight(key, e.target.value)}
                              className="em-fusion-range"
                            />
                          )}
                          {!canWrite && (
                            <div className="em-fusion-range-readonly" style={{ width: `${pct}%` }} />
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Min Confidence */}
              <div className="em-fusion-section">
                <div className="em-fusion-section-header">
                  <Shield size={18} />
                  <div>
                    <h4>Min Confidence</h4>
                    <span className="em-fusion-badge">{Object.keys(fusionConfig.min_confidence).length} sources</span>
                  </div>
                </div>
                <div className="em-fusion-sliders">
                  {Object.entries(fusionConfig.min_confidence).map(([key, value]) => {
                    const pct = value * 100
                    return (
                      <div key={key} className="em-fusion-slider-row">
                        <div className="em-fusion-slider-label">
                          <span>{key.replace(/_/g, ' ')}</span>
                          <span className="em-fusion-value">{value.toFixed(2)}</span>
                        </div>
                        <div className="em-fusion-slider-track">
                          <div className="em-fusion-bar em-fusion-bar-green" style={{ width: `${pct}%` }} />
                          {canWrite && (
                            <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.05"
                              value={value}
                              onChange={(e) => updateMinConf(key, e.target.value)}
                              className="em-fusion-range"
                            />
                          )}
                          {!canWrite && (
                            <div className="em-fusion-range-readonly" style={{ width: `${pct}%` }} />
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>

            {/* Live Preview */}
            <div className="em-fusion-preview">
              <div className="em-fusion-preview-header">
                <Sparkles size={16} />
                <span>Live Preview</span>
              </div>
              <div className="em-fusion-preview-body">
                {(() => {
                  const { entries, winner } = computePreview()
                  return (
                    <>
                      <div className="em-fusion-preview-scenario">
                        <div><strong>Text</strong> angry @ 0.6 × {fusionConfig.source_weights.text.toFixed(1)} = <TrendingUp size={12} /> {(0.6 * fusionConfig.source_weights.text).toFixed(2)}</div>
                        <div><strong>Visual</strong> sad @ 0.7 × {fusionConfig.source_weights.visual_pre.toFixed(1)} = <TrendingUp size={12} /> {(0.7 * fusionConfig.source_weights.visual_pre).toFixed(2)}</div>
                      </div>
                      <div className="em-fusion-preview-winner">
                        Winner: <span className="em-fusion-winner-emotion">{winner[0]}</span> <span className="em-fusion-winner-score">(score {winner[1].toFixed(2)})</span>
                      </div>
                      <div className="em-fusion-preview-bar">
                        {entries.map(([emo, score]) => (
                          <div key={emo} className="em-fusion-preview-segment" style={{ width: `${Math.max((score / (entries[0][1] || 1)) * 100, 10)}%` }}>
                            <div className="em-fusion-preview-label">{emo}</div>
                            <div className="em-fusion-preview-score">{score.toFixed(2)}</div>
                          </div>
                        ))}
                      </div>
                    </>
                  )
                })()}
              </div>
            </div>

            {/* Actions */}
            {canWrite && (
              <div className="em-fusion-actions">
                <button className="em-btn em-btn-ghost" onClick={handleResetFusion} disabled={fusionSaving}>
                  <RotateCcw size={14} />
                  Reset to Defaults
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {(Object.keys(MODEL_TYPE_META) as ModelType[]).map((mtype) => {
        const meta = MODEL_TYPE_META[mtype]
        const Icon = meta.icon
        const section = data?.[mtype]
        const models = section?.models ?? []
        const active = section?.active
        const defaultFilename = section?.default_filename ?? ''
        const isUploading = uploading === mtype
        const df = downloadForm[mtype]

        return (
          <div key={mtype} className="em-card">
            <div className="em-card-header">
              <Icon size={20} />
              <div>
                <h3>{meta.label}</h3>
                <p className="em-card-desc">{meta.desc}</p>
              </div>
            </div>

            {/* Active model info */}
            <div className="em-active-row">
              <span className="em-active-label">Active:</span>
              <span className="em-active-value">
                {active ? (
                  <>
                    <CheckCircle size={14} className="em-active-icon" />
                    {active}
                  </>
                ) : (
                  <span className="em-inactive">None (default: {defaultFilename})</span>
                )}
              </span>
            </div>

            {/* Model list */}
            {models.length > 0 && (
              <div className="em-list">
                {models.map((m: EmotionModelEntry) => {
                  const isActive = active === m.filename
                  return (
                    <div key={m.filename} className={`em-list-item${isActive ? ' em-list-item-active' : ''}`}>
                      <div className="em-list-item-info">
                        <span className="em-list-item-name">{m.filename}</span>
                        <span className="em-list-item-meta">{formatBytes(m.size)}</span>
                      </div>
                      <div className="em-list-item-actions">
                        {!isActive && canWrite && (
                          <button
                            className="em-btn em-btn-sm em-btn-primary"
                            onClick={() => handleActivate(mtype, m.filename)}
                            title="Activate"
                          >
                            <CheckCircle size={14} />
                            Activate
                          </button>
                        )}
                        {canWrite && (
                          <button
                            className="em-btn em-btn-sm em-btn-danger"
                            onClick={() => handleDelete(mtype, m.filename)}
                            title="Delete"
                          >
                            <Trash2 size={14} />
                          </button>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}

            {/* Upload + Download */}
            {canWrite && (
              <div className="em-actions-row">
                <button
                  className="em-btn em-btn-secondary"
                  onClick={() => inputRefs[mtype].current?.click()}
                  disabled={isUploading}
                >
                  <Upload size={14} />
                  {isUploading ? 'Uploading...' : 'Upload ONNX'}
                </button>
                <input
                  ref={inputRefs[mtype]}
                  type="file"
                  accept=".onnx"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) handleUpload(mtype, file)
                    e.target.value = ''
                  }}
                />
                <button
                  className="em-btn em-btn-ghost"
                  onClick={() =>
                    setDownloadForm((prev) => ({
                      ...prev,
                      [mtype]: { ...prev[mtype], open: !prev[mtype].open },
                    }))
                  }
                >
                  <Download size={14} />
                  Download from URL
                </button>
              </div>
            )}

            {/* Download form */}
            {df.open && canWrite && (
              <div className="em-download-form">
                <input
                  type="text"
                  placeholder="https://example.com/model.onnx"
                  value={df.url}
                  onChange={(e) =>
                    setDownloadForm((prev) => ({
                      ...prev,
                      [mtype]: { ...prev[mtype], url: e.target.value },
                    }))
                  }
                  className="em-input"
                />
                <button className="em-btn em-btn-primary" onClick={() => handleDownload(mtype)}>
                  <Save size={14} />
                  Download
                </button>
                <button
                  className="em-btn em-btn-ghost"
                  onClick={() =>
                    setDownloadForm((prev) => ({
                      ...prev,
                      [mtype]: { url: '', open: false },
                    }))
                  }
                >
                  <X size={14} />
                </button>
              </div>
            )}
          </div>
        )
      })}

      <UnsavedChangesBar
        dirty={fusionDirty && canWrite}
        saving={fusionSaving}
        onSave={handleSaveFusion}
        onDiscard={handleDiscardFusion}
      />
    </div>
  )
}
