import { useEffect, useState, useRef } from 'react'
import { useAuth } from '../api/AuthContext'
import UnsavedChangesBar, { useUnsavedGuard } from './UnsavedChangesBar'
import ConfirmModal from './ConfirmModal'
import { useData } from '../api/DataContext'
import { useAppPreferences } from '../ui/AppPreferences'
import { useT } from '../i18n'
import {
  runtimeConfig,
  providers,
  apiKeys,
  config,
  configDescriptions,
  invalidateCache,
  type ProviderEntry,
  type ProviderInput,
  type RuntimeConfigResponse,
  type ApiKeyEntry,
  type ConfigDescription,
} from '../api/client'
import {
  Save,
  Download,
  Upload,
  AlertCircle,
  CheckCircle,
  Plus,
  Trash2,
  Play,
  Eye,
  EyeOff,
  ChevronDown,
  Settings,
  Server,
  KeyRound,
  Pencil,
  Globe,
  Clock,
  Zap,
  LayoutGrid,
  ShieldAlert,
  Check,
  Copy,
  X,
  Info,
} from 'lucide-react'

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

type ProviderForm = ProviderInput

/* ------------------------------------------------------------------ */
/* Field definitions for runtime config                                */
/* ------------------------------------------------------------------ */

const CONFIG_SECTIONS: { title: string; keys: string[] }[] = [
  {
    title: 'Vector & Embeddings',
    keys: [
      'vector_search_k',
      'vector_batch_size',
      'vector_force_rebuild',
      'embedding_model',
      'embeddings_api_url',
      'embeddings_request_timeout',
      'embeddings_max_retries',
      'embeddings_backoff_factor',
    ],
  },
  {
    title: 'History',
    keys: [
      'max_history_messages',
      'rolling_history_enabled',
      'rolling_history_context_window',
      'rolling_history_max_output_tokens',
      'rolling_history_rag_reserved_tokens',
      'rolling_history_min_recent_turns',
      'rolling_history_safety_margin_ratio',
      'rolling_history_summary_timeout_seconds',
    ],
  },
  {
    title: 'Retrieval',
    keys: [
      'retrieval_hybrid_enabled',
      'hybrid_vector_candidates',
      'hybrid_keyword_candidates',
      'hybrid_rrf_k',
      'hybrid_max_chunks_per_source',
      'hybrid_query_expansion_enabled',
      'hybrid_max_expanded_queries',
      'hybrid_rerank_enabled',
      'hybrid_min_semantic_similarity',
      'retrieval_history_enabled',
      'retrieval_history_user_messages',
      'retrieval_history_max_chars',
      'retrieval_history_standalone_detection',
      'contextualize_question_enabled',
      'contextualize_question_model',
      'contextualize_question_temperature',
      'contextualize_question_disable_reasoning',
    ],
  },
  {
    title: 'Indexing & Processing',
    keys: [
      'system_slow',
      'loader_max_concurrency',
      'chunk_size',
      'chunk_overlap',
      'indexing_incremental_enabled',
    ],
  },
  {
    title: 'Request Validation',
    keys: [
      'max_request_bytes',
      'max_messages',
      'max_message_chars',
      'max_total_chars',
      'require_last_user',
      'allow_non_string_content',
      'validation_control_chars_mode',
      'validation_enforce',
    ],
  },
  {
    title: 'Runtime & API',
    keys: [
      'api_reload',
      'auto_initialize',
      'user_agent',
      'stream_idle_timeout_seconds',
      'retriever_timeout_seconds',
    ],
  },
  {
    title: 'Logging',
    keys: [
      'log_level',
      'log_file_max_bytes',
    ],
  },
  {
    title: 'OCR',
    keys: [
      'tesseract_cmd',
      'ocr_language',
      'ocr_mode',
      'ocr_llm_prompt',
      'ocr_llm_concurrency',
      'pdf_image_extraction_enabled',
      'pdf_image_extraction_mode',
      'pptx_slide_analysis_enabled',
      'pptx_slide_analysis_mode',
    ],
  },
  {
    title: 'Chat Images',
    keys: [
      'chat_image_upload_enabled',
      'chat_image_max_size_bytes',
      'chat_image_max_count',
      'chat_image_allowed_types',
    ],
  },
  {
    title: 'Chat Files',
    keys: [
      'chat_file_upload_enabled',
      'chat_file_max_size_bytes',
      'chat_file_max_count',
      'chat_file_allowed_exts',
      'chat_file_max_extract_chars',
      'chat_file_max_total_chars',
      'chat_file_extract_timeout_seconds',
      'chat_files_orphan_ttl_days',
      'chat_files_retention_days',
    ],
  },
  {
    title: 'Audio',
    keys: [
      'audio_stt_enabled',
      'audio_tts_enabled',
      'audio_tts_streaming_enabled',
      'audio_tts_voices_dir',
      'audio_tts_kokoro_model_dir',
      'audio_tts_piper_binary',
      'hf_cache_dir',
      'hf_token',
    ],
  },
  {
    title: 'Emotion Detection',
    keys: [
      'emotion_detection_enabled',
      'emotion_text_model_enabled',
      'emotion_text_model_url',
      'emotion_text_model_download_url',
      'emotion_text_tokenizer_url',
      'emotion_text_tokenizer_download_url',
      'emotion_webcam_enabled',
      'emotion_prompt_steering_enabled',
      'emotion_analytics_enabled',
      'emotion_min_confidence',
      'emotion_webcam_runtime',
      'emotion_webcam_execution_providers',
      'emotion_webcam_face_detector',
      'emotion_webcam_snippet_window_ms',
      'emotion_webcam_sample_fps',
      'emotion_webcam_max_frames_per_snippet',
      'emotion_webcam_min_valid_face_frames',
      'emotion_webcam_frame_upload_enabled',
      'emotion_audio_enabled',
    ],
  },
]

function labelForKey(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') return String(value)
  if (typeof value === 'string') return value
  return JSON.stringify(value)
}

function isBooleanValue(value: unknown): boolean {
  return typeof value === 'boolean'
}

function parseValue(raw: string, original: unknown): unknown {
  const trimmed = raw.trim()
  if (trimmed === '') return ''
  if (typeof original === 'boolean') {
    return trimmed.toLowerCase() === 'true'
  }
  if (typeof original === 'number') {
    if (trimmed.includes('.')) return parseFloat(trimmed)
    return parseInt(trimmed, 10)
  }
  return trimmed
}

/* ------------------------------------------------------------------ */
/* Main Component                                                      */
/* ------------------------------------------------------------------ */

export default function RuntimeConfig() {
  const { language } = useAppPreferences()
  const $t = useT(language)
  const { hasPermission } = useAuth()
  const canWriteConfig = hasPermission('runtime_config:write')
  const canApplyConfig = hasPermission('runtime_config:apply')
  const canWriteProviders = hasPermission('providers:write')
  const canWriteApiKeys = hasPermission('api_keys:write')

  const [configData, setConfigData] = useState<RuntimeConfigResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [applyStatus, setApplyStatus] = useState<'idle' | 'loading' | 'done' | 'error'>('idle')

  const [providerList, setProviderList] = useState<ProviderEntry[]>([])
  const [providerLoading, setProviderLoading] = useState(true)
  const [providerError, setProviderError] = useState<string | null>(null)

  const [apiKeyList, setApiKeyList] = useState<ApiKeyEntry[]>([])
  const [apiKeyLoading, setApiKeyLoading] = useState(true)
  const [apiKeyError, setApiKeyError] = useState<string | null>(null)
  const [editingApiKey, setEditingApiKey] = useState<ApiKeyEntry | null>(null)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [newKeyValue, setNewKeyValue] = useState<string | null>(null)

  const [activeTab, setActiveTab] = useState<'config' | 'providers' | 'apikeys'>('config')
  const [pendingConfirm, setPendingConfirm] = useState<{ message: string; action: () => void } | null>(null)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set())
  const [editingProvider, setEditingProvider] = useState<ProviderForm | null>(null)
  const [showSecretInput, setShowSecretInput] = useState<string | null>(null)
  const [secretValue, setSecretValue] = useState('')
  const [providerModels, setProviderModels] = useState<Record<string, { name: string; models: string[] }>>({})
  const [providerModelsLoading, setProviderModelsLoading] = useState(true)

  const [uploadError, setUploadError] = useState<string | null>(null)
  const uploadInputRef = useRef<HTMLInputElement>(null)

  const initialDataRef = useRef<string>('')
  const baselineConfigRef = useRef<RuntimeConfigResponse | null>(null)

  const [descriptions, setDescriptions] = useState<Record<string, ConfigDescription>>({})
  const [expandedHelp, setExpandedHelp] = useState<Set<string>>(new Set())

  const { getProviders } = useData()

  // Load config, providers, api keys, provider-models, and descriptions
  useEffect(() => {
    Promise.all([
      runtimeConfig.get().catch((err) => {
        setError(String(err))
        return null
      }),
      getProviders().catch((err) => {
        setProviderError(String(err))
        return []
      }),
      apiKeys.list().catch((err) => {
        setApiKeyError(String(err))
        return []
      }),
      apiKeys.models().catch((err) => {
        setApiKeyError(String(err))
        return []
      }),
      configDescriptions.get().catch(() => ({} as Record<string, ConfigDescription>)),
      config.listModelsByProvider().catch(() => ({})),
    ])
      .then(([cfg, provs, keys, models, descs, provModels]) => {
        if (cfg) {
          setConfigData(cfg)
          initialDataRef.current = JSON.stringify(cfg.stored)
          baselineConfigRef.current = cfg
        }
        setProviderList(provs)
        setApiKeyList(keys)
        setAvailableModels(models)
        if (descs) setDescriptions(descs)
        if (provModels) setProviderModels(provModels)
        setLoading(false)
        setProviderLoading(false)
        setApiKeyLoading(false)
        setProviderModelsLoading(false)
      })
      .catch((err) => {
        setError(String(err))
        setLoading(false)
        setProviderLoading(false)
        setApiKeyLoading(false)
        setProviderModelsLoading(false)
      })
  }, [])

  const isDirty = Boolean(
    configData && JSON.stringify(configData.stored) !== initialDataRef.current
  )

  useUnsavedGuard(isDirty, $t('common.leaveWarning'))

  const handleSave = async () => {
    if (!configData) return
    setSaveStatus('saving')
    try {
      await runtimeConfig.save(configData.stored)
      setSaveStatus('saved')
      initialDataRef.current = JSON.stringify(configData.stored)
      baselineConfigRef.current = configData
      invalidateCache('/admin/runtime-config')
      setTimeout(() => setSaveStatus('idle'), 2000)
    } catch (err) {
      setSaveStatus('error')
      setError(String(err))
    }
  }

  const handleDiscard = () => {
    if (baselineConfigRef.current) {
      setConfigData(baselineConfigRef.current)
    }
  }

  const toggleSection = (title: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev)
      if (next.has(title)) next.delete(title)
      else next.add(title)
      return next
    })
  }

  const handleValueChange = (key: string, raw: string | boolean) => {
    if (!configData) return
    const entry = configData.effective[key]
    if (!entry?.editable) return
    let parsed: unknown
    if (typeof raw === 'boolean') {
      parsed = raw
    } else {
      parsed = parseValue(raw, entry.value)
    }
    setConfigData((prev) => {
      if (!prev) return prev
      return {
        ...prev,
        stored: { ...prev.stored, [key]: parsed },
        effective: {
          ...prev.effective,
          [key]: { ...entry, value: parsed, source: 'db' },
        },
      }
    })
  }

  const handleApply = async () => {
    setApplyStatus('loading')
    try {
      await runtimeConfig.apply()
      invalidateCache('/admin/runtime-config')
      invalidateCache('/admin/config')
      setApplyStatus('done')
      setTimeout(() => setApplyStatus('idle'), 2000)
    } catch (err) {
      setApplyStatus('error')
      setError(String(err))
    }
  }

  const handleExport = async () => {
    try {
      const blob = await runtimeConfig.export()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'runtime_config.json'
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(String(err))
    }
  }

  const handleImport = async (file: File) => {
    setUploadError(null)
    try {
      const text = await file.text()
      const data = JSON.parse(text)
      const result = await runtimeConfig.import(data)
      if (result.imported_keys.length > 0) {
        const refreshed = await runtimeConfig.get()
        setConfigData(refreshed)
        initialDataRef.current = JSON.stringify(refreshed.stored)
        baselineConfigRef.current = refreshed
      }
    } catch (err) {
      setUploadError(String(err))
    }
  }

  const { refreshProviders: refreshProvidersCtx } = useData()

  const refreshProviders = async () => {
    setProviderLoading(true)
    try {
      const list = await refreshProvidersCtx()
      setProviderList(list)
      setProviderError(null)
    } catch (err) {
      setProviderError(String(err))
    } finally {
      setProviderLoading(false)
    }
  }

  const handleCreateProvider = () => {
    setProviderError(null)
    setSecretValue('')
    setShowSecretInput(null)
    setEditingProvider({
      id: '',
      name: '',
      base_url: 'http://127.0.0.1:5001',
      enabled: true,
      is_default: false,
      provider_type: 'openai-compatible',
      timeout_seconds: 30,
      max_retries: 3,
      extra_headers: null,
      supports_native_tool_calls: null,
    })
  }

  const handleEditProvider = (p: ProviderEntry) => {
    if (p.source === 'env') return
    setProviderError(null)
    setSecretValue('')
    setShowSecretInput(null)
    setEditingProvider({
      id: p.id,
      name: p.name,
      base_url: p.base_url,
      enabled: p.enabled,
      is_default: p.is_default,
      provider_type: p.provider_type,
      timeout_seconds: p.timeout_seconds,
      max_retries: p.max_retries,
      extra_headers: p.extra_headers,
      supports_native_tool_calls: p.supports_native_tool_calls,
      _originalId: p.id,
    } as any)
  }

  const handleSaveProvider = async () => {
    if (!editingProvider) return
    if (!editingProvider.id.trim()) {
      setProviderError($t('runtime.providerIdEmpty'))
      return
    }
    if (!editingProvider.name.trim()) {
      setProviderError($t('runtime.providerNameEmpty'))
      return
    }
    if (!editingProvider.base_url.trim()) {
      setProviderError($t('runtime.baseUrlEmpty'))
      return
    }
    try {
      const existing = providerList.find((p) => p.id === editingProvider.id)
      // Track if we renamed from another ID
      const originalId = (editingProvider as any)._originalId as string | undefined
      let targetId: string
      if (existing && existing.source !== 'env') {
        if (originalId && originalId !== editingProvider.id) {
          // ID changed: delete old, create new
          await providers.delete(originalId)
          await providers.create(editingProvider)
          targetId = editingProvider.id
        } else {
          await providers.update(editingProvider.id, editingProvider)
          targetId = editingProvider.id
        }
      } else {
        await providers.create(editingProvider)
        targetId = editingProvider.id
      }
      invalidateCache('/admin/providers')
      if (secretValue.trim()) {
        await providers.rotateSecret(targetId, secretValue)
        setSecretValue('')
      }
      setEditingProvider(null)
      await refreshProviders()
    } catch (err) {
      setProviderError(String(err))
    }
  }

  const handleDeleteProvider = async (id: string) => {
    if (id === 'env-default') return
    if (!id.trim()) {
      setProviderError($t('runtime.invalidProviderId'))
      return
    }
    setPendingConfirm({ message: $t('runtime.confirmDeleteProvider'), action: () => void doDeleteProvider(id) })
  }

  const doDeleteProvider = async (id: string) => {
    try {
      await providers.delete(id)
      invalidateCache('/admin/providers')
      await refreshProviders()
    } catch (err) {
      setProviderError(String(err))
    }
  }

  const handleCheckProvider = async (id: string) => {
    if (id === 'env-default') return
    try {
      const result = await providers.check(id)
      setProviderList((prev) =>
        prev.map((p) =>
          p.id === id
            ? { ...p, last_checked_at: new Date().toISOString(), last_check_status: result.status }
            : p
        )
      )
    } catch (err) {
      setProviderError(String(err))
    }
  }

  const handleExportProviders = async () => {
    try {
      const blob = await providers.export()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'providers.json'
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setProviderError(String(err))
    }
  }

  const refreshApiKeys = async () => {
    setApiKeyLoading(true)
    try {
      const list = await apiKeys.list()
      setApiKeyList(list)
      setApiKeyError(null)
    } catch (err) {
      setApiKeyError(String(err))
    } finally {
      setApiKeyLoading(false)
    }
  }

  const handleCreateApiKey = async (name: string, allowedModels: string[], enabled: boolean, rpm: number, rpd: number) => {
    try {
      const result = await apiKeys.create({ name, allowed_models: allowedModels, enabled, rate_limit_rpm: rpm, rate_limit_rpd: rpd })
      invalidateCache('/admin/api-keys')
      setNewKeyValue(result.api_key)
      await refreshApiKeys()
    } catch (err) {
      setApiKeyError(String(err))
    }
  }

  const handleUpdateApiKey = async (id: string, name: string, allowedModels: string[], enabled: boolean, rpm: number, rpd: number) => {
    try {
      await apiKeys.update(id, { name, allowed_models: allowedModels, enabled, rate_limit_rpm: rpm, rate_limit_rpd: rpd })
      invalidateCache('/admin/api-keys')
      setEditingApiKey(null)
      await refreshApiKeys()
    } catch (err) {
      setApiKeyError(String(err))
    }
  }

  const handleDeleteApiKey = async (id: string) => {
    setPendingConfirm({ message: $t('common.confirmDelete'), action: () => void doDeleteApiKey(id) })
  }

  const doDeleteApiKey = async (id: string) => {
    try {
      await apiKeys.delete(id)
      invalidateCache('/admin/api-keys')
      await refreshApiKeys()
    } catch (err) {
      setApiKeyError(String(err))
    }
  }

  if (loading) {
    return (
      <div className="loading">
        <div>{$t('common.loading')}</div>
      </div>
    )
  }

  return (
    <div className="rc-page">
      <h2>{$t('nav.configRuntime')}</h2>

      {error && (
        <div className="error-box">
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* --- Tabs --- */}
      <div className="rc-tabs">
        <button
          className={`rc-tab ${activeTab === 'config' ? 'active' : ''}`}
          onClick={() => setActiveTab('config')}
        >
          <Settings size={16} /> {$t('runtime.tabConfig')}
        </button>
        <button
          className={`rc-tab ${activeTab === 'providers' ? 'active' : ''}`}
          onClick={() => setActiveTab('providers')}
        >
          <Server size={16} /> {$t('runtime.tabProviders')}
        </button>
        <button
          className={`rc-tab ${activeTab === 'apikeys' ? 'active' : ''}`}
          onClick={() => setActiveTab('apikeys')}
        >
          <KeyRound size={16} /> {$t('runtime.tabApiKeys')}
        </button>
      </div>

      {activeTab === 'config' && (
        <>
          {/* --- Action bar --- */}
          <div className="rc-action-bar">
            {canApplyConfig && (
              <button className="rc-btn primary" onClick={handleApply} disabled={applyStatus === 'loading'}>
                <Zap size={14} />
                {applyStatus === 'loading' ? $t('runtime.applying') : applyStatus === 'done' ? $t('runtime.applied') : $t('common.apply')}
              </button>
            )}
            <button className="rc-btn secondary" onClick={handleExport}>
              <Download size={14} /> Export
            </button>
            {canWriteConfig && (
              <button className="rc-btn secondary" onClick={() => uploadInputRef.current?.click()}>
                <Upload size={14} /> Import
              </button>
            )}
            <input
              ref={uploadInputRef}
              type="file"
              accept=".json"
              style={{ display: 'none' }}
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) handleImport(file)
                e.target.value = ''
              }}
            />
            {saveStatus === 'saving' && (
              <span className="rc-status saving">
                <Save size={14} /> {$t('runtime.saving')}
              </span>
            )}
            {saveStatus === 'saved' && (
              <span className="rc-status saved">
                <CheckCircle size={14} /> {$t('runtime.saved')}
              </span>
            )}
            {saveStatus === 'error' && (
              <span className="rc-status error">
                <AlertCircle size={14} /> {$t('runtime.error')}
              </span>
            )}
          </div>

          {uploadError && <div className="error-box">{uploadError}</div>}

          {/* --- Runtime Config Sections --- */}
          {CONFIG_SECTIONS.map((section) => (
            <div key={section.title} className="rc-card">
              <div className="rc-section-header" onClick={() => toggleSection(section.title)}>
                <h3>
                  <LayoutGrid size={16} />
                  {section.title}
                  <span className="rc-section-count">{section.keys.length}</span>
                </h3>
                <span className={`rc-section-chevron ${expandedSections.has(section.title) ? 'rotated' : ''}`}>
                  <ChevronDown size={18} />
                </span>
              </div>
              {expandedSections.has(section.title) && (
                <div className="rc-field-grid">
                  {section.keys.map((key) => {
                    const entry = configData?.effective[key]
                    if (!entry) return null
                    return (
                      <div key={key} className={`rc-field ${!entry.editable ? 'locked' : ''}`}>
                        <div className="rc-field-label">
                          {descriptions[key]?.label || labelForKey(key)}
                          {entry.source === 'env' && <span className="env-badge">env</span>}
                          {entry.source === 'default' && <span className="default-badge">default</span>}
                          {entry.source === 'db' && <span className="db-badge">db</span>}
                          {descriptions[key] && (
                            <button
                              className="rc-help-btn"
                              onClick={() =>
                                setExpandedHelp((prev) => {
                                  const next = new Set(prev)
                                  if (next.has(key)) next.delete(key)
                                  else next.add(key)
                                  return next
                                })
                              }
                              title={$t('runtime.showDescription')}
                              type="button"
                            >
                              <Info size={14} />
                            </button>
                          )}
                        </div>
                        {expandedHelp.has(key) && descriptions[key] && (
                          <div className="rc-field-help">
                            <p>{descriptions[key].description}</p>
                            <div className="rc-field-help-meta">
                              <span>{$t('runtime.description')}: <code>{descriptions[key].type}</code></span>
                              <span>{$t('runtime.defaultValue')}: <code>{formatValue(descriptions[key].default)}</code></span>
                            </div>
                          </div>
                        )}
                        {key === 'embeddings_api_url' && (
                          <select
                            value={(() => {
                              const url = String(entry.value || '')
                              const match = providerList.find((p) => p.base_url === url)
                              return match ? match.id : ''
                            })()}
                            onChange={(e) => {
                              const pid = e.target.value
                              if (!pid) return
                              const prov = providerList.find((p) => p.id === pid)
                              if (prov) {
                                handleValueChange('embeddings_api_url', prov.base_url)
                              }
                            }}
                            disabled={!entry.editable || !canWriteConfig}
                            style={{ marginBottom: 6 }}
                          >
                            <option value="">{$t('runtime.customUrl') || 'Custom URL'}</option>
                            {providerList.map((p) => (
                              <option key={p.id} value={p.id}>
                                {p.name}
                              </option>
                            ))}
                          </select>
                        )}
                        {key === 'embedding_model' ? (() => {
                          const urlEntry = configData?.effective['embeddings_api_url']
                          const url = String(urlEntry?.value || '')
                          const prov = providerList.find((p) => p.base_url === url)
                          const pm = prov ? providerModels[prov.id] : null
                          if (pm && pm.models.length > 0) {
                            return (
                              <select
                                value={String(entry.value || '')}
                                onChange={(e) => handleValueChange(key, e.target.value)}
                                disabled={!entry.editable || !canWriteConfig}
                              >
                                {pm.models.map((m: string) => (
                                  <option key={m} value={m}>{m}</option>
                                ))}
                              </select>
                            )
                          }
                          return (
                            <input
                              type="text"
                              value={formatValue(entry.value)}
                              onChange={(e) => handleValueChange(key, e.target.value)}
                              disabled={!entry.editable || !canWriteConfig}
                              title={entry.source === 'env' ? $t('runtime.envOverride') : undefined}
                            />
                          )
                        })() : descriptions[key]?.type === 'select' && descriptions[key]?.options ? (
                          <select
                            value={String(entry.value ?? '')}
                            onChange={(e) => handleValueChange(key, e.target.value)}
                            disabled={!entry.editable || !canWriteConfig}
                            title={entry.source === 'env' ? $t('runtime.envOverride') : undefined}
                          >
                            {descriptions[key].options.map((opt: { value: string; label: string }) => (
                              <option key={opt.value} value={opt.value}>{opt.label}</option>
                            ))}
                          </select>
                        ) : isBooleanValue(entry.value) ? (
                          <label className={`rc-toggle ${!entry.editable ? 'locked' : ''}`}>
                            <input
                              type="checkbox"
                              checked={entry.value === true}
                              onChange={(e) => handleValueChange(key, e.target.checked)}
                              disabled={!entry.editable || !canWriteConfig}
                            />
                            <span className="rc-toggle-track">
                              <span className="rc-toggle-knob" />
                            </span>
                            <span className="rc-toggle-label">
                              {entry.value === true ? $t('common.on') : $t('common.off')}
                            </span>
                          </label>
                        ) : typeof entry.value === 'number' ? (
                          <input
                            type="number"
                            value={formatValue(entry.value)}
                            onChange={(e) => handleValueChange(key, e.target.value)}
                            disabled={!entry.editable || !canWriteConfig}
                            title={entry.source === 'env' ? $t('runtime.envOverride') : undefined}
                          />
                        ) : (
                          <input
                            type="text"
                            value={formatValue(entry.value)}
                            onChange={(e) => handleValueChange(key, e.target.value)}
                            disabled={!entry.editable || !canWriteConfig}
                            title={entry.source === 'env' ? $t('runtime.envOverride') : undefined}
                          />
                        )}
                        <div className="rc-field-meta">
                          {entry.default_value !== undefined && (
                            <span className="meta-default">{$t('runtime.defaultValue')}: {formatValue(entry.default_value)}</span>
                          )}
                          {entry.env_value !== undefined && entry.env_value !== null && (
                            <span className="meta-env">Env: {formatValue(entry.env_value)}</span>
                          )}
                          {entry.source === 'env' && configData?.stored[key] !== undefined && (
                            <span className="meta-db">DB: {formatValue(configData.stored[key])}</span>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          ))}
        </>
      )}

      {activeTab === 'providers' && (
        <>
          <div className="rc-action-bar">
            {canWriteProviders && (
              <button className="rc-btn primary" onClick={handleCreateProvider}>
                <Plus size={14} /> {$t('common.add')}
              </button>
            )}
            <button className="rc-btn secondary" onClick={handleExportProviders}>
              <Download size={14} /> Export
            </button>
            {canWriteProviders && (
              <button className="rc-btn secondary" onClick={() => uploadInputRef.current?.click()}>
                <Upload size={14} /> Import
              </button>
            )}
            <input
              ref={uploadInputRef}
              type="file"
              accept=".json"
              style={{ display: 'none' }}
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) handleImport(file)
                e.target.value = ''
              }}
            />
          </div>

          {providerError && (
            <div className="error-box">
              <AlertCircle size={16} />
              <span>{providerError}</span>
            </div>
          )}

          {providerLoading || providerModelsLoading ? (
            <div className="loading">{$t('common.loading')}</div>
          ) : providerList.length === 0 ? (
            <div className="rc-empty">
              <Server size={32} style={{ opacity: 0.4 }} />
              <p>{$t('runtime.noProviders')}</p>
            </div>
          ) : (
            <>
              {(() => {
                const allModelSet = new Set<string>()
                Object.values(providerModels).forEach((pm) => pm.models.forEach((m) => allModelSet.add(m)))
                const total = allModelSet.size
                const count = Object.values(providerModels).filter((pm) => pm.models.length > 0).length
                return (
                  <div className="rc-card" style={{ marginBottom: 16, padding: 12 }}>
                    <div style={{ fontSize: 14, color: 'var(--muted)' }}>
                      {total === 0 ? $t('runtime.modelsAvailableNone') : $t('runtime.totalModels', { total, count })}
                    </div>
                  </div>
                )
              })()}
              <div className="rc-provider-grid">
                {providerList.map((p) => {
                  const pm = providerModels[p.id]
                  const modelCount = pm?.models.length ?? 0
                  return (
                    <div key={p.id} className={`rc-provider-card ${p.enabled ? '' : 'disabled'}`}>
                      <div className="rc-provider-card-header">
                        <div>
                          <div className="rc-provider-name">
                            {p.name}
                          </div>
                          <div className="rc-provider-badges">
                            {p.is_default && (
                              <span className="rc-badge default">
                                <Check size={10} /> default
                              </span>
                            )}
                            {!p.enabled && (
                              <span className="rc-badge disabled">disabled</span>
                            )}
                            {p.source === 'env' && (
                              <span className="rc-badge env">env</span>
                            )}
                            {p.has_api_key ? (
                              <span className="rc-badge secret-ok">
                                <KeyRound size={10} /> Key
                              </span>
                            ) : (
                              <span className="rc-badge secret-missing">
                                <ShieldAlert size={10} /> {$t('runtime.noKey')}
                              </span>
                            )}
                            {p.supports_native_tool_calls === true && (
                              <span className="rc-badge status-ok">
                                <Check size={10} /> native tools
                              </span>
                            )}
                            {p.supports_native_tool_calls === false && (
                              <span className="rc-badge status-error">
                                <AlertCircle size={10} /> JSON fallback
                              </span>
                            )}
                            {p.supports_native_tool_calls === null && (
                              <span className="rc-badge" style={{ background: '#fef3c7', color: '#92400e' }}>
                                <Info size={10} /> auto-detect
                              </span>
                            )}
                            {p.last_check_status && (
                              <span className={`rc-badge status-${p.last_check_status === 'ok' ? 'ok' : 'error'}`}>
                                {p.last_check_status === 'ok' ? <Check size={10} /> : <AlertCircle size={10} />}
                                {p.last_check_status}
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="rc-provider-actions">
                          {p.source !== 'env' && canWriteProviders && (
                            <>
                              <button className="rc-icon-btn" onClick={() => handleCheckProvider(p.id)} title={$t('runtime.test')}>
                                <Play size={14} />
                              </button>
                              <button className="rc-icon-btn" onClick={() => handleEditProvider(p)} title={$t('common.edit')}>
                                <Pencil size={14} />
                              </button>
                              <button className="rc-icon-btn danger" onClick={() => handleDeleteProvider(p.id)} title={$t('common.delete')}>
                                <Trash2 size={14} />
                              </button>
                            </>
                          )}
                          {p.source === 'env' && (
                            <span className="rc-badge env" style={{ marginTop: 4 }}>{$t('runtime.readonly')}</span>
                          )}
                        </div>
                      </div>
                      <div className="rc-provider-meta">
                        <span><Globe size={13} /> {p.base_url}</span>
                        <span><Clock size={13} /> {p.timeout_seconds}s</span>
                      </div>
                      <div style={{ marginTop: 8, fontSize: 12, color: 'var(--muted)' }}>
                        {modelCount === 0
                          ? $t('runtime.modelsAvailableNone')
                          : $t('runtime.modelsAvailable', { count: modelCount })}
                      </div>
                    </div>
                  )
                })}
              </div>
            </>
          )}
        </>
      )}

      {activeTab === 'apikeys' && (
        <>
          <div className="rc-card" style={{ marginBottom: 16, padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontWeight: 600, fontSize: 15 }}>{$t('runtime.apiKeyEnforcement')}</div>
                <div style={{ fontSize: 13, color: 'var(--muted)', marginTop: 4 }}>
                  {$t('runtime.apiKeyEnforcementDesc')}
                </div>
              </div>
              {(() => {
                const entry = configData?.effective.api_keys_enabled
                const editable = entry?.editable !== false
                const isOn = entry?.value !== false
                return (
                  <label className={`rc-toggle ${!editable ? 'locked' : ''}`} title={!editable ? $t('runtime.envOverride') : undefined}>
                    <input
                      type="checkbox"
                      checked={isOn}
                      onChange={async (e) => {
                        if (!editable) return
                        try {
                          await runtimeConfig.save({ api_keys_enabled: e.target.checked })
                          invalidateCache('/admin/runtime-config')
                          const refreshed = await runtimeConfig.get()
                          if (refreshed) setConfigData(refreshed)
                        } catch (err) {
                          setApiKeyError(String(err))
                        }
                      }}
                      disabled={!editable}
                    />
                    <span className="rc-toggle-track">
                      <span className="rc-toggle-knob" />
                    </span>
                    <span className="rc-toggle-label">
                      {isOn ? $t('common.on') : $t('common.off')}
                    </span>
                    {!editable && entry?.source === 'env' && <span className="env-badge" style={{ marginLeft: 8 }}>env</span>}
                  </label>
                )
              })()}
            </div>
          </div>

          <div className="rc-action-bar">
            {canWriteApiKeys && (
              <button
                className="rc-btn primary"
                onClick={() => {
                  setApiKeyError(null)
                  setNewKeyValue(null)
                  setEditingApiKey({ id: '', name: '', allowed_models: [], enabled: true, rate_limit_rpm: 0, rate_limit_rpd: 0 })
                }}
              >
                <Plus size={14} /> {$t('common.create')}
              </button>
            )}
          </div>

          {newKeyValue && (
            <div className="rc-form-group" style={{ background: '#ecfdf5', border: '1px solid #10b981', borderRadius: 10, padding: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <strong style={{ color: '#065f46' }}>{$t('runtime.newApiKey')}</strong>
                <button className="rc-icon-btn" onClick={() => setNewKeyValue(null)} title={$t('common.close')}>
                  <X size={14} />
                </button>
              </div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <code style={{ flex: 1, background: '#fff', padding: '8px 12px', borderRadius: 6, fontSize: 13, wordBreak: 'break-all' }}>
                  {newKeyValue}
                </code>
                <button
                  className="rc-btn secondary"
                  onClick={() => navigator.clipboard.writeText(newKeyValue)}
                >
                  <Copy size={14} /> {$t('common.copy')}
                </button>
              </div>
            </div>
          )}

          {apiKeyError && (
            <div className="error-box">
              <AlertCircle size={16} />
              <span>{apiKeyError}</span>
            </div>
          )}

          {apiKeyLoading ? (
            <div className="loading">{$t('common.loading')}</div>
          ) : apiKeyList.length === 0 ? (
            <div className="rc-empty">
              <KeyRound size={32} style={{ opacity: 0.4 }} />
              <p>{$t('runtime.noApiKeys')}</p>
            </div>
          ) : (
            <div className="rc-provider-grid">
              {apiKeyList.map((k) => (
                <div key={k.id} className={`rc-provider-card ${k.enabled ? '' : 'disabled'}`} style={k.source === 'env' ? { opacity: 0.7 } : undefined}>
                  <div className="rc-provider-card-header">
                    <div>
                      <div className="rc-provider-name">{k.name}</div>
                      <div className="rc-provider-badges">
                        {!k.enabled && <span className="rc-badge disabled">disabled</span>}
                        {k.source === 'env' && <span className="rc-badge env">env</span>}
                        {k.allowed_models.length === 0 ? (
                          <span className="rc-badge default">{$t('runtime.allModels')}</span>
                        ) : (
                          <span className="rc-badge env">{$t('runtime.modelsCount', { count: k.allowed_models.length })}</span>
                        )}
                        {(k.rate_limit_rpm > 0 || k.rate_limit_rpd > 0) && (
                          <span className="rc-badge env" title="rate limit">
                            {k.rate_limit_rpm > 0 ? `${k.rate_limit_rpm}/min` : ''}{k.rate_limit_rpm > 0 && k.rate_limit_rpd > 0 ? ' · ' : ''}{k.rate_limit_rpd > 0 ? `${k.rate_limit_rpd}/d` : ''}
                          </span>
                        )}
                      </div>
                      {k.allowed_models.length > 0 && (
                        <div style={{ marginTop: 6, fontSize: 12, color: 'var(--muted)', display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                          {k.allowed_models.map((m) => (
                            <span key={m} className="rc-badge env" style={{ fontSize: 11 }}>{m}</span>
                          ))}
                        </div>
                      )}
                    </div>
                    <div className="rc-provider-actions">
                      {canWriteApiKeys && (
                        <button
                          className="rc-icon-btn"
                          onClick={() => {
                            setApiKeyError(null)
                            setNewKeyValue(null)
                            setEditingApiKey(k)
                          }}
                          title={$t('runtime.edit')}
                        >
                          <Pencil size={14} />
                        </button>
                      )}
                      {canWriteApiKeys && k.source !== 'env' && (
                        <button
                          className="rc-icon-btn danger"
                          onClick={() => handleDeleteApiKey(k.id)}
                          title={$t('common.delete')}
                        >
                          <Trash2 size={14} />
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* --- Provider Edit Modal --- */}
      {editingProvider && (
        <div className="rc-modal-overlay" onClick={() => setEditingProvider(null)}>
          <div className="rc-modal" onClick={(e) => e.stopPropagation()}>
            <h3>{providerList.some((p) => p.id === editingProvider.id) ? $t('runtime.editProvider') : $t('runtime.addProvider')}</h3>
            <div className="rc-form-group">
              <label className="rc-required">ID</label>
              <input
                type="text"
                value={editingProvider.id}
                onChange={(e) => setEditingProvider({ ...editingProvider, id: e.target.value })}
                required
              />
            </div>
            <div className="rc-form-group">
              <label className="rc-required">Name</label>
              <input
                type="text"
                value={editingProvider.name}
                onChange={(e) => setEditingProvider({ ...editingProvider, name: e.target.value })}
                required
              />
            </div>
            <div className="rc-form-group">
              <label className="rc-required">Base URL</label>
              <input
                type="text"
                value={editingProvider.base_url}
                onChange={(e) => setEditingProvider({ ...editingProvider, base_url: e.target.value })}
                required
              />
            </div>
            <div className="rc-form-group">
              <label>Provider Typ</label>
              <select
                value={editingProvider.provider_type}
                onChange={(e) => setEditingProvider({ ...editingProvider, provider_type: e.target.value })}
              >
                <option value="openai-compatible">OpenAI Compatible</option>
                <option value="ollama">Ollama</option>
                <option value="custom">Custom</option>
              </select>
            </div>
            <div className="rc-form-group">
              <label>Native Tool Calling</label>
              <select
                value={editingProvider.supports_native_tool_calls === null ? '' : String(editingProvider.supports_native_tool_calls)}
                onChange={(e) => {
                  const val = e.target.value
                  setEditingProvider({
                    ...editingProvider,
                    supports_native_tool_calls: val === '' ? null : val === 'true',
                  })
                }}
              >
                <option value="">Auto-detect</option>
                <option value="true">Enabled</option>
                <option value="false">Disabled / JSON-Fallback</option>
              </select>
            </div>
            <div className="rc-form-row">
              <div className="rc-form-group">
                <label>Timeout (s)</label>
                <input
                  type="number"
                  value={editingProvider.timeout_seconds}
                  onChange={(e) =>
                    setEditingProvider({ ...editingProvider, timeout_seconds: parseInt(e.target.value, 10) || 0 })
                  }
                />
              </div>
              <div className="rc-form-group">
                <label>Max Retries</label>
                <input
                  type="number"
                  value={editingProvider.max_retries}
                  onChange={(e) =>
                    setEditingProvider({ ...editingProvider, max_retries: parseInt(e.target.value, 10) || 0 })
                  }
                />
              </div>
            </div>
            <div className="rc-checkbox-group">
              <label className="rc-checkbox-label">
                <input
                  type="checkbox"
                  checked={editingProvider.enabled}
                  onChange={(e) => setEditingProvider({ ...editingProvider, enabled: e.target.checked })}
                />
                {$t('runtime.enabled')}
              </label>
              <label className="rc-checkbox-label">
                <input
                  type="checkbox"
                  checked={editingProvider.is_default}
                  onChange={(e) => setEditingProvider({ ...editingProvider, is_default: e.target.checked })}
                />
                {$t('runtime.defaultModel')}
              </label>
            </div>
            <div className="rc-form-group">
              <label>API Key</label>
              <div className="rc-secret-row">
                <input
                  type={showSecretInput === editingProvider.id ? 'text' : 'password'}
                  placeholder={`${$t('common.optional')}: API Key`}
                  value={secretValue}
                  onChange={(e) => setSecretValue(e.target.value)}
                />
                <button
                  className="rc-icon-btn"
                  onClick={() =>
                    setShowSecretInput((prev) => (prev === editingProvider.id ? null : editingProvider.id))
                  }
                >
                  {showSecretInput === editingProvider.id ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </div>
            <div className="rc-modal-actions">
              {canWriteProviders && (
                <button className="rc-btn primary" onClick={handleSaveProvider}>
                  {$t('runtime.save')}
                </button>
              )}
              <button className="rc-btn ghost" onClick={() => setEditingProvider(null)}>
                {$t('runtime.cancel')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* --- API Key Edit Modal --- */}
      {editingApiKey && (
        <div className="rc-modal-overlay" onClick={() => setEditingApiKey(null)}>
          <div className="rc-modal" onClick={(e) => e.stopPropagation()}>
            <h3>{editingApiKey.id ? $t('runtime.apiKeyEdit') : $t('runtime.apiKeyCreate')}</h3>
            <div className="rc-form-group">
              <label className="rc-required">Name</label>
              <input
                type="text"
                value={editingApiKey.name}
                onChange={(e) => setEditingApiKey({ ...editingApiKey, name: e.target.value })}
                disabled={editingApiKey.source === 'env'}
                required
              />
            </div>
            <div className="rc-form-group">
              <label>{$t('runtime.allowedModels')}</label>
              <div style={{ maxHeight: 200, overflowY: 'auto', border: '1px solid var(--border)', borderRadius: 10, padding: 12 }}>
                {availableModels.length === 0 ? (
                  <span style={{ color: 'var(--muted)', fontSize: 13 }}>{$t('runtime.noModelsAvailable')}</span>
                ) : (
                  availableModels.map((model) => (
                    <label key={model} className="rc-checkbox-label" style={{ display: 'block', marginBottom: 6 }}>
                      <input
                        type="checkbox"
                        checked={editingApiKey.allowed_models.includes(model)}
                        onChange={(e) => {
                          const next = e.target.checked
                            ? [...editingApiKey.allowed_models, model]
                            : editingApiKey.allowed_models.filter((m) => m !== model)
                          setEditingApiKey({ ...editingApiKey, allowed_models: next })
                        }}
                        disabled={editingApiKey.source === 'env'}
                      />
                      {model}
                    </label>
                  ))
                )}
              </div>
              <div style={{ marginTop: 6, fontSize: 12, color: 'var(--muted)' }}>
                {$t('runtime.noSelectionAllModels')}
              </div>
            </div>
            <div className="rc-checkbox-group">
              <label className="rc-checkbox-label">
                <input
                  type="checkbox"
                  checked={editingApiKey.enabled}
                  onChange={(e) => setEditingApiKey({ ...editingApiKey, enabled: e.target.checked })}
                />
                {$t('runtime.enabled')}
              </label>
            </div>
            <div className="rc-form-group">
              <label>{$t('runtime.rateLimits')}</label>
              <div style={{ display: 'flex', gap: 12 }}>
                <div style={{ flex: 1 }}>
                  <input
                    type="number"
                    min={0}
                    value={editingApiKey.rate_limit_rpm ?? 0}
                    onChange={(e) => setEditingApiKey({ ...editingApiKey, rate_limit_rpm: Math.max(0, parseInt(e.target.value) || 0) })}
                    placeholder="0 = unlimited"
                    disabled={editingApiKey.source === 'env'}
                  />
                  <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 4 }}>{$t('runtime.requestsPerMinute')}</div>
                </div>
                <div style={{ flex: 1 }}>
                  <input
                    type="number"
                    min={0}
                    value={editingApiKey.rate_limit_rpd ?? 0}
                    onChange={(e) => setEditingApiKey({ ...editingApiKey, rate_limit_rpd: Math.max(0, parseInt(e.target.value) || 0) })}
                    placeholder="0 = unlimited"
                    disabled={editingApiKey.source === 'env'}
                  />
                  <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 4 }}>{$t('runtime.requestsPerDay')}</div>
                </div>
              </div>
            </div>
            <div className="rc-modal-actions">
              {canWriteApiKeys && (
                <button
                  className="rc-btn primary"
                  onClick={async () => {
                    if (editingApiKey.source !== 'env' && !editingApiKey.name.trim()) {
                      setApiKeyError($t('runtime.nameEmpty'))
                      return
                    }
                    try {
                      if (editingApiKey.id) {
                        await handleUpdateApiKey(editingApiKey.id, editingApiKey.name, editingApiKey.allowed_models, editingApiKey.enabled, editingApiKey.rate_limit_rpm ?? 0, editingApiKey.rate_limit_rpd ?? 0)
                      } else {
                        await handleCreateApiKey(editingApiKey.name, editingApiKey.allowed_models, editingApiKey.enabled, editingApiKey.rate_limit_rpm ?? 0, editingApiKey.rate_limit_rpd ?? 0)
                      }
                      setEditingApiKey(null)
                    } catch (err) {
                      setApiKeyError(String(err))
                    }
                  }}
                >
                  {$t('runtime.save')}
                </button>
              )}
              <button className="rc-btn ghost" onClick={() => setEditingApiKey(null)}>
                {$t('runtime.cancel')}
              </button>
            </div>
          </div>
        </div>
      )}

      <UnsavedChangesBar
        dirty={isDirty && canWriteConfig}
        saving={saveStatus === 'saving'}
        onSave={handleSave}
        onDiscard={handleDiscard}
      />
      <ConfirmModal
        open={pendingConfirm !== null}
        title={$t('common.delete')}
        message={pendingConfirm?.message || ''}
        onConfirm={() => {
          const action = pendingConfirm?.action
          setPendingConfirm(null)
          action?.()
        }}
        onCancel={() => setPendingConfirm(null)}
      />
    </div>
  )
}
