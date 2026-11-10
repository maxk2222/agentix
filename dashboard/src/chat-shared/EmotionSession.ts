import { createVideoSnippetBuffer } from '../emotion/videoSnippetBuffer'
import { createAudioSnippetBuffer } from '../emotion/audioSnippetBuffer'
import { fuseEmotions } from '../emotion/emotionAggregator'
import type {
  EmotionContextPayload,
  EmotionRuntimeStatus,
  SnippetConfig,
  EmotionResult,
  PhaseEmotion,
  EmotionPhase,
} from '../emotion/types'

export interface EmotionSessionConfig {
  enabled: boolean
  text: { enabled: boolean; rules_enabled: boolean }
  webcam: {
    enabled: boolean
    button_visible: boolean
    runtime: string
    execution_providers: string[]
    face_detector: string
    face_detector_type?: 'mediapipe' | 'onnx'
    face_model_url: string
    expression_model_url: string
    expression_model_type?: string
    expression_input_size?: number
    snippet_window_ms: number
    sample_fps: number
    max_frames_per_snippet: number
    min_valid_face_frames: number
    frame_upload_enabled: boolean
  }
  audio: {
    enabled: boolean
    model_url: string
    min_snr_db?: number
    noise_gate_min?: number
    noise_gate_max?: number
  }
  prompt_steering: { enabled: boolean; min_confidence: number }
  analytics: { enabled: boolean }
}

export interface EmotionSessionCallbacks {
  onStatusChange?: (status: EmotionRuntimeStatus) => void
  onWebcamActiveChange?: (active: boolean) => void
  onWebcamConsentChange?: (consented: boolean) => void
  /** UI hook: must resolve to the user's webcam-consent decision (modal). */
  onWebcamConsentRequired?: () => Promise<boolean>
}

export class EmotionSession {
  private config: EmotionSessionConfig | null = null
  private callbacks: EmotionSessionCallbacks

  private _webcamConsent = false
  private _webcamActive = false
  private _emotionRuntimeStatus: EmotionRuntimeStatus = 'uninitialized'

  private snippetBufferRef: ReturnType<typeof createVideoSnippetBuffer> | null = null
  private audioSnippetBufferRef: ReturnType<typeof createAudioSnippetBuffer> | null = null
  private continuousAudioStreamRef: MediaStream | null = null

  private isStartingWebcam = false
  private isStartingAudio = false

  private emotionTimelineRef: {
    phases: Record<EmotionPhase, { snapshots: EmotionResult[]; startTime: number; endTime?: number }>
    activePhase: EmotionPhase
    startTime: number
  } | null = null

  private emotionPollIntervalRef: ReturnType<typeof setInterval> | null = null
  private hasTransitionedToTypingRef = false
  private audioSnapshotsRef: EmotionResult[] = []

  constructor(config: EmotionSessionConfig | null, callbacks: EmotionSessionCallbacks = {}) {
    this.config = config
    this.callbacks = callbacks
  }

  /* ------------------------------------------------------------------ */
  /*  Internal state helpers                                            */
  /* ------------------------------------------------------------------ */
  private setStatus(status: EmotionRuntimeStatus) {
    this._emotionRuntimeStatus = status
    this.callbacks.onStatusChange?.(status)
  }

  private setWebcamActive(active: boolean) {
    this._webcamActive = active
    this.callbacks.onWebcamActiveChange?.(active)
  }

  private setWebcamConsent(consented: boolean) {
    this._webcamConsent = consented
    this.callbacks.onWebcamConsentChange?.(consented)
  }

  /* ------------------------------------------------------------------ */
  /*  Webcam                                                            */
  /* ------------------------------------------------------------------ */
  async startWebcam(): Promise<void> {
    if (this._webcamActive) return
    if (this.isStartingWebcam) return
    if (!this.config?.webcam?.enabled) return

    if (!this._webcamConsent) {
      const agreed = (await this.callbacks.onWebcamConsentRequired?.()) ?? false
      if (!agreed) return
      this.setWebcamConsent(true)
    }

    this.isStartingWebcam = true
    this.setStatus('loading')
    try {
      const cfg = this.config.webcam
      const snippetConfig: SnippetConfig = {
        windowMs: cfg.snippet_window_ms,
        sampleFps: cfg.sample_fps,
        maxFrames: cfg.max_frames_per_snippet,
        minValidFaceFrames: cfg.min_valid_face_frames,
        faceModelUrl: cfg.face_model_url,
        faceDetectorType: (cfg.face_detector?.includes('mediapipe') ? 'mediapipe' : 'onnx') as 'mediapipe' | 'onnx',
        expressionModelUrl: cfg.expression_model_url,
        expressionModelType: (cfg.expression_model_type as 'fer' | 'affectnet') ?? 'affectnet',
        expressionInputSize: cfg.expression_input_size ?? 224,
        executionProviders: cfg.execution_providers,
      }
      this.snippetBufferRef = createVideoSnippetBuffer(snippetConfig)
      await this.snippetBufferRef.start()
      if (this.isStartingWebcam) {
        this.setWebcamActive(true)
        this.setStatus('ready')
      }
    } catch {
      this.setStatus('error')
      this.setWebcamActive(false)
    } finally {
      this.isStartingWebcam = false
    }
  }

  stopWebcam(): void {
    this.isStartingWebcam = false
    if (this.snippetBufferRef) {
      this.snippetBufferRef.stop()
      this.snippetBufferRef = null
    }
    this.stopContinuousAudio()
    this.setWebcamActive(false)
    this.setStatus('uninitialized')
  }

  async toggleWebcam(): Promise<void> {
    if (this._webcamActive || this.isStartingWebcam) {
      this.stopWebcam()
    } else {
      await this.startWebcam().catch(() => {})
      if (this.config?.audio?.enabled) {
        await this.startContinuousAudio().catch(() => {})
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /*  Continuous Audio (independent stream)                               */
  /* ------------------------------------------------------------------ */
  async startContinuousAudio(): Promise<void> {
    if (this.audioSnippetBufferRef?.isActive()) return
    if (this.isStartingAudio) return
    if (!this.config?.audio?.enabled) {
      console.info('[EmotionSession] Audio emotion disabled in config')
      return
    }
    this.isStartingAudio = true
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      if (!this.isStartingAudio) {
        stream.getTracks().forEach((t) => t.stop())
        return
      }
      this.continuousAudioStreamRef = stream
      const cfg = this.config.webcam
      const audioCfg: SnippetConfig = {
        windowMs: cfg?.snippet_window_ms || 3000,
        sampleFps: cfg?.sample_fps || 3,
        maxFrames: cfg?.max_frames_per_snippet || 12,
        minValidFaceFrames: cfg?.min_valid_face_frames || 3,
        faceModelUrl: cfg?.face_model_url || '/models/emotion/face-detector.onnx',
        faceDetectorType: 'mediapipe',
        expressionModelUrl: this.config.audio.model_url || '/models/emotion/audio-emotion.onnx',
        expressionModelType: 'affectnet',
        expressionInputSize: 224,
        executionProviders: cfg?.execution_providers || ['wasm'],
        minSnrDb: this.config.audio.min_snr_db ?? 10,
        noiseGateMin: this.config.audio.noise_gate_min ?? 0.001,
        noiseGateMax: this.config.audio.noise_gate_max ?? 0.5,
      }
      console.info('[EmotionSession] Starting continuous audio emotion with modelUrl:', audioCfg.expressionModelUrl)
      this.audioSnippetBufferRef = createAudioSnippetBuffer(audioCfg)
      this.audioSnippetBufferRef.start(stream)
      console.info('[EmotionSession] Continuous audio emotion started')
    } catch (err) {
      console.warn('[EmotionSession] Audio emotion start failed:', err)
    } finally {
      this.isStartingAudio = false
    }
  }

  stopContinuousAudio(): void {
    this.isStartingAudio = false
    if (this.audioSnippetBufferRef) {
      this.audioSnippetBufferRef.stop()
      this.audioSnippetBufferRef = null
    }
    if (this.continuousAudioStreamRef) {
      this.continuousAudioStreamRef.getTracks().forEach((t) => t.stop())
      this.continuousAudioStreamRef = null
    }
  }

  /* ------------------------------------------------------------------ */
  /*  Pre-send capture                                                  */
  /* ------------------------------------------------------------------ */
  async capturePreSend(): Promise<EmotionResult | null> {
    if (!this.snippetBufferRef || !this.snippetBufferRef.isActive()) return null
    return this.snippetBufferRef.captureSnapshot()
  }

  /* ------------------------------------------------------------------ */
  /*  Payload building                                                  */
  /* ------------------------------------------------------------------ */
  buildPayload(options: {
    preSendEmotion?: EmotionResult | null
    audioProsody?: EmotionResult | null
    fused?: EmotionResult | null
    phases?: PhaseEmotion[]
    forceAudioOptIn?: boolean
  } = {}): EmotionContextPayload | undefined {
    if (!this.config?.enabled) return undefined

    const payload: EmotionContextPayload = {}

    if (this.config.webcam.enabled && this._webcamActive && this._webcamConsent) {
      // visual_pre_send added separately via preSendEmotion
    }

    payload.consent = {
      webcam_opt_in: this._webcamConsent,
      frames_sent_to_backend: false,
      audio_opt_in: this.audioSnippetBufferRef?.isActive() ?? false,
    }

    if (options.preSendEmotion) {
      payload.visual_pre_send = options.preSendEmotion
    }
    if (options.audioProsody) {
      payload.audio_prosody = options.audioProsody
    }
    if (options.fused) {
      payload.fused = options.fused
    }
    if (options.phases && options.phases.length > 0) {
      payload.phases = options.phases
    }

    // Best accumulated audio snapshot overrides STT-provided one
    if (this.audioSnapshotsRef.length > 0) {
      const bestAudio = this.audioSnapshotsRef.reduce((best, curr) =>
        curr.confidence > best.confidence ? curr : best
      )
      payload.audio_prosody = bestAudio
      this.audioSnapshotsRef = []
    }

    if (options.forceAudioOptIn || options.audioProsody || payload.audio_prosody) {
      if (payload.consent) {
        payload.consent.audio_opt_in = true
      }
    }

    return payload
  }

  /* ------------------------------------------------------------------ */
  /*  Phase-based polling                                               */
  /* ------------------------------------------------------------------ */
  startPolling(initialPhase: EmotionPhase = 'waiting_for_response'): void {
    if (this.emotionPollIntervalRef) return
    this.hasTransitionedToTypingRef = false
    this.audioSnapshotsRef = []
    const phases = this.createEmptyPhases()
    phases[initialPhase].startTime = Date.now()
    this.emotionTimelineRef = {
      phases,
      activePhase: initialPhase,
      startTime: Date.now(),
    }
    this.emotionPollIntervalRef = setInterval(() => {
      this.processEmotionBatch().catch(() => {})
    }, 2000)
  }

  async stopPolling(): Promise<void> {
    if (this.emotionPollIntervalRef) {
      clearInterval(this.emotionPollIntervalRef)
      this.emotionPollIntervalRef = null
    }
    await this.processEmotionBatch()
  }

  switchPhase(nextPhase: EmotionPhase): void {
    const tl = this.emotionTimelineRef
    if (!tl || tl.activePhase === nextPhase) return

    const current = tl.phases[tl.activePhase]
    if (current.snapshots.length > 0) {
      const fused = fuseEmotions(current.snapshots)
      if (fused) {
        tl.phases[tl.activePhase] = {
          snapshots: [fused],
          startTime: current.startTime,
          endTime: Date.now(),
        }
      } else {
        current.endTime = Date.now()
      }
    } else {
      current.endTime = Date.now()
    }

    tl.phases[nextPhase].startTime = Date.now()
    tl.activePhase = nextPhase
  }

  private createEmptyPhases(): Record<EmotionPhase, { snapshots: EmotionResult[]; startTime: number; endTime?: number }> {
    return {
      waiting_for_response: { snapshots: [], startTime: Date.now() },
      reading: { snapshots: [], startTime: 0 },
      typing: { snapshots: [], startTime: 0 },
      speaking: { snapshots: [], startTime: 0 },
    }
  }

  private async processEmotionBatch(): Promise<void> {
    const tl = this.emotionTimelineRef
    if (!tl) return

    const promises: Promise<void>[] = []
    if (this.snippetBufferRef?.isActive()) {
      promises.push(
        this.snippetBufferRef.captureSnapshot().then((res) => {
          if (res) tl.phases[tl.activePhase].snapshots.push(res)
        }).catch(() => {})
      )
    }
    if (this.audioSnippetBufferRef?.isActive()) {
      promises.push(
        this.audioSnippetBufferRef.captureSnapshot().then((res) => {
          if (res) {
            tl.phases[tl.activePhase].snapshots.push(res)
            this.audioSnapshotsRef.push(res)
          }
        }).catch(() => {})
      )
    }
    await Promise.all(promises)
  }

  /* ------------------------------------------------------------------ */
  /*  Phase computation                                                 */
  /* ------------------------------------------------------------------ */
  buildEmotionPhases(): PhaseEmotion[] {
    const tl = this.emotionTimelineRef
    if (!tl) return []

    const result: PhaseEmotion[] = []
    const phaseOrder: EmotionPhase[] = ['waiting_for_response', 'reading', 'typing', 'speaking']

    for (const phase of phaseOrder) {
      const data = tl.phases[phase]
      if (data.snapshots.length === 0 && !data.endTime && phase !== tl.activePhase) {
        continue
      }
      if (data.snapshots.length > 0) {
        const fused = fuseEmotions(data.snapshots)
        if (fused) {
          result.push({
            phase,
            primary_emotion: fused.primary_emotion,
            confidence: fused.confidence,
            intensity: fused.intensity,
            source: `phase__${phase}`,
            timestamp_ms: data.startTime || Date.now(),
            sampled_frames: fused.sampled_frames,
            valid_face_frames: fused.valid_face_frames,
            face_detection_rate: fused.face_detection_rate,
            window_ms: fused.window_ms,
          })
          continue
        }
      }
      if (data.startTime > 0 || phase === tl.activePhase) {
        result.push({
          phase,
          primary_emotion: 'unknown',
          confidence: 0,
          intensity: 'low',
          source: `phase__${phase}`,
          timestamp_ms: data.startTime || Date.now(),
        })
      }
    }
    return result
  }

  computeOverallFused(phases: PhaseEmotion[]): EmotionResult | null {
    if (phases.length === 0) return null
    const asResults: EmotionResult[] = phases.map((p) => ({
      primary_emotion: p.primary_emotion,
      confidence: p.confidence,
      intensity: p.intensity,
      source: p.source,
      timestamp_ms: p.timestamp_ms,
      sampled_frames: p.sampled_frames,
      valid_face_frames: p.valid_face_frames,
      face_detection_rate: p.face_detection_rate,
      window_ms: p.window_ms,
    }))
    return fuseEmotions(asResults)
  }

  /* ------------------------------------------------------------------ */
  /*  Getters                                                           */
  /* ------------------------------------------------------------------ */
  get status(): EmotionRuntimeStatus {
    return this._emotionRuntimeStatus
  }

  get isWebcamActive(): boolean {
    return this._webcamActive
  }

  get isWebcamConsented(): boolean {
    return this._webcamConsent
  }

  get isAudioActive(): boolean {
    return this.audioSnippetBufferRef?.isActive() ?? false
  }

  get hasTransitionedToTyping(): boolean {
    return this.hasTransitionedToTypingRef
  }

  setHasTransitionedToTyping(value: boolean) {
    this.hasTransitionedToTypingRef = value
  }

  /* ------------------------------------------------------------------ */
  /*  Cleanup                                                           */
  /* ------------------------------------------------------------------ */
  destroy(): void {
    this.stopWebcam()
    if (this.emotionPollIntervalRef) {
      clearInterval(this.emotionPollIntervalRef)
      this.emotionPollIntervalRef = null
    }
    this.emotionTimelineRef = null
    this.audioSnapshotsRef = []
  }
}
