import type { EmotionResult, SnippetConfig } from './types'
import { AudioEmotionClassifier } from './audioEmotionClassifier'
import { applyHighpassFilter, estimateNoiseFloor, applyNoiseGate, computeSNR } from './audioFilter'

export interface AudioSnippetBuffer {
  start(stream: MediaStream): void
  stop(): void
  captureSnapshot(): Promise<EmotionResult | null>
  isActive(): boolean
}

interface AudioFeatures {
  rms: number
  zcr: number
  timestamp: number
}

class AudioSnippetBufferImpl implements AudioSnippetBuffer {
  private audioCtx: AudioContext | null = null
  private analyser: AnalyserNode | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private intervalId: ReturnType<typeof setInterval> | null = null
  private features: AudioFeatures[] = []
  private rawSamples: Float32Array[] = []
  private active = false
  private config: SnippetConfig
  private classifier: AudioEmotionClassifier
  private sampleRate: number = 44100
  private modelLoaded = false

  constructor(config: SnippetConfig) {
    this.config = config
    this.classifier = new AudioEmotionClassifier(undefined, config.noiseGateMin, config.noiseGateMax)
  }

  start(stream: MediaStream): void {
    if (this.active) return
    try {
      this.audioCtx = new AudioContext()
      this.sampleRate = this.audioCtx.sampleRate
      this.analyser = this.audioCtx.createAnalyser()
      this.analyser.fftSize = 2048
      this.source = this.audioCtx.createMediaStreamSource(stream)
      this.source.connect(this.analyser)

      this.active = true
      this.features = []
      this.rawSamples = []
      this.modelLoaded = false

      // Attempt to load ONNX audio emotion model
      const modelUrl = this.config.expressionModelUrl
      if (modelUrl) {
        this.classifier.load(modelUrl).then((ok) => {
          this.modelLoaded = ok
          if (ok) {
            console.info('[Emotion] Audio ONNX model loaded:', modelUrl)
          } else {
            console.warn('[Emotion] Audio ONNX model failed to load, using heuristic fallback')
          }
        }).catch((err) => {
          this.modelLoaded = false
          console.warn('[Emotion] Audio ONNX model load threw error:', err)
        })
      }

      // Collect RMS + ZCR every 100ms, also accumulate filtered samples
      this.intervalId = setInterval(() => {
        if (!this.analyser || !this.active) return
        const bufferLength = this.analyser.fftSize
        const dataArray = new Float32Array(bufferLength)
        this.analyser.getFloatTimeDomainData(dataArray)

        // Apply highpass filter to remove low-frequency rumble (HVAC, fans)
        const filtered = applyHighpassFilter(dataArray, this.sampleRate, 80)

        // Accumulate filtered samples for ONNX inference
        this.rawSamples.push(new Float32Array(filtered))
        // Keep roughly last 3 seconds of audio
        const maxRawBuffers = Math.ceil(3000 / 100)
        while (this.rawSamples.length > maxRawBuffers) {
          this.rawSamples.shift()
        }

        // RMS Energy on filtered signal
        const sumSq = filtered.reduce((acc, val) => acc + val * val, 0)
        const rms = Math.sqrt(sumSq / bufferLength)

        // Zero-Crossing Rate on filtered signal
        let zcr = 0
        for (let i = 1; i < bufferLength; i++) {
          if (filtered[i - 1] * filtered[i] < 0) {
            zcr++
          }
        }
        zcr /= bufferLength

        this.features.push({ rms, zcr, timestamp: Date.now() })
      }, 100)
    } catch (err) {
      console.warn('[Emotion] Audio emotion start failed:', err)
      this.active = false
    }
  }

  stop(): void {
    this.active = false
    this.modelLoaded = false
    if (this.intervalId) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }
    if (this.source) {
      this.source.disconnect()
      this.source = null
    }
    if (this.analyser) {
      this.analyser.disconnect()
      this.analyser = null
    }
    if (this.audioCtx && this.audioCtx.state !== 'closed') {
      this.audioCtx.close()
      this.audioCtx = null
    }
    this.features = []
    this.rawSamples = []
  }

  isActive(): boolean {
    return this.active
  }

  async captureSnapshot(): Promise<EmotionResult | null> {
    // Try ONNX model first if loaded
    if (this.modelLoaded && this.classifier.isReady() && this.rawSamples.length > 0) {
      try {
        // Concatenate all filtered samples into one buffer
        const totalSamples = this.rawSamples.reduce((sum, buf) => sum + buf.length, 0)
        const combined = new Float32Array(totalSamples)
        let offset = 0
        for (const buf of this.rawSamples) {
          combined.set(buf, offset)
          offset += buf.length
        }

        // Estimate noise floor and apply noise gate on the combined buffer
        const noiseFloorRMS = estimateNoiseFloor(combined)
        const gated = applyNoiseGate(
          combined, noiseFloorRMS, 2.5, 5, 50, this.sampleRate,
          this.config.noiseGateMin, this.config.noiseGateMax,
        )

        // Compute SNR from gated signal
        let sumSq = 0
        for (let i = 0; i < gated.length; i++) {
          sumSq += gated[i] * gated[i]
        }
        const signalRMS = Math.sqrt(sumSq / gated.length)
        const snrDb = computeSNR(signalRMS, noiseFloorRMS)

        const minSnrDb = this.config.minSnrDb ?? 10
        if (snrDb < minSnrDb) {
          console.info('[Emotion] Audio SNR too low:', snrDb.toFixed(1), 'dB <', minSnrDb, 'dB')
          return null
        }

        const result = await this.classifier.classify(gated, this.sampleRate)
        console.info('[Emotion] ONNX audio classify result:', result)
        if (result.emotion !== 'unknown' && result.confidence >= 0.2) {
          let intensity: 'low' | 'medium' | 'high' = 'low'
          if (result.confidence >= 0.7) intensity = 'high'
          else if (result.confidence >= 0.4) intensity = 'medium'

          const validFrames = this.features.length
          return {
            primary_emotion: result.emotion,
            confidence: result.confidence,
            intensity,
            source: 'audio_prosody_browser_onnx',
            timestamp_ms: Date.now(),
            sampled_frames: validFrames,
            valid_face_frames: validFrames,
            face_detection_rate: 1.0,
            window_ms: this.config.windowMs,
          }
        } else {
          console.warn('[Emotion] ONNX classify returned unknown or low confidence:', result)
        }
      } catch (err) {
        console.warn('[Emotion] ONNX audio inference failed:', err)
      }
    } else {
      console.info('[Emotion] ONNX not used: modelLoaded=', this.modelLoaded, 'isReady=', this.classifier.isReady(), 'rawSamples=', this.rawSamples.length)
    }

    // Fallback: heuristic-based prosody emotion detection
    if (this.features.length < 1) {
      console.info('[Emotion] Heuristic fallback skipped: features.length=', this.features.length)
      return null
    }

    const avgRms = this.features.reduce((a, b) => a + b.rms, 0) / this.features.length
    const avgZcr = this.features.reduce((a, b) => a + b.zcr, 0) / this.features.length
    const maxRms = Math.max(...this.features.map((f) => f.rms))
    const rmsVariance = this.features.reduce((a, b) => a + Math.pow(b.rms - avgRms, 2), 0) / this.features.length

    const isHighEnergy = maxRms > 0.025
    const isVeryHighEnergy = maxRms > 0.07
    const isHighZcr = avgZcr > 0.025
    const isVariable = rmsVariance > 0.0004
    const isLowEnergy = avgRms < 0.004

    let primaryEmotion = 'neutral'
    let confidence = 0.2
    let intensity: 'low' | 'medium' | 'high' = 'low'

    if (isVeryHighEnergy && isHighZcr) {
      primaryEmotion = 'angry'
      confidence = Math.min(maxRms * 8, 0.9)
      intensity = 'high'
    } else if (isHighEnergy && isHighZcr && isVariable) {
      primaryEmotion = 'frustrated'
      confidence = Math.min(maxRms * 6, 0.75)
      intensity = 'medium'
    } else if (isHighEnergy && !isHighZcr) {
      primaryEmotion = 'urgent'
      confidence = Math.min(maxRms * 6, 0.7)
      intensity = 'high'
    } else if (isLowEnergy && !isHighZcr) {
      primaryEmotion = 'sad'
      confidence = Math.min(0.5 + (0.005 - avgRms) * 40, 0.7)
      intensity = 'medium'
    } else if (isVariable && avgRms > 0.01) {
      primaryEmotion = 'anxious'
      confidence = Math.min(rmsVariance * 400, 0.65)
      intensity = 'medium'
    }

    if (confidence < 0.2) {
      return null
    }

    const validFrames = this.features.length
    const sampledFrames = validFrames

    return {
      primary_emotion: primaryEmotion,
      confidence: Math.round(confidence * 100) / 100,
      intensity,
      source: 'audio_prosody_browser',
      timestamp_ms: Date.now(),
      sampled_frames: sampledFrames,
      valid_face_frames: validFrames,
      face_detection_rate: 1.0,
      window_ms: this.config.windowMs,
    }
  }
}

export function createAudioSnippetBuffer(config: SnippetConfig): AudioSnippetBuffer {
  return new AudioSnippetBufferImpl(config)
}
