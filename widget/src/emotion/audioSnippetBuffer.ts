import type { EmotionResult, SnippetConfig } from './types'
import { applyHighpassFilter, computeSNR } from './audioFilter'

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
  private active = false
  private config: SnippetConfig

  constructor(config: SnippetConfig) {
    this.config = config
  }

  start(stream: MediaStream): void {
    if (this.active) return
    try {
      this.audioCtx = new AudioContext()
      this.analyser = this.audioCtx.createAnalyser()
      this.analyser.fftSize = 2048
      this.source = this.audioCtx.createMediaStreamSource(stream)
      this.source.connect(this.analyser)

      this.active = true
      this.features = []

      // Collect RMS + ZCR every 100ms on filtered samples
      this.intervalId = setInterval(() => {
        if (!this.analyser || !this.active) return
        const bufferLength = this.analyser.fftSize
        const dataArray = new Float32Array(bufferLength)
        this.analyser.getFloatTimeDomainData(dataArray)

        // Apply highpass filter to remove low-frequency rumble (HVAC, fans)
        const filtered = applyHighpassFilter(dataArray, this.audioCtx!.sampleRate, 80)

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
  }

  isActive(): boolean {
    return this.active
  }

  async captureSnapshot(): Promise<EmotionResult | null> {
    if (this.features.length < 10) {
      return null
    }

    // Estimate noise floor from collected RMS features and apply SNR gate
    const rmsValues = this.features.map((f) => f.rms)
    const sortedRms = [...rmsValues].sort((a, b) => a - b)
    const noiseFloorRMS = sortedRms[Math.floor(sortedRms.length * 0.1)] || 0.0001
    const signalRMS = rmsValues.reduce((a, b) => a + b, 0) / rmsValues.length
    const snrDb = computeSNR(signalRMS, noiseFloorRMS)

    const minSnrDb = this.config.minSnrDb ?? 10
    if (snrDb < minSnrDb) {
      console.info('[Emotion] Audio SNR too low:', snrDb.toFixed(1), 'dB <', minSnrDb, 'dB')
      return null
    }

    // Simple heuristic-based prosody emotion detection
    // High RMS + high ZCR often correlate with anger/stress
    // Low RMS + low ZCR often correlate with sadness
    // Very high RMS + moderate ZCR = urgent/anxious

    const avgRms = this.features.reduce((a, b) => a + b.rms, 0) / this.features.length
    const avgZcr = this.features.reduce((a, b) => a + b.zcr, 0) / this.features.length
    const maxRms = Math.max(...this.features.map((f) => f.rms))
    const rmsVariance = this.features.reduce((a, b) => a + Math.pow(b.rms - avgRms, 2), 0) / this.features.length

    // Normalize thresholds (heuristic) – lowered slightly because highpass removes some energy
    const isHighEnergy = maxRms > 0.12
    const isVeryHighEnergy = maxRms > 0.25
    const isHighZcr = avgZcr > 0.12
    const isVariable = rmsVariance > 0.004
    const isLowEnergy = avgRms < 0.015

    let primaryEmotion = 'neutral'
    let confidence = 0.3
    let intensity: 'low' | 'medium' | 'high' = 'low'

    if (isVeryHighEnergy && isHighZcr) {
      primaryEmotion = 'angry'
      confidence = Math.min(maxRms * 2.5, 0.9)
      intensity = 'high'
    } else if (isHighEnergy && isHighZcr && isVariable) {
      primaryEmotion = 'frustrated'
      confidence = Math.min(maxRms * 2.0, 0.75)
      intensity = 'medium'
    } else if (isHighEnergy && !isHighZcr) {
      primaryEmotion = 'urgent'
      confidence = Math.min(maxRms * 2.0, 0.7)
      intensity = 'high'
    } else if (isLowEnergy && !isHighZcr) {
      primaryEmotion = 'sad'
      confidence = Math.min(0.5 + (0.015 - avgRms) * 20, 0.7)
      intensity = 'medium'
    } else if (isVariable && avgRms > 0.025) {
      primaryEmotion = 'anxious'
      confidence = Math.min(rmsVariance * 40, 0.65)
      intensity = 'medium'
    }

    if (confidence < 0.3) {
      return null // Too weak to report
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
