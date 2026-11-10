import * as ort from 'onnxruntime-web'
import { estimateNoiseFloor, applyNoiseGate } from './audioFilter'

// Configure ONNX Runtime WASM for model compatibility
ort.env.wasm.numThreads = 1
ort.env.wasm.simd = true

export interface AudioEmotionResult {
  emotion: string
  confidence: number
}

// Common speech emotion labels mapped to our taxonomy
const DEFAULT_LABEL_MAP: Record<string, string> = {
  neutral: 'neutral',
  happy: 'satisfied',
  sad: 'sad',
  angry: 'angry',
  fear: 'anxious',
  disgust: 'frustrated',
  surprised: 'surprised',
  // Variants
  happiness: 'satisfied',
  sadness: 'sad',
  anger: 'angry',
  anxiety: 'anxious',
  frustration: 'frustrated',
  calm: 'neutral',
  excited: 'satisfied',
  bored: 'neutral',
}

export class AudioEmotionClassifier {
  private session: ort.InferenceSession | null = null
  private modelUrl: string | null = null
  private inferring = false
  private labelMap: Record<string, string>
  private expectedInputShape: number[] | null = null
  private inputType: 'spectrogram' | 'raw' | 'features' | null = null
  private noiseGateMin?: number
  private noiseGateMax?: number

  constructor(labelMap?: Record<string, string>, noiseGateMin?: number, noiseGateMax?: number) {
    this.labelMap = labelMap || DEFAULT_LABEL_MAP
    this.noiseGateMin = noiseGateMin
    this.noiseGateMax = noiseGateMax
  }

  async load(modelUrl: string): Promise<boolean> {
    if (this.session && this.modelUrl === modelUrl) return true
    try {
      const eps = ['webgpu', 'wasm'].filter((p) => {
        if (p === 'webgpu' && 'gpu' in navigator) return true
        return p === 'wasm'
      })
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: eps.length > 0 ? eps : ['wasm'],
      })
      this.modelUrl = modelUrl

      // Inspect input shape to determine preprocessing strategy
      const inputMeta = this.session.inputMetadata
      const firstInput = Object.values(inputMeta)[0]
      if (firstInput && 'dims' in firstInput) {
        this.expectedInputShape = (firstInput as any).dims as number[]
        const dims = this.expectedInputShape.length
        if (dims === 4) {
          this.inputType = 'spectrogram' // [1, 1, freq, time] or [1, freq, time, 1]
        } else if (dims === 3) {
          this.inputType = 'spectrogram' // [1, time, freq]
        } else if (dims === 2) {
          this.inputType = 'features' // [1, feature_count]
        } else if (dims === 1) {
          this.inputType = 'raw' // [time]
        }
      }

      console.info('[AudioEmotion] Model loaded, input shape:', this.expectedInputShape, 'type:', this.inputType)
      return true
    } catch (err) {
      console.warn('[AudioEmotion] Model load failed:', err)
      this.session = null
      this.modelUrl = null
      this.expectedInputShape = null
      this.inputType = null
      return false
    }
  }

  isReady(): boolean {
    return this.session !== null
  }

  async classify(samples: Float32Array, sampleRate: number): Promise<AudioEmotionResult> {
    if (!this.session || this.inferring) {
      return { emotion: 'unknown', confidence: 0 }
    }
    this.inferring = true

    try {
      // Early abort if signal is too weak to contain meaningful speech
      let maxAmp = 0
      for (let i = 0; i < samples.length; i++) {
        const a = Math.abs(samples[i])
        if (a > maxAmp) maxAmp = a
      }
      if (maxAmp < 0.005) {
        return { emotion: 'unknown', confidence: 0 }
      }

      const tensor = this._preprocess(samples, sampleRate)
      if (!tensor) {
        return { emotion: 'unknown', confidence: 0 }
      }

      const inputName = this.session.inputNames[0] || 'input'
      const feeds: Record<string, ort.Tensor> = { [inputName]: tensor }
      const results = await this.session.run(feeds)

      // Try common output names
      const output =
        results.output ||
        results['output0'] ||
        results['probabilities'] ||
        results['scores'] ||
        results['logits'] ||
        Object.values(results)[0]

      if (!output || !(output instanceof ort.Tensor)) {
        console.warn('[AudioEmotion] No valid output tensor')
        return { emotion: 'unknown', confidence: 0 }
      }

      const data = output.data as Float32Array
      if (!data || data.length === 0) {
        return { emotion: 'unknown', confidence: 0 }
      }

      // Find argmax
      let maxIdx = 0
      let maxVal = data[0]
      for (let i = 1; i < data.length; i++) {
        if (data[i] > maxVal) {
          maxVal = data[i]
          maxIdx = i
        }
      }

      // If output looks like logits (can be negative), apply softmax
      let confidence = maxVal
      if (confidence < 0 || confidence > 1.01) {
        const expVals = Array.from(data).map((v) => Math.exp(v - Math.max(...data)))
        const sumExp = expVals.reduce((a, b) => a + b, 0)
        confidence = expVals[maxIdx] / sumExp
      }

      confidence = Math.round(Math.min(Math.max(confidence, 0), 1) * 100) / 100

      // Try to map index to label
      // Many models don't provide label metadata, so we use index-based mapping
      const knownLabels = Object.keys(this.labelMap)
      let rawLabel: string
      if (maxIdx < knownLabels.length) {
        rawLabel = knownLabels[maxIdx]
      } else {
        rawLabel = `class_${maxIdx}`
      }

      const mappedEmotion = this.labelMap[rawLabel] || rawLabel
      return { emotion: mappedEmotion, confidence }
    } catch (err) {
      console.warn('[AudioEmotion] Inference failed:', err)
      return { emotion: 'unknown', confidence: 0 }
    } finally {
      this.inferring = false
    }
  }

  private _preprocess(samples: Float32Array, sampleRate: number): ort.Tensor | null {
    if (!this.session || !this.expectedInputShape) return null

    try {
      // Normalize to [-1, 1]
      let normalized = new Float32Array(samples.length)
      let localMaxAmp = 0
      for (let i = 0; i < samples.length; i++) {
        const a = Math.abs(samples[i])
        if (a > localMaxAmp) localMaxAmp = a
      }
      const scale = localMaxAmp > 0 ? 1 / localMaxAmp : 1
      for (let i = 0; i < samples.length; i++) {
        normalized[i] = samples[i] * scale
      }

      // Additional noise-gate defense: suppress anything near the noise floor
      const noiseFloorRMS = estimateNoiseFloor(normalized)
      const gated = applyNoiseGate(normalized, noiseFloorRMS, 2.5, 5, 50, sampleRate, this.noiseGateMin, this.noiseGateMax)

      if (this.inputType === 'spectrogram') {
        return this._makeSpectrogramTensor(gated, sampleRate)
      } else if (this.inputType === 'features') {
        return this._makeFeatureTensor(gated, sampleRate)
      } else if (this.inputType === 'raw') {
        return this._makeRawTensor(gated)
      }

      // Unknown input type: try spectrogram as default
      return this._makeSpectrogramTensor(gated, sampleRate)
    } catch (err) {
      console.warn('[AudioEmotion] Preprocessing failed:', err)
      return null
    }
  }

  private _makeRawTensor(samples: Float32Array): ort.Tensor | null {
    if (!this.expectedInputShape) return null
    const targetLen = this.expectedInputShape[this.expectedInputShape.length - 1]
    const resized = this._resize1D(samples, targetLen)
    return new ort.Tensor('float32', resized, this.expectedInputShape)
  }

  private _makeFeatureTensor(samples: Float32Array, sampleRate: number): ort.Tensor | null {
    if (!this.expectedInputShape) return null
    const featureCount = this.expectedInputShape[this.expectedInputShape.length - 1]

    // Compute basic audio features
    const features = this._extractFeatures(samples, sampleRate)

    // Pad or truncate to expected feature count
    const out = new Float32Array(featureCount)
    const len = Math.min(features.length, featureCount)
    out.set(features.subarray(0, len))

    const shape = [...this.expectedInputShape]
    shape[shape.length - 1] = featureCount
    return new ort.Tensor('float32', out, shape)
  }

  private _makeSpectrogramTensor(samples: Float32Array, sampleRate: number): ort.Tensor | null {
    if (!this.expectedInputShape) return null

    // Compute a simple mel-like spectrogram
    const fftSize = 512
    const hopLength = 256
    const numMels = 64
    const windowDuration = 2.0 // seconds
    const targetSamples = Math.min(samples.length, Math.floor(windowDuration * sampleRate))
    const clipped = samples.subarray(0, targetSamples)

    const numFrames = Math.floor((clipped.length - fftSize) / hopLength) + 1
    if (numFrames <= 0) return null

    // Simple STFT magnitude spectrogram
    const spec: number[][] = []
    for (let f = 0; f < numFrames; f++) {
      const frame = new Float32Array(fftSize)
      for (let i = 0; i < fftSize; i++) {
        const idx = f * hopLength + i
        frame[i] = idx < clipped.length ? clipped[idx] * this._hannWindow(i, fftSize) : 0
      }
      const magnitudes = this._fftMagnitude(frame)
      spec.push(magnitudes)
    }

    // Determine expected shape from model
    const shape = [...this.expectedInputShape]
    // Most common shapes for audio models:
    // [1, 1, numMels, numFrames]
    // [1, numFrames, numMels]
    // [1, numMels, numFrames]

    const expectedFrames = shape.includes(numFrames) ? numFrames : (shape[shape.length - 1] || numFrames)
    const expectedMels = shape.includes(numMels) ? numMels : (shape[shape.length - 2] || numMels)

    // Resize spectrogram
    const resizedSpec = this._resize2D(spec, expectedMels, expectedFrames)

    // Normalize
    let maxVal = 0
    for (let i = 0; i < resizedSpec.length; i++) {
      for (let j = 0; j < resizedSpec[i].length; j++) {
        if (resizedSpec[i][j] > maxVal) maxVal = resizedSpec[i][j]
      }
    }
    const normScale = maxVal > 0 ? 1 / maxVal : 1

    // Build flat array based on shape
    const totalSize = shape.reduce((a, b) => a * b, 1)
    const flat = new Float32Array(totalSize)

    // Common layout: [batch=1, channels=1, mels, frames]
    if (shape.length === 4) {
      for (let m = 0; m < expectedMels; m++) {
        for (let f = 0; f < expectedFrames; f++) {
          flat[f + m * expectedFrames] = (resizedSpec[m]?.[f] || 0) * normScale
        }
      }
    } else if (shape.length === 3) {
      // [batch, frames, mels] or [batch, mels, frames]
      if (shape[1] === expectedFrames) {
        // [1, frames, mels]
        for (let f = 0; f < expectedFrames; f++) {
          for (let m = 0; m < expectedMels; m++) {
            flat[f * expectedMels + m] = (resizedSpec[m]?.[f] || 0) * normScale
          }
        }
      } else {
        // [1, mels, frames]
        for (let m = 0; m < expectedMels; m++) {
          for (let f = 0; f < expectedFrames; f++) {
            flat[m * expectedFrames + f] = (resizedSpec[m]?.[f] || 0) * normScale
          }
        }
      }
    } else if (shape.length === 2) {
      for (let m = 0; m < expectedMels; m++) {
        for (let f = 0; f < expectedFrames; f++) {
          flat[m * expectedFrames + f] = (resizedSpec[m]?.[f] || 0) * normScale
        }
      }
    }

    return new ort.Tensor('float32', flat, shape)
  }

  private _hannWindow(n: number, size: number): number {
    return 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (size - 1))
  }

  private _fftMagnitude(frame: Float32Array): number[] {
    // Very simple DFT magnitude for first 256 bins
    const N = frame.length
    const magnitudes: number[] = []
    for (let k = 0; k < N / 2; k++) {
      let real = 0
      let imag = 0
      for (let n = 0; n < N; n++) {
        const angle = (-2 * Math.PI * k * n) / N
        real += frame[n] * Math.cos(angle)
        imag += frame[n] * Math.sin(angle)
      }
      magnitudes.push(Math.sqrt(real * real + imag * imag))
    }
    return magnitudes
  }

  private _resize1D(src: Float32Array, targetLen: number): Float32Array {
    if (src.length === targetLen) return src
    const out = new Float32Array(targetLen)
    const ratio = src.length / targetLen
    for (let i = 0; i < targetLen; i++) {
      const srcIdx = Math.floor(i * ratio)
      out[i] = src[Math.min(srcIdx, src.length - 1)]
    }
    return out
  }

  private _resize2D(src: number[][], targetRows: number, targetCols: number): number[][] {
    const out: number[][] = []
    for (let r = 0; r < targetRows; r++) {
      const srcRow = src[Math.floor((r / targetRows) * src.length)] || []
      const row: number[] = []
      for (let c = 0; c < targetCols; c++) {
        row.push(srcRow[Math.floor((c / targetCols) * srcRow.length)] || 0)
      }
      out.push(row)
    }
    return out
  }

  private _extractFeatures(samples: Float32Array, sampleRate: number): Float32Array {
    // Basic audio features: RMS, ZCR, spectral centroid approx, duration
    const frameSize = 512
    const hop = 256
    const numFrames = Math.floor((samples.length - frameSize) / hop) + 1

    const rmsValues: number[] = []
    const zcrValues: number[] = []
    const spectralCentroids: number[] = []

    for (let f = 0; f < numFrames; f++) {
      const frame = new Float32Array(frameSize)
      for (let i = 0; i < frameSize; i++) {
        frame[i] = samples[f * hop + i] || 0
      }

      // RMS
      const rms = Math.sqrt(frame.reduce((a, b) => a + b * b, 0) / frameSize)
      rmsValues.push(rms)

      // ZCR
      let zcr = 0
      for (let i = 1; i < frameSize; i++) {
        if (frame[i - 1] * frame[i] < 0) zcr++
      }
      zcrValues.push(zcr / frameSize)

      // Spectral centroid approximation using simple FFT
      const mags = this._fftMagnitude(frame)
      let sumMag = 0
      let sumWeighted = 0
      for (let i = 0; i < mags.length; i++) {
        sumMag += mags[i]
        sumWeighted += i * mags[i]
      }
      spectralCentroids.push(sumMag > 0 ? sumWeighted / sumMag : 0)
    }

    // Aggregate stats per feature
    const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length
    const std = (arr: number[]) => {
      const m = mean(arr)
      return Math.sqrt(arr.reduce((a, b) => a + (b - m) * (b - m), 0) / arr.length)
    }
    const max = (arr: number[]) => Math.max(...arr)
    const min = (arr: number[]) => Math.min(...arr)

    return new Float32Array([
      mean(rmsValues),
      std(rmsValues),
      max(rmsValues),
      min(rmsValues),
      mean(zcrValues),
      std(zcrValues),
      max(zcrValues),
      min(zcrValues),
      mean(spectralCentroids),
      std(spectralCentroids),
      max(spectralCentroids),
      min(spectralCentroids),
      samples.length / sampleRate, // duration
      sampleRate,
    ])
  }
}
