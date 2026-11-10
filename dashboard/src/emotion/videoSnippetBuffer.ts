import type { EmotionResult, SnippetConfig } from './types'
import { FaceDetector } from './faceDetector'
import { FaceDetectorMediaPipe } from './faceDetectorMediaPipe'
import { ExpressionClassifier } from './expressionClassifier'
import { aggregateFrameResults } from './emotionAggregator'

type FaceDetectorUnion = FaceDetector | FaceDetectorMediaPipe

export interface VideoSnippetBuffer {
  start(): Promise<void>
  stop(): void
  captureSnapshot(): Promise<EmotionResult | null>
  isActive(): boolean
}

interface BufferEntry {
  imageData: ImageData
  timestamp: number
  faceResults: EmotionResult[]
}

class SnippetBufferImpl implements VideoSnippetBuffer {
  private video: HTMLVideoElement | null = null
  private stream: MediaStream | null = null
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private intervalId: ReturnType<typeof setInterval> | null = null
  private rollingBuffer: BufferEntry[] = []
  private config: SnippetConfig
  private active = false
  private faceDetector: FaceDetectorUnion
  private expressionClassifier: ExpressionClassifier
  private modelsLoaded = false
  private isSampling = false
  private stopPromise: Promise<void> | null = null
  private stopResolve: (() => void) | null = null
  private abortController: AbortController | null = null

  constructor(config: SnippetConfig) {
    this.config = config
    this.canvas = document.createElement('canvas')
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })!
    // Default to MediaPipe; fallback to ONNX if explicitly configured
    this.faceDetector = (config.faceDetectorType === 'onnx')
      ? new FaceDetector()
      : new FaceDetectorMediaPipe()
    this.expressionClassifier = new ExpressionClassifier()
  }

  async start(): Promise<void> {
    if (this.active) return
    // Wait for any previous stop to fully complete
    if (this.stopPromise) {
      await this.stopPromise
    }
    // Clear any orphaned interval
    if (this.intervalId) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, frameRate: { ideal: 10 } },
        audio: false,
      })
      this.video = document.createElement('video')
      this.video.srcObject = this.stream
      this.video.play()
      this.video.muted = true
      this.video.playsInline = true

      // Wait for video to be ready
      await new Promise<void>((resolve) => {
        if (!this.video) return
        this.video.onloadedmetadata = () => resolve()
        setTimeout(resolve, 1000)
      })

      // Load face detector and expression classifier
      const faceLoad = (this.config.faceDetectorType === 'onnx')
        ? this.faceDetector.load(this.config.faceModelUrl)
        : (this.faceDetector as FaceDetectorMediaPipe).load()
      const [faceOk, exprOk] = await Promise.all([
        faceLoad,
        this.expressionClassifier.load({
          modelType: this.config.expressionModelType ?? 'affectnet',
          inputSize: this.config.expressionInputSize ?? 224,
          modelUrl: this.config.expressionModelUrl,
          executionProviders: this.config.executionProviders,
        }),
      ])
      this.modelsLoaded = faceOk && exprOk
      if (!this.modelsLoaded) {
        console.warn('[Emotion] ONNX models failed to load, falling back to placeholder')
      } else {
        console.info('[Emotion] Models loaded. Running diagnostic tests...')
        await this.expressionClassifier.runDiagnosticTests()
      }

      this.active = true
      const sampleIntervalMs = 1000 / this.config.sampleFps
      this.intervalId = setInterval(() => this._sampleFrame(), sampleIntervalMs)
    } catch (err) {
      console.warn('[Emotion] Webcam access failed:', err)
      this.active = false
    }
  }

  stop(): void {
    this.active = false
    this.modelsLoaded = false
    // Abort any in-flight detection immediately
    if (this.abortController) {
      this.abortController.abort()
      this.abortController = null
    }
    if (this.intervalId) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop())
      this.stream = null
    }
    if (this.video) {
      this.video.pause()
      this.video.srcObject = null
      this.video = null
    }
    this.rollingBuffer = []
    this.isSampling = false

    // If sampling is still running, create a promise that resolves when it finishes
    if (this.isSampling) {
      this.stopPromise = new Promise((resolve) => { this.stopResolve = resolve })
    }
  }

  isActive(): boolean {
    return this.active
  }

  private _cropFace(imageData: ImageData, box: { x: number; y: number; w: number; h: number }): ImageData {
    const srcCanvas = document.createElement('canvas')
    srcCanvas.width = imageData.width
    srcCanvas.height = imageData.height
    const srcCtx = srcCanvas.getContext('2d')!
    srcCtx.putImageData(imageData, 0, 0)

    // Add margin around face to match EmotiEffLib training preprocessing
    const marginRatio = 0.2
    const marginX = Math.round(box.w * marginRatio)
    const marginY = Math.round(box.h * marginRatio)
    const x = Math.max(0, box.x - marginX)
    const y = Math.max(0, box.y - marginY)
    const w = Math.min(imageData.width - x, box.w + marginX * 2)
    const h = Math.min(imageData.height - y, box.h + marginY * 2)

    const cropCanvas = document.createElement('canvas')
    cropCanvas.width = w
    cropCanvas.height = h
    const cropCtx = cropCanvas.getContext('2d')!
    cropCtx.drawImage(srcCanvas, x, y, w, h, 0, 0, w, h)
    return cropCtx.getImageData(0, 0, w, h)
  }

  private async _sampleFrame(): Promise<void> {
    if (!this.video || !this.active || this.video.readyState < 2 || this.isSampling) return
    this.isSampling = true
    this.abortController = new AbortController()
    try {
      const w = this.video.videoWidth || 320
      const h = this.video.videoHeight || 240
      this.canvas.width = w
      this.canvas.height = h
      this.ctx.drawImage(this.video, 0, 0, w, h)
      const imageData = this.ctx.getImageData(0, 0, w, h)

      let faceResults: EmotionResult[] = []

      if (this.modelsLoaded && this.active) {
        try {
          const faces = await this.faceDetector.detect(imageData)
          if (!this.active) return
          for (const face of faces) {
            const cropped = this._cropFace(imageData, face)
            const expr = await this.expressionClassifier.classify(cropped)
            if (!this.active) return

            // Skip weak neutral predictions
            if (expr.emotion === 'neutral' && expr.confidence < 0.4) {
              continue
            }

            // Determine intensity from confidence (now proper probabilities after softmax)
            let intensity: 'low' | 'medium' | 'high' = 'low'
            if (expr.confidence >= 0.65) intensity = 'high'
            else if (expr.confidence >= 0.35) intensity = 'medium'

            faceResults.push({
              primary_emotion: expr.emotion,
              confidence: expr.confidence,
              intensity,
              source: 'webcam_browser_onnx',
              timestamp_ms: Date.now(),
              valid_face_frames: 1,
              sampled_frames: 1,
              face_detection_rate: 1.0,
            })
          }
        } catch (err) {
          if (!this.active) return
          console.warn('[Emotion] Frame analysis failed:', err)
        }
      }

      this.rollingBuffer.push({ imageData, timestamp: Date.now(), faceResults })

      // Keep only the last windowMs worth of frames
      const cutoff = Date.now() - this.config.windowMs
      while (this.rollingBuffer.length > 0 && this.rollingBuffer[0].timestamp < cutoff) {
        this.rollingBuffer.shift()
      }
      // Also cap at maxFrames
      while (this.rollingBuffer.length > this.config.maxFrames) {
        this.rollingBuffer.shift()
      }
    } finally {
      this.isSampling = false
      if (this.stopResolve) {
        this.stopResolve()
        this.stopResolve = null
        this.stopPromise = null
      }
    }
  }

  async captureSnapshot(): Promise<EmotionResult | null> {
    // Trigger one more analysis if buffer is empty or models not loaded yet
    if (this.rollingBuffer.length === 0 && this.video && this.active) {
      const w = this.video.videoWidth || 320
      const h = this.video.videoHeight || 240
      this.canvas.width = w
      this.canvas.height = h
      this.ctx.drawImage(this.video, 0, 0, w, h)
      const imageData = this.ctx.getImageData(0, 0, w, h)

      let faceResults: EmotionResult[] = []
      if (this.modelsLoaded) {
        try {
          const faces = await this.faceDetector.detect(imageData)
          for (const face of faces) {
            const cropped = this._cropFace(imageData, face)
            const expr = await this.expressionClassifier.classify(cropped)
            if (expr.emotion === 'neutral' && expr.confidence < 0.4) {
              continue
            }
            let intensity: 'low' | 'medium' | 'high' = 'low'
            if (expr.confidence >= 0.65) intensity = 'high'
            else if (expr.confidence >= 0.35) intensity = 'medium'
            faceResults.push({
              primary_emotion: expr.emotion,
              confidence: expr.confidence,
              intensity,
              source: 'webcam_browser_onnx',
              timestamp_ms: Date.now(),
              valid_face_frames: 1,
              sampled_frames: 1,
              face_detection_rate: 1.0,
            })
          }
        } catch (err) {
          console.warn('[Emotion] Immediate capture failed:', err)
        }
      }
      this.rollingBuffer.push({ imageData, timestamp: Date.now(), faceResults })
    }

    const allFaceResults = this.rollingBuffer.flatMap((b) => b.faceResults)

    if (allFaceResults.length > 0) {
      return aggregateFrameResults(allFaceResults)
    }

    // Fallback: models not loaded or no faces detected
    const validFrames = this.rollingBuffer.length
    if (validFrames < this.config.minValidFaceFrames) {
      return {
        primary_emotion: 'unknown',
        confidence: 0.0,
        intensity: 'low',
        source: 'webcam_browser_onnx',
        timestamp_ms: Date.now(),
        sampled_frames: validFrames,
        valid_face_frames: 0,
        face_detection_rate: 0,
        window_ms: this.config.windowMs,
      }
    }

    return {
      primary_emotion: 'neutral',
      confidence: 0.0,
      intensity: 'low',
      source: 'webcam_browser_onnx',
      timestamp_ms: Date.now(),
      sampled_frames: validFrames,
      valid_face_frames: 0,
      face_detection_rate: 0,
      window_ms: this.config.windowMs,
    }
  }
}

export function createVideoSnippetBuffer(config: SnippetConfig): VideoSnippetBuffer {
  return new SnippetBufferImpl(config)
}
