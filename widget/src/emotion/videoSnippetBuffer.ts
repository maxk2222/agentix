import type { EmotionResult, SnippetConfig } from './types'
import { FaceDetector } from './faceDetector'
import { ExpressionClassifier } from './expressionClassifier'
import { aggregateFrameResults } from './emotionAggregator'

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
  private faceDetector: FaceDetector
  private expressionClassifier: ExpressionClassifier
  private modelsLoaded = false

  constructor(config: SnippetConfig) {
    this.config = config
    this.canvas = document.createElement('canvas')
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })!
    this.faceDetector = new FaceDetector()
    this.expressionClassifier = new ExpressionClassifier()
  }

  async start(): Promise<void> {
    if (this.active) return
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

      // Load ONNX models
      const [faceOk, exprOk] = await Promise.all([
        this.faceDetector.load(this.config.faceModelUrl),
        this.expressionClassifier.load(this.config.expressionModelUrl),
      ])
      this.modelsLoaded = faceOk && exprOk
      if (!this.modelsLoaded) {
        console.warn('[Emotion] ONNX models failed to load, falling back to placeholder')
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

    const cropCanvas = document.createElement('canvas')
    cropCanvas.width = box.w
    cropCanvas.height = box.h
    const cropCtx = cropCanvas.getContext('2d')!
    cropCtx.drawImage(srcCanvas, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h)
    return cropCtx.getImageData(0, 0, box.w, box.h)
  }

  private async _sampleFrame(): Promise<void> {
    if (!this.video || !this.active || this.video.readyState < 2) return
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

          // Determine intensity from confidence
          let intensity: 'low' | 'medium' | 'high' = 'low'
          if (expr.confidence >= 0.7) intensity = 'high'
          else if (expr.confidence >= 0.4) intensity = 'medium'

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
            let intensity: 'low' | 'medium' | 'high' = 'low'
            if (expr.confidence >= 0.7) intensity = 'high'
            else if (expr.confidence >= 0.4) intensity = 'medium'
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
