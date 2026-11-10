// Configure ONNX Runtime WASM for model compatibility
type ORT = typeof import('onnxruntime-web')
const ort = (): ORT => (window as any).ort
;(() => {
  const o = (window as any).ort as ORT | undefined
  if (o) {
    o.env.wasm.numThreads = 1
    o.env.wasm.simd = true
  }
})()

type InferenceSession = import('onnxruntime-web').InferenceSession
type Tensor = import('onnxruntime-web').Tensor

export interface ExpressionResult {
  emotion: string
  confidence: number
}

const FER_LABELS = [
  'neutral',
  'happy',
  'sad',
  'surprised',
  'fear',
  'disgust',
  'angry',
]

// Map FER labels to our emotion taxonomy
const LABEL_MAP: Record<string, string> = {
  neutral: 'neutral',
  happy: 'satisfied',
  sad: 'sad',
  surprised: 'surprised',
  fear: 'anxious',
  disgust: 'frustrated',
  angry: 'angry',
}

export class ExpressionClassifier {
  private session: InferenceSession | null = null
  private inputSize = 48 // FER+ standard size
  private modelUrl: string | null = null

  async load(modelUrl: string): Promise<boolean> {
    if (this.session && this.modelUrl === modelUrl) return true
    try {
      const eps = ['webgpu', 'wasm'].filter((p) => {
        if (p === 'webgpu' && 'gpu' in navigator) return true
        return p === 'wasm'
      })
      this.session = await ort().InferenceSession.create(modelUrl, {
        executionProviders: eps.length > 0 ? eps : ['wasm'],
      })
      this.modelUrl = modelUrl
      return true
    } catch (err) {
      console.warn('[Emotion] Expression classifier load failed:', err)
      this.session = null
      this.modelUrl = null
      return false
    }
  }

  isReady(): boolean {
    return this.session !== null
  }

  async classify(croppedFace: ImageData): Promise<ExpressionResult> {
    if (!this.session) return { emotion: 'unknown', confidence: 0 }

    // Resize to 48x48, grayscale, normalize
    const canvas = document.createElement('canvas')
    canvas.width = this.inputSize
    canvas.height = this.inputSize
    const ctx = canvas.getContext('2d')!
    const srcCanvas = document.createElement('canvas')
    srcCanvas.width = croppedFace.width
    srcCanvas.height = croppedFace.height
    const srcCtx = srcCanvas.getContext('2d')!
    srcCtx.putImageData(croppedFace, 0, 0)
    ctx.drawImage(srcCanvas, 0, 0, croppedFace.width, croppedFace.height, 0, 0, this.inputSize, this.inputSize)

    const resized = ctx.getImageData(0, 0, this.inputSize, this.inputSize)
    const grayData = new Float32Array(this.inputSize * this.inputSize)

    for (let i = 0; i < this.inputSize * this.inputSize; i++) {
      const r = resized.data[i * 4]
      const g = resized.data[i * 4 + 1]
      const b = resized.data[i * 4 + 2]
      // ITU-R BT.601 luma
      grayData[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    }

    try {
      const tensor = new (ort().Tensor)('float32', grayData, [1, 1, this.inputSize, this.inputSize])
      const feeds: Record<string, Tensor> = { input: tensor }
      const results = await this.session.run(feeds)

      // Try common output names
      const output = results.output || results['output0'] || results['probabilities'] || Object.values(results)[0]
      if (!output || !(output instanceof ort().Tensor)) {
        return { emotion: 'unknown', confidence: 0 }
      }

      const probs = output.data as Float32Array
      let maxIdx = 0
      let maxVal = probs[0]
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > maxVal) {
          maxVal = probs[i]
          maxIdx = i
        }
      }

      const rawLabel = FER_LABELS[maxIdx] || 'unknown'
      const mappedEmotion = LABEL_MAP[rawLabel] || rawLabel

      return { emotion: mappedEmotion, confidence: Math.round(maxVal * 100) / 100 }
    } catch (err) {
      console.warn('[Emotion] Expression classification inference failed:', err)
      return { emotion: 'unknown', confidence: 0 }
    }
  }
}
