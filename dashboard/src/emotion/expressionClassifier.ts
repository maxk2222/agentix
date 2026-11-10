import * as ort from 'onnxruntime-web'

// Configure ONNX Runtime WASM for model compatibility
ort.env.wasm.numThreads = 1
ort.env.wasm.simd = true

export interface ExpressionResult {
  emotion: string
  confidence: number
}

export interface ClassifierConfig {
  modelType: 'fer' | 'affectnet'
  inputSize: number
  modelUrl: string
  executionProviders?: string[]
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

const AFFECTNET_LABELS = [
  'anger',
  'contempt',
  'disgust',
  'fear',
  'happiness',
  'neutral',
  'sadness',
  'surprise',
]

// Map FER labels to our emotion taxonomy
const FER_LABEL_MAP: Record<string, string> = {
  neutral: 'neutral',
  happy: 'satisfied',
  sad: 'sad',
  surprised: 'surprised',
  fear: 'anxious',
  disgust: 'frustrated',
  angry: 'angry',
}

// Map AffectNet labels to our emotion taxonomy
const AFFECTNET_LABEL_MAP: Record<string, string> = {
  anger: 'angry',
  contempt: 'frustrated',
  disgust: 'frustrated',
  fear: 'anxious',
  happiness: 'satisfied',
  neutral: 'neutral',
  sadness: 'sad',
  surprise: 'surprised',
}

function getLabels(modelType: 'fer' | 'affectnet'): string[] {
  return modelType === 'affectnet' ? AFFECTNET_LABELS : FER_LABELS
}

function getLabelMap(modelType: 'fer' | 'affectnet'): Record<string, string> {
  return modelType === 'affectnet' ? AFFECTNET_LABEL_MAP : FER_LABEL_MAP
}

export class ExpressionClassifier {
  private session: ort.InferenceSession | null = null
  private config: ClassifierConfig | null = null
  private inferring = false

  async load(config: ClassifierConfig): Promise<boolean> {
    if (this.session && this.config?.modelUrl === config.modelUrl && this.config.modelType === config.modelType) {
      return true
    }
    try {
      const eps = config.executionProviders ?? ['webgpu', 'wasm'].filter((p) => {
        if (p === 'webgpu' && 'gpu' in navigator) return true
        return p === 'wasm'
      })
      this.session = await ort.InferenceSession.create(config.modelUrl, {
        executionProviders: eps.length > 0 ? eps : ['wasm'],
      })
      this.config = config
      return true
    } catch (err) {
      console.warn('[Emotion] Expression classifier load failed:', err)
      this.session = null
      this.config = null
      return false
    }
  }

  isReady(): boolean {
    return this.session !== null && this.config !== null
  }

  private preprocessFER(croppedFace: ImageData, inputSize: number): ort.Tensor {
    const canvas = document.createElement('canvas')
    canvas.width = inputSize
    canvas.height = inputSize
    const ctx = canvas.getContext('2d')!
    const srcCanvas = document.createElement('canvas')
    srcCanvas.width = croppedFace.width
    srcCanvas.height = croppedFace.height
    const srcCtx = srcCanvas.getContext('2d')!
    srcCtx.putImageData(croppedFace, 0, 0)
    ctx.drawImage(srcCanvas, 0, 0, croppedFace.width, croppedFace.height, 0, 0, inputSize, inputSize)

    const resized = ctx.getImageData(0, 0, inputSize, inputSize)
    const grayData = new Float32Array(inputSize * inputSize)

    for (let i = 0; i < inputSize * inputSize; i++) {
      const r = resized.data[i * 4]
      const g = resized.data[i * 4 + 1]
      const b = resized.data[i * 4 + 2]
      // ITU-R BT.601 luma
      grayData[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    }

    return new ort.Tensor('float32', grayData, [1, 1, inputSize, inputSize])
  }

  private preprocessAffectNet(croppedFace: ImageData, inputSize: number): ort.Tensor {
    const canvas = document.createElement('canvas')
    canvas.width = inputSize
    canvas.height = inputSize
    const ctx = canvas.getContext('2d')!
    const srcCanvas = document.createElement('canvas')
    srcCanvas.width = croppedFace.width
    srcCanvas.height = croppedFace.height
    const srcCtx = srcCanvas.getContext('2d')!
    srcCtx.putImageData(croppedFace, 0, 0)
    ctx.drawImage(srcCanvas, 0, 0, croppedFace.width, croppedFace.height, 0, 0, inputSize, inputSize)

    const resized = ctx.getImageData(0, 0, inputSize, inputSize)
    const rgbData = new Float32Array(3 * inputSize * inputSize)
    const mean = [0.485, 0.456, 0.406]
    const std = [0.229, 0.224, 0.225]

    for (let i = 0; i < inputSize * inputSize; i++) {
      const r = resized.data[i * 4] / 255.0
      const g = resized.data[i * 4 + 1] / 255.0
      const b = resized.data[i * 4 + 2] / 255.0
      rgbData[i] = (r - mean[0]) / std[0]
      rgbData[inputSize * inputSize + i] = (g - mean[1]) / std[1]
      rgbData[2 * inputSize * inputSize + i] = (b - mean[2]) / std[2]
    }

    return new ort.Tensor('float32', rgbData, [1, 3, inputSize, inputSize])
  }

  async classify(croppedFace: ImageData): Promise<ExpressionResult> {
    if (!this.session || !this.config || this.inferring) return { emotion: 'unknown', confidence: 0 }
    this.inferring = true

    const { modelType, inputSize } = this.config
    const labels = getLabels(modelType)
    const labelMap = getLabelMap(modelType)

    try {
      const tensor = modelType === 'affectnet'
        ? this.preprocessAffectNet(croppedFace, inputSize)
        : this.preprocessFER(croppedFace, inputSize)

      const inputName = this.session.inputNames[0] || 'input'
      const feeds: Record<string, ort.Tensor> = { [inputName]: tensor }
      const results = await this.session.run(feeds)

      // Try common output names
      const output = results.output || results['output0'] || results['probabilities'] || Object.values(results)[0]
      if (!output || !(output instanceof ort.Tensor)) {
        return { emotion: 'unknown', confidence: 0 }
      }

      const logits = output.data as Float32Array
      // Convert logits to probabilities via softmax
      const maxLogit = Math.max(...Array.from(logits))
      const exps = Array.from(logits).map((v) => Math.exp(v - maxLogit))
      const sumExps = exps.reduce((a, b) => a + b, 0)
      const probs = exps.map((v) => v / sumExps)

      let maxIdx = 0
      let maxVal = probs[0]
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > maxVal) {
          maxVal = probs[i]
          maxIdx = i
        }
      }

      const rawLabel = labels[maxIdx] || 'unknown'
      const mappedEmotion = labelMap[rawLabel] || rawLabel

      // Debug: log probabilities to help verify label alignment
      console.debug('[Emotion] probs:', probs.map((v, i) => `${labels[i]}:${v.toFixed(3)}`).join(', '), '->', rawLabel, `(${mappedEmotion})`)

      return { emotion: mappedEmotion, confidence: Math.round(maxVal * 100) / 100 }
    } catch (err) {
      console.warn('[Emotion] Expression classification inference failed:', err)
      return { emotion: 'unknown', confidence: 0 }
    } finally {
      this.inferring = false
    }
  }

  // Diagnostic: run the classifier on synthetic test images to verify it works
  async runDiagnosticTests(): Promise<void> {
    if (!this.session || !this.config) {
      console.warn('[Emotion] Cannot run tests: classifier not loaded')
      return
    }
    const { inputSize } = this.config
    const makeImage = (r: number, g: number, b: number): ImageData => {
      const canvas = document.createElement('canvas')
      canvas.width = inputSize
      canvas.height = inputSize
      const ctx = canvas.getContext('2d')!
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fillRect(0, 0, inputSize, inputSize)
      return ctx.getImageData(0, 0, inputSize, inputSize)
    }
    const tests = [
      { name: 'black', r: 0, g: 0, b: 0 },
      { name: 'white', r: 255, g: 255, b: 255 },
      { name: 'gray', r: 128, g: 128, b: 128 },
      { name: 'red', r: 255, g: 0, b: 0 },
      { name: 'green', r: 0, g: 255, b: 0 },
      { name: 'blue', r: 0, g: 0, b: 255 },
    ]
    console.info('[Emotion] === Diagnostic Tests ===')
    for (const t of tests) {
      const result = await this.classify(makeImage(t.r, t.g, t.b))
      console.info(`[Emotion] Test ${t.name}:`, result)
    }
    console.info('[Emotion] === End Tests ===')
  }
}
