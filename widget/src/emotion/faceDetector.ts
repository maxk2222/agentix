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
type OnnxValueMapType = import('onnxruntime-web').InferenceSession.OnnxValueMapType

export interface FaceBox {
  x: number
  y: number
  w: number
  h: number
  conf: number
}

export class FaceDetector {
  private session: InferenceSession | null = null
  private inputSize = 320
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
      console.warn('[Emotion] Face detector load failed:', err)
      this.session = null
      this.modelUrl = null
      return false
    }
  }

  isReady(): boolean {
    return this.session !== null
  }

  async detect(imageData: ImageData): Promise<FaceBox[]> {
    if (!this.session) return []

    // Resize to model input size (square)
    const { width, height } = imageData
    const canvas = document.createElement('canvas')
    canvas.width = this.inputSize
    canvas.height = this.inputSize
    const ctx = canvas.getContext('2d')!

    // Draw and resize maintaining aspect ratio with letterboxing
    const scale = Math.min(this.inputSize / width, this.inputSize / height)
    const newW = Math.round(width * scale)
    const newH = Math.round(height * scale)
    const dx = (this.inputSize - newW) / 2
    const dy = (this.inputSize - newH) / 2

    const imgCanvas = document.createElement('canvas')
    imgCanvas.width = width
    imgCanvas.height = height
    const imgCtx = imgCanvas.getContext('2d')!
    imgCtx.putImageData(imageData, 0, 0)

    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, this.inputSize, this.inputSize)
    ctx.drawImage(imgCanvas, 0, 0, width, height, dx, dy, newW, newH)

    const resized = ctx.getImageData(0, 0, this.inputSize, this.inputSize)
    const floatData = new Float32Array(this.inputSize * this.inputSize * 3)

    // Normalize to [0,1] and HWC -> CHW
    for (let y = 0; y < this.inputSize; y++) {
      for (let x = 0; x < this.inputSize; x++) {
        const srcIdx = (y * this.inputSize + x) * 4
        const dstIdx = y * this.inputSize + x
        floatData[dstIdx] = resized.data[srcIdx] / 255.0
        floatData[this.inputSize * this.inputSize + dstIdx] = resized.data[srcIdx + 1] / 255.0
        floatData[2 * this.inputSize * this.inputSize + dstIdx] = resized.data[srcIdx + 2] / 255.0
      }
    }

    try {
      const tensor = new (ort().Tensor)('float32', floatData, [1, 3, this.inputSize, this.inputSize])
      const feeds: Record<string, Tensor> = { input: tensor }
      const results = await this.session.run(feeds)

      // Postprocess depends on model format
      // YuNet outputs: [batch, num_boxes, 15] where each box is [batch_idx, x1, y1, x2, y2, conf, ...landmarks]
      // Generic: try common formats
      return this._postprocess(results, scale, dx, dy, width, height)
    } catch (err) {
      console.warn('[Emotion] Face detection inference failed:', err)
      return []
    }
  }

  private _postprocess(
    results: OnnxValueMapType,
    scale: number,
    dx: number,
    dy: number,
    origW: number,
    origH: number,
  ): FaceBox[] {
    // Try to find output tensor (common names)
    const output = results.output || results['output0'] || results['detections'] || Object.values(results)[0]
    if (!output || !(output instanceof ort().Tensor)) return []

    const data = output.data as Float32Array
    const dims = output.dims
    const boxes: FaceBox[] = []

    // Common ONNX face detector output shapes:
    // [1, num_boxes, 15] (YuNet-style)
    // [1, 25200, 85] (YOLO-style)

    if (dims.length === 3 && dims[2] >= 5) {
      const [, numBoxes, boxDim] = dims as [number, number, number]
      const confThreshold = 0.5
      const nmsThreshold = 0.4
      const candidates: Array<[number, number, number, number, number]> = []

      for (let i = 0; i < numBoxes; i++) {
        const offset = i * boxDim
        const conf = data[offset + 4]
        if (conf > confThreshold) {
          const x1 = data[offset]
          const y1 = data[offset + 1]
          const x2 = data[offset + 2]
          const y2 = data[offset + 3]
          candidates.push([x1, y1, x2, y2, conf])
        }
      }

      // Simple NMS
      candidates.sort((a, b) => b[4] - a[4])
      const picked: Array<[number, number, number, number, number]> = []

      for (const box of candidates) {
        let shouldPick = true
        for (const p of picked) {
          if (this._iou(box, p) > nmsThreshold) {
            shouldPick = false
            break
          }
        }
        if (shouldPick) picked.push(box)
      }

      for (const [x1, y1, x2, y2, conf] of picked) {
        // Convert back to original image coordinates
        const cx = (x1 - dx) / scale
        const cy = (y1 - dy) / scale
        const cw = (x2 - x1) / scale
        const ch = (y2 - y1) / scale
        boxes.push({
          x: Math.max(0, Math.round(cx)),
          y: Math.max(0, Math.round(cy)),
          w: Math.min(origW, Math.round(cw)),
          h: Math.min(origH, Math.round(ch)),
          conf,
        })
      }
    }

    return boxes.slice(0, 5) // max 5 faces
  }

  private _iou(a: [number, number, number, number, number], b: [number, number, number, number, number]): number {
    const x1 = Math.max(a[0], b[0])
    const y1 = Math.max(a[1], b[1])
    const x2 = Math.min(a[2], b[2])
    const y2 = Math.min(a[3], b[3])
    const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
    const areaA = (a[2] - a[0]) * (a[3] - a[1])
    const areaB = (b[2] - b[0]) * (b[3] - b[1])
    return interArea / (areaA + areaB - interArea + 1e-6)
  }
}
