import * as ort from 'onnxruntime-web'

// Configure ONNX Runtime WASM for model compatibility
ort.env.wasm.numThreads = 1
ort.env.wasm.simd = true

export interface FaceBox {
  x: number
  y: number
  w: number
  h: number
  conf: number
}

export class FaceDetector {
  private session: ort.InferenceSession | null = null
  private inputSize = 640
  private modelUrl: string | null = null
  private inferring = false

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
    if (!this.session || this.inferring) return []
    this.inferring = true

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

    // YOLOv5 preprocessing: [0, 1] normalization
    // HWC -> CHW
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
      const tensor = new ort.Tensor('float32', floatData, [1, 3, this.inputSize, this.inputSize])
      const feeds: Record<string, ort.Tensor> = { input: tensor }
      const results = await this.session.run(feeds)

      // Debug: log output tensor shapes
      for (const [name, value] of Object.entries(results)) {
        if (value && value instanceof ort.Tensor) {
          console.debug('[Emotion] FaceDetector output', name, 'shape:', value.dims?.join('x') || 'unknown')
        }
      }

      const faces = this._postprocess(results, scale, dx, dy, width, height)
      console.debug('[Emotion] FaceDetector found', faces.length, 'faces:', faces.map(f => `(${Math.round(f.x)},${Math.round(f.y)} ${Math.round(f.w)}x${Math.round(f.h)} conf=${f.conf.toFixed(2)})`).join(', '))
      return faces
    } catch (err) {
      console.warn('[Emotion] Face detection inference failed:', err)
      return []
    } finally {
      this.inferring = false
    }
  }

  private _postprocess(
    results: ort.InferenceSession.OnnxValueMapType,
    scale: number,
    dx: number,
    dy: number,
    origW: number,
    origH: number,
  ): FaceBox[] {
    // Try YuNet single-tensor format first: [batch, num_boxes, 15]
    // where each box is [batch_idx, class_id, score, x1, y1, x2, y2, ...landmarks]
    const allKeys = Object.keys(results)
    const singleTensorKey = allKeys.find(k => {
      const t = results[k] as ort.Tensor | undefined
      return t && t.dims && t.dims.length === 3
    })
    if (singleTensorKey) {
      return this._parseSingleTensor(results[singleTensorKey] as ort.Tensor, scale, dx, dy, origW, origH)
    }

    // Fallback: YOLOv5-Face multi-scale decoding
    return this._parseYOLOv5(results, scale, dx, dy, origW, origH)
  }

  private _parseSingleTensor(
    tensor: ort.Tensor,
    scale: number,
    dx: number,
    dy: number,
    origW: number,
    origH: number,
  ): FaceBox[] {
    const data = tensor.data as Float32Array
    const dims = tensor.dims as number[]
    const numBoxes = dims[1]
    const boxLen = dims[2] // typically 15
    const boxes: FaceBox[] = []

    for (let i = 0; i < numBoxes; i++) {
      const off = i * boxLen
      const conf = data[off + 2] // score at index 2
      if (conf < 0.5) continue
      // YuNet bbox coords are normalized to model input size (640)
      const x1 = data[off + 3]
      const y1 = data[off + 4]
      const x2 = data[off + 5]
      const y2 = data[off + 6]
      const w = Math.min(origW, Math.round((x2 - x1) / scale))
      const h = Math.min(origH, Math.round((y2 - y1) / scale))
      if (w < 40 || h < 40) continue
      boxes.push({
        x: Math.max(0, Math.round((x1 - dx) / scale)),
        y: Math.max(0, Math.round((y1 - dy) / scale)),
        w,
        h,
        conf,
      })
    }

    if (boxes.length === 0) return []
    boxes.sort((a, b) => (b.w * b.h) - (a.w * a.h))
    return boxes.slice(0, 1)
  }

  private _parseYOLOv5(
    results: ort.InferenceSession.OnnxValueMapType,
    scale: number,
    dx: number,
    dy: number,
    origW: number,
    origH: number,
  ): FaceBox[] {
    const scales = [
      { name: '8', stride: 8, grid: 80, anchor: [10, 13] },
      { name: '16', stride: 16, grid: 40, anchor: [30, 61] },
      { name: '32', stride: 32, grid: 20, anchor: [116, 90] },
    ]

    const candidates: Array<[number, number, number, number, number]> = []

    for (const s of scales) {
      const bbox = results[`bbox_${s.name}`] as ort.Tensor | undefined
      const cls = results[`cls_${s.name}`] as ort.Tensor | undefined
      const obj = results[`obj_${s.name}`] as ort.Tensor | undefined

      if (!bbox || !cls || !obj) continue

      const bboxData = bbox.data as Float32Array
      const clsData = cls.data as Float32Array
      const objData = obj.data as Float32Array
      const total = s.grid * s.grid

      for (let i = 0; i < total; i++) {
        const objScore = this._sigmoid(objData[i])
        const clsScore = this._sigmoid(clsData[i])
        const conf = objScore * clsScore
        if (conf < 0.5) continue

        const offset = i * 4
        const tx = bboxData[offset]
        const ty = bboxData[offset + 1]
        const tw = bboxData[offset + 2]
        const th = bboxData[offset + 3]

        const gridY = Math.floor(i / s.grid)
        const gridX = i % s.grid

        const cx = (this._sigmoid(tx) + gridX) * s.stride
        const cy = (this._sigmoid(ty) + gridY) * s.stride
        // Clamp before exp() to prevent Infinity/NaN
        const w = Math.exp(Math.min(Math.max(tw, -10), 10)) * s.anchor[0]
        const h = Math.exp(Math.min(Math.max(th, -10), 10)) * s.anchor[1]

        const x1 = cx - w / 2
        const y1 = cy - h / 2
        const x2 = cx + w / 2
        const y2 = cy + h / 2

        // Guard against NaN/Infinity
        if (!isFinite(x1) || !isFinite(y1) || !isFinite(x2) || !isFinite(y2)) continue

        candidates.push([x1, y1, x2, y2, conf])
      }
    }

    // NMS
    candidates.sort((a, b) => b[4] - a[4])
    const picked: Array<[number, number, number, number, number]> = []

    for (const box of candidates) {
      let shouldPick = true
      for (const p of picked) {
        if (this._iou(box, p) > 0.5) {
          shouldPick = false
          break
        }
      }
      if (shouldPick) picked.push(box)
    }

    const boxes: FaceBox[] = []
    let debugLogged = false
    for (const [x1, y1, x2, y2, conf] of picked) {
      const rawCx = (x1 - dx) / scale
      const rawCy = (y1 - dy) / scale
      const cw = (x2 - x1) / scale
      const ch = (y2 - y1) / scale
      const w = Math.min(origW, Math.round(cw))
      const h = Math.min(origH, Math.round(ch))
      // Debug first box
      if (!debugLogged) {
        console.debug('[Emotion] YOLO raw:', { x1: x1.toFixed(1), y1: y1.toFixed(1), x2: x2.toFixed(1), y2: y2.toFixed(1), scale, dx, dy })
        console.debug('[Emotion] YOLO mapped:', { rawCx, rawCy, cw, ch, w, h })
        debugLogged = true
      }
      if (w < 40 || h < 40) continue
      if (!isFinite(rawCx) || !isFinite(rawCy)) continue
      boxes.push({
        x: Math.max(0, Math.round(rawCx)),
        y: Math.max(0, Math.round(rawCy)),
        w,
        h,
        conf,
      })
    }

    if (boxes.length === 0) return []
    boxes.sort((a, b) => (b.w * b.h) - (a.w * a.h))
    return boxes.slice(0, 1)
  }

  private _sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x))
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
