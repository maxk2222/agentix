export interface FaceBox {
  x: number
  y: number
  w: number
  h: number
  conf: number
}

// MediaPipe FaceDetection loaded dynamically from CDN
declare global {
  interface Window {
    FaceDetection?: any
  }
}

export class FaceDetectorMediaPipe {
  private detector: any = null
  private inferring = false
  private lastResults: any = null
  private pendingResolve: ((value: any) => void) | null = null

  async load(): Promise<boolean> {
    try {
      // Dynamically load MediaPipe FaceDetection from CDN
      if (!window.FaceDetection) {
        await this._loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/face_detection.js')
      }

      if (!window.FaceDetection) {
        console.warn('[Emotion] MediaPipe FaceDetection not available after loading script')
        return false
      }

      const faceDetection = new window.FaceDetection({
        locateFile: (file: string) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
        }
      })

      await faceDetection.setOptions({
        model: 'short',
        minDetectionConfidence: 0.5
      })

      await faceDetection.initialize()
      this.detector = faceDetection
      // Register callback once during load so it's ready before send()
      // Note: onResults is a METHOD, not a property assignment
      this.detector.onResults((r: any) => {
        this.lastResults = r
        if (this.pendingResolve) {
          this.pendingResolve(r)
          this.pendingResolve = null
        }
      })
      console.info('[Emotion] MediaPipe FaceDetection loaded successfully')
      return true
    } catch (err) {
      console.warn('[Emotion] MediaPipe FaceDetection load failed:', err)
      this.detector = null
      return false
    }
  }

  isReady(): boolean {
    return this.detector !== null
  }

  async detect(imageData: ImageData): Promise<FaceBox[]> {
    if (!this.detector || this.inferring) return []
    this.inferring = true

    try {
      // Convert ImageData to HTMLCanvasElement for MediaPipe
      const canvas = document.createElement('canvas')
      canvas.width = imageData.width
      canvas.height = imageData.height
      const ctx = canvas.getContext('2d')!
      ctx.putImageData(imageData, 0, 0)

      // MediaPipe may need canvas attached to DOM
      canvas.style.position = 'absolute'
      canvas.style.left = '-9999px'
      document.body.appendChild(canvas)

      // MediaPipe uses callback-based API: send({image}) triggers onResults
      // onResults was already registered in load(), so just call send and wait
      this.lastResults = null
      await this.detector.send({ image: canvas })

      // Remove from DOM after detection
      document.body.removeChild(canvas)
      const results = await new Promise<any>((resolve, reject) => {
        if (this.lastResults) {
          resolve(this.lastResults)
          return
        }
        this.pendingResolve = resolve
        setTimeout(() => {
          if (this.pendingResolve) {
            this.pendingResolve = null
            reject(new Error('MediaPipe detection timeout'))
          }
        }, 5000)
      })

      const detections = results?.detections || []

      const faces: FaceBox[] = []
      for (const d of detections) {
        if (!d.boundingBox || !d.V || !d.V[0]) continue
        const bb = d.boundingBox
        // MediaPipe v0.4 returns CENTER format: {xCenter, yCenter, width, height}
        const xMin = bb.xCenter - bb.width / 2
        const yMin = bb.yCenter - bb.height / 2
        const xMax = bb.xCenter + bb.width / 2
        const yMax = bb.yCenter + bb.height / 2

        const x = Math.round(xMin * imageData.width)
        const y = Math.round(yMin * imageData.height)
        const w = Math.round((xMax - xMin) * imageData.width)
        const h = Math.round((yMax - yMin) * imageData.height)
        const conf = d.V[0].ga || 0.5

        if (w < 40 || h < 40) continue
        faces.push({ x, y, w, h, conf })
      }

      if (faces.length > 0) {
        console.debug('[Emotion] MediaPipe found', faces.length, 'faces:', faces.map(f => `(${f.x},${f.y} ${f.w}x${f.h} conf=${f.conf.toFixed(2)})`).join(', '))
      }

      // Return only the largest face
      if (faces.length === 0) return []
      faces.sort((a, b) => (b.w * b.h) - (a.w * a.h))
      return faces.slice(0, 1)
    } catch (err) {
      console.warn('[Emotion] MediaPipe face detection failed:', err)
      return []
    } finally {
      this.inferring = false
    }
  }

  private _loadScript(src: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (document.querySelector(`script[src="${src}"]`)) {
        resolve()
        return
      }
      const script = document.createElement('script')
      script.src = src
      script.crossOrigin = 'anonymous'
      script.onload = () => resolve()
      script.onerror = () => reject(new Error(`Failed to load ${src}`))
      document.head.appendChild(script)
    })
  }
}
