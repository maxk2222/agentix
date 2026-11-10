import type { SnippetConfig } from './types'

type ORT = typeof import('onnxruntime-web')
const ort = (): ORT => (window as any).ort

type InferenceSession = import('onnxruntime-web').InferenceSession
type Tensor = import('onnxruntime-web').Tensor
type OnnxValueMapType = import('onnxruntime-web').InferenceSession.OnnxValueMapType

let ortSession: InferenceSession | null = null
let modelLoaded = false
let modelUrl: string | null = null

export async function initOnnxRuntime(config: SnippetConfig): Promise<boolean> {
  if (modelLoaded && modelUrl === config.expressionModelUrl) {
    return true
  }
  try {
    const eps = config.executionProviders
      .map((p) => {
        if (p === 'webgpu' && 'gpu' in navigator) return 'webgpu'
        return 'wasm'
      })
      .filter((v, i, a) => a.indexOf(v) === i)

    ortSession = await ort().InferenceSession.create(config.expressionModelUrl, {
      executionProviders: eps.length > 0 ? eps : ['wasm'],
    })
    modelUrl = config.expressionModelUrl
    modelLoaded = true
    return true
  } catch (err) {
    console.warn('[Emotion] ONNX runtime init failed:', err)
    ortSession = null
    modelLoaded = false
    return false
  }
}

export function isOnnxReady(): boolean {
  return modelLoaded && ortSession !== null
}

export function resetOnnxRuntime(): void {
  ortSession = null
  modelLoaded = false
  modelUrl = null
}

export async function runInference(inputTensor: Tensor): Promise<OnnxValueMapType | null> {
  if (!ortSession) return null
  try {
    const feeds: Record<string, Tensor> = { input: inputTensor }
    return await ortSession.run(feeds)
  } catch (err) {
    console.warn('[Emotion] ONNX inference failed:', err)
    return null
  }
}
