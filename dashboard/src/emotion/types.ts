export interface EmotionResult {
  primary_emotion: string
  confidence: number
  intensity: 'low' | 'medium' | 'high'
  source: string
  signals?: string[]
  timestamp_ms: number
  sampled_frames?: number
  valid_face_frames?: number
  face_detection_rate?: number
  window_ms?: number
}

export type EmotionPhase = 'waiting_for_response' | 'reading' | 'typing' | 'speaking'

export interface PhaseEmotion {
  phase: EmotionPhase
  primary_emotion: string
  confidence: number
  intensity: 'low' | 'medium' | 'high'
  source: string
  timestamp_ms: number
  sampled_frames?: number
  valid_face_frames?: number
  face_detection_rate?: number
  window_ms?: number
}

export interface EmotionContextPayload {
  visual_pre_send?: EmotionResult
  audio_prosody?: EmotionResult
  fused?: EmotionResult
  phases?: PhaseEmotion[]
  consent?: {
    webcam_opt_in: boolean
    frames_sent_to_backend: boolean
    audio_opt_in: boolean
  }
}

export interface SnippetConfig {
  windowMs: number
  sampleFps: number
  maxFrames: number
  minValidFaceFrames: number
  faceModelUrl: string
  faceDetectorType?: 'mediapipe' | 'onnx'
  expressionModelUrl: string
  expressionModelType?: 'fer' | 'affectnet'
  expressionInputSize?: number
  executionProviders: string[]
  minSnrDb?: number
  noiseGateMin?: number
  noiseGateMax?: number
}

export type EmotionRuntimeStatus =
  | 'uninitialized'
  | 'loading'
  | 'ready'
  | 'error'
  | 'no-face'
  | 'disabled'
