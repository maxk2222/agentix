import type { EmotionResult } from './types'

/**
 * Generic aggregation of multiple EmotionResults.
 * Works for any source type (webcam, audio, text, etc.).
 */
export function aggregateResults(results: EmotionResult[], sourceLabel?: string): EmotionResult | null {
  if (results.length === 0) return null
  if (results.length === 1) return results[0]

  // Count emotion occurrences
  const counts: Record<string, number> = {}
  const confidences: Record<string, number[]> = {}
  for (const r of results) {
    const emo = r.primary_emotion
    counts[emo] = (counts[emo] || 0) + 1
    if (!confidences[emo]) confidences[emo] = []
    confidences[emo].push(r.confidence)
  }

  // Majority vote for primary emotion
  const primary = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]
  const avgConfidence =
    confidences[primary].reduce((a, b) => a + b, 0) / confidences[primary].length

  // Determine intensity by majority among results matching the primary emotion
  const intensityCounts: Record<string, number> = {}
  for (const r of results) {
    if (r.primary_emotion === primary) {
      const int = r.intensity
      intensityCounts[int] = (intensityCounts[int] || 0) + 1
    }
  }
  const intensity = (Object.entries(intensityCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ||
    'low') as 'low' | 'medium' | 'high'

  const validFaceFrames = results.reduce((sum, r) => sum + (r.valid_face_frames || 0), 0)
  const sampledFrames = results.reduce((sum, r) => sum + (r.sampled_frames || 1), 0)

  return {
    primary_emotion: primary,
    confidence: Math.round(avgConfidence * 100) / 100,
    intensity,
    source: sourceLabel || results[0].source || 'aggregate',
    timestamp_ms: Date.now(),
    sampled_frames: sampledFrames,
    valid_face_frames: validFaceFrames,
    face_detection_rate:
      sampledFrames > 0 ? Math.round((validFaceFrames / sampledFrames) * 100) / 100 : 0,
  }
}

/**
 * Legacy webcam-specific wrapper around aggregateResults.
 */
export function aggregateFrameResults(results: EmotionResult[]): EmotionResult | null {
  const aggregated = aggregateResults(results, 'webcam_browser_onnx')
  if (!aggregated) return null
  return aggregated
}

export interface SourceWeight {
  source: string
  weight: number
}

/**
 * Fuse emotions from multiple modalities (e.g. visual + audio + text)
 * into a single unified EmotionResult.
 *
 * Default weights favor visual > prosody > text, but can be overridden.
 */
export function fuseEmotions(
  results: EmotionResult[],
  weights?: SourceWeight[],
): EmotionResult | null {
  if (results.length === 0) return null
  if (results.length === 1) return results[0]

  const defaultWeights: SourceWeight[] = [
    { source: 'webcam_browser_onnx', weight: 1.0 },
    { source: 'audio_prosody_browser', weight: 0.8 },
    { source: 'text_rules', weight: 0.6 },
  ]

  const weightMap = new Map<string, number>()
  for (const w of weights || defaultWeights) {
    weightMap.set(w.source, w.weight)
  }

  // Weighted score per emotion
  const scores: Record<string, number> = {}
  const intensities: Record<string, { low: number; medium: number; high: number }> = {}

  for (const r of results) {
    const w = weightMap.get(r.source) ?? 0.5
    const emo = r.primary_emotion
    scores[emo] = (scores[emo] || 0) + r.confidence * w

    if (!intensities[emo]) intensities[emo] = { low: 0, medium: 0, high: 0 }
    intensities[emo][r.intensity] += w
  }

  const primary = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0]

  // Normalize confidence by total weight of that emotion
  const totalWeightForPrimary = results
    .filter((r) => r.primary_emotion === primary)
    .reduce((sum, r) => sum + (weightMap.get(r.source) ?? 0.5), 0)

  const rawConfidence = scores[primary]
  const confidence =
    totalWeightForPrimary > 0
      ? Math.round(Math.min(rawConfidence / totalWeightForPrimary, 1) * 100) / 100
      : 0

  // Intensity by weighted majority for the primary emotion
  const primaryIntensities = intensities[primary] || { low: 1, medium: 0, high: 0 }
  const intensity = (Object.entries(primaryIntensities).sort((a, b) => b[1] - a[1])[0][0] ||
    'low') as 'low' | 'medium' | 'high'

  // Collect signals from all contributing results
  const allSignals = results.flatMap((r) => r.signals || [])

  return {
    primary_emotion: primary,
    confidence,
    intensity,
    source: 'multimodal_fusion',
    timestamp_ms: Date.now(),
    signals: Array.from(new Set(allSignals)).slice(0, 8),
  }
}
