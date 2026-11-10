/**
 * Browser-side audio preprocessing utilities for noise reduction.
 * Used by the audio emotion pipeline and STT recording paths.
 */

/**
 * Apply a 1st-order IIR high-pass filter to remove low-frequency rumble
 * (HVAC, fan noise, electrical hum).
 *
 * @param samples      Input audio samples (Float32Array, [-1, 1])
 * @param sampleRate   Sample rate in Hz
 * @param cutoffHz     Cutoff frequency (default 80 Hz)
 * @returns            Filtered samples (new Float32Array)
 */
export function applyHighpassFilter(
  samples: Float32Array,
  sampleRate: number,
  cutoffHz: number = 80
): Float32Array {
  const rc = 1.0 / (2.0 * Math.PI * cutoffHz)
  const dt = 1.0 / sampleRate
  const alpha = rc / (rc + dt)

  const out = new Float32Array(samples.length)
  let prevIn = 0
  let prevOut = 0

  for (let i = 0; i < samples.length; i++) {
    const x = samples[i]
    const y = alpha * (prevOut + x - prevIn)
    out[i] = y
    prevIn = x
    prevOut = y
  }

  return out
}

/**
 * Estimate the noise floor from a buffer of audio samples.
 *
 * Splits the buffer into short frames, computes per-frame RMS, then
 * returns the given percentile (default 10th) of those RMS values.
 * This percentile is a robust estimator of the background noise level.
 *
 * @param samples    Input audio samples
 * @param frameSize  Frame size in samples (default 512)
 * @param percentile Percentile to use (0..1, default 0.1 = 10th)
 * @returns          Estimated noise floor RMS
 */
export function estimateNoiseFloor(
  samples: Float32Array,
  frameSize: number = 512,
  percentile: number = 0.1
): number {
  const frameCount = Math.floor(samples.length / frameSize)
  if (frameCount === 0) {
    // Single-frame fallback
    let sumSq = 0
    for (let i = 0; i < samples.length; i++) {
      sumSq += samples[i] * samples[i]
    }
    return Math.sqrt(sumSq / Math.max(samples.length, 1))
  }

  const rmsValues: number[] = []
  for (let f = 0; f < frameCount; f++) {
    let sumSq = 0
    const start = f * frameSize
    for (let i = 0; i < frameSize; i++) {
      const v = samples[start + i]
      sumSq += v * v
    }
    rmsValues.push(Math.sqrt(sumSq / frameSize))
  }

  rmsValues.sort((a, b) => a - b)
  const idx = Math.floor((rmsValues.length - 1) * Math.max(0, Math.min(1, percentile)))
  return rmsValues[idx] || 0
}

/**
 * Apply a simple noise gate with attack/release smoothing.
 *
 * The gate opens when the absolute sample value exceeds
 * `noiseFloorRMS * ratio` and closes smoothly when it drops below.
 *
 * @param samples        Input audio samples
 * @param noiseFloorRMS  Estimated noise floor RMS
 * @param ratio          How many times above the noise floor the
 *                       threshold is (default 2.5 ≈ +8 dB)
 * @param attackMs       Attack time in milliseconds (default 5)
 * @param releaseMs      Release time in milliseconds (default 50)
 * @param sampleRate     Sample rate in Hz
 * @returns              Gated samples (new Float32Array)
 */
export function applyNoiseGate(
  samples: Float32Array,
  noiseFloorRMS: number,
  ratio: number = 2.5,
  attackMs: number = 5,
  releaseMs: number = 50,
  sampleRate: number = 44100,
  minThreshold?: number,
  maxThreshold?: number,
): Float32Array {
  let threshold = noiseFloorRMS * ratio
  if (minThreshold !== undefined) {
    threshold = Math.max(threshold, minThreshold)
  }
  if (maxThreshold !== undefined) {
    threshold = Math.min(threshold, maxThreshold)
  }
  if (threshold <= 0) {
    // Nothing to gate – return a copy to keep the API contract uniform
    return new Float32Array(samples)
  }

  const attackStep = 1.0 / Math.max(1, (attackMs / 1000) * sampleRate)
  const releaseStep = 1.0 / Math.max(1, (releaseMs / 1000) * sampleRate)

  const out = new Float32Array(samples.length)
  let gain = 0.0

  for (let i = 0; i < samples.length; i++) {
    const absSample = Math.abs(samples[i])
    if (absSample > threshold) {
      gain = Math.min(1.0, gain + attackStep)
    } else {
      gain = Math.max(0.0, gain - releaseStep)
    }
    out[i] = samples[i] * gain
  }

  return out
}

/**
 * Compute Signal-to-Noise Ratio in decibels.
 *
 * @param signalRMS     RMS of the signal of interest
 * @param noiseFloorRMS Estimated noise floor RMS
 * @returns             SNR in dB (Infinity if noiseFloorRMS === 0)
 */
export function computeSNR(signalRMS: number, noiseFloorRMS: number): number {
  if (noiseFloorRMS <= 0 || signalRMS <= 0) {
    return signalRMS > 0 ? Infinity : 0
  }
  return 20 * Math.log10(signalRMS / noiseFloorRMS)
}

/**
 * Convenience helper: highpass → noise gate in one call.
 *
 * @param samples      Raw audio samples
 * @param sampleRate   Sample rate in Hz
 * @param cutoffHz     Highpass cutoff (default 80)
 * @param gateRatio    Noise-gate ratio above noise floor (default 2.5)
 * @returns            Object with filtered samples, noiseFloorRMS, and SNR
 */
export function preprocessAudio(
  samples: Float32Array,
  sampleRate: number,
  cutoffHz: number = 80,
  gateRatio: number = 2.5,
  minThreshold?: number,
  maxThreshold?: number,
): { filtered: Float32Array; noiseFloorRMS: number; snrDb: number } {
  const highpassed = applyHighpassFilter(samples, sampleRate, cutoffHz)
  const noiseFloorRMS = estimateNoiseFloor(highpassed)
  const gated = applyNoiseGate(highpassed, noiseFloorRMS, gateRatio, 5, 50, sampleRate, minThreshold, maxThreshold)

  // Compute signal RMS from gated output
  let sumSq = 0
  for (let i = 0; i < gated.length; i++) {
    sumSq += gated[i] * gated[i]
  }
  const signalRMS = Math.sqrt(sumSq / gated.length)
  const snrDb = computeSNR(signalRMS, noiseFloorRMS)

  return { filtered: gated, noiseFloorRMS, snrDb }
}
