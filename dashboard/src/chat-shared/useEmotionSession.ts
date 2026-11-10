import { useRef, useState, useCallback } from 'react'
import { EmotionSession, EmotionSessionConfig, EmotionSessionCallbacks } from './EmotionSession'
import type { EmotionRuntimeStatus, PhaseEmotion, EmotionPhase } from '../emotion/types'

export function useEmotionSession(config: EmotionSessionConfig | null) {
  const [status, setStatus] = useState<EmotionRuntimeStatus>('uninitialized')
  const [webcamActive, setWebcamActive] = useState(false)
  const [webcamConsent, setWebcamConsent] = useState(false)

  const callbacks: EmotionSessionCallbacks = {
    onStatusChange: setStatus,
    onWebcamActiveChange: setWebcamActive,
    onWebcamConsentChange: setWebcamConsent,
  }

  const sessionRef = useRef<EmotionSession | null>(null)

  // Recreate session when config changes
  if (!sessionRef.current || sessionRef.current['config'] !== config) {
    sessionRef.current = new EmotionSession(config, callbacks)
  }

  const session = sessionRef.current

  // Wrapped async methods that keep the session reference stable
  const startWebcam = useCallback(() => session.startWebcam(), [session])
  const stopWebcam = useCallback(() => session.stopWebcam(), [session])
  const toggleWebcam = useCallback(() => session.toggleWebcam(), [session])

  const startContinuousAudio = useCallback(() => session.startContinuousAudio(), [session])
  const stopContinuousAudio = useCallback(() => session.stopContinuousAudio(), [session])

  const capturePreSend = useCallback(() => session.capturePreSend(), [session])
  const buildPayload = useCallback(
    (opts: Parameters<typeof session.buildPayload>[0] = {}) => session.buildPayload(opts),
    [session]
  )

  const startPolling = useCallback((phase?: EmotionPhase) => session.startPolling(phase), [session])
  const stopPolling = useCallback(() => session.stopPolling(), [session])
  const switchPhase = useCallback((phase: EmotionPhase) => session.switchPhase(phase), [session])

  const buildEmotionPhases = useCallback(() => session.buildEmotionPhases(), [session])
  const computeOverallFused = useCallback(
    (phases: PhaseEmotion[]) => session.computeOverallFused(phases),
    [session]
  )

  const destroy = useCallback(() => session.destroy(), [session])

  return {
    session,
    status,
    webcamActive,
    webcamConsent,
    isAudioActive: session.isAudioActive,
    startWebcam,
    stopWebcam,
    toggleWebcam,
    startContinuousAudio,
    stopContinuousAudio,
    capturePreSend,
    buildPayload,
    startPolling,
    stopPolling,
    switchPhase,
    buildEmotionPhases,
    computeOverallFused,
    destroy,
  }
}
