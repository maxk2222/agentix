import type { EmotionResult } from './types'

const TEXT_KEYWORDS: Record<string, Record<string, string[]>> = {
  de: {
    frustrated: [
      'frustriert', 'ärgerlich', 'verdammt', 'verflixt', 'nicht funktioniert',
      'geht nicht', 'schon wieder', 'immer wieder', 'langsam', 'hängt',
      'abgestürzt', 'mühsam', 'nervig', 'lästig', 'unglaublich',
      'katastrophe', 'irre', 'unzumutbar', 'geht mir auf die nerven',
      'nervt', 'nervig', 'unmöglich',
    ],
    angry: [
      'wütend', 'furchtbar', 'scheiße', 'mist', 'unverschämt',
      'frech', 'absurd', 'unmöglich', 'ganz schlecht', 'schlimm',
      'hässlich', 'blöd', 'doof', 'arschloch', 'idiotisch',
      'sinnlos', 'fürchterlich', 'zum kotzen', 'verdammt',
    ],
    anxious: [
      'besorgt', 'ängstlich', 'panik', 'hilfe', 'dringend', 'notfall',
      'schnell', 'sofort', 'bitte helfen', 'was tun', 'was soll ich',
    ],
    confused: [
      'verwirrt', 'verstanden nicht', 'nicht klar', 'unklar',
      'was meinst du', 'verstehe nicht', 'kapier ich nicht', 'wie genau',
      'was genau', 'erkläre', 'nochmal', 'bitte wiederholen',
    ],
    satisfied: [
      'danke', 'perfekt', 'super', 'genau', 'klar', 'verstanden',
      'hilfreich', 'gut', 'bestens', 'prima', 'top', 'klasse',
    ],
    sad: [
      'traurig', 'enttäuscht', 'schade', 'bedauerlich', 'hilflos',
      'verloren', 'nicht gut', 'schwierig', 'überfordert',
      'deprimiert', 'hoffnungslos', 'sinnlos', 'alles scheiße',
    ],
    surprised: [
      'überrascht', 'wow', 'echt', 'unglaublich', 'krass',
      'das wusste ich nicht', 'interessant', 'faszinierend',
    ],
    urgent: [
      'sofort', 'dringend', 'eilig', 'critical', 'wichtig',
      'jetzt', 'schnellstmöglich', 'asap', 'zeitkritisch',
    ],
  },
  en: {
    frustrated: [
      'frustrated', 'annoying', 'not working', 'broken', 'again',
      'slow', 'crashes', 'painful', 'stuck', 'fed up', 'irritating',
    ],
    angry: [
      'angry', 'furious', 'terrible', 'awful', 'ridiculous',
      'outrageous', 'unacceptable', 'worst', 'hate', 'disgusting',
    ],
    anxious: [
      'worried', 'anxious', 'panic', 'urgent', 'emergency',
      'help', 'scared', 'nervous', 'stress', 'please help',
    ],
    confused: [
      'confused', 'unclear', 'don\'t understand', 'what do you mean',
      'explain', 'repeat', 'not sure', 'lost', 'what exactly',
    ],
    satisfied: [
      'thanks', 'thank you', 'perfect', 'great', 'helpful',
      'clear', 'understood', 'good', 'awesome', 'nice', 'love it',
    ],
    sad: [
      'sad', 'disappointed', 'sorry', 'helpless', 'lost',
      'not good', 'difficult', 'overwhelmed', 'struggling',
    ],
    surprised: [
      'surprised', 'wow', 'amazing', 'didn\'t know', 'incredible',
      'interesting', 'fascinating', 'unexpected', 'shocked',
    ],
    urgent: [
      'urgent', 'immediately', 'asap', 'critical', 'important',
      'now', 'quickly', 'time-sensitive', 'emergency',
    ],
  },
}

const INTENSITY_PATTERNS = {
  high: /!{2,}|\b(sehr|extrem|total|absolut|wütend|furious|völlig|unglaublich|complete|utterly)\b/i,
  medium: /!|\b(ziemlich|recht|echt|voll|ganz|fairly|quite|rather|pretty)\b/i,
}

const NEGATION_WORDS = new Set(['nicht', 'kein', 'no', 'not', 'never', 'none', 'nichts', 'nothing'])

function _negationNearby(textLower: string, keyword: string): boolean {
  const idx = textLower.indexOf(keyword)
  if (idx === -1) return false
  const window = textLower.slice(Math.max(0, idx - 30), idx)
  const words = window.split(/\s+/)
  return words.slice(-3).some((w) => NEGATION_WORDS.has(w))
}

export function detectTextEmotion(text: string, languageHint?: string): EmotionResult {
  if (!text || !text.trim()) {
    return _makeResult('unknown', 0.0, 'low', 'text_rules')
  }

  const textLower = text.toLowerCase()
  let lang = languageHint
  if (!lang) {
    const germanMarkers = ['der', 'die', 'das', 'und', 'ist', 'nicht', 'zu'].filter((w) => textLower.includes(w)).length
    const englishMarkers = ['the', 'and', 'is', 'not', 'to', 'a', 'of'].filter((w) => textLower.includes(w)).length
    lang = germanMarkers > englishMarkers ? 'de' : 'en'
  }

  const keywords = TEXT_KEYWORDS[lang] || TEXT_KEYWORDS.en
  const scores: Record<string, number> = {}
  const matchedSignals: string[] = []

  for (const [emotion, words] of Object.entries(keywords)) {
    let score = 0
    for (const word of words) {
      if (textLower.includes(word)) {
        score += 0.5
        matchedSignals.push(word)
      }
    }
    if (score > 0) {
      scores[emotion] = Math.min(score, 1.0)
    }
  }

  // Negation dampening (context-sensitive)
  if (Object.keys(scores).length > 0) {
    for (const emotion of Object.keys(scores)) {
      if (['frustrated', 'angry', 'anxious', 'sad'].includes(emotion)) {
        const words = keywords[emotion] || []
        const hasNearbyNegation = words.some((w) => textLower.includes(w) && _negationNearby(textLower, w))
        if (hasNearbyNegation) {
          scores[emotion] *= 0.5
        }
      }
    }
  }

  // Intensity + confidence boost
  let intensity: 'low' | 'medium' | 'high' = 'low'
  let intensityBoost = 0
  if (INTENSITY_PATTERNS.high.test(text)) {
    intensity = 'high'
    intensityBoost = 0.2
  } else if (INTENSITY_PATTERNS.medium.test(text)) {
    intensity = 'medium'
    intensityBoost = 0.1
  }
  if (text === text.toUpperCase() && text.length > 5) {
    intensity = 'high'
    intensityBoost = Math.max(intensityBoost, 0.2)
  }
  if (Object.keys(scores).length > 0 && intensityBoost > 0) {
    for (const emotion of Object.keys(scores)) {
      scores[emotion] = Math.min(scores[emotion] + intensityBoost, 1.0)
    }
  }

  if (Object.keys(scores).length === 0) {
    return _makeResult('neutral', 0.0, intensity, 'text_rules')
  }

  const primary = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0]
  let confidence = scores[primary]
  if (text.trim().length < 10) {
    confidence *= 0.7
  }
  confidence = Math.round(Math.min(Math.max(confidence, 0), 1) * 100) / 100

  return _makeResult(
    primary,
    confidence,
    intensity,
    'text_rules',
    Array.from(new Set(matchedSignals)).slice(0, 5),
  )
}

function _makeResult(
  primary: string,
  confidence: number,
  intensity: string,
  source: string,
  signals?: string[],
): EmotionResult {
  const result: EmotionResult = {
    primary_emotion: primary,
    confidence,
    intensity: intensity as 'low' | 'medium' | 'high',
    source,
    timestamp_ms: Date.now(),
  }
  if (signals && signals.length > 0) {
    result.signals = signals
  }
  return result
}
