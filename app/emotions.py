"""Emotion detection: ONNX-based text analysis, visual payload validation,
fusion logic, and prompt-steering helpers.
"""
from __future__ import annotations

import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class EmotionResult(TypedDict, total=False):
    primary_emotion: str
    confidence: float
    intensity: str
    source: str
    signals: list[str]
    timestamp_ms: int
    sampled_frames: Optional[int]
    valid_face_frames: Optional[int]
    face_detection_rate: Optional[float]
    window_ms: Optional[int]


class EmotionContext(TypedDict, total=False):
    text_emotion: EmotionResult
    visual_pre_send: EmotionResult
    fused_emotion: EmotionResult
    phases: dict[str, EmotionResult]
    consent: dict[str, Any]


# ---------------------------------------------------------------------------
# ONNX text emotion classifier (German GoEmotions)
# ---------------------------------------------------------------------------

_GOEMOTIONS_LABELS: list[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

_GOEMOTIONS_TO_AGENTIX: dict[str, list[str]] = {
    "neutral": ["neutral", "curiosity", "realization", "pride"],
    "angry": ["anger", "disgust", "disapproval"],
    "frustrated": ["annoyance"],
    "anxious": ["fear", "nervousness"],
    "confused": ["confusion"],
    "satisfied": [
        "joy", "amusement", "excitement", "admiration", "approval",
        "gratitude", "love", "optimism", "relief", "caring", "desire",
    ],
    "sad": ["sadness", "disappointment", "grief", "remorse", "embarrassment"],
    "surprised": ["surprise"],
}

_URGENT_KEYWORDS = re.compile(
    r"\b(sofort|dringend|eilig|critical|wichtig|jetzt|schnellstmöglich|asap|zeitkritisch|"
    r"urgent|immediately|emergency|quickly|time-sensitive)\b",
    re.IGNORECASE,
)


class TextEmotionClassifier:
    """Lazy-loaded ONNX GoEmotions classifier for German text."""

    def __init__(self, model_path: Path | None = None, tokenizer_path: Path | None = None) -> None:
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path
        self._session: ort.InferenceSession | None = None
        self._tokenizer: Any = None

    def _load(self) -> bool:
        if self._session is not None:
            return True
        if self._model_path is None or not self._model_path.exists():
            logger.warning("Text emotion ONNX model not found at %s", self._model_path)
            return False
        if self._tokenizer_path is None or not self._tokenizer_path.exists():
            logger.warning("Text emotion tokenizer not found at %s", self._tokenizer_path)
            return False
        try:
            from tokenizers import Tokenizer

            self._session = ort.InferenceSession(str(self._model_path))
            self._tokenizer = Tokenizer.from_file(str(self._tokenizer_path))
            self._tokenizer.enable_padding(length=128, pad_id=1, pad_token="<pad>")
            self._tokenizer.enable_truncation(max_length=128)
            logger.info("Loaded text emotion ONNX model from %s", self._model_path)
            return True
        except Exception as exc:
            logger.error("Failed to load text emotion model: %s", exc)
            return False

    def predict(self, text: str) -> dict[str, float]:
        """Return GoEmotions label -> probability mapping."""
        if not self._load():
            return {}
        enc = self._tokenizer.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        logits = self._session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0][0]
        # Stable softmax
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)
        return {label: float(probs[i]) for i, label in enumerate(_GOEMOTIONS_LABELS)}


# Module-level cache so the model is loaded only once per process.
_text_classifier: TextEmotionClassifier | None = None
_last_model_path: Path | None = None
_last_tokenizer_path: Path | None = None


def _get_classifier(model_path: Path | None, tokenizer_path: Path | None) -> TextEmotionClassifier | None:
    """Return a cached classifier, recreating it if the paths changed."""
    global _text_classifier, _last_model_path, _last_tokenizer_path
    if _text_classifier is None or model_path != _last_model_path or tokenizer_path != _last_tokenizer_path:
        _text_classifier = TextEmotionClassifier(model_path=model_path, tokenizer_path=tokenizer_path)
        _last_model_path = model_path
        _last_tokenizer_path = tokenizer_path
    return _text_classifier


def _aggregate_to_agentix(go_probs: dict[str, float]) -> tuple[str, float]:
    """Aggregate 28 GoEmotions probabilities into agentix labels."""
    agentix_scores: dict[str, float] = {}
    for agentix_label, go_sources in _GOEMOTIONS_TO_AGENTIX.items():
        score = sum(go_probs.get(src, 0.0) for src in go_sources)
        if score > 0:
            agentix_scores[agentix_label] = score
    if not agentix_scores:
        return "neutral", 0.0
    primary = max(agentix_scores, key=agentix_scores.get)
    return primary, agentix_scores[primary]


def detect_text_emotion(
    text: str,
    language_hint: Optional[str] = None,
    *,
    config: Any | None = None,
) -> EmotionResult:
    """Detect text emotion using the German GoEmotions ONNX model.

    Falls back to neutral/unknown if the model is not available.
    """
    if not text or not text.strip():
        return _make_result("unknown", 0.0, "low", "text_rules")

    model_path: Path | None = None
    tokenizer_path: Path | None = None
    if config is not None:
        models_dir = Path(config.data_dir) / "models" / "emotion"
        model_url = getattr(config, "emotion_text_model_url", "/models/emotion/text-emotion.onnx")
        tokenizer_url = getattr(config, "emotion_text_tokenizer_url", "/models/emotion/text-emotion-tokenizer.json")
        model_path = models_dir / Path(model_url).name
        tokenizer_path = models_dir / Path(tokenizer_url).name

    clf = _get_classifier(model_path, tokenizer_path)
    go_probs = clf.predict(text) if clf is not None else {}

    if not go_probs:
        # Model not loaded — check urgent keywords first, else return neutral so fusion isn't broken
        if _URGENT_KEYWORDS.search(text):
            return _make_result("urgent", 0.6, "high", "text_rules")
        return _make_result("neutral", 0.0, "low", "text_rules")

    primary, confidence = _aggregate_to_agentix(go_probs)

    # Intensity derived from confidence
    intensity = "low"
    if confidence >= 0.7:
        intensity = "high"
    elif confidence >= 0.4:
        intensity = "medium"

    # Urgent keyword override
    if _URGENT_KEYWORDS.search(text):
        primary = "urgent"
        confidence = max(confidence, 0.6)
        intensity = "high"

    # Penalize very short messages
    if len(text.strip()) < 10:
        confidence *= 0.7

    confidence = round(min(max(confidence, 0.0), 1.0), 2)

    # Include top-3 GoEmotions as signals for debugging
    top3 = sorted(go_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    signals = [f"{label}({prob:.2f})" for label, prob in top3]

    return _make_result(primary, confidence, intensity, "text_rules", signals=signals)


def _make_result(
    primary: str,
    confidence: float,
    intensity: str,
    source: str,
    signals: Optional[list[str]] = None,
    **kwargs: Any,
) -> EmotionResult:
    import time
    result: EmotionResult = {
        "primary_emotion": primary,
        "confidence": confidence,
        "intensity": intensity,
        "source": source,
        "timestamp_ms": int(time.time() * 1000),
    }
    if signals:
        result["signals"] = signals
    result.update(kwargs)
    return result


# ---------------------------------------------------------------------------
# Visual payload validation
# ---------------------------------------------------------------------------

_ALLOWED_VISUAL_FIELDS = {
    "source", "runtime", "face_detector", "window_ms", "sampled_frames",
    "valid_face_frames", "face_detection_rate", "primary_emotion",
    "confidence", "intensity",
}

_ALLOWED_PRIMARY_EMOTIONS = {
    "neutral", "frustrated", "angry", "anxious", "confused",
    "satisfied", "sad", "surprised", "urgent", "unknown",
}


def validate_visual_emotion(payload: Optional[dict], config: Any) -> Optional[EmotionResult]:
    """Sanitize and validate a visual emotion payload from the frontend."""
    if not payload or not isinstance(payload, dict):
        return None

    # Strip any unexpected fields
    cleaned = {k: v for k, v in payload.items() if k in _ALLOWED_VISUAL_FIELDS}

    primary = str(cleaned.get("primary_emotion", "unknown")).lower()
    if primary not in _ALLOWED_PRIMARY_EMOTIONS:
        primary = "unknown"

    confidence = float(cleaned.get("confidence", 0.0))
    confidence = min(max(confidence, 0.0), 1.0)

    intensity = str(cleaned.get("intensity", "low")).lower()
    if intensity not in ("low", "medium", "high"):
        intensity = "low"

    # Penalize confidence if too few valid face frames
    valid_face_frames = cleaned.get("valid_face_frames")
    min_valid = getattr(config, "emotion_webcam_min_valid_face_frames", 3)
    if isinstance(valid_face_frames, (int, float)) and valid_face_frames < min_valid:
        confidence *= 0.5

    result = _make_result(
        primary,
        round(confidence, 2),
        intensity,
        str(cleaned.get("source", "webcam_browser")),
        sampled_frames=cleaned.get("sampled_frames"),
        valid_face_frames=valid_face_frames,
        face_detection_rate=cleaned.get("face_detection_rate"),
        window_ms=cleaned.get("window_ms"),
    )
    return result


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


_DEFAULT_SOURCE_WEIGHTS = {
    "text": 1.0,
    "visual_pre": 1.0,
    "audio_prosody": 0.9,
    "phase_typing": 0.8,
    "phase_waiting_for_response": 0.6,
    "phase_reading": 0.4,
}

_DEFAULT_SOURCE_MIN_CONFIDENCE = {
    "text": 0.55,
    "visual_pre": 0.55,
    "audio_prosody": 0.55,
    "phase": 0.55,
}

_INTENSITY_RANK = {"low": 0, "medium": 1, "high": 2}


def _resolve_source_weights(config: Any) -> dict[str, float]:
    """Return effective source weights, merging config overrides with defaults."""
    raw = getattr(config, "emotion_fusion_source_weights", None) if config else None
    if not raw or not isinstance(raw, dict):
        return dict(_DEFAULT_SOURCE_WEIGHTS)
    merged = dict(_DEFAULT_SOURCE_WEIGHTS)
    for k, v in raw.items():
        try:
            merged[k] = float(v)
        except (ValueError, TypeError):
            pass
    return merged


def _resolve_source_min_confidence(config: Any) -> dict[str, float]:
    """Return effective per-source min confidence, merging config overrides with defaults."""
    raw = getattr(config, "emotion_fusion_min_confidence", None) if config else None
    if not raw or not isinstance(raw, dict):
        return dict(_DEFAULT_SOURCE_MIN_CONFIDENCE)
    merged = dict(_DEFAULT_SOURCE_MIN_CONFIDENCE)
    for k, v in raw.items():
        try:
            merged[k] = float(v)
        except (ValueError, TypeError):
            pass
    return merged


def _phase_decay_weight(
    phase_result: EmotionResult,
    base_weight: float,
    now_ms: int,
    half_life_seconds: float = 60.0,
) -> float:
    """Apply temporal decay to phase weights based on age."""
    ts = phase_result.get("timestamp_ms")
    if not ts:
        return base_weight
    age_seconds = max(0.0, (now_ms - ts) / 1000.0)
    if age_seconds <= 0:
        return base_weight
    decay = 0.5 ** (age_seconds / half_life_seconds)
    return base_weight * max(decay, 0.25)


def fuse_emotions(
    text_emotion: Optional[EmotionResult] = None,
    visual_pre_send: Optional[EmotionResult] = None,
    audio_prosody: Optional[EmotionResult] = None,
    phases: Optional[dict[str, EmotionResult]] = None,
    config: Any = None,
) -> EmotionResult:
    """Fuse text, visual, audio, and phase emotion signals using weighted confidence aggregation.

    Sources are filtered by per-source min_confidence, then each qualifying source contributes
    a weighted score to its primary emotion. The emotion with the highest weighted score wins.
    Final confidence is the weighted average of all agreeing sources.
    """
    source_weights = _resolve_source_weights(config)
    source_min_conf = _resolve_source_min_confidence(config)
    now_ms = int(time.time() * 1000)

    # Gather all qualifying sources
    sources: list[tuple[str, EmotionResult, float]] = []

    def _add_source(label: str, result: Optional[EmotionResult]) -> None:
        if not result:
            return
        if result.get("primary_emotion") == "unknown":
            return
        threshold = source_min_conf.get(label, source_min_conf.get("phase", 0.55))
        if result.get("confidence", 0) < threshold:
            return
        weight = source_weights.get(label, 0.5)
        if label.startswith("phase_"):
            weight = _phase_decay_weight(result, weight, now_ms)
        sources.append((label, result, weight))

    if text_emotion:
        _add_source("text", text_emotion)
    if visual_pre_send:
        _add_source("visual_pre", visual_pre_send)
    if audio_prosody:
        _add_source("audio_prosody", audio_prosody)
    if phases:
        for phase_name, phase_result in phases.items():
            label = f"phase_{phase_name}"
            _add_source(label, phase_result)

    if not sources:
        return _make_result("neutral", 0.0, "low", "fusion")

    # Separate neutral and non-neutral sources
    neutral_sources = [s for s in sources if s[1].get("primary_emotion") == "neutral"]
    non_neutral_sources = [s for s in sources if s[1].get("primary_emotion") != "neutral"]

    # If there are non-neutral sources, they take precedence
    if non_neutral_sources:
        # Weighted score per emotion
        scores: Dict[str, float] = {}
        intensities: Dict[str, Dict[str, float]] = {}
        best_confidence: Dict[str, float] = {}

        for label, src, weight in non_neutral_sources:
            emo = src["primary_emotion"]
            conf = src["confidence"]
            scores[emo] = scores.get(emo, 0.0) + conf * weight

            if emo not in intensities:
                intensities[emo] = {"low": 0.0, "medium": 0.0, "high": 0.0}
            intensities[emo][src.get("intensity", "low")] += weight

            if emo not in best_confidence or conf > best_confidence[emo]:
                best_confidence[emo] = conf

        # Pick emotion with highest weighted score; tie-break by highest single confidence
        sorted_emotions = sorted(
            scores.keys(),
            key=lambda e: (scores[e], best_confidence.get(e, 0.0)),
            reverse=True,
        )
        primary_emotion = sorted_emotions[0]

        # Weighted average confidence for the primary emotion
        agreeing_sources = [(s, w) for _, s, w in non_neutral_sources if s["primary_emotion"] == primary_emotion]
        total_weighted_conf = sum(s["confidence"] * w for s, w in agreeing_sources)
        total_weight = sum(w for _, w in agreeing_sources)
        confidence = total_weighted_conf / total_weight if total_weight > 0 else 0.0
        confidence = min(max(confidence, 0.0), 1.0)

        # Intensity by weighted majority among agreeing sources
        primary_intensities = intensities.get(primary_emotion, {"low": 1.0, "medium": 0.0, "high": 0.0})
        intensity = max(
            primary_intensities.keys(),
            key=lambda i: (primary_intensities.get(i, 0.0), _INTENSITY_RANK.get(i, 0)),
        )

        return _make_result(primary_emotion, round(confidence, 2), intensity, "fusion")

    # All sources are neutral — weighted average confidence and intensity
    total_weighted_conf = sum(s["confidence"] * w for _, s, w in neutral_sources)
    total_weight = sum(w for _, _, w in neutral_sources)
    avg_conf = total_weighted_conf / total_weight if total_weight > 0 else 0.0

    neutral_intensities: Dict[str, float] = {"low": 0.0, "medium": 0.0, "high": 0.0}
    for _, src, weight in neutral_sources:
        neutral_intensities[src.get("intensity", "low")] += weight
    intensity = max(
        neutral_intensities.keys(),
        key=lambda i: (neutral_intensities.get(i, 0.0), _INTENSITY_RANK.get(i, 0)),
    )

    return _make_result("neutral", round(avg_conf, 2), intensity, "fusion")


# ---------------------------------------------------------------------------
# Prompt steering
# ---------------------------------------------------------------------------

_PROMPT_HINTS: Dict[str, str] = {
    "frustrated": (
        "Detected affect signal: the user may be frustrated. "
        "Acknowledge friction briefly, stay calm, and be concise."
    ),
    "angry": (
        "Detected affect signal: the user may be upset. "
        "Stay calm, do not escalate, keep responses short and respectful."
    ),
    "anxious": (
        "Detected affect signal: the user may be anxious or under pressure. "
        "Reassure with clear, actionable steps and a calm tone."
    ),
    "confused": (
        "Detected affect signal: the user may be confused. "
        "Be extra clear, break down explanations, and confirm understanding."
    ),
    "sad": (
        "Detected affect signal: the user may be discouraged. "
        "Be supportive, offer constructive next steps, and keep it gentle."
    ),
    "urgent": (
        "Detected affect signal: the user may perceive urgency. "
        "Prioritize actionable, concise guidance without alarm."
    ),
}


def _fmt_emotion(label: str, emotion: Optional[EmotionResult]) -> str | None:
    if not emotion:
        return None
    primary = emotion.get("primary_emotion", "unknown")
    confidence = emotion.get("confidence", 0)
    intensity = emotion.get("intensity", "low")
    source = emotion.get("source", "unknown")
    return f"{label}: {primary} (confidence: {confidence}, intensity: {intensity}, source: {source})"


def build_emotion_prompt_hint(
    fused_emotion: Optional[EmotionResult],
    text_emotion: Optional[EmotionResult] = None,
    visual_pre: Optional[EmotionResult] = None,
    audio_prosody: Optional[EmotionResult] = None,
    phases: Optional[dict[str, EmotionResult]] = None,
) -> Optional[str]:
    """Return a structured emotion context block for the LLM system prompt.
    All available sources are shown regardless of confidence or emotion type."""
    has_any = bool(
        fused_emotion or text_emotion or visual_pre or audio_prosody or phases
    )
    if not has_any:
        return None

    lines: list[str] = ["[Emotion Context]"]

    if fused_emotion:
        lines.append(_fmt_emotion("Fused", fused_emotion))

    if text_emotion:
        lines.append(_fmt_emotion("Text", text_emotion))
        signals = text_emotion.get("signals")
        if signals:
            lines.append(f"Text signals: {', '.join(signals)}")

    if audio_prosody:
        lines.append(_fmt_emotion("Audio (prosody)", audio_prosody))

    if visual_pre:
        lines.append(_fmt_emotion("Webcam", visual_pre))
        vfr = visual_pre.get("valid_face_frames")
        sfr = visual_pre.get("sampled_frames")
        fdr = visual_pre.get("face_detection_rate")
        parts: list[str] = []
        if vfr is not None:
            parts.append(f"valid faces: {vfr}")
        if sfr is not None:
            parts.append(f"sampled: {sfr}")
        if fdr is not None:
            parts.append(f"detection rate: {fdr:.0%}")
        if parts:
            lines.append(f"Webcam details: {', '.join(parts)}")

    if phases:
        phase_labels = {
            "waiting_for_response": "while waiting for response",
            "reading": "while reading",
            "typing": "while typing",
            "speaking": "while speaking",
        }
        phase_lines: list[str] = []
        for phase_name, entry in phases.items():
            emo = entry.get("primary_emotion", "unknown")
            conf = entry.get("confidence", 0)
            label = phase_labels.get(phase_name, phase_name)
            phase_lines.append(f"{label}: {emo} (confidence: {conf})")
        if phase_lines:
            lines.append("Phase emotions: " + " | ".join(phase_lines))

    # Append instruction hint only if fused emotion matches a known hint
    if fused_emotion:
        hint = _PROMPT_HINTS.get(fused_emotion.get("primary_emotion"))
        if hint:
            lines.append(f"Instruction: {hint}")

    return "\n".join(lines)
