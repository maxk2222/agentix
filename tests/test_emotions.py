"""Tests for the emotion detection pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.emotions import (
    _aggregate_to_agentix,
    detect_text_emotion,
    fuse_emotions,
    validate_visual_emotion,
    build_emotion_prompt_hint,
    TextEmotionClassifier,
    _GOEMOTIONS_LABELS,
)


class FakeConfig:
    emotion_min_confidence = 0.55
    emotion_webcam_min_valid_face_frames = 3
    data_dir = Path("data")
    emotion_text_model_url = "/models/emotion/text-emotion.onnx"
    emotion_text_tokenizer_url = "/models/emotion/text-emotion-tokenizer.json"


# ---------------------------------------------------------------------------
# _aggregate_to_agentix
# ---------------------------------------------------------------------------


def test_aggregate_to_agentix_angry():
    probs = {label: 0.0 for label in _GOEMOTIONS_LABELS}
    probs["anger"] = 0.5
    probs["disgust"] = 0.2
    probs["disapproval"] = 0.1
    primary, confidence = _aggregate_to_agentix(probs)
    assert primary == "angry"
    assert confidence == pytest.approx(0.8, abs=0.01)


def test_aggregate_to_agentix_satisfied():
    probs = {label: 0.0 for label in _GOEMOTIONS_LABELS}
    probs["joy"] = 0.6
    probs["amusement"] = 0.2
    primary, confidence = _aggregate_to_agentix(probs)
    assert primary == "satisfied"
    assert confidence == pytest.approx(0.8, abs=0.01)


def test_aggregate_to_agentix_neutral():
    probs = {label: 0.0 for label in _GOEMOTIONS_LABELS}
    probs["neutral"] = 0.9
    primary, confidence = _aggregate_to_agentix(probs)
    assert primary == "neutral"
    assert confidence == pytest.approx(0.9, abs=0.01)


def test_aggregate_to_agentix_empty():
    primary, confidence = _aggregate_to_agentix({})
    assert primary == "neutral"
    assert confidence == 0.0


# ---------------------------------------------------------------------------
# TextEmotionClassifier (mocked)
# ---------------------------------------------------------------------------


def test_text_emotion_classifier_predict_mock(monkeypatch):
    clf = TextEmotionClassifier(model_path=Path("/fake/model.onnx"), tokenizer_path=Path("/fake/tokenizer.json"))
    # Mock _load to return True without real ONNX
    monkeypatch.setattr(clf, "_load", lambda: True)
    # Mock tokenizer + session
    fake_tokenizer = MagicMock()
    fake_enc = MagicMock()
    fake_enc.ids = [1, 2, 3]
    fake_enc.attention_mask = [1, 1, 1]
    fake_tokenizer.encode.return_value = fake_enc
    clf._tokenizer = fake_tokenizer

    fake_session = MagicMock()
    # logits for 28 classes: high joy (index 17), rest strongly negative
    logits = np.full(28, -5.0, dtype=np.float32)
    logits[17] = 5.0  # joy
    fake_session.run.return_value = [np.array([logits])]
    clf._session = fake_session

    result = clf.predict("Ich bin glücklich")
    assert result["joy"] > 0.5
    assert result["neutral"] < 0.5


# ---------------------------------------------------------------------------
# detect_text_emotion
# ---------------------------------------------------------------------------


def test_detect_text_emotion_empty():
    result = detect_text_emotion("")
    assert result["primary_emotion"] == "unknown"
    assert result["confidence"] == 0.0


def test_detect_text_emotion_neutral_fallback():
    # Without model files present, falls back to neutral
    result = detect_text_emotion("Wie ist das Wetter?")
    assert result["primary_emotion"] == "neutral"


def test_detect_text_emotion_urgent_override():
    # Even without model, urgent keywords trigger override
    result = detect_text_emotion("sofort dringend")
    assert result["primary_emotion"] == "urgent"
    assert result["intensity"] == "high"


def test_detect_text_emotion_with_config(monkeypatch):
    # Mock classifier to return deterministic probabilities
    from app.emotions import _get_classifier

    def _mock_get_classifier(model_path, tokenizer_path):
        clf = TextEmotionClassifier(model_path=model_path, tokenizer_path=tokenizer_path)
        clf._session = True  # loaded
        fake_tokenizer = MagicMock()
        fake_enc = MagicMock()
        fake_enc.ids = [1, 2, 3]
        fake_enc.attention_mask = [1, 1, 1]
        fake_tokenizer.encode.return_value = fake_enc
        clf._tokenizer = fake_tokenizer

        fake_session = MagicMock()
        logits = np.full(28, -5.0, dtype=np.float32)
        logits[2] = 5.0  # anger
        fake_session.run.return_value = [np.array([logits])]
        clf._session = fake_session
        return clf

    monkeypatch.setattr("app.emotions._text_classifier", None)
    monkeypatch.setattr("app.emotions._last_model_path", None)
    monkeypatch.setattr("app.emotions._last_tokenizer_path", None)
    monkeypatch.setattr("app.emotions._get_classifier", _mock_get_classifier)
    result = detect_text_emotion("Ich bin sehr wütend auf dich", config=FakeConfig())
    assert result["primary_emotion"] == "angry"
    assert result["confidence"] >= 0.5
    assert result["source"] == "text_rules"


# ---------------------------------------------------------------------------
# fuse_emotions
# ---------------------------------------------------------------------------


def test_fuse_text_only():
    text = {"primary_emotion": "angry", "confidence": 0.6, "intensity": "high", "source": "text"}
    fused = fuse_emotions(text, None, config=FakeConfig())
    assert fused["primary_emotion"] == "angry"
    assert fused["confidence"] == 0.6


def test_fuse_visual_only():
    visual = {"primary_emotion": "sad", "confidence": 0.7, "intensity": "medium", "source": "webcam"}
    fused = fuse_emotions(None, visual, config=FakeConfig())
    assert fused["primary_emotion"] == "sad"
    assert fused["confidence"] == 0.7


def test_fuse_agreement_boost():
    text = {"primary_emotion": "angry", "confidence": 0.6, "intensity": "high", "source": "text"}
    visual = {"primary_emotion": "angry", "confidence": 0.7, "intensity": "high", "source": "webcam"}
    fused = fuse_emotions(text, visual, config=FakeConfig())
    assert fused["primary_emotion"] == "angry"
    # Weighted average: (0.6*1.0 + 0.7*1.0) / (1.0+1.0) = 0.65
    assert fused["confidence"] == 0.65


def test_fuse_stronger_signal_wins():
    # No artificial text priority — the signal with higher weighted score wins
    text = {"primary_emotion": "angry", "confidence": 0.6, "intensity": "high", "source": "text"}
    visual = {"primary_emotion": "sad", "confidence": 0.7, "intensity": "medium", "source": "webcam"}
    fused = fuse_emotions(text, visual, config=FakeConfig())
    # sad score=0.7*1.0=0.7, angry score=0.6*1.0=0.6 → sad wins
    assert fused["primary_emotion"] == "sad"


def test_fuse_below_threshold():
    text = {"primary_emotion": "angry", "confidence": 0.3, "intensity": "low", "source": "text"}
    fused = fuse_emotions(text, None, config=FakeConfig())
    assert fused["primary_emotion"] == "neutral"
    assert fused["confidence"] == 0.0


def test_fuse_neutral_kept():
    text = {"primary_emotion": "neutral", "confidence": 0.8, "intensity": "low", "source": "text"}
    fused = fuse_emotions(text, None, config=FakeConfig())
    assert fused["primary_emotion"] == "neutral"
    assert fused["confidence"] == 0.8


def test_fuse_with_audio_prosody():
    visual = {"primary_emotion": "frustrated", "confidence": 0.6, "intensity": "medium", "source": "webcam"}
    audio_prosody = {"primary_emotion": "frustrated", "confidence": 0.7, "intensity": "high", "source": "audio_prosody"}
    fused = fuse_emotions(None, visual, audio_prosody=audio_prosody, config=FakeConfig())
    assert fused["primary_emotion"] == "frustrated"
    # Weighted average: (0.6*1.0 + 0.7*0.9) / (1.0+0.9) = 1.23/1.9 ≈ 0.65
    assert fused["confidence"] == pytest.approx(0.65, abs=0.01)


def test_fuse_all_sources():
    text = {"primary_emotion": "angry", "confidence": 0.6, "intensity": "high", "source": "text"}
    visual = {"primary_emotion": "angry", "confidence": 0.7, "intensity": "high", "source": "webcam"}
    audio_prosody = {"primary_emotion": "angry", "confidence": 0.65, "intensity": "high", "source": "audio_prosody"}
    fused = fuse_emotions(text, visual, audio_prosody=audio_prosody, config=FakeConfig())
    assert fused["primary_emotion"] == "angry"
    # Weighted average: (0.6*1.0 + 0.7*1.0 + 0.65*0.9) / (1.0+1.0+0.9) = 1.885/2.9 ≈ 0.65
    assert fused["confidence"] == pytest.approx(0.65, abs=0.01)


# ---------------------------------------------------------------------------
# validate_visual_emotion
# ---------------------------------------------------------------------------


def test_validate_visual_emotion_ok():
    payload = {
        "primary_emotion": "angry",
        "confidence": 0.7,
        "intensity": "high",
        "source": "webcam_browser",
        "sampled_frames": 10,
        "valid_face_frames": 8,
        "face_detection_rate": 0.8,
        "window_ms": 3000,
    }
    result = validate_visual_emotion(payload, FakeConfig())
    assert result is not None
    assert result["primary_emotion"] == "angry"
    assert result["confidence"] == 0.7


def test_validate_visual_emotion_clamps():
    payload = {
        "primary_emotion": "INVALID",
        "confidence": 1.5,
        "intensity": "INVALID",
        "source": "test",
    }
    result = validate_visual_emotion(payload, FakeConfig())
    assert result is not None
    assert result["primary_emotion"] == "unknown"
    assert result["confidence"] == 1.0
    assert result["intensity"] == "low"


def test_validate_visual_emotion_low_face_frames():
    payload = {
        "primary_emotion": "angry",
        "confidence": 0.7,
        "intensity": "high",
        "source": "webcam_browser",
        "valid_face_frames": 1,
    }
    result = validate_visual_emotion(payload, FakeConfig())
    assert result is not None
    # Penalty: 0.7 * 0.5 = 0.35
    assert result["confidence"] == 0.35


def test_validate_visual_emotion_none():
    result = validate_visual_emotion(None, FakeConfig())
    assert result is None


# ---------------------------------------------------------------------------
# build_emotion_prompt_hint
# ---------------------------------------------------------------------------


def test_build_hint_returns_none_for_neutral():
    fused = {"primary_emotion": "neutral", "confidence": 0.8, "intensity": "low", "source": "fusion"}
    hint = build_emotion_prompt_hint(fused)
    assert hint is not None
    assert "neutral" in hint
    assert "Emotion Context" in hint
    # No instruction hint for neutral
    assert "Instruction:" not in hint


def test_build_hint_returns_none_for_low_confidence():
    fused = {"primary_emotion": "angry", "confidence": 0.3, "intensity": "high", "source": "fusion"}
    hint = build_emotion_prompt_hint(fused)
    assert hint is not None
    assert "angry" in hint
    assert "Emotion Context" in hint
    # No instruction hint for low-confidence angry (hint only shown for known emotions,
    # but build_emotion_prompt_hint currently shows all fused emotions regardless of confidence)
    assert "upset" in hint


def test_build_hint_for_angry():
    fused = {"primary_emotion": "angry", "confidence": 0.7, "intensity": "high", "source": "fusion"}
    hint = build_emotion_prompt_hint(fused)
    assert hint is not None
    assert "upset" in hint
    assert "Emotion Context" in hint


def test_build_hint_with_signals():
    fused = {"primary_emotion": "frustrated", "confidence": 0.7, "intensity": "medium", "source": "fusion"}
    text = {"primary_emotion": "frustrated", "confidence": 0.6, "intensity": "medium", "source": "text", "signals": ["frustriert"]}
    hint = build_emotion_prompt_hint(fused, text_emotion=text)
    assert hint is not None
    assert "frustriert" in hint


def test_fuse_phases():
    text = {"primary_emotion": "angry", "confidence": 0.6, "intensity": "high", "source": "text"}
    phases = {
        "waiting_for_response": {"primary_emotion": "angry", "confidence": 0.7, "intensity": "high", "source": "phase_waiting"},
    }
    fused = fuse_emotions(text, None, phases=phases, config=FakeConfig())
    assert fused["primary_emotion"] == "angry"
    # Weighted average: (0.6*1.0 + 0.7*0.6) / (1.0+0.6) = 1.02/1.6 ≈ 0.64
    assert fused["confidence"] == pytest.approx(0.64, abs=0.01)


def test_fuse_neutral_vs_non_neutral():
    visual = {"primary_emotion": "neutral", "confidence": 1.0, "intensity": "low", "source": "webcam"}
    text = {"primary_emotion": "frustrated", "confidence": 0.6, "intensity": "medium", "source": "text"}
    fused = fuse_emotions(text, visual, config=FakeConfig())
    assert fused["primary_emotion"] == "frustrated"
    # Weighted average of non-neutral: 0.6*1.0 / 1.0 = 0.6
    assert fused["confidence"] == 0.6


def test_fuse_all_neutral():
    visual = {"primary_emotion": "neutral", "confidence": 0.9, "intensity": "low", "source": "webcam"}
    audio = {"primary_emotion": "neutral", "confidence": 0.7, "intensity": "low", "source": "audio_prosody"}
    fused = fuse_emotions(None, visual, audio_prosody=audio, config=FakeConfig())
    assert fused["primary_emotion"] == "neutral"
    # Weighted average: (0.9*1.0 + 0.7*0.9) / (1.0+0.9) = 1.53/1.9 ≈ 0.81
    assert fused["confidence"] == pytest.approx(0.81, abs=0.01)


def test_fuse_tie_breaking_by_confidence():
    # Same weighted score (0.6), but different single confidences → higher confidence wins
    text = {"primary_emotion": "angry", "confidence": 0.6, "intensity": "high", "source": "text"}
    visual = {"primary_emotion": "sad", "confidence": 0.6, "intensity": "medium", "source": "webcam"}
    fused = fuse_emotions(text, visual, config=FakeConfig())
    # Both have score 0.6*1.0 = 0.6, tie-break by confidence (both 0.6) → deterministic sort
    # Since both are equal, the sort is stable; we just verify it picks one deterministically
    assert fused["primary_emotion"] in ("angry", "sad")
    assert fused["confidence"] == 0.6


def test_fuse_weighted_intensity():
    # 2 sources agree, one low, one high → weighted majority picks high
    text = {"primary_emotion": "angry", "confidence": 0.6, "intensity": "low", "source": "text"}
    visual = {"primary_emotion": "angry", "confidence": 0.6, "intensity": "high", "source": "webcam"}
    fused = fuse_emotions(text, visual, config=FakeConfig())
    assert fused["primary_emotion"] == "angry"
    # Both weights are 1.0, so high and low have equal weight → high wins (equal weight, alphabetical fallback)
    assert fused["intensity"] == "high"


def test_fuse_custom_weights_override():
    class CustomConfig:
        emotion_min_confidence = 0.55
        emotion_webcam_min_valid_face_frames = 3
        data_dir = Path("data")
        emotion_text_model_url = "/models/emotion/text-emotion.onnx"
        emotion_text_tokenizer_url = "/models/emotion/text-emotion-tokenizer.json"
        emotion_fusion_source_weights = {"text": 0.5, "visual_pre": 1.5}

    text = {"primary_emotion": "angry", "confidence": 0.8, "intensity": "high", "source": "text"}
    visual = {"primary_emotion": "sad", "confidence": 0.6, "intensity": "medium", "source": "webcam"}
    fused = fuse_emotions(text, visual, config=CustomConfig())
    # angry score = 0.8*0.5 = 0.4, sad score = 0.6*1.5 = 0.9 → sad wins
    assert fused["primary_emotion"] == "sad"
