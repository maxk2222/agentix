"""Tests for the public emotion router."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.routers.emotions import emotion_config, detect_text_emotion_endpoint


class FakeConfig:
    emotion_detection_enabled = True
    emotion_text_model_enabled = True
    emotion_text_model_url = "/models/emotion/text-emotion.onnx"
    emotion_text_model_download_url = "https://example.com/model.onnx"
    emotion_text_tokenizer_url = "/models/emotion/text-emotion-tokenizer.json"
    emotion_text_tokenizer_download_url = "https://example.com/tokenizer.json"
    emotion_webcam_enabled = False
    emotion_webcam_runtime = "onnxruntime-web"
    emotion_webcam_execution_providers = "webgpu,wasm"
    emotion_webcam_face_detector = "onnx"
    emotion_webcam_face_model_url = "/models/emotion/face-detector.onnx"
    emotion_webcam_face_model_download_url = "https://example.com/face.onnx"
    emotion_webcam_expression_model_url = "/models/emotion/expression-affectnet.onnx"
    emotion_webcam_expression_model_download_url = "https://example.com/expr.onnx"
    emotion_webcam_expression_model_type = "affectnet"
    emotion_webcam_expression_input_size = 224
    emotion_webcam_snippet_window_ms = 3000
    emotion_webcam_sample_fps = 3
    emotion_webcam_max_frames_per_snippet = 12
    emotion_webcam_min_valid_face_frames = 3
    emotion_webcam_frame_upload_enabled = False
    emotion_prompt_steering_enabled = True
    emotion_min_confidence = 0.55
    emotion_analytics_enabled = True
    emotion_audio_enabled = False
    emotion_audio_model_url = "/models/emotion/audio-emotion.onnx"
    emotion_audio_model_download_url = "https://example.com/audio.onnx"


class FakeRequest:
    def __init__(self, config, assistants=None) -> None:
        rag_application = None
        if assistants is not None:
            rag_application = SimpleNamespace(instructions={"assistants": assistants})
        self.app = SimpleNamespace(
            state=SimpleNamespace(app_config=config, rag_application=rag_application)
        )


@pytest.mark.asyncio
async def test_emotion_config_returns_text_fields():
    req = FakeRequest(FakeConfig())
    result = await emotion_config(req)
    # Without a model param assistant_enabled is False, so overall enabled is False
    assert result["enabled"] is False
    assert result["text"]["enabled"] is False
    assert result["text"]["model_enabled"] is True
    assert result["text"]["model_url"] == "/models/emotion/text-emotion.onnx"
    assert result["text"]["model_download_url"] == "https://example.com/model.onnx"
    assert result["text"]["tokenizer_url"] == "/models/emotion/text-emotion-tokenizer.json"
    assert result["text"]["tokenizer_download_url"] == "https://example.com/tokenizer.json"


@pytest.mark.asyncio
async def test_disabled_assistant_hides_widget_emotion_controls():
    """Assistant master-off must override modality-level webcam/audio flags."""
    config = FakeConfig()
    config.emotion_webcam_enabled = True
    config.emotion_audio_enabled = True
    req = FakeRequest(
        config,
        assistants=[
            {
                "name": "DisabledAssistant",
                "emotion_detection_enabled": False,
                "emotion_webcam_enabled": True,
                "emotion_audio_enabled": True,
            }
        ],
    )

    result = await emotion_config(req, model="DisabledAssistant")

    assert result["enabled"] is False
    assert result["webcam"]["enabled"] is False
    assert result["webcam"]["button_visible"] is False
    assert result["audio"]["enabled"] is False
    assert result["text"]["enabled"] is False
    assert result["prompt_steering"]["enabled"] is False
    assert result["analytics"]["enabled"] is False


@pytest.mark.asyncio
async def test_enabled_assistant_can_enable_webcam_and_audio_controls():
    config = FakeConfig()
    config.emotion_webcam_enabled = True
    config.emotion_audio_enabled = True
    req = FakeRequest(
        config,
        assistants=[
            {
                "name": "EnabledAssistant",
                "emotion_detection_enabled": True,
                "emotion_webcam_enabled": True,
                "emotion_audio_enabled": True,
            }
        ],
    )

    result = await emotion_config(req, model="EnabledAssistant")

    assert result["enabled"] is True
    assert result["webcam"]["enabled"] is True
    assert result["webcam"]["button_visible"] is True
    assert result["audio"]["enabled"] is True


@pytest.mark.asyncio
async def test_detect_text_emotion_endpoint(monkeypatch):
    req = FakeRequest(FakeConfig())
    payload = SimpleNamespace(text="Ich bin wütend")

    # Mock detect_text_emotion to avoid loading real ONNX
    def _mock_detect(text, *, config=None):
        return {
            "primary_emotion": "angry",
            "confidence": 0.85,
            "intensity": "high",
            "source": "text_rules",
            "timestamp_ms": 12345,
            "signals": ["anger(0.85)"],
        }

    monkeypatch.setattr("app.emotions.detect_text_emotion", _mock_detect)
    result = await detect_text_emotion_endpoint(req, payload)
    assert result["primary_emotion"] == "angry"
    assert result["confidence"] == 0.85
    assert result["source"] == "text_rules"
