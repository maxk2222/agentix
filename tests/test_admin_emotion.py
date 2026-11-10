"""Tests for the admin emotion model management router."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.routers.admin_emotion import (
    MODEL_TYPE_KEYS,
    MODEL_TYPE_FILES,
    admin_list_emotion_models,
    admin_activate_emotion_model,
    admin_delete_emotion_model,
)


class FakeConfig:
    data_dir = Path("/tmp/test_models")
    emotion_webcam_face_model_url = "/models/emotion/face-detector.onnx"
    emotion_webcam_expression_model_url = "/models/emotion/expression-affectnet.onnx"
    emotion_webcam_expression_model_type = "affectnet"
    emotion_audio_model_url = "/models/emotion/audio-emotion.onnx"
    emotion_text_model_url = "/models/emotion/text-emotion.onnx"
    dashboard_enabled = True
    dashboard_session_secret = "test-secret"


class FakeRuntimeStore:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def set(self, key: str, value: str) -> None:
        self._data[key] = value


class FakeRequest:
    def __init__(self, config, store=None) -> None:
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                app_config=config,
                _runtime_config_store=store,
            )
        )
        self.cookies = {}


@pytest.mark.asyncio
async def test_model_type_keys_includes_text_emotion():
    assert "text_emotion" in MODEL_TYPE_KEYS
    assert MODEL_TYPE_KEYS["text_emotion"] == "emotion_text_model_url"
    assert MODEL_TYPE_FILES["text_emotion"] == "text-emotion.onnx"


@pytest.mark.asyncio
async def test_list_models_includes_text_emotion(monkeypatch, tmp_path):
    config = FakeConfig()
    config.data_dir = tmp_path
    models_dir = tmp_path / "models" / "emotion"
    models_dir.mkdir(parents=True)
    (models_dir / "text-emotion.onnx").write_bytes(b"fake")

    req = FakeRequest(config)
    monkeypatch.setattr("app.routers.admin_emotion.require_permission", AsyncMock())
    result = await admin_list_emotion_models(req)
    assert "text_emotion" in result
    assert result["text_emotion"]["active"] == "text-emotion.onnx"
    assert result["text_emotion"]["default_filename"] == "text-emotion.onnx"


@pytest.mark.asyncio
async def test_activate_text_emotion_model(monkeypatch, tmp_path):
    config = FakeConfig()
    config.data_dir = tmp_path
    models_dir = tmp_path / "models" / "emotion"
    models_dir.mkdir(parents=True)
    (models_dir / "text-emotion.onnx").write_bytes(b"fake")

    store = FakeRuntimeStore()
    req = FakeRequest(config, store)

    # monkeypatch permission check to always succeed
    monkeypatch.setattr("app.routers.admin_emotion.require_permission", AsyncMock())

    result = await admin_activate_emotion_model(req, "text_emotion", "text-emotion.onnx")
    assert result["success"] is True
    assert result["model_type"] == "text_emotion"
    assert result["filename"] == "text-emotion.onnx"
    assert store._data.get("emotion_text_model_url") == "/models/emotion/text-emotion.onnx"


@pytest.mark.asyncio
async def test_delete_text_emotion_model(monkeypatch, tmp_path):
    config = FakeConfig()
    config.data_dir = tmp_path
    models_dir = tmp_path / "models" / "emotion"
    models_dir.mkdir(parents=True)
    (models_dir / "text-emotion.onnx").write_bytes(b"fake")

    store = FakeRuntimeStore()
    req = FakeRequest(config, store)

    monkeypatch.setattr("app.routers.admin_emotion.require_permission", AsyncMock())

    result = await admin_delete_emotion_model(req, "text_emotion", "text-emotion.onnx")
    assert result["success"] is True
    assert not (models_dir / "text-emotion.onnx").exists()
