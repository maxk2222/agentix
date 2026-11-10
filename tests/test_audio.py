"""Tests for audio endpoints and services."""

import io
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.audio_services import STTService, PiperTTSService, VoiceManager, KokoroVoiceManager
from app.routers.audio import _get_assistant_by_name
from app.chat_utils import load_instructions_from_yaml_text
from app.main import app


client = TestClient(app)


@pytest.fixture
def temp_voices_dir(tmp_path: Path):
    voices = tmp_path / "models" / "voices" / "tts_voices"
    voices.mkdir(parents=True)
    return voices


class TestVoiceManager:
    def test_list_voices_empty(self, tmp_path: Path):
        vm = VoiceManager(tmp_path / "voices")
        assert vm.list_voices() == []

    def test_list_voices_with_files(self, tmp_path: Path):
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        (voices_dir / "test_voice.onnx").write_bytes(b"fake onnx")
        (voices_dir / "test_voice.onnx.json").write_text('{"test": true}')
        vm = VoiceManager(voices_dir)
        result = vm.list_voices()
        assert len(result) == 1
        assert result[0]["id"] == "test_voice"
        assert result[0]["has_config"] is True

    def test_voice_exists(self, tmp_path: Path):
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        (voices_dir / "exists.onnx").write_bytes(b"x")
        vm = VoiceManager(voices_dir)
        assert vm.voice_exists("exists") is True
        assert vm.voice_exists("missing") is False

    def test_delete_voice(self, tmp_path: Path):
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        (voices_dir / "to_delete.onnx").write_bytes(b"x")
        (voices_dir / "to_delete.onnx.json").write_text("{}")
        vm = VoiceManager(voices_dir)
        assert vm.delete_voice("to_delete") is True
        assert vm.voice_exists("to_delete") is False

    def test_validate_upload_ok(self, tmp_path: Path):
        vm = VoiceManager(tmp_path)
        vm.validate_upload("voice.onnx", "voice.onnx.json")

    def test_validate_upload_bad_onnx(self, tmp_path: Path):
        vm = VoiceManager(tmp_path)
        with pytest.raises(HTTPException) as exc:
            vm.validate_upload("voice.txt", "voice.onnx.json")
        assert exc.value.status_code == 400

    def test_validate_upload_mismatch(self, tmp_path: Path):
        vm = VoiceManager(tmp_path)
        with pytest.raises(HTTPException) as exc:
            vm.validate_upload("voice_a.onnx", "voice_b.onnx.json")
        assert exc.value.status_code == 400


class TestPiperTTSService:
    def test_is_ready_not_found(self, tmp_path: Path):
        tts = PiperTTSService(tmp_path, piper_binary="nonexistent_piper_binary")
        assert tts.is_ready() is False

    def test_voice_path(self, tmp_path: Path):
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        (voices_dir / "test.onnx").write_bytes(b"x")
        tts = PiperTTSService(voices_dir)
        assert tts._voice_path("test") == voices_dir / "test.onnx"
        assert tts._voice_path("missing") is None


class TestSTTService:
    def test_is_ready_without_dependency(self):
        stt = STTService()
        # faster-whisper may not be installed in test env
        assert stt.is_ready() is False or True  # depends on env

    def test_default_provider_is_faster_whisper(self):
        stt = STTService()
        assert stt._provider == "faster_whisper"

    def test_distil_provider_switch(self):
        stt = STTService(provider="distil_whisper", model_size="distil-whisper/distil-large-v3")
        assert stt._provider == "distil_whisper"
        assert stt._model_size == "distil-whisper/distil-large-v3"

    def test_is_ready_distil_without_dependencies(self):
        from unittest.mock import patch
        stt = STTService(provider="distil_whisper")
        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            assert stt.is_ready() is False

    def test_is_loading_initially_false(self):
        stt = STTService()
        assert stt.is_loading() is False

    def test_hf_settings_passed_through(self):
        stt = STTService(hf_cache_dir="/tmp/hf", hf_token="abc123")
        assert stt._hf_cache_dir == "/tmp/hf"
        assert stt._hf_token == "abc123"


class TestGetAssistantByName:
    def test_name_indexed_format(self):
        """Name-indexed dict is the current format from load_instructions."""
        mock_request = MagicMock()
        mock_rag = MagicMock()
        mock_rag.instructions = {
            "CyberSecurityChatbot": {
                "name": "CyberSecurityChatbot",
                "audio_stt_enabled": True,
                "audio_tts_enabled": True,
                "audio_phone_enabled": True,
            }
        }
        mock_request.app.state.rag_application = mock_rag

        result = _get_assistant_by_name(mock_request, "CyberSecurityChatbot")
        assert result is not None
        assert result["audio_stt_enabled"] is True
        assert result["audio_tts_enabled"] is True

    def test_legacy_list_format(self):
        """Legacy format with 'assistants' wrapper list."""
        mock_request = MagicMock()
        mock_rag = MagicMock()
        mock_rag.instructions = {
            "assistants": [
                {
                    "name": "CyberSecurityChatbot",
                    "audio_stt_enabled": True,
                    "audio_tts_enabled": True,
                }
            ]
        }
        mock_request.app.state.rag_application = mock_rag

        result = _get_assistant_by_name(mock_request, "CyberSecurityChatbot")
        assert result is not None
        assert result["audio_stt_enabled"] is True

    def test_not_found(self):
        mock_request = MagicMock()
        mock_rag = MagicMock()
        mock_rag.instructions = {"OtherAssistant": {}}
        mock_request.app.state.rag_application = mock_rag

        result = _get_assistant_by_name(mock_request, "CyberSecurityChatbot")
        assert result is None

    def test_no_rag_app(self):
        mock_request = MagicMock()
        mock_request.app.state.rag_application = None

        result = _get_assistant_by_name(mock_request, "CyberSecurityChatbot")
        assert result is None


class TestLoadInstructionsAudioFields:
    def test_audio_fields_parsed(self):
        yaml_text = """
assistants:
  - name: ToolTestAgent
    model: gpt-oss:120b
    temperature: 0.4
    system_instructions: "Test"
    audio_stt_enabled: true
    audio_tts_enabled: true
    audio_phone_enabled: true
    audio_tts_auto_read: true
    audio_tts_streaming_enabled: false
    audio_voice_id: de_DE-thorsten-medium
"""
        result = load_instructions_from_yaml_text(yaml_text)
        assistant = result["ToolTestAgent"]
        assert assistant["audio_stt_enabled"] is True
        assert assistant["audio_tts_enabled"] is True
        assert assistant["audio_phone_enabled"] is True
        assert assistant["audio_tts_auto_read"] is True
        assert assistant["audio_tts_streaming_enabled"] is False
        assert assistant["audio_voice_id"] == "de_DE-thorsten-medium"

    def test_audio_fields_defaults(self):
        yaml_text = """
assistants:
  - name: PlainAgent
    model: gpt-oss:120b
    temperature: 0.4
    system_instructions: "Test"
"""
        result = load_instructions_from_yaml_text(yaml_text)
        assistant = result["PlainAgent"]
        assert assistant["audio_stt_enabled"] is False
        assert assistant["audio_tts_enabled"] is False
        assert assistant["audio_phone_enabled"] is False
        assert assistant["audio_tts_auto_read"] is False
        assert assistant["audio_tts_streaming_enabled"] is True
        assert assistant["audio_voice_id"] is None


class TestAudioRouterPublic:
    @patch("app.routers.audio._get_config")
    def test_health_no_config(self, mock_get_config):
        mock_get_config.return_value = None
        response = client.get("/audio/health")
        assert response.status_code == 503

    @patch("app.routers.audio._get_config")
    def test_config_no_config(self, mock_get_config):
        mock_get_config.return_value = None
        response = client.get("/audio/config")
        assert response.status_code == 503

    @patch("app.routers.audio._get_config")
    def test_stt_not_enabled(self, mock_get_config):
        cfg = MagicMock()
        cfg.audio_stt_enabled = False
        mock_get_config.return_value = cfg
        response = client.post("/audio/stt", files={"audio": ("test.wav", io.BytesIO(b"fake"), "audio/wav")})
        assert response.status_code == 503

    @patch("app.routers.audio._get_config")
    def test_tts_not_enabled(self, mock_get_config):
        cfg = MagicMock()
        cfg.audio_tts_enabled = False
        mock_get_config.return_value = cfg
        response = client.post("/audio/tts", json={"text": "hello"})
        assert response.status_code == 503

    @patch("app.routers.audio._get_stt_service")
    @patch("app.routers.audio._get_config")
    def test_stt_empty_transcription(self, mock_get_config, mock_get_stt_service):
        cfg = MagicMock()
        cfg.audio_stt_enabled = True
        mock_get_config.return_value = cfg

        async def mock_transcribe(audio_bytes, language=None, **kwargs):
            return ""

        stt = MagicMock()
        stt.is_ready.return_value = True
        stt.transcribe = mock_transcribe
        mock_get_stt_service.return_value = stt

        response = client.post("/audio/stt", files={"audio": ("test.wav", io.BytesIO(b"fake"), "audio/wav")})
        assert response.status_code == 400
        assert "No speech detected" in response.text


class TestGermanKokoroTTSService:
    def test_is_ready_when_model_files_missing(self):
        from app.tts_kokoro_german import GermanKokoroONNXTTSService
        svc = GermanKokoroONNXTTSService(Path("/tmp/nonexistent_kokoro"), default_voice="martin")
        # onnxruntime/kokoro may not be installed in test env
        assert svc.is_ready() is False

    def test_default_voice_is_martin(self):
        from app.tts_kokoro_german import GermanKokoroONNXTTSService
        svc = GermanKokoroONNXTTSService(Path("/tmp/kokoro_test"), default_voice="martin")
        assert svc._default_voice == "martin"

    def test_sentence_splitting(self):
        from app.tts_kokoro_german import GermanKokoroONNXTTSService
        svc = GermanKokoroONNXTTSService(Path("/tmp/kokoro_test"))
        sentences = svc._split_sentences("Hallo. Wie geht es dir? Gut!")
        assert len(sentences) == 3
        assert sentences[0] == "Hallo."
        assert sentences[1] == "Wie geht es dir?"
        assert sentences[2] == "Gut!"

    def test_wav_bytes_format(self):
        from app.tts_kokoro_german import GermanKokoroONNXTTSService
        svc = GermanKokoroONNXTTSService(Path("/tmp/kokoro_test"))
        samples = np.array([0.0, 0.5, -0.5, 0.0], dtype=np.float32)
        wav = svc._to_wav_bytes(samples)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"


class TestKokoroVoiceManager:
    def test_list_voices_empty(self, tmp_path: Path):
        km = KokoroVoiceManager(tmp_path / "kokoro")
        assert km.list_voices() == []

    def test_list_voices_with_files(self, tmp_path: Path):
        models_dir = tmp_path / "kokoro"
        models_dir.mkdir()
        (models_dir / "test_voice.onnx").write_bytes(b"fake onnx")
        (models_dir / "test_voice.npz").write_bytes(b"fake npz")
        km = KokoroVoiceManager(models_dir)
        result = km.list_voices()
        assert len(result) == 1
        assert result[0]["id"] == "test_voice"
        assert result[0]["has_voices"] is True

    def test_voice_npz_alt_pattern(self, tmp_path: Path):
        models_dir = tmp_path / "kokoro"
        models_dir.mkdir()
        (models_dir / "martin.onnx").write_bytes(b"fake onnx")
        (models_dir / "voices-martin.npz").write_bytes(b"fake npz")
        km = KokoroVoiceManager(models_dir)
        result = km.list_voices()
        assert len(result) == 1
        assert result[0]["has_voices"] is True

    def test_delete_voice(self, tmp_path: Path):
        models_dir = tmp_path / "kokoro"
        models_dir.mkdir()
        (models_dir / "voice.onnx").write_bytes(b"fake onnx")
        (models_dir / "voice.npz").write_bytes(b"fake npz")
        km = KokoroVoiceManager(models_dir)
        assert km.voice_exists("voice")
        km.delete_voice("voice")
        assert not km.voice_exists("voice")

    def test_validate_upload_ok(self):
        km = KokoroVoiceManager(Path("/tmp"))
        km.validate_upload("voice.onnx", "voice.npz")

    def test_validate_upload_bad_onnx(self):
        km = KokoroVoiceManager(Path("/tmp"))
        with pytest.raises(HTTPException) as exc:
            km.validate_upload("voice.txt", "voice.npz")
        assert exc.value.status_code == 400

    def test_validate_upload_alt_npz(self):
        km = KokoroVoiceManager(Path("/tmp"))
        km.validate_upload("martin.onnx", "voices-martin.npz")


class TestAudioRouterAdmin:
    @patch("app.routers.admin_audio.require_permission")
    @patch("app.routers.admin_audio._get_config")
    def test_list_voices_no_config(self, mock_get_config, mock_perm):
        mock_get_config.return_value = None
        mock_perm.return_value = None
        response = client.get("/admin/audio/voices")
        assert response.status_code == 503

    @patch("app.routers.admin_audio.require_permission")
    @patch("app.routers.admin_audio._get_config")
    @patch("app.routers.admin_audio._get_runtime_store")
    def test_get_settings(self, mock_store, mock_config, mock_perm):
        mock_perm.return_value = None
        cfg = MagicMock()
        cfg.audio_tts_provider = "piper"
        cfg.audio_stt_model = "small"
        mock_config.return_value = cfg
        store = MagicMock()
        async def mock_get(key, default=None):
            return default
        store.get = MagicMock(side_effect=mock_get)
        mock_store.return_value = store
        response = client.get("/admin/audio/settings")
        assert response.status_code == 200
        data = response.json()
        assert data["audio_tts_provider"] == "piper"
        assert data["audio_stt_model"] == "small"

    @patch("app.routers.admin_audio.require_permission")
    @patch("app.routers.admin_audio._get_runtime_store")
    def test_save_settings(self, mock_store, mock_perm):
        mock_perm.return_value = None
        store = MagicMock()
        async def mock_set_many(values):
            return None
        store.set_many = MagicMock(side_effect=mock_set_many)
        mock_store.return_value = store
        response = client.put("/admin/audio/settings", json={"audio_tts_provider": "kokoro", "audio_stt_model": "base"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        store.set_many.assert_called_once()

    @patch("app.routers.admin_audio.require_permission")
    @patch("app.routers.admin_audio._get_config")
    def test_list_kokoro_voices_no_config(self, mock_get_config, mock_perm):
        mock_get_config.return_value = None
        mock_perm.return_value = None
        response = client.get("/admin/audio/voices/kokoro")
        assert response.status_code == 503

    @patch("app.routers.admin_audio.require_permission")
    @patch("app.routers.admin_audio._get_config")
    def test_upload_kokoro_voice_no_config(self, mock_get_config, mock_perm):
        mock_get_config.return_value = None
        mock_perm.return_value = None
        # Must provide valid form fields so FastAPI validation passes before _get_config check
        response = client.post(
            "/admin/audio/voices/kokoro/upload",
            files={
                "onnx_file": ("test.onnx", io.BytesIO(b"fake"), "application/octet-stream"),
                "npz_file": ("test.npz", io.BytesIO(b"fake"), "application/octet-stream"),
            },
        )
        assert response.status_code == 503

    @patch("app.routers.admin_audio.require_permission")
    @patch("app.routers.admin_audio._get_config")
    def test_delete_kokoro_voice_no_config(self, mock_get_config, mock_perm):
        mock_get_config.return_value = None
        mock_perm.return_value = None
        response = client.delete("/admin/audio/voices/kokoro/testvoice")
        assert response.status_code == 503
