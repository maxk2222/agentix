"""Audio router for TTS/STT endpoints and voice management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app.audio_services import STTService, UnifiedTTSService, VoiceManager
from app.auth import require_permission
from app.runtime_config import RuntimeConfigStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["Audio"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_config(request: Request):
    return getattr(request.app.state, "app_config", None)


def _get_runtime_store(request: Request) -> Optional[RuntimeConfigStore]:
    return getattr(request.app.state, "_runtime_config_store", None)


def _get_rag_app(request: Request):
    return getattr(request.app.state, "rag_application", None)


def _get_voice_manager(config) -> VoiceManager:
    return VoiceManager(config.tts_voices_dir)


async def _get_tts_service(request: Request) -> UnifiedTTSService:
    config = _get_config(request)
    store = _get_runtime_store(request)
    if store is not None:
        provider = await store.get("audio_tts_provider", config.audio_tts_provider)
        kokoro_voice = await store.get("audio_tts_kokoro_voice", config.audio_tts_kokoro_voice)
        kokoro_model_dir = await store.get("audio_tts_kokoro_model_dir", config.audio_tts_kokoro_model_dir)
        piper_binary = await store.get("audio_tts_piper_binary", config.audio_tts_piper_binary)
        hf_token = await store.get("hf_token", config.hf_token)
    else:
        provider = config.audio_tts_provider
        kokoro_voice = config.audio_tts_kokoro_voice
        kokoro_model_dir = config.audio_tts_kokoro_model_dir
        piper_binary = config.audio_tts_piper_binary
        hf_token = config.hf_token

    return UnifiedTTSService(
        provider=provider,
        kokoro_model_dir=kokoro_model_dir,
        kokoro_voice=kokoro_voice,
        voices_dir=config.tts_voices_dir,
        piper_binary=piper_binary,
        hf_token=hf_token,
        lang=getattr(config, "audio_tts_language", "de"),
    )


async def _get_stt_service(request: Request) -> STTService:
    config = _get_config(request)
    store = _get_runtime_store(request)
    if store is not None:
        model = await store.get("audio_stt_model", config.audio_stt_model)
        lang = await store.get("audio_stt_language", config.audio_stt_language)
        provider = await store.get("audio_stt_provider", config.audio_stt_provider)
        hf_cache_dir = await store.get("hf_cache_dir", config.hf_cache_dir)
        hf_token = await store.get("hf_token", config.hf_token)
    else:
        model = config.audio_stt_model
        lang = config.audio_stt_language
        provider = config.audio_stt_provider
        hf_cache_dir = config.hf_cache_dir
        hf_token = config.hf_token

    cached = getattr(request.app.state, "_stt_service", None)
    cached_key = getattr(request.app.state, "_stt_service_key", None)
    current_key = (model, lang, provider, hf_cache_dir, hf_token)
    if cached is not None and cached_key == current_key:
        return cached

    svc = STTService(
        model_size=model,
        language=lang,
        provider=provider,
        hf_cache_dir=hf_cache_dir,
        hf_token=hf_token,
    )
    request.app.state._stt_service = svc
    request.app.state._stt_service_key = current_key
    return svc


def _get_runtime_store_from_config(config) -> Optional[RuntimeConfigStore]:
    # This is a helper for service creation where we don't have a request
    return None


async def _resolve_voice_id(config, assistant: Optional[dict], request: Optional[Request] = None) -> Optional[str]:
    """Return the effective voice ID considering assistant and runtime overrides."""
    if assistant and assistant.get("audio_voice_id"):
        voice_id = assistant["audio_voice_id"]
    else:
        voice_id = config.audio_tts_default_voice
        # Check runtime config override (e.g. set via admin voice activation)
        if request is not None:
            store = _get_runtime_store(request)
            if store is not None:
                try:
                    runtime_voice_id = await store.get("audio_tts_default_voice")
                    if runtime_voice_id is not None:
                        voice_id = runtime_voice_id
                except Exception:
                    pass
    if not voice_id:
        return None
    # When using Kokoro, voice IDs are string names (e.g. "martin") and do not
    # need to exist as ONNX files on disk.
    provider = getattr(config, "audio_tts_provider", "kokoro")
    if provider == "kokoro":
        return voice_id
    vm = VoiceManager(config.tts_voices_dir)
    if not vm.voice_exists(voice_id):
        logger.warning("Voice '%s' not found in %s", voice_id, config.tts_voices_dir)
        return None
    return voice_id


def _get_assistant_by_name(request: Request, model_name: str) -> Optional[dict]:
    """Look up an assistant by name in the loaded instructions.

    Instructions are stored as a name-indexed dict (keys are assistant names,
    values are the assistant dicts).  We try that format first, then fall back
    to the legacy ``{"assistants": [...]}`` wrapper for safety.
    """
    rag_app = _get_rag_app(request)
    if rag_app is None:
        return None
    instructions = getattr(rag_app, "instructions", None)
    if instructions is None:
        return None
    # Name-indexed format (current)
    assistant = instructions.get(model_name)
    if assistant is not None:
        return assistant
    # Legacy wrapper format
    assistants = instructions.get("assistants", [])
    for assistant in assistants:
        if assistant.get("name") == model_name:
            return assistant
    return None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AudioConfigResponse(BaseModel):
    stt: dict
    tts: dict
    phone: dict


class HealthResponse(BaseModel):
    stt_ready: bool = False
    stt_loading: bool = False
    tts_ready: bool = False
    active_voice: Optional[str] = None
    voices_count: int = 0
    piper_found: bool = False
    piper_path: str = "piper"
    kokoro_ready: bool = False
    tts_provider: str = "piper"


class STTRequest(BaseModel):
    language: Optional[str] = None


class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    stream: bool = False


class TTSSegmentRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Public audio endpoints (no admin required, may be used by widget)
# ---------------------------------------------------------------------------

@router.get("/health")
async def audio_health(request: Request) -> HealthResponse:
    config = _get_config(request)
    if config is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    vm = _get_voice_manager(config)
    tts = await _get_tts_service(request)
    stt = await _get_stt_service(request)

    voices = vm.list_voices()
    store = _get_runtime_store(request)
    active = config.audio_tts_default_voice
    provider = config.audio_tts_provider
    if store is not None:
        active = await store.get("audio_tts_default_voice", active)
        provider = await store.get("audio_tts_provider", provider)
    if provider != "kokoro" and active and not vm.voice_exists(active):
        active = None

    return HealthResponse(
        stt_ready=stt.is_ready() and config.audio_stt_enabled,
        stt_loading=stt.is_loading(),
        tts_ready=tts.is_ready() and config.audio_tts_enabled and (provider == "kokoro" or len(voices) > 0),
        active_voice=active,
        voices_count=len(voices),
        piper_found=tts._piper.is_ready(),
        piper_path=config.audio_tts_piper_binary,
        kokoro_ready=tts._kokoro.is_ready(),
        tts_provider=provider,
    )


@router.get("/config")
async def audio_config(request: Request, model: Optional[str] = None) -> dict:
    config = _get_config(request)
    if config is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    vm = _get_voice_manager(config)
    tts_svc = await _get_tts_service(request)
    stt_svc = await _get_stt_service(request)

    store = _get_runtime_store(request)
    provider = config.audio_tts_provider
    if store is not None:
        provider = await store.get("audio_tts_provider", provider)
    global_stt_enabled = config.audio_stt_enabled and stt_svc.is_ready()
    global_tts_enabled = config.audio_tts_enabled and tts_svc.is_ready() and (provider == "kokoro" or len(vm.list_voices()) > 0)

    assistant: Optional[dict] = None
    if model:
        assistant = _get_assistant_by_name(request, model)

    assistant_stt = bool(assistant and assistant.get("audio_stt_enabled"))
    assistant_tts = bool(assistant and assistant.get("audio_tts_enabled"))
    assistant_phone = bool(assistant and assistant.get("audio_phone_enabled"))
    assistant_streaming = bool(
        assistant is None or assistant.get("audio_tts_streaming_enabled", True)
    )

    voice_id = await _resolve_voice_id(config, assistant, request)

    stt_visible = global_stt_enabled and assistant_stt
    tts_visible = global_tts_enabled and assistant_tts
    phone_visible = assistant_phone
    # For Kokoro, use the configured voice even if no ONNX files are present
    if provider == "kokoro" and voice_id is None:
        voice_id = getattr(config, "audio_tts_kokoro_voice", "martin")

    store = _get_runtime_store(request)
    runtime_lang = config.audio_stt_language
    if store is not None:
        runtime_lang = await store.get("audio_stt_language", config.audio_stt_language)

    return {
        "stt": {
            "enabled": global_stt_enabled,
            "ready": stt_svc.is_ready(),
            "button_visible": stt_visible,
            "language": runtime_lang,
            "provider": config.audio_stt_provider,
            "model": config.audio_stt_model,
        },
        "tts": {
            "enabled": global_tts_enabled,
            "ready": tts_svc.is_ready(),
            "button_visible": tts_visible,
            "streaming_enabled": config.audio_tts_streaming_enabled and assistant_streaming,
            "voice_id": voice_id,
            "auto_read": bool(assistant and assistant.get("audio_tts_auto_read")),
        },
        "phone": {
            "enabled": assistant_phone,
            "button_visible": phone_visible,
        },
    }


@router.post("/stt")
async def audio_stt(
    request: Request,
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
) -> dict:
    config = _get_config(request)
    if config is None or not config.audio_stt_enabled:
        raise HTTPException(status_code=503, detail="STT not enabled")

    stt = await _get_stt_service(request)
    if not stt.is_ready():
        raise HTTPException(status_code=503, detail="STT service not ready")

    audio_bytes = await audio.read()
    if len(audio_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large (max 50 MB)")

    store = _get_runtime_store(request)
    min_snr_db = getattr(config, "audio_min_snr_db", None)
    noise_gate_min = getattr(config, "audio_noise_gate_min", None)
    noise_gate_max = getattr(config, "audio_noise_gate_max", None)
    if store is not None:
        min_snr_db = await store.get("audio_min_snr_db", min_snr_db)
        noise_gate_min = await store.get("audio_noise_gate_min", noise_gate_min)
        noise_gate_max = await store.get("audio_noise_gate_max", noise_gate_max)

    text = await stt.transcribe(
        audio_bytes,
        language=language,
        min_snr_db=min_snr_db,
        noise_gate_min=noise_gate_min,
        noise_gate_max=noise_gate_max,
    )
    if not text.strip():
        raise HTTPException(status_code=400, detail="No speech detected")
    return {"text": text}


@router.post("/tts")
async def audio_tts(request: Request, payload: TTSRequest):
    config = _get_config(request)
    if config is None or not config.audio_tts_enabled:
        raise HTTPException(status_code=503, detail="TTS not enabled")

    tts = await _get_tts_service(request)
    if not tts.is_ready():
        raise HTTPException(status_code=503, detail="TTS service not ready")

    store = _get_runtime_store(request)
    provider = config.audio_tts_provider
    kokoro_voice = config.audio_tts_kokoro_voice
    default_voice = config.audio_tts_default_voice
    if store is not None:
        provider = await store.get("audio_tts_provider", provider)
        kokoro_voice = await store.get("audio_tts_kokoro_voice", kokoro_voice)
        default_voice = await store.get("audio_tts_default_voice", default_voice)
    if payload.voice_id:
        voice_id = payload.voice_id
    elif provider == "kokoro":
        voice_id = kokoro_voice
    else:
        voice_id = default_voice
    if not voice_id:
        raise HTTPException(status_code=400, detail="No voice configured")

    if provider != "kokoro":
        vm = _get_voice_manager(config)
        if not vm.voice_exists(voice_id):
            raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")

    if len(payload.text) > 5000:
        raise HTTPException(status_code=413, detail="Text too long (max 5000 chars)")

    if payload.stream:
        boundary = "wav-segment"

        async def _stream_parts():
            async for wav_bytes in tts.synthesize_streaming(payload.text, voice_id):
                yield (
                    f"--{boundary}\r\n"
                    f"Content-Type: audio/wav\r\n"
                    f"Content-Length: {len(wav_bytes)}\r\n"
                    f"\r\n".encode("utf-8")
                )
                yield wav_bytes
                yield b"\r\n"
            yield f"--{boundary}--\r\n".encode("utf-8")

        return StreamingResponse(
            _stream_parts(),
            media_type=f"multipart/mixed; boundary={boundary}",
        )

    wav_bytes = await tts.synthesize(payload.text, voice_id)
    return Response(content=wav_bytes, media_type="audio/wav")


@router.post("/tts/segment")
async def audio_tts_segment(request: Request, payload: TTSSegmentRequest) -> Response:
    config = _get_config(request)
    if config is None or not config.audio_tts_enabled:
        raise HTTPException(status_code=503, detail="TTS not enabled")

    tts = await _get_tts_service(request)
    if not tts.is_ready():
        raise HTTPException(status_code=503, detail="TTS service not ready")

    store = _get_runtime_store(request)
    provider = config.audio_tts_provider
    kokoro_voice = config.audio_tts_kokoro_voice
    default_voice = config.audio_tts_default_voice
    if store is not None:
        provider = await store.get("audio_tts_provider", provider)
        kokoro_voice = await store.get("audio_tts_kokoro_voice", kokoro_voice)
        default_voice = await store.get("audio_tts_default_voice", default_voice)
    if payload.voice_id:
        voice_id = payload.voice_id
    elif provider == "kokoro":
        voice_id = kokoro_voice
    else:
        voice_id = default_voice
    if not voice_id:
        raise HTTPException(status_code=400, detail="No voice configured")

    if provider != "kokoro":
        vm = _get_voice_manager(config)
        if not vm.voice_exists(voice_id):
            raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")

    if len(payload.text) > 1000:
        raise HTTPException(status_code=413, detail="Segment too long (max 1000 chars)")

    wav_bytes = await tts.synthesize(payload.text, voice_id)
    return Response(content=wav_bytes, media_type="audio/wav")


