"""Admin audio voice management endpoints under /admin/audio."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app.audio_services import KokoroVoiceManager, PiperTTSService, VoiceManager
from app.auth import require_permission
from app.runtime_config import RuntimeConfigStore
from app.tts_kokoro_german import GermanKokoroONNXTTSService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/audio", tags=["Admin Audio"])


# ---------------------------------------------------------------------------
# Keys that the audio page manages (written to RuntimeConfigStore)
# ---------------------------------------------------------------------------
AUDIO_SETTING_KEYS = {
    "audio_tts_provider",
    "audio_stt_provider",
    "audio_stt_model",
    "audio_stt_language",
    "audio_tts_default_voice",
    "audio_tts_kokoro_voice",
    "audio_min_snr_db",
    "audio_noise_gate_min",
    "audio_noise_gate_max",
}


def _get_config(request: Request):
    return getattr(request.app.state, "app_config", None)


def _get_runtime_store(request: Request) -> Optional[RuntimeConfigStore]:
    return getattr(request.app.state, "_runtime_config_store", None)


def _get_voice_manager(config) -> VoiceManager:
    return VoiceManager(config.tts_voices_dir)


def _get_kokoro_manager(config) -> KokoroVoiceManager:
    return KokoroVoiceManager(config.audio_tts_kokoro_model_dir)


class VoiceDownloadRequest(BaseModel):
    onnx_url: str
    config_url: str
    voice_id: str


class AudioSettingsPayload(BaseModel):
    audio_tts_provider: Optional[str] = None
    audio_stt_provider: Optional[str] = None
    audio_stt_model: Optional[str] = None
    audio_stt_language: Optional[str] = None
    audio_tts_default_voice: Optional[str] = None
    audio_tts_kokoro_voice: Optional[str] = None
    audio_min_snr_db: Optional[float] = None
    audio_noise_gate_min: Optional[float] = None
    audio_noise_gate_max: Optional[float] = None


@router.get("/voices")
async def admin_list_voices(request: Request) -> dict:
    await require_permission(request, "instructions:read")
    config = _get_config(request)
    if config is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    vm = _get_voice_manager(config)
    voices = vm.list_voices()
    active = config.audio_tts_default_voice
    for v in voices:
        v["active"] = v["id"] == active
    return {
        "voices_dir": str(config.tts_voices_dir),
        "voices": voices,
    }


@router.post("/voices/upload")
async def admin_upload_voice(
    request: Request,
    onnx_file: UploadFile = File(...),
    config_file: UploadFile = File(...),
) -> dict:
    await require_permission(request, "instructions:write")
    cfg = _get_config(request)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    vm = _get_voice_manager(cfg)
    onnx_name = onnx_file.filename or ""
    config_name = config_file.filename or ""
    vm.validate_upload(onnx_name, config_name)

    voice_id = Path(onnx_name).stem
    onnx_bytes = await onnx_file.read()
    config_bytes = await config_file.read()

    if len(onnx_bytes) > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="ONNX file too large (max 500 MB)")
    if len(config_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Config file too large (max 5 MB)")

    vm.save_voice_files(onnx_bytes, config_bytes, voice_id)
    return {"success": True, "voice_id": voice_id}


@router.post("/voices/download")
async def admin_download_voice(request: Request, payload: VoiceDownloadRequest) -> dict:
    await require_permission(request, "instructions:write")
    cfg = _get_config(request)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    vm = _get_voice_manager(cfg)
    await vm.download_voice(payload.onnx_url, payload.config_url, payload.voice_id)
    return {"success": True, "voice_id": payload.voice_id}


@router.delete("/voices/{voice_id}")
async def admin_delete_voice(request: Request, voice_id: str) -> dict:
    await require_permission(request, "instructions:write")
    cfg = _get_config(request)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    vm = _get_voice_manager(cfg)
    if not vm.voice_exists(voice_id):
        raise HTTPException(status_code=404, detail="Voice not found")

    if cfg.audio_tts_default_voice == voice_id:
        store = _get_runtime_store(request)
        if store is not None:
            await store.set("audio_tts_default_voice", None)

    vm.delete_voice(voice_id)
    return {"success": True}


@router.post("/voices/{voice_id}/activate")
async def admin_activate_voice(request: Request, voice_id: str) -> dict:
    await require_permission(request, "instructions:write")
    cfg = _get_config(request)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    vm = _get_voice_manager(cfg)
    if not vm.voice_exists(voice_id):
        raise HTTPException(status_code=404, detail="Voice not found")

    store = _get_runtime_store(request)
    if store is not None:
        await store.set("audio_tts_default_voice", voice_id)
    return {"success": True, "voice_id": voice_id}


@router.post("/voices/test")
async def admin_test_voice(
    request: Request,
    voice_id: str = Form(...),
    text: str = Form(default="Hallo, das ist ein Test."),
) -> Response:
    await require_permission(request, "instructions:read")
    cfg = _get_config(request)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    vm = _get_voice_manager(cfg)
    if not vm.voice_exists(voice_id):
        raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")

    if len(text) > 500:
        raise HTTPException(status_code=413, detail="Text too long for test (max 500 chars)")

    piper = PiperTTSService(
        vm._voices_dir,
        getattr(cfg, "audio_tts_piper_binary", "piper"),
    )
    if not piper.is_ready():
        raise HTTPException(status_code=503, detail="Piper TTS not ready")

    wav_bytes = await piper.synthesize(text, voice_id)
    return Response(content=wav_bytes, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Audio settings (managed on the audio page, stored in RuntimeConfigStore)
# ---------------------------------------------------------------------------

@router.get("/settings")
async def admin_get_audio_settings(request: Request) -> dict:
    await require_permission(request, "instructions:read")
    config = _get_config(request)
    if config is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    store = _get_runtime_store(request)
    settings: dict[str, Optional[object]] = {}
    for key in AUDIO_SETTING_KEYS:
        default = getattr(config, key, None)
        if store is not None:
            settings[key] = await store.get(key, default)
        else:
            settings[key] = default
    return settings


@router.put("/settings")
async def admin_save_audio_settings(request: Request, payload: AudioSettingsPayload) -> dict:
    await require_permission(request, "runtime_config:write")
    store = _get_runtime_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Runtime config store unavailable")

    values: dict[str, object] = {}
    for key, value in payload.model_dump(exclude_none=True).items():
        values[key] = value
    if values:
        await store.set_many(values)
    return {"success": True, "updated_keys": list(values.keys())}


# ---------------------------------------------------------------------------
# Kokoro voices
# ---------------------------------------------------------------------------

@router.get("/voices/kokoro")
async def admin_list_kokoro_voices(request: Request) -> dict:
    await require_permission(request, "instructions:read")
    config = _get_config(request)
    if config is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    km = _get_kokoro_manager(config)
    voices = km.list_voices()
    active = getattr(config, "audio_tts_kokoro_voice", "martin")
    for v in voices:
        v["active"] = v["id"] == active
    return {
        "models_dir": str(config.audio_tts_kokoro_model_dir),
        "voices": voices,
    }


@router.post("/voices/kokoro/upload")
async def admin_upload_kokoro_voice(
    request: Request,
    onnx_file: UploadFile = File(...),
    npz_file: UploadFile = File(...),
) -> dict:
    await require_permission(request, "instructions:write")
    cfg = _get_config(request)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    km = _get_kokoro_manager(cfg)
    onnx_name = onnx_file.filename or ""
    npz_name = npz_file.filename or ""
    km.validate_upload(onnx_name, npz_name)

    voice_id = Path(onnx_name).stem
    onnx_bytes = await onnx_file.read()
    npz_bytes = await npz_file.read()

    if len(onnx_bytes) > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="ONNX file too large (max 500 MB)")
    if len(npz_bytes) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Voices file too large (max 100 MB)")

    km.save_voice_files(onnx_bytes, npz_bytes, voice_id)
    return {"success": True, "voice_id": voice_id}


@router.delete("/voices/kokoro/{voice_id}")
async def admin_delete_kokoro_voice(request: Request, voice_id: str) -> dict:
    await require_permission(request, "instructions:write")
    cfg = _get_config(request)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    km = _get_kokoro_manager(cfg)
    if not km.voice_exists(voice_id):
        raise HTTPException(status_code=404, detail="Kokoro voice not found")

    kokoro_voice = getattr(cfg, "audio_tts_kokoro_voice", "martin")
    if kokoro_voice == voice_id:
        store = _get_runtime_store(request)
        if store is not None:
            await store.set("audio_tts_kokoro_voice", None)

    km.delete_voice(voice_id)
    return {"success": True}


@router.post("/voices/kokoro/{voice_id}/activate")
async def admin_activate_kokoro_voice(request: Request, voice_id: str) -> dict:
    await require_permission(request, "instructions:write")
    cfg = _get_config(request)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    km = _get_kokoro_manager(cfg)
    if not km.voice_exists(voice_id):
        raise HTTPException(status_code=404, detail="Kokoro voice not found")

    store = _get_runtime_store(request)
    if store is not None:
        await store.set("audio_tts_kokoro_voice", voice_id)
    return {"success": True, "voice_id": voice_id}


@router.post("/voices/kokoro/test")
async def admin_test_kokoro_voice(
    request: Request,
    voice_id: str = Form(...),
    text: str = Form(default="Hallo, das ist ein Test."),
) -> Response:
    await require_permission(request, "instructions:read")
    cfg = _get_config(request)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    km = _get_kokoro_manager(cfg)
    if not km.voice_exists(voice_id):
        raise HTTPException(status_code=404, detail=f"Kokoro voice not found: {voice_id}")

    if len(text) > 500:
        raise HTTPException(status_code=413, detail="Text too long for test (max 500 chars)")

    onnx_path, npz_path = km.get_voice_files(voice_id)
    if onnx_path is None or npz_path is None:
        raise HTTPException(status_code=404, detail="Kokoro model files incomplete")

    # Resolve actual voice name from .npz keys (ONNX filename != voice name inside .npz)
    try:
        with np.load(npz_path) as npz:
            available_voices = list(npz.files)
    except Exception as exc:
        logger.warning("Failed to read voice names from %s: %s", npz_path, exc)
        available_voices = []

    actual_voice = voice_id
    if voice_id not in available_voices:
        # Try suffix match, e.g. kokoro-martin -> martin
        for v in available_voices:
            if voice_id.endswith(v):
                actual_voice = v
                break
        else:
            if available_voices:
                actual_voice = available_voices[0]

    kokoro = GermanKokoroONNXTTSService(
        model_dir=onnx_path.parent,
        default_voice=actual_voice,
        model_filename=onnx_path.name,
        voices_filename=npz_path.name,
    )
    if not kokoro.is_ready():
        raise HTTPException(status_code=503, detail="Kokoro TTS not ready (pykokoro missing?)")

    wav_bytes = await kokoro.synthesize(text, actual_voice)
    return Response(content=wav_bytes, media_type="audio/wav")
