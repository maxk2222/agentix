"""Emotion detection router for effective config and optional vision analysis."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.auth import require_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/emotion", tags=["Emotion"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_config(request: Request):
    return getattr(request.app.state, "app_config", None)


def _get_rag_app(request: Request):
    return getattr(request.app.state, "rag_application", None)


def _get_assistant_by_name(request: Request, model_name: str) -> Optional[dict]:
    """Look up an assistant by name in the loaded instructions."""
    rag_app = _get_rag_app(request)
    if rag_app is None:
        return None
    instructions = getattr(rag_app, "instructions", None)
    if instructions is None:
        return None
    assistant = instructions.get(model_name)
    if assistant is not None:
        return assistant
    assistants = instructions.get("assistants", [])
    for a in assistants:
        if a.get("name") == model_name:
            return a
    return None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class EmotionConfigResponse(BaseModel):
    enabled: bool = False
    text: dict = {}
    webcam: dict = {}
    prompt_steering: dict = {}
    analytics: dict = {}


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------


@router.get("/config")
async def emotion_config(request: Request, model: Optional[str] = None) -> dict:
    """Return effective emotion config for a given model/assistant."""
    config = _get_config(request)
    if config is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    assistant: Optional[dict] = None
    if model:
        assistant = _get_assistant_by_name(request, model)

    global_enabled = config.emotion_detection_enabled
    assistant_enabled = bool(assistant and assistant.get("emotion_detection_enabled"))
    enabled = global_enabled and assistant_enabled

    global_text = getattr(config, "emotion_text_model_enabled", True)
    assistant_text = bool(assistant is None or assistant.get("emotion_text_enabled", True))
    text_enabled = enabled and global_text and assistant_text

    global_webcam = config.emotion_webcam_enabled
    assistant_webcam = bool(assistant is None or assistant.get("emotion_webcam_enabled", False))
    # The modality toggles are subordinate to the assistant-level master
    # switch. Otherwise a disabled assistant can still expose the widget
    # emotion button when webcam/audio is independently enabled.
    webcam_visible = enabled and global_webcam and assistant_webcam

    global_steering = config.emotion_prompt_steering_enabled
    assistant_steering = bool(assistant is None or assistant.get("emotion_prompt_steering_enabled", True))
    steering_enabled = enabled and global_steering and assistant_steering

    global_analytics = config.emotion_analytics_enabled
    assistant_analytics = bool(assistant is None or assistant.get("emotion_analytics_enabled", True))
    analytics_enabled = enabled and global_analytics and assistant_analytics

    global_audio = config.emotion_audio_enabled
    assistant_audio = bool(assistant is None or assistant.get("emotion_audio_enabled", global_audio))
    audio_visible = enabled and global_audio and assistant_audio

    # Determine expression model URL based on model type
    model_type = config.emotion_webcam_expression_model_type
    if model_type == "fer":
        expression_model_url = "/models/emotion/expression-fer.onnx"
    else:
        expression_model_url = "/models/emotion/expression-affectnet.onnx"

    return {
        "enabled": enabled,
        "text": {
            "enabled": text_enabled,
            "model_enabled": global_text,
            "model_url": getattr(config, "emotion_text_model_url", "/models/emotion/text-emotion.onnx"),
            "model_download_url": getattr(config, "emotion_text_model_download_url", None),
            "tokenizer_url": getattr(config, "emotion_text_tokenizer_url", "/models/emotion/text-emotion-tokenizer.json"),
            "tokenizer_download_url": getattr(config, "emotion_text_tokenizer_download_url", None),
        },
        "webcam": {
            "enabled": webcam_visible,
            "button_visible": webcam_visible,
            "runtime": config.emotion_webcam_runtime,
            "execution_providers": config.emotion_webcam_execution_providers.split(","),
            "face_detector": config.emotion_webcam_face_detector,
            "face_model_url": config.emotion_webcam_face_model_url,
            "face_model_download_url": config.emotion_webcam_face_model_download_url,
            "expression_model_url": expression_model_url,
            "expression_model_download_url": config.emotion_webcam_expression_model_download_url,
            "expression_model_type": config.emotion_webcam_expression_model_type,
            "expression_input_size": config.emotion_webcam_expression_input_size,
            "snippet_window_ms": config.emotion_webcam_snippet_window_ms,
            "sample_fps": config.emotion_webcam_sample_fps,
            "max_frames_per_snippet": config.emotion_webcam_max_frames_per_snippet,
            "min_valid_face_frames": config.emotion_webcam_min_valid_face_frames,
            "frame_upload_enabled": config.emotion_webcam_frame_upload_enabled,
        },
        "audio": {
            "enabled": audio_visible,
            "model_url": config.emotion_audio_model_url,
            "model_download_url": config.emotion_audio_model_download_url,
            "min_snr_db": getattr(config, "audio_min_snr_db", None) or 10,
            "noise_gate_min": getattr(config, "audio_noise_gate_min", None) or 0.001,
            "noise_gate_max": getattr(config, "audio_noise_gate_max", None) or 0.5,
        },
        "prompt_steering": {
            "enabled": steering_enabled,
            "min_confidence": config.emotion_min_confidence,
        },
        "analytics": {
            "enabled": analytics_enabled,
        },
        "fusion": {
            "source_weights": getattr(config, "emotion_fusion_source_weights", None) or {
                "text": 1.0,
                "visual_pre": 1.0,
                "audio_prosody": 0.9,
                "phase_typing": 0.8,
                "phase_waiting_for_response": 0.6,
                "phase_reading": 0.4,
            },
            "min_confidence": getattr(config, "emotion_fusion_min_confidence", None) or {
                "text": 0.55,
                "visual_pre": 0.55,
                "audio_prosody": 0.55,
                "phase": 0.55,
            },
        },
    }


class TextEmotionDetectRequest(BaseModel):
    text: str


@router.post("/detect-text")
async def detect_text_emotion_endpoint(request: Request, payload: TextEmotionDetectRequest) -> dict:
    """Run the ONNX text emotion classifier on the provided text."""
    config = _get_config(request)
    if config is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    from app.emotions import detect_text_emotion

    result = detect_text_emotion(payload.text, config=config)
    return dict(result)
