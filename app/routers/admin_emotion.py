"""Admin emotion model management endpoints under /admin/emotion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.auth import require_permission
from app.runtime_config import RuntimeConfigStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/emotion", tags=["Admin Emotion"])

# ---------------------------------------------------------------------------
# Runtime config keys that this router manages
# ---------------------------------------------------------------------------

MODEL_TYPE_KEYS = {
    "face_detector": "emotion_webcam_face_model_url",
    "expression": "emotion_webcam_expression_model_url",
    "audio_emotion": "emotion_audio_model_url",
    "text_emotion": "emotion_text_model_url",
}

MODEL_TYPE_FILES = {
    "face_detector": "face-detector.onnx",
    "expression": "expression-affectnet.onnx",
    "audio_emotion": "audio-emotion.onnx",
    "text_emotion": "text-emotion.onnx",
}


def _get_default_filename(config, model_type: str) -> Optional[str]:
    if model_type == "expression":
        mtype = getattr(config, "emotion_webcam_expression_model_type", "affectnet")
        return "expression-fer.onnx" if mtype == "fer" else "expression-affectnet.onnx"
    return MODEL_TYPE_FILES.get(model_type)


class ModelDownloadRequest(BaseModel):
    model_type: str
    url: str


def _get_config(request: Request):
    return getattr(request.app.state, "app_config", None)


def _get_runtime_store(request: Request) -> Optional[RuntimeConfigStore]:
    return getattr(request.app.state, "_runtime_config_store", None)


def _get_models_dir(config) -> Path:
    return Path(config.data_dir) / "models" / "emotion"


def _list_models(models_dir: Path, model_type: str) -> list[dict]:
    suffix = ".onnx"
    files = sorted(
        [f for f in models_dir.iterdir() if f.is_file() and f.suffix == suffix],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return [
        {
            "filename": f.name,
            "size": f.stat().st_size,
            "modified_at": f.stat().st_mtime,
        }
        for f in files
    ]


def _get_active_filename(config, model_type: str) -> Optional[str]:
    key = MODEL_TYPE_KEYS.get(model_type)
    if not key:
        return None
    url = getattr(config, key, "")
    if not url:
        return None
    # URL may be /models/emotion/filename.onnx
    return Path(url).name


@router.get("/models")
async def admin_list_emotion_models(request: Request) -> dict:
    await require_permission(request, "instructions:read")
    config = _get_config(request)
    if config is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    models_dir = _get_models_dir(config)
    models_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, dict] = {}
    for mtype in MODEL_TYPE_KEYS:
        result[mtype] = {
            "models": _list_models(models_dir, mtype),
            "active": _get_active_filename(config, mtype),
            "default_filename": _get_default_filename(config, mtype),
        }
    return {"models_dir": str(models_dir), **result}


@router.post("/models/upload")
async def admin_upload_emotion_model(
    request: Request,
    model_type: str = Form(...),
    onnx_file: UploadFile = File(...),
) -> dict:
    await require_permission(request, "instructions:write")
    cfg = _get_config(request)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    if model_type not in MODEL_TYPE_KEYS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type. Allowed: {', '.join(MODEL_TYPE_KEYS.keys())}",
        )

    onnx_name = onnx_file.filename or ""
    if not onnx_name.endswith(".onnx"):
        raise HTTPException(status_code=400, detail="File must have .onnx extension")

    models_dir = _get_models_dir(cfg)
    models_dir.mkdir(parents=True, exist_ok=True)

    onnx_bytes = await onnx_file.read()
    if len(onnx_bytes) > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="ONNX file too large (max 500 MB)")

    dest = models_dir / onnx_name
    dest.write_bytes(onnx_bytes)
    logger.info("Uploaded emotion model %s to %s", model_type, dest)

    return {"success": True, "filename": onnx_name}


@router.post("/models/{model_type}/activate")
async def admin_activate_emotion_model(
    request: Request, model_type: str, filename: str = Form(...)
) -> dict:
    await require_permission(request, "instructions:write")
    cfg = _get_config(request)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    if model_type not in MODEL_TYPE_KEYS:
        raise HTTPException(status_code=400, detail="Invalid model_type")

    models_dir = _get_models_dir(cfg)
    target = models_dir / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    key = MODEL_TYPE_KEYS[model_type]
    url = f"/models/emotion/{filename}"

    store = _get_runtime_store(request)
    if store is not None:
        await store.set(key, url)
    else:
        raise HTTPException(status_code=503, detail="Runtime config store unavailable")

    logger.info("Activated emotion model %s -> %s", model_type, url)
    return {"success": True, "model_type": model_type, "filename": filename, "url": url}


@router.delete("/models/{model_type}/{filename}")
async def admin_delete_emotion_model(
    request: Request, model_type: str, filename: str
) -> dict:
    await require_permission(request, "instructions:write")
    cfg = _get_config(request)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    if model_type not in MODEL_TYPE_KEYS:
        raise HTTPException(status_code=400, detail="Invalid model_type")

    models_dir = _get_models_dir(cfg)
    target = models_dir / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    target.unlink()
    logger.info("Deleted emotion model %s/%s", model_type, filename)

    # If this was the active model, clear the runtime config
    active = _get_active_filename(cfg, model_type)
    if active == filename:
        store = _get_runtime_store(request)
        if store is not None:
            key = MODEL_TYPE_KEYS[model_type]
            await store.set(key, "")
            logger.info("Cleared active emotion model %s because file was deleted", model_type)

    return {"success": True}


@router.post("/models/download")
async def admin_download_emotion_model(
    request: Request, payload: ModelDownloadRequest
) -> dict:
    await require_permission(request, "instructions:write")
    cfg = _get_config(request)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Config unavailable")

    if payload.model_type not in MODEL_TYPE_KEYS:
        raise HTTPException(status_code=400, detail="Invalid model_type")

    if not payload.url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Only HTTPS URLs are allowed")

    models_dir = _get_models_dir(cfg)
    models_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(payload.url).name or f"{payload.model_type}.onnx"
    dest = models_dir / filename

    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
        try:
            resp = await client.get(payload.url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Download failed: {exc}") from exc

    if len(resp.content) > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="ONNX file too large (max 500 MB)")

    dest.write_bytes(resp.content)
    logger.info("Downloaded emotion model %s to %s", payload.model_type, dest)
    return {"success": True, "filename": filename}
