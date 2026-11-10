"""Async model download utility with simple file-lock race protection."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


async def download_model(url: str, dest: Path, timeout: float = 300.0) -> bool:
    """Download a model from *url* to *dest* if the file does not already exist.

    Uses a sibling ``.lock`` file to avoid racing downloads during uvicorn reload.
    Returns ``True`` when the file exists (either already or after download).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Model already present: %s", dest)
        return True

    lock_path = dest.with_suffix(dest.suffix + ".lock")

    # Simple file-based lock (non-blocking)
    try:
        fd = lock_path.open("x")
    except FileExistsError:
        # Another process is downloading; wait until it finishes
        logger.info("Model download in progress by another worker: %s", dest.name)
        for _ in range(int(timeout)):
            await asyncio.sleep(1)
            if dest.exists() and dest.stat().st_size > 0:
                return True
            if not lock_path.exists():
                break
        return dest.exists() and dest.stat().st_size > 0

    try:
        logger.info("Downloading model %s -> %s", url, dest)
        start = time.perf_counter()
        headers = {"Accept": "application/octet-stream"}
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                total_bytes = 0
                with dest.open("wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
                        total_bytes += len(chunk)
        elapsed = time.perf_counter() - start
        logger.info(
            "Model download complete: %s (%d bytes in %.1fs)",
            dest.name,
            total_bytes,
            elapsed,
        )
        return True
    except Exception as exc:
        logger.error("Model download failed for %s: %s", url, exc)
        # Clean up partial file
        if dest.exists():
            dest.unlink()
        return False
    finally:
        try:
            fd.close()
            lock_path.unlink()
        except Exception:
            pass


async def ensure_emotion_models(config) -> None:
    """Download missing ONNX emotion models based on *config* fields.

    Checks the three model files under ``data_dir / "models" / "emotion"``.
    If a file is missing and its corresponding ``*_download_url`` is set,
    the model is fetched asynchronously.

    For the expression model, the download URL and filename are chosen
    based on ``emotion_webcam_expression_model_type`` so that FER and
    AffectNet models can coexist.
    """
    models_dir = Path(config.data_dir) / "models" / "emotion"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Determine expression model filename based on model type
    model_type = getattr(config, "emotion_webcam_expression_model_type", "affectnet")
    if model_type == "fer":
        expr_filename = "expression-fer.onnx"
        # Re-use the existing legacy FER model on disk if available
        existing_fer = models_dir / "expression.onnx"
        if existing_fer.exists() and existing_fer.stat().st_size > 0:
            expr_download_url = None  # Skip download, existing model will be used
        else:
            expr_download_url = getattr(
                config,
                "emotion_webcam_expression_model_download_url",
                "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx",
            )
    else:
        expr_filename = "expression-affectnet.onnx"
        expr_download_url = getattr(
            config,
            "emotion_webcam_expression_model_download_url",
            "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx",
        )

    # If FER is selected and the legacy model exists, copy it to the expected filename
    if model_type == "fer" and expr_download_url is None:
        fer_dest = models_dir / expr_filename
        if not fer_dest.exists():
            import shutil
            shutil.copy2(models_dir / "expression.onnx", fer_dest)
            logger.info("Copied legacy expression.onnx to %s", fer_dest)

    models = [
        (config.emotion_webcam_face_model_download_url, models_dir / "face-detector.onnx"),
        (expr_download_url, models_dir / expr_filename),
        (config.emotion_audio_model_download_url, models_dir / "audio-emotion.onnx"),
        (getattr(config, "emotion_text_model_download_url", None), models_dir / "text-emotion.onnx"),
        (getattr(config, "emotion_text_tokenizer_download_url", None), models_dir / "text-emotion-tokenizer.json"),
    ]

    tasks = []
    for download_url, dest in models:
        if download_url and not (dest.exists() and dest.stat().st_size > 0):
            tasks.append(download_model(download_url, dest))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for exc in results:
            if isinstance(exc, Exception):
                logger.error("Emotion model download error: %s", exc)
