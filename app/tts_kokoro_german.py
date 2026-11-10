"""German Kokoro ONNX TTS using the Godelaune/Kokoro-82M-ONNX-German-Martin model."""

from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
import wave
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import HTTPException

logger = logging.getLogger(__name__)

REPO_ID = "Godelaune/Kokoro-82M-ONNX-German-Martin"
DEFAULT_ONNX_FILENAME = "kokoro-martin.onnx"
DEFAULT_VOICES_FILENAME = "voices-martin.npz"
SAMPLE_RATE = 24000


class GermanKokoroONNXTTSService:
    """Local German TTS using pykokoro (ONNX runtime).

    Downloads model files from HuggingFace Hub on first use and caches them
    locally. Uses the ``pykokoro`` Python package for phonemization,
    tokenization and ONNX inference orchestration.

    Supports arbitrary Kokoro ONNX models by passing explicit
    ``model_filename`` and ``voices_filename``.
    """

    def __init__(
        self,
        model_dir: Path,
        default_voice: str = "martin",
        speed: float = 1.0,
        lang: str = "de",
        hf_token: Optional[str] = None,
        model_filename: Optional[str] = None,
        voices_filename: Optional[str] = None,
    ) -> None:
        self._model_dir = model_dir
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._default_voice = default_voice
        self._speed = speed
        self._lang = lang
        self._hf_token = hf_token
        self._model_filename = model_filename or DEFAULT_ONNX_FILENAME
        self._voices_filename = voices_filename or DEFAULT_VOICES_FILENAME
        self._pipeline = None
        self._lock = threading.Lock()

    @property
    def model_path(self) -> Path:
        return self._model_dir / self._model_filename

    @property
    def voices_path(self) -> Path:
        return self._model_dir / self._voices_filename

    def is_ready(self) -> bool:
        """Return True if pykokoro is importable and model files exist."""
        try:
            from pykokoro import KokoroPipeline  # noqa: F401
        except ImportError:
            return False
        return self.model_path.exists() and self.voices_path.exists()

    def _ensure_model(self):
        """Lazy-download ONNX + voice files and create the pykokoro pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            try:
                from pykokoro import KokoroPipeline, PipelineConfig, GenerationConfig
            except ImportError as exc:
                logger.error("Failed to import pykokoro: %s", exc)
                raise HTTPException(
                    status_code=503,
                    detail=f"pykokoro not installed: {exc}. Run: pip install pykokoro",
                ) from exc

            onnx_path = self.model_path
            voices_path = self.voices_path

            if not onnx_path.exists() or not voices_path.exists():
                # Only auto-download the default German Martin model
                if (
                    self._model_filename == DEFAULT_ONNX_FILENAME
                    and self._voices_filename == DEFAULT_VOICES_FILENAME
                ):
                    try:
                        from huggingface_hub import hf_hub_download
                    except ImportError as exc:
                        raise HTTPException(
                            status_code=503,
                            detail="huggingface_hub not installed; cannot download model",
                        ) from exc

                    logger.info(
                        "Downloading German Kokoro ONNX model from %s to %s ...",
                        REPO_ID,
                        self._model_dir,
                    )
                    hf_hub_download(
                        repo_id=REPO_ID,
                        filename=DEFAULT_ONNX_FILENAME,
                        local_dir=str(self._model_dir),
                        token=self._hf_token,
                    )
                    hf_hub_download(
                        repo_id=REPO_ID,
                        filename=DEFAULT_VOICES_FILENAME,
                        local_dir=str(self._model_dir),
                        token=self._hf_token,
                    )
                    logger.info("German Kokoro ONNX model files downloaded.")
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Kokoro model files not found: {onnx_path.name}, {voices_path.name}",
                    )

            logger.info("Loading Kokoro ONNX model via pykokoro ...")

            # Resolve actual voice name from .npz keys (ONNX filename != voice name inside .npz)
            actual_voice = self._default_voice
            if voices_path.exists():
                try:
                    with np.load(voices_path) as npz:
                        available_voices = list(npz.files)
                    if actual_voice not in available_voices:
                        for v in available_voices:
                            if actual_voice.endswith(v):
                                actual_voice = v
                                break
                        else:
                            if available_voices:
                                actual_voice = available_voices[0]
                except Exception as exc:
                    logger.warning("Failed to read voice names from %s: %s", voices_path, exc)

            try:
                config = PipelineConfig(
                    voice=actual_voice,
                    model_path=onnx_path,
                    voices_path=voices_path,
                    generation=GenerationConfig(lang=self._lang),
                )
                self._pipeline = KokoroPipeline(config)
            except Exception as exc:
                logger.exception("Failed to initialize pykokoro pipeline")
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load Kokoro ONNX model via pykokoro: {exc}",
                ) from exc

            logger.info("Kokoro ONNX TTS ready (voice=%s, model=%s).", actual_voice, onnx_path.name)
            return self._pipeline

    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> bytes:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._run_synthesize, text, voice_id or self._default_voice
        )

    async def synthesize_streaming(
        self, text: str, voice_id: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """Stream WAV bytes per sentence, yielding as soon as each sentence is synthesized."""
        loop = asyncio.get_event_loop()
        voice = voice_id or self._default_voice
        logger.info("[Kokoro] streaming start: voice=%s text_len=%d", voice, len(text))

        # Ensure model is loaded (blocking call in thread pool)
        await loop.run_in_executor(None, self._ensure_model)
        pipe = self._pipeline

        sentences = self._split_sentences(text)
        logger.info("[Kokoro] split into %d sentences", len(sentences))
        if not sentences:
            raise HTTPException(status_code=400, detail="No synthesizable text after normalization")

        yielded_any = False
        for sentence in sentences:
            audio = await loop.run_in_executor(
                None, self._synthesize_sentence, pipe, sentence, voice
            )
            if audio is not None and len(audio) > 0:
                wav = self._to_wav_bytes(audio)
                logger.info("[Kokoro] streaming sentence: samples=%d wav=%d bytes", len(audio), len(wav))
                yielded_any = True
                yield wav

        if not yielded_any:
            logger.error("[Kokoro] all %d sentences produced no audio", len(sentences))
            raise HTTPException(status_code=500, detail="Kokoro produced no audio")

    def _run_synthesize(self, text: str, voice: str) -> bytes:
        logger.info("[Kokoro] synthesize start: voice=%s text_len=%d", voice, len(text))
        pipe = self._ensure_model()

        sentences = self._split_sentences(text)
        logger.info("[Kokoro] split into %d sentences", len(sentences))
        if not sentences:
            raise HTTPException(status_code=400, detail="No synthesizable text after normalization")

        audio_segments: list[np.ndarray] = []
        for sentence in sentences:
            audio = self._synthesize_sentence(pipe, sentence, voice)
            if audio is not None and len(audio) > 0:
                audio_segments.append(audio)

        if not audio_segments:
            logger.error("[Kokoro] all %d sentences produced no audio", len(sentences))
            raise HTTPException(status_code=500, detail="Kokoro produced no audio")

        full_audio = np.concatenate(audio_segments)
        logger.info("[Kokoro] total audio samples=%d", len(full_audio))
        return self._to_wav_bytes(full_audio)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text at real sentence boundaries for sentence-level synthesis."""
        text = re.sub(r"\n\s*\n", "\n\n", text)
        parts = re.split(r"(?<=[.!?])\s+(?=\S)", text)
        result: list[str] = []
        for part in parts:
            for line in part.split("\n"):
                line = line.strip()
                if line and len(line) > 1:
                    result.append(line)
        return result

    def _synthesize_sentence(
        self,
        pipe,
        sentence: str,
        voice: str,
    ) -> Optional[np.ndarray]:
        """Synthesize a single sentence using pykokoro pipeline."""
        # Newlines confuse the espeak wrapper
        sentence = sentence.replace("\n", " ").strip()
        # Replace typographic quotes/dashes that can trip espeak
        safe = re.sub(r'[«‹–""]', "'", sentence).strip()
        if not safe:
            return None

        logger.info("[Kokoro] sentence: %r", safe)

        try:
            result = pipe.run(safe)
            audio = result.audio
            if audio is None or len(audio) == 0:
                logger.warning("[Kokoro] sentence produced no audio")
                return None
            arr = np.asarray(audio, dtype=np.float32)
            logger.info("[Kokoro] sentence OK: samples=%d", len(arr))
            return arr
        except ValueError as exc:
            err_str = str(exc)
            logger.warning("[Kokoro] sentence ValueError: %s", err_str)
            if "number of lines in input and output must be equal" in err_str:
                # espeak choked on an abbreviation; strip exotic chars
                very_safe = re.sub(
                    r"[^\w\säöüßÄÖÜ.,:;!?\"'\-]", " ", safe
                ).strip()
                if very_safe:
                    try:
                        result = pipe.run(very_safe)
                        audio = result.audio
                        if audio is not None and len(audio) > 0:
                            arr = np.asarray(audio, dtype=np.float32)
                            logger.info("[Kokoro] fallback sentence OK: samples=%d", len(arr))
                            return arr
                    except Exception:
                        pass
            logger.exception("[Kokoro] sentence synthesis failed (ValueError)")
            return None
        except Exception as exc:
            logger.exception("[Kokoro] sentence synthesis failed: %s", exc)
            return None

    def _to_wav_bytes(self, samples: np.ndarray) -> bytes:
        """Convert float32 [-1, 1] samples to mono 16-bit WAV bytes."""
        samples = np.clip(samples, -1.0, 1.0)
        pcm = (samples * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()
