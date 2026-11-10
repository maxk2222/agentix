"""Local TTS/STT audio services using Piper and faster-whisper."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import subprocess
import threading
from pathlib import Path
from collections.abc import AsyncGenerator
from typing import Optional, Tuple

import httpx
from fastapi import HTTPException

from app.tts_kokoro_german import GermanKokoroONNXTTSService as KokoroTTSService
from app.tts_text_normalization import normalize_tts_text
from app.utils.audio_cleaner import clean_audio

logger = logging.getLogger(__name__)


class VoiceManager:
    """Manages Piper voice files under a persistent directory."""

    def __init__(self, voices_dir: Path) -> None:
        self._voices_dir = voices_dir
        self._voices_dir.mkdir(parents=True, exist_ok=True)

    @property
    def voices_dir(self) -> Path:
        return self._voices_dir

    def list_voices(self) -> list[dict]:
        """Return all voices with their config files."""
        voices: list[dict] = []
        for onnx_file in sorted(self._voices_dir.glob("*.onnx")):
            config_file = onnx_file.with_suffix(".onnx.json")
            voices.append({
                "id": onnx_file.stem,
                "onnx_path": str(onnx_file),
                "config_path": str(config_file) if config_file.exists() else None,
                "has_config": config_file.exists(),
            })
        return voices

    def get_voice_path(self, voice_id: str) -> Optional[Path]:
        onnx = self._voices_dir / f"{voice_id}.onnx"
        return onnx if onnx.exists() else None

    def voice_exists(self, voice_id: str) -> bool:
        return (self._voices_dir / f"{voice_id}.onnx").exists()

    def delete_voice(self, voice_id: str) -> bool:
        deleted = False
        for suffix in (".onnx", ".onnx.json"):
            path = self._voices_dir / f"{voice_id}{suffix}"
            if path.exists():
                path.unlink()
                deleted = True
        return deleted

    def validate_upload(self, onnx_name: str, config_name: str) -> None:
        if not onnx_name.endswith(".onnx"):
            raise HTTPException(status_code=400, detail="ONNX file must have .onnx extension")
        if not config_name.endswith(".onnx.json"):
            raise HTTPException(status_code=400, detail="Config file must have .onnx.json extension")
        onnx_base = Path(onnx_name).stem
        config_base = Path(config_name).with_suffix("").stem
        if onnx_base != config_base:
            raise HTTPException(status_code=400, detail="ONNX and config file base names must match")

    def save_voice_files(self, onnx_bytes: bytes, config_bytes: bytes, voice_id: str) -> None:
        onnx_path = self._voices_dir / f"{voice_id}.onnx"
        config_path = self._voices_dir / f"{voice_id}.onnx.json"
        onnx_path.write_bytes(onnx_bytes)
        config_path.write_bytes(config_bytes)
        logger.info("Saved voice files for %s", voice_id)

    async def download_voice(self, onnx_url: str, config_url: str, voice_id: str) -> None:
        if not onnx_url.startswith("https://") or not config_url.startswith("https://"):
            raise HTTPException(status_code=400, detail="Only HTTPS URLs are allowed")

        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            try:
                onnx_resp = await client.get(onnx_url)
                onnx_resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"Failed to download ONNX: {exc}") from exc

            if len(onnx_resp.content) > 500 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="ONNX file too large (max 500 MB)")

            try:
                config_resp = await client.get(config_url)
                config_resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"Failed to download config: {exc}") from exc

            if len(config_resp.content) > 5 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Config file too large (max 5 MB)")

            try:
                json.loads(config_resp.content)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSON config: {exc}") from exc

        self.save_voice_files(onnx_resp.content, config_resp.content, voice_id)


class KokoroVoiceManager:
    """Manages Kokoro ONNX voice files under a persistent directory.

    Each voice is an ``.onnx`` model file paired with a ``.npz`` voices file.
    Files are expected either flat in the directory or in per-voice sub-folders.
    """

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._models_dir.mkdir(parents=True, exist_ok=True)

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def _voice_npz_path(self, onnx_path: Path) -> Optional[Path]:
        """Find the matching voices .npz file for an ONNX model."""
        # Same directory, same stem
        npz = onnx_path.with_suffix(".npz")
        if npz.exists():
            return npz
        # voices-<stem>.npz pattern (used by the German Martin model)
        alt = onnx_path.parent / f"voices-{onnx_path.stem}.npz"
        if alt.exists():
            return alt
        # For hyphenated stems like kokoro-martin, try voices-<last_part>.npz
        if "-" in onnx_path.stem:
            suffix = onnx_path.stem.split("-")[-1]
            alt2 = onnx_path.parent / f"voices-{suffix}.npz"
            if alt2.exists():
                return alt2
        # Generic fallback: any voices-*.npz in the same directory
        for candidate in onnx_path.parent.glob("voices-*.npz"):
            return candidate
        return None

    def list_voices(self) -> list[dict]:
        """Return all Kokoro voices with their model files."""
        voices: list[dict] = []
        # Search recursively for .onnx files
        onnx_files = sorted(self._models_dir.rglob("*.onnx"))
        for onnx_file in onnx_files:
            npz_file = self._voice_npz_path(onnx_file)
            # Use relative path stem as ID, or folder name if nested
            rel = onnx_file.relative_to(self._models_dir)
            voice_id = str(rel.with_suffix(""))
            voices.append({
                "id": voice_id,
                "onnx_path": str(onnx_file),
                "voices_path": str(npz_file) if npz_file else None,
                "has_voices": npz_file is not None,
            })
        return voices

    def get_voice_files(self, voice_id: str) -> Tuple[Optional[Path], Optional[Path]]:
        """Return (onnx_path, npz_path) for a voice ID, or (None, None)."""
        # Try flat first
        onnx = self._models_dir / f"{voice_id}.onnx"
        if onnx.exists():
            npz = self._voice_npz_path(onnx)
            return onnx, npz
        # Try nested path (voice_id may contain / from relative path)
        onnx = self._models_dir / voice_id / f"{Path(voice_id).name}.onnx"
        if onnx.exists():
            npz = self._voice_npz_path(onnx)
            return onnx, npz
        # Try generic: voice_id as relative path to .onnx
        onnx = self._models_dir / f"{voice_id}.onnx"
        if onnx.exists():
            npz = self._voice_npz_path(onnx)
            return onnx, npz
        return None, None

    def voice_exists(self, voice_id: str) -> bool:
        onnx, _ = self.get_voice_files(voice_id)
        return onnx is not None

    def delete_voice(self, voice_id: str) -> bool:
        onnx, npz = self.get_voice_files(voice_id)
        deleted = False
        if onnx and onnx.exists():
            onnx.unlink()
            deleted = True
        if npz and npz.exists():
            npz.unlink()
            deleted = True
        # Also clean up empty parent dir if nested
        if "/" in voice_id or "\\" in voice_id:
            parent = self._models_dir / voice_id
            if parent.exists() and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        return deleted

    def validate_upload(self, onnx_name: str, npz_name: str) -> None:
        if not onnx_name.endswith(".onnx"):
            raise HTTPException(status_code=400, detail="ONNX file must have .onnx extension")
        if not npz_name.endswith(".npz"):
            raise HTTPException(status_code=400, detail="Voices file must have .npz extension")
        onnx_base = Path(onnx_name).stem
        npz_base = Path(npz_name).stem
        # Allow voices-<stem>.npz as well as <stem>.npz
        if npz_base != onnx_base and not npz_base.endswith(f"-{onnx_base}"):
            raise HTTPException(
                status_code=400,
                detail="ONNX and voices file base names must match (or use voices-<stem>.npz)"
            )

    def save_voice_files(self, onnx_bytes: bytes, npz_bytes: bytes, voice_id: str) -> Tuple[Path, Path]:
        onnx_path = self._models_dir / f"{voice_id}.onnx"
        npz_path = self._models_dir / f"{voice_id}.npz"
        onnx_path.write_bytes(onnx_bytes)
        npz_path.write_bytes(npz_bytes)
        logger.info("Saved Kokoro voice files for %s", voice_id)
        return onnx_path, npz_path


class PiperTTSService:
    """Local TTS using Piper CLI."""

    def __init__(self, voices_dir: Path, piper_binary: str = "piper") -> None:
        self._voices_dir = voices_dir
        self._piper_binary = piper_binary
        self._resolved_binary: Optional[str] = None

    def _voice_path(self, voice_id: str) -> Optional[Path]:
        path = self._voices_dir / f"{voice_id}.onnx"
        return path if path.exists() else None

    def _resolve_piper_binary(self) -> Optional[str]:
        """Find the actual piper executable, caching the result."""
        if self._resolved_binary is not None:
            return self._resolved_binary

        import sys

        candidates = [self._piper_binary]
        # If the configured name is just "piper", also try the venv binary path
        if self._piper_binary == "piper" and hasattr(sys, "executable"):
            venv_bin = Path(sys.executable).parent / "piper"
            if str(venv_bin) not in candidates:
                candidates.append(str(venv_bin))

        for cmd in candidates:
            try:
                result = subprocess.run(
                    [cmd, "--help"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    self._resolved_binary = cmd
                    return cmd
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        self._resolved_binary = ""
        return None

    def is_ready(self) -> bool:
        """Check if Piper binary is available."""
        return self._resolve_piper_binary() is not None

    async def synthesize(self, text: str, voice_id: str) -> bytes:
        voice_path = self._voice_path(voice_id)
        if voice_path is None:
            raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")

        config_path = voice_path.with_suffix(".onnx.json")
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Voice config not found: {config_path.name}")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._run_piper, text, str(voice_path), str(config_path)
        )

    def _run_piper(self, text: str, voice_path: str, config_path: str) -> bytes:
        binary = self._resolve_piper_binary()
        if not binary:
            raise HTTPException(
                status_code=503,
                detail="Piper binary not found. Install piper-tts or set audio_tts_piper_binary to the full path."
            )
        cmd = [
            binary,
            "--model", voice_path,
            "--config", config_path,
            "--output_file", "-",
        ]
        result = subprocess.run(
            cmd, input=text.encode("utf-8"), capture_output=True, timeout=60
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[:500]
            raise HTTPException(
                status_code=500,
                detail=f"Piper TTS failed: {stderr}"
            )
        return result.stdout


class UnifiedTTSService:
    """Facade that prefers Kokoro and falls back to Piper automatically."""

    def __init__(
        self,
        provider: str,
        kokoro_model_dir: Path,
        kokoro_voice: str,
        voices_dir: Path,
        piper_binary: str = "piper",
        hf_token: Optional[str] = None,
        lang: str = "de",
    ) -> None:
        self._provider = provider
        self._lang = lang
        self._kokoro = KokoroTTSService(
            model_dir=kokoro_model_dir,
            default_voice=kokoro_voice,
            hf_token=hf_token,
        )
        self._piper = PiperTTSService(voices_dir, piper_binary)

    def is_ready(self) -> bool:
        primary = self._kokoro if self._provider == "kokoro" else self._piper
        fallback = self._piper if self._provider == "kokoro" else self._kokoro
        if primary.is_ready():
            return True
        return fallback.is_ready()

    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> bytes:
        """Try the configured primary engine, then fall back to the other.

        Voice-not-found errors (404) are NOT retried on the fallback because
        voice IDs are provider-specific and the fallback would fail with the
        same missing voice. Engine-level errors (500, 503) still trigger fallback.
        """
        text = normalize_tts_text(text, self._lang)
        primary = self._kokoro if self._provider == "kokoro" else self._piper
        fallback = self._piper if self._provider == "kokoro" else self._kokoro

        try:
            return await primary.synthesize(text, voice_id)
        except HTTPException as exc:
            logger.error(
                "Primary TTS (%s) failed with HTTPException: status=%d detail=%s",
                self._provider,
                exc.status_code,
                exc.detail,
            )
            # Do not fall back on voice-not-found; voice IDs are provider-specific
            if exc.status_code == 404:
                raise
            if fallback.is_ready():
                logger.warning("Primary TTS (%s) failed, falling back to %s", self._provider, "piper" if primary is self._kokoro else "kokoro")
                return await fallback.synthesize(text, voice_id)
            raise
        except Exception as exc:
            logger.exception("Primary TTS (%s) raised unexpected error", self._provider)
            if fallback.is_ready():
                logger.warning("Primary TTS (%s) raised unexpected error, falling back: %s", self._provider, exc)
                return await fallback.synthesize(text, voice_id)
            raise

    async def synthesize_streaming(
        self, text: str, voice_id: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """Stream WAV bytes per sentence from the primary engine.

        Falls back to non-streaming ``synthesize()`` for engines that do not
        support sentence-level streaming (e.g. Piper). In that case the
        entire audio is yielded as a single chunk.
        """
        text = normalize_tts_text(text, self._lang)
        primary = self._kokoro if self._provider == "kokoro" else self._piper
        fallback = self._piper if self._provider == "kokoro" else self._kokoro

        try:
            if hasattr(primary, "synthesize_streaming"):
                async for chunk in primary.synthesize_streaming(text, voice_id):
                    yield chunk
                return
            # Fallback: synthesize everything and yield as one chunk
            wav = await primary.synthesize(text, voice_id)
            yield wav
            return
        except HTTPException as exc:
            logger.error(
                "Primary TTS (%s) failed with HTTPException: status=%d detail=%s",
                self._provider,
                exc.status_code,
                exc.detail,
            )
            if exc.status_code == 404:
                raise
            if fallback.is_ready():
                logger.warning("Primary TTS (%s) failed, falling back to %s", self._provider, "piper" if primary is self._kokoro else "kokoro")
                if hasattr(fallback, "synthesize_streaming"):
                    async for chunk in fallback.synthesize_streaming(text, voice_id):
                        yield chunk
                else:
                    wav = await fallback.synthesize(text, voice_id)
                    yield wav
                return
            raise
        except Exception as exc:
            logger.exception("Primary TTS (%s) raised unexpected error", self._provider)
            if fallback.is_ready():
                logger.warning("Primary TTS (%s) raised unexpected error, falling back: %s", self._provider, exc)
                if hasattr(fallback, "synthesize_streaming"):
                    async for chunk in fallback.synthesize_streaming(text, voice_id):
                        yield chunk
                else:
                    wav = await fallback.synthesize(text, voice_id)
                    yield wav
                return
            raise


class STTService:
    """Local STT using faster-whisper or distil-whisper via transformers.

    Model loading is asynchronous and non-blocking.  is_ready() only checks
    whether the required Python dependencies are installed; it does NOT
    download or load the model.  The first call to transcribe() triggers the
    actual load in a background thread so the event loop stays responsive.
    """

    def __init__(
        self,
        model_size: str = "small",
        language: str = "de",
        provider: str = "faster_whisper",
        hf_cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self._model_size = model_size
        self._language = language
        self._provider = provider
        self._hf_cache_dir = hf_cache_dir
        self._hf_token = hf_token
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._loading_event = threading.Event()
        self._loading_future: Optional[asyncio.Future] = None
        self._load_error: Optional[Exception] = None

    def is_ready(self) -> bool:
        """Return True if the required Python packages are installed.

        Does NOT download or load the model so that health checks stay fast.
        """
        if self._provider == "distil_whisper":
            try:
                import transformers  # noqa: F401
                import torch  # noqa: F401
                return True
            except ImportError:
                return False
        else:
            try:
                import faster_whisper  # noqa: F401
                return True
            except ImportError:
                return False

    def is_loading(self) -> bool:
        """Return True while a model download/load is in progress."""
        return self._loading_event.is_set() and self._model is None

    async def _ensure_model_loaded(self):
        """Asynchronously load the model if it has not been loaded yet.

        Parallel callers wait on the same future so the model is loaded exactly
        once even when multiple requests arrive at the same time.
        """
        if self._model is not None:
            return

        if self._load_error is not None:
            raise self._load_error

        # Fast-path: another coroutine is already loading
        if self._loading_event.is_set() and self._loading_future is not None:
            await asyncio.wrap_future(self._loading_future)
            return

        with self._lock:
            # Double-checked locking
            if self._model is not None:
                return
            if self._load_error is not None:
                raise self._load_error
            if self._loading_event.is_set() and self._loading_future is not None:
                # Another thread started loading while we waited for the lock
                await asyncio.wrap_future(self._loading_future)
                return

            self._loading_event.set()
            loop = asyncio.get_event_loop()
            self._loading_future = loop.run_in_executor(None, self._load_model_sync)

        try:
            await asyncio.wrap_future(self._loading_future)
        except Exception as exc:
            self._load_error = exc
            raise

    def _load_model_sync(self):
        """Synchronous model loading – runs in a worker thread."""
        try:
            if self._provider == "distil_whisper":
                self._load_distil_model()
            else:
                self._load_faster_whisper_model()
        finally:
            self._loading_event.clear()

    def _apply_hf_auth(self):
        """Configure HuggingFace cache dir and optional authentication token."""
        if self._hf_cache_dir:
            import os
            os.environ["TRANSFORMERS_CACHE"] = self._hf_cache_dir
            os.environ["HF_HOME"] = self._hf_cache_dir
        if self._hf_token:
            try:
                from huggingface_hub import login
                login(token=self._hf_token)
            except Exception as exc:
                logger.warning("HF login failed (non-fatal): %s", exc)

    def _load_distil_model(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch

        self._apply_hf_auth()
        logger.info("Loading Distil-Whisper model '%s'...", self._model_size)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_size,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(self._model_size)
        self._model = model
        self._processor = processor
        logger.info("Distil-Whisper model '%s' loaded on %s.", self._model_size, device)

    def _load_faster_whisper_model(self):
        from faster_whisper import WhisperModel
        logger.info("Loading Whisper model '%s' (this may take a while on first use)...", self._model_size)
        self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
        logger.info("Whisper model '%s' loaded.", self._model_size)

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        min_snr_db: Optional[float] = None,
        noise_gate_min: Optional[float] = None,
        noise_gate_max: Optional[float] = None,
    ) -> str:
        await self._ensure_model_loaded()
        loop = asyncio.get_event_loop()
        logger.info(
            "Starting STT transcription (%d bytes, provider=%s, lang=%s)...",
            len(audio_bytes),
            self._provider,
            language or self._language,
        )
        result = await loop.run_in_executor(
            None,
            self._run_transcription,
            audio_bytes,
            language or self._language,
            min_snr_db,
            noise_gate_min,
            noise_gate_max,
        )
        logger.info("STT transcription result: %r", result)
        return result

    def _run_transcription(
        self,
        audio_bytes: bytes,
        language: str,
        min_snr_db: Optional[float] = None,
        noise_gate_min: Optional[float] = None,
        noise_gate_max: Optional[float] = None,
    ) -> str:
        # Server-side noise reduction (highpass + noise gate)
        cleaned_bytes, snr_db = clean_audio(
            audio_bytes, noise_gate_min=noise_gate_min, noise_gate_max=noise_gate_max
        )
        logger.info("STT audio cleaned (SNR=%.1f dB).", snr_db)
        threshold = min_snr_db if min_snr_db is not None else 5.0
        if snr_db < threshold:
            logger.warning("STT skipped: audio SNR too low (%.1f dB < %.1f dB).", snr_db, threshold)
            raise HTTPException(status_code=400, detail="No speech detected")

        if self._provider == "distil_whisper":
            return self._run_distil_transcription(cleaned_bytes, language)
        return self._run_faster_whisper_transcription(cleaned_bytes, language)

    def _run_faster_whisper_transcription(self, audio_bytes: bytes, language: str) -> str:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="faster-whisper not installed") from exc

        if self._model is None:
            raise HTTPException(status_code=503, detail="Whisper model not loaded")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with self._lock:
            try:
                segments, info = self._model.transcribe(tmp_path, language=language, beam_size=5)
                text = " ".join([segment.text for segment in segments]).strip()
                logger.info("STT transcribed successfully (detected lang=%s): %r", info.language, text)
                return text
            except Exception as exc:
                logger.error("STT transcription failed: %s", exc)
                raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    def _load_audio_as_array(self, file_path: str):
        """Load an audio file into a numpy array without requiring ffmpeg.

        Attempts stdlib ``wave`` for WAV files, then ``torchaudio`` if available.
        Raises HTTPException with a helpful message when neither works.
        """
        import numpy as np
        import wave

        # 1) stdlib wave – works for genuine WAV files (no ffmpeg needed)
        try:
            with wave.open(file_path, "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

                if sample_width == 1:
                    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                elif sample_width == 2:
                    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 4:
                    arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")

                if n_channels > 1:
                    arr = arr.reshape(-1, n_channels).mean(axis=1)
                return arr, sample_rate
        except wave.Error:
            pass

        # 2) torchaudio (often bundled with torch) – supports many formats
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(file_path)
            arr = waveform.mean(dim=0).numpy().astype(np.float32)
            return arr, sample_rate
        except Exception:
            pass

        raise HTTPException(
            status_code=503,
            detail="ffmpeg is required to load this audio format. Please install ffmpeg (e.g. 'brew install ffmpeg' on macOS, 'apt install ffmpeg' on Debian/Ubuntu), or ensure the uploaded audio is a standard WAV file.",
        )

    def _run_distil_transcription(self, audio_bytes: bytes, language: str) -> str:
        try:
            import torch
            import numpy as np
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="transformers/torch not installed for distil_whisper") from exc

        if self._model is None or self._processor is None:
            raise HTTPException(status_code=503, detail="Distil-Whisper model not loaded")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with self._lock:
            try:
                audio_array, sample_rate = self._load_audio_as_array(tmp_path)
                logger.debug("Audio loaded: shape=%s, sample_rate=%d", audio_array.shape, sample_rate)

                # Whisper expects exactly 16 kHz
                if sample_rate != 16000:
                    import torchaudio.functional as F
                    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
                    audio_tensor = F.resample(audio_tensor, sample_rate, 16000)
                    audio_array = audio_tensor.squeeze(0).numpy()
                    sample_rate = 16000

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if device != "cpu" else torch.float32

                # Extract features using the processor's feature extractor
                inputs = self._processor(
                    audio_array,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                )
                input_features = inputs.input_features.to(device).to(torch_dtype)

                # Build explicit decoder_input_ids to force the target language
                forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                    language=language, task="transcribe"
                )
                # Sort by position and extract token IDs
                sorted_ids = sorted(forced_decoder_ids, key=lambda x: x[0])
                prompt_ids = [token_id for _, token_id in sorted_ids]
                # Prepend BOS / decoder_start_token_id
                decoder_start = self._processor.tokenizer.bos_token_id
                decoder_input_ids = torch.tensor([[decoder_start] + prompt_ids], device=device)

                # Generate tokens directly on the model
                generated_ids = self._model.generate(
                    input_features,
                    decoder_input_ids=decoder_input_ids,
                )

                # Decode to text
                transcription = self._processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                text = transcription.strip()
                logger.info("STT transcribed successfully with distil_whisper: %r", text)
                return text
            except Exception as exc:
                logger.error("STT transcription failed: %s", exc)
                raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
            finally:
                Path(tmp_path).unlink(missing_ok=True)
