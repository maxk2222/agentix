"""Server-side audio preprocessing for STT noise reduction.

Uses only stdlib ``wave`` and ``numpy`` (already pulled in via torch).
"""
from __future__ import annotations

import io
import wave
from typing import Tuple

import numpy as np


def load_wav_to_numpy(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Load WAV bytes into a mono float32 numpy array in [-1, 1]."""
    with io.BytesIO(wav_bytes) as bio:
        with wave.open(bio, "rb") as wf:
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


def save_numpy_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Write a mono float32 array in [-1, 1] back to WAV bytes."""
    # Clamp and convert to int16
    clipped = np.clip(samples, -1.0, 1.0)
    int16_arr = (clipped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_arr.tobytes())
    return buf.getvalue()


def apply_highpass(
    samples: np.ndarray,
    sample_rate: float,
    cutoff_hz: float = 80.0,
) -> np.ndarray:
    """Apply a 1st-order IIR high-pass filter."""
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = rc / (rc + dt)

    out = np.empty_like(samples)
    prev_in = 0.0
    prev_out = 0.0
    for i in range(samples.size):
        x = float(samples[i])
        y = alpha * (prev_out + x - prev_in)
        out[i] = y
        prev_in = x
        prev_out = y
    return out


def estimate_noise_floor(
    samples: np.ndarray,
    percentile: float = 0.1,
) -> float:
    """Estimate background noise floor from the given percentile of absolute sample values."""
    abs_samples = np.abs(samples)
    return float(np.percentile(abs_samples, percentile * 100))


def apply_noise_gate(
    samples: np.ndarray,
    noise_floor_rms: float,
    ratio: float = 2.5,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
    sample_rate: float = 44100.0,
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None,
) -> np.ndarray:
    """Simple noise gate with attack/release smoothing and optional threshold clamping."""
    threshold = noise_floor_rms * ratio
    if min_threshold is not None:
        threshold = max(threshold, min_threshold)
    if max_threshold is not None:
        threshold = min(threshold, max_threshold)
    if threshold <= 0.0:
        return samples.copy()

    attack_step = 1.0 / max(1.0, (attack_ms / 1000.0) * sample_rate)
    release_step = 1.0 / max(1.0, (release_ms / 1000.0) * sample_rate)

    out = np.empty_like(samples)
    gain = 0.0
    for i in range(samples.size):
        if abs(float(samples[i])) > threshold:
            gain = min(1.0, gain + attack_step)
        else:
            gain = max(0.0, gain - release_step)
        out[i] = samples[i] * gain
    return out


def compute_snr(signal_rms: float, noise_floor_rms: float) -> float:
    """Return SNR in decibels."""
    if noise_floor_rms <= 0.0 or signal_rms <= 0.0:
        return float("inf") if signal_rms > 0.0 else 0.0
    return 20.0 * np.log10(signal_rms / noise_floor_rms)


def clean_audio(
    wav_bytes: bytes,
    cutoff_hz: float = 80.0,
    noise_gate_min: Optional[float] = None,
    noise_gate_max: Optional[float] = None,
) -> Tuple[bytes, float]:
    """Full cleaning pipeline: load → highpass → noise_gate → save.

    Returns the cleaned WAV bytes and the estimated SNR in dB.
    """
    samples, sample_rate = load_wav_to_numpy(wav_bytes)
    samples = apply_highpass(samples, sample_rate, cutoff_hz)
    noise_floor = estimate_noise_floor(samples)
    gated = apply_noise_gate(
        samples, noise_floor, ratio=2.5, sample_rate=sample_rate,
        min_threshold=noise_gate_min, max_threshold=noise_gate_max,
    )

    signal_rms = float(np.sqrt(np.mean(gated ** 2)))
    snr_db = compute_snr(signal_rms, noise_floor)

    cleaned_bytes = save_numpy_to_wav(gated, sample_rate)
    return cleaned_bytes, snr_db
