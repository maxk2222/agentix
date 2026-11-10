"""Tests for app.utils.audio_cleaner."""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from app.utils.audio_cleaner import (
    apply_highpass,
    apply_noise_gate,
    compute_snr,
    estimate_noise_floor,
    load_wav_to_numpy,
    save_numpy_to_wav,
    clean_audio,
)


def _make_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Helper to create WAV bytes from a float32 array."""
    int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())
    return buf.getvalue()


class TestLoadSave:
    def test_roundtrip(self):
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
        wav_bytes = save_numpy_to_wav(samples, 16000)
        loaded, sr = load_wav_to_numpy(wav_bytes)
        assert sr == 16000
        assert loaded.shape == samples.shape
        # Allow small numerical error from int16 quantisation
        np.testing.assert_allclose(loaded, samples, atol=1e-4)

    def test_load_8bit(self):
        arr = np.random.randint(-100, 100, size=1000, dtype=np.int8)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)
            wf.setframerate(16000)
            wf.writeframes((arr.astype(np.int16) + 128).astype(np.uint8).tobytes())
        loaded, sr = load_wav_to_numpy(buf.getvalue())
        assert sr == 16000
        assert loaded.dtype == np.float32


class TestHighpass:
    def test_removes_dc_offset(self):
        samples = np.ones(1000, dtype=np.float32) * 0.5
        filtered = apply_highpass(samples, 16000, cutoff_hz=80)
        # After settling, DC should be strongly attenuated
        assert abs(filtered[-100:].mean()) < 0.05

    def test_preserves_high_freq(self):
        t = np.linspace(0, 1, 16000)
        samples = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        filtered = apply_highpass(samples, 16000, cutoff_hz=80)
        # High frequency signal should survive mostly intact after settling
        assert np.std(filtered[500:]) > 0.5


class TestNoiseGate:
    def test_silences_below_threshold(self):
        # Very quiet noise (std=0.001) against a noise floor of 0.005
        samples = np.random.randn(1000).astype(np.float32) * 0.001
        gated = apply_noise_gate(samples, noise_floor_rms=0.005, ratio=2.5)
        # Most samples should be zeroed out (threshold = 0.0125 >> noise amplitude)
        assert np.mean(gated == 0.0) > 0.5

    def test_preserves_above_threshold(self):
        samples = np.ones(1000, dtype=np.float32) * 0.5
        gated = apply_noise_gate(samples, noise_floor_rms=0.01, ratio=2.5)
        # Should pass through after attack
        assert np.mean(gated[100:] > 0.4) > 0.9

    def test_min_threshold_clamps_up(self):
        # noise_floor_rms * ratio = 0.001 * 2.5 = 0.0025, but min_threshold = 0.01 forces it up
        samples = np.random.randn(1000).astype(np.float32) * 0.003
        gated = apply_noise_gate(samples, noise_floor_rms=0.001, ratio=2.5, min_threshold=0.01)
        # Most samples should be zeroed out because threshold is clamped to 0.01
        assert np.mean(gated == 0.0) > 0.5

    def test_max_threshold_clamps_down(self):
        # noise_floor_rms * ratio = 0.1 * 2.5 = 0.25, but max_threshold = 0.05 forces it down
        samples = np.ones(1000, dtype=np.float32) * 0.1
        gated = apply_noise_gate(samples, noise_floor_rms=0.1, ratio=2.5, max_threshold=0.05)
        # Threshold should be clamped to 0.05, so samples of 0.1 pass through
        assert np.mean(gated[100:] > 0.05) > 0.9

    def test_min_max_combined(self):
        # noise_floor_rms * ratio = 0.01 * 2.5 = 0.025 -> clamped to [0.015, 0.03]
        samples = np.ones(1000, dtype=np.float32) * 0.2
        gated = apply_noise_gate(
            samples, noise_floor_rms=0.01, ratio=2.5, min_threshold=0.015, max_threshold=0.03
        )
        # 0.2 > 0.03 (clamped threshold), so samples should pass through after attack
        assert np.mean(gated[100:] > 0.1) > 0.9


class TestSNR:
    def test_snr_computation(self):
        # Signal RMS = 1.0, noise RMS = 0.1 => 20 dB
        assert compute_snr(1.0, 0.1) == pytest.approx(20.0, rel=0.01)

    def test_infinity_when_no_noise(self):
        assert compute_snr(1.0, 0.0) == float("inf")

    def test_zero_when_no_signal(self):
        assert compute_snr(0.0, 0.1) == 0.0


class TestEstimateNoiseFloor:
    def test_on_silence(self):
        samples = np.zeros(1024, dtype=np.float32)
        assert estimate_noise_floor(samples) == pytest.approx(0.0, abs=1e-6)

    def test_on_noisy_signal(self):
        # First 50% silence, second 50% loud signal
        silence = np.zeros(512, dtype=np.float32)
        signal = np.ones(512, dtype=np.float32) * 0.5
        samples = np.concatenate([silence, signal])
        floor = estimate_noise_floor(samples)
        # Noise floor should be close to zero (silence dominates low percentile)
        assert floor < 0.05


class TestCleanAudio:
    def test_returns_bytes_and_snr(self):
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
        wav_bytes = _make_wav_bytes(samples)
        cleaned, snr = clean_audio(wav_bytes)
        assert isinstance(cleaned, bytes)
        assert snr > 0.0

    def test_low_snr_returns_poor_snr(self):
        # Pure silence → SNR should be 0 or very low
        wav_bytes = _make_wav_bytes(np.zeros(16000, dtype=np.float32))
        cleaned, snr = clean_audio(wav_bytes)
        assert snr < 5.0

    def test_cleaned_audio_is_valid_wav(self):
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
        wav_bytes = _make_wav_bytes(samples)
        cleaned, _ = clean_audio(wav_bytes)
        loaded, sr = load_wav_to_numpy(cleaned)
        assert sr == 16000
        assert loaded.size == 16000

    def test_clean_audio_applies_noise_gate_clamping(self):
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32) * 0.5
        wav_bytes = _make_wav_bytes(samples)
        cleaned, snr = clean_audio(wav_bytes, noise_gate_min=0.01, noise_gate_max=0.2)
        assert isinstance(cleaned, bytes)
        assert snr > 0.0
