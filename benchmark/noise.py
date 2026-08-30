from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags


def mix_white_noise(audio: np.ndarray, target_snr_db: float, seed: int) -> tuple[np.ndarray, dict]:
    signal = np.asarray(audio, dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(signal.shape).astype(np.float32)
    signal_rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
    raw_noise_rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float64))))
    desired_noise_rms = signal_rms / (10 ** (target_snr_db / 20))
    scale = desired_noise_rms / raw_noise_rms if raw_noise_rms else 0.0
    scaled_noise = noise * scale
    mixture = signal + scaled_noise
    peak = float(np.max(np.abs(mixture))) if mixture.size else 0.0
    anti_clip_scale = 0.999 / peak if peak > 0.999 else 1.0
    mixture *= anti_clip_scale
    scaled_noise *= anti_clip_scale
    scaled_signal = signal * anti_clip_scale
    achieved = 20 * math.log10(
        float(np.sqrt(np.mean(scaled_signal.astype(np.float64) ** 2))) /
        float(np.sqrt(np.mean(scaled_noise.astype(np.float64) ** 2)))
    )
    return mixture, {"signal_rms": signal_rms, "raw_noise_rms": raw_noise_rms,
                     "noise_scaling_factor": scale, "anti_clip_scale": anti_clip_scale,
                     "target_snr_db": target_snr_db, "achieved_snr_db": achieved}


def mix_environmental_noise(audio: np.ndarray, noise: np.ndarray, target_snr_db: float,
                            seed: int) -> tuple[np.ndarray, dict]:
    """Mix a deterministic segment of real noise at a measured target SNR."""
    signal = np.asarray(audio, dtype=np.float32)
    background = np.asarray(noise, dtype=np.float32)
    if signal.ndim != 1 or background.ndim != 1 or not signal.size or not background.size:
        raise ValueError("signal and noise must be non-empty mono arrays")
    rng = np.random.default_rng(seed)
    if len(background) < len(signal):
        repeats = int(np.ceil(len(signal) / len(background)))
        background = np.tile(background, repeats)
    maximum_start = len(background) - len(signal)
    start = int(rng.integers(0, maximum_start + 1)) if maximum_start else 0
    segment = background[start:start + len(signal)].copy()
    segment -= float(np.mean(segment))
    signal_rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))
    noise_rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
    if noise_rms == 0:
        raise ValueError("selected environmental noise segment is silent")
    desired_noise_rms = signal_rms / (10 ** (target_snr_db / 20))
    scaled_noise = segment * (desired_noise_rms / noise_rms)
    mixture = signal + scaled_noise
    peak = float(np.max(np.abs(mixture)))
    anti_clip_scale = 0.999 / peak if peak > 0.999 else 1.0
    mixture *= anti_clip_scale
    scaled_noise *= anti_clip_scale
    scaled_signal = signal * anti_clip_scale
    achieved = snr_db(scaled_signal, scaled_signal + scaled_noise)
    return mixture, {"start_sample": start, "signal_rms": signal_rms,
                     "raw_noise_rms": noise_rms,
                     "noise_scaling_factor": desired_noise_rms / noise_rms,
                     "anti_clip_scale": anti_clip_scale,
                     "target_snr_db": target_snr_db, "achieved_snr_db": achieved}


def write_noisy_variants(source: Path, output_dir: Path, sample_id: str,
                         snrs: tuple[int, ...], seed: int) -> list[dict]:
    audio, sample_rate = sf.read(source, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, snr in enumerate(snrs):
        mixed, metadata = mix_white_noise(audio, snr, seed + index)
        path = output_dir / f"{sample_id}_{snr}db.wav"
        sf.write(path, mixed, sample_rate, subtype="PCM_16")
        rows.append({"sample_id": sample_id, "snr_db": snr, "path": str(path),
                     "sample_rate": sample_rate, **metadata})
    return rows


def snr_db(reference: np.ndarray, degraded: np.ndarray) -> float:
    size = min(len(reference), len(degraded))
    reference = np.asarray(reference[:size], dtype=np.float64)
    degraded = np.asarray(degraded[:size], dtype=np.float64)
    noise = degraded - reference
    signal_power = float(np.mean(reference ** 2))
    noise_power = float(np.mean(noise ** 2))
    if noise_power == 0:
        return float("inf")
    return 10 * math.log10(signal_power / noise_power) if signal_power else float("-inf")


def align_to_reference(reference: np.ndarray, degraded: np.ndarray,
                       max_lag_samples: int = 2048) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove fixed algorithmic delay before intrusive signal comparison."""
    reference = np.asarray(reference, dtype=np.float32)
    degraded = np.asarray(degraded, dtype=np.float32)
    correlations = correlate(degraded, reference, mode="full", method="fft")
    lags = correlation_lags(len(degraded), len(reference), mode="full")
    allowed = np.abs(lags) <= max_lag_samples
    lag = int(lags[allowed][np.argmax(correlations[allowed])])
    if lag >= 0:
        aligned_degraded = degraded[lag:]
        aligned_reference = reference[:len(aligned_degraded)]
    else:
        aligned_reference = reference[-lag:]
        aligned_degraded = degraded[:len(aligned_reference)]
    size = min(len(aligned_reference), len(aligned_degraded))
    return aligned_reference[:size], aligned_degraded[:size], lag
