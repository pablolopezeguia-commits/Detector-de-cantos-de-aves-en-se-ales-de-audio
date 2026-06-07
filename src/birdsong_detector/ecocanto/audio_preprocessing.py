"""Audio preprocessing used by the final detector."""

from __future__ import annotations

import pathlib
from collections.abc import Iterator

import librosa
import numpy as np
import soundfile as sf

from .config import (
    PREPROCESS_EPS,
    PREPROCESS_FRAME_SEC,
    PREPROCESS_HEADROOM,
    PREPROCESS_HOP_SEC,
    PREPROCESS_NORMALIZATION_MODE,
    PREPROCESS_TARGET_RMS,
)


def read_mono_audio(path: str | pathlib.Path) -> tuple[np.ndarray, int, int]:
    """Read audio as mono while preserving the original sample rate."""
    try:
        audio_2d, sr = sf.read(str(path), dtype="float32", always_2d=True)
        channels = int(audio_2d.shape[1])
        if channels == 1:
            mono = audio_2d[:, 0]
        else:
            mono = audio_2d.mean(axis=1, dtype=np.float32)
        return np.ascontiguousarray(mono, dtype=np.float32), int(sr), channels
    except Exception:
        audio, sr = librosa.load(str(path), sr=None, mono=True)
        return np.ascontiguousarray(audio, dtype=np.float32), int(sr), 1


def iter_raw_windows(
    audio: np.ndarray,
    sr: int,
    window_s: float,
    overlap: float,
) -> Iterator[tuple[float, float, np.ndarray]]:
    """Yield complete windows only, matching the training contract."""
    win_samples = int(round(window_s * sr))
    step_samples = int(round(win_samples * (1.0 - overlap)))
    if win_samples <= 0 or step_samples <= 0:
        raise ValueError("Invalid window/overlap configuration.")

    start = 0
    while start + win_samples <= len(audio):
        end = start + win_samples
        chunk = audio[start:end]
        t_ini = start / sr
        t_fin = end / sr
        yield t_ini, t_fin, np.ascontiguousarray(chunk, dtype=np.float32)
        start += step_samples


def count_raw_windows(n_samples: int, sr: int, window_s: float, overlap: float) -> int:
    win_samples = int(round(window_s * sr))
    step_samples = int(round(win_samples * (1.0 - overlap)))
    if win_samples <= 0 or step_samples <= 0:
        raise ValueError("Invalid window/overlap configuration.")
    if n_samples < win_samples:
        return 0
    return int((n_samples - win_samples) // step_samples + 1)


def preprocess_window_v2(
    audio: np.ndarray,
    sr: int,
    target_rms: float = PREPROCESS_TARGET_RMS,
    headroom: float = PREPROCESS_HEADROOM,
    normalization_mode: str = PREPROCESS_NORMALIZATION_MODE,
    frame_sec: float = PREPROCESS_FRAME_SEC,
    hop_sec: float = PREPROCESS_HOP_SEC,
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate the local noise floor and normalize the window by peak level."""
    if len(audio) == 0:
        raise ValueError("Cannot preprocess an empty audio window.")

    audio = np.ascontiguousarray(audio, dtype=np.float32)
    frame_len = max(1, int(round(frame_sec * sr)))
    hop_len = max(1, int(round(hop_sec * sr)))
    if len(audio) < frame_len:
        frame_len = len(audio)
        hop_len = max(1, frame_len)

    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=0))

    noise_floor_rms = float(np.percentile(frame_rms, 10))
    signal_peak_rms = float(np.percentile(frame_rms, 90))
    snr_estimated = float((signal_peak_rms + PREPROCESS_EPS) / (noise_floor_rms + PREPROCESS_EPS))

    input_peak_abs = float(np.max(np.abs(audio))) if len(audio) else 0.0
    noise_floor_gain = 1.0
    limiter_gain = 1.0

    if normalization_mode == "peak":
        if input_peak_abs > PREPROCESS_EPS:
            limiter_gain = float(headroom / input_peak_abs)
        normalized = (audio * limiter_gain).astype(np.float32)
    elif normalization_mode == "noise_floor_limited":
        if noise_floor_rms > PREPROCESS_EPS:
            noise_floor_gain = float(target_rms / noise_floor_rms)
        normalized = (audio * noise_floor_gain).astype(np.float32)
        peak_after_noise_floor = float(np.max(np.abs(normalized))) if len(normalized) else 0.0
        if peak_after_noise_floor > headroom:
            limiter_gain = float(headroom / peak_after_noise_floor)
            normalized = (normalized * limiter_gain).astype(np.float32)
    else:
        raise ValueError(f"Unsupported normalization_mode: {normalization_mode}")

    final_gain = float(noise_floor_gain * limiter_gain)
    metrics = {
        "frame_len_samples": float(frame_len),
        "hop_len_samples": float(hop_len),
        "noise_floor_rms": noise_floor_rms,
        "signal_peak_rms": signal_peak_rms,
        "snr_estimated": snr_estimated,
        "noise_floor_gain": noise_floor_gain,
        "limiter_gain": limiter_gain,
        "gain": final_gain,
        "input_rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        "output_rms": float(np.sqrt(np.mean(normalized.astype(np.float64) ** 2))),
        "input_peak_abs": input_peak_abs,
        "output_peak_abs": float(np.max(np.abs(normalized))) if len(normalized) else 0.0,
    }
    return normalized, metrics


def prepare_window_for_yamnet(
    audio: np.ndarray,
    source_sr: int,
    target_sr: int,
    window_s: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Preprocess one window and resample it to the YAMNet input rate."""
    processed, metrics = preprocess_window_v2(audio, source_sr)
    if source_sr != target_sr:
        processed = librosa.resample(
            processed.astype(np.float32),
            orig_sr=source_sr,
            target_sr=target_sr,
        ).astype(np.float32)

    target_len = int(round(window_s * target_sr))
    if len(processed) < target_len:
        processed = np.pad(processed, (0, target_len - len(processed)))
    elif len(processed) > target_len:
        processed = processed[:target_len]

    metrics["model_sample_rate"] = float(target_sr)
    metrics["model_n_samples"] = float(len(processed))
    return np.ascontiguousarray(processed, dtype=np.float32), metrics
