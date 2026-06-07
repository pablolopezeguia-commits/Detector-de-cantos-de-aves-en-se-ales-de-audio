"""Shared audio utilities for training, evaluation, and inference."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def read_mono(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Read an audio file as float32 mono, optionally resampling it."""
    try:
        audio_2d, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio_2d[:, 0] if audio_2d.shape[1] == 1 else audio_2d.mean(axis=1, dtype=np.float32)
    except Exception:
        audio, sr = librosa.load(str(path), sr=None, mono=True)

    audio = np.ascontiguousarray(audio, dtype=np.float32)
    sr = int(sr)
    if target_sr is not None and sr != int(target_sr):
        audio = librosa.resample(audio, orig_sr=sr, target_sr=int(target_sr)).astype(np.float32)
        sr = int(target_sr)
    return np.ascontiguousarray(audio, dtype=np.float32), sr


def iter_complete_windows(
    audio: np.ndarray,
    sr: int,
    window_s: float = 3.0,
    overlap: float = 0.5,
) -> Iterator[tuple[int, float, float, np.ndarray]]:
    """Yield complete fixed-length windows and discard any incomplete tail."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in the [0, 1) range.")

    window_samples = int(round(window_s * sr))
    step_samples = int(round(window_samples * (1.0 - overlap)))
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("window_s and overlap produce an invalid window step.")

    start = 0
    index = 0
    while start + window_samples <= len(audio):
        end = start + window_samples
        yield index, start / sr, end / sr, np.ascontiguousarray(audio[start:end], dtype=np.float32)
        start += step_samples
        index += 1


def count_complete_windows(n_samples: int, sr: int, window_s: float = 3.0, overlap: float = 0.5) -> int:
    """Return the number of complete windows produced by iter_complete_windows."""
    window_samples = int(round(window_s * sr))
    step_samples = int(round(window_samples * (1.0 - overlap)))
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("window_s and overlap produce an invalid window step.")
    if n_samples < window_samples:
        return 0
    return int((n_samples - window_samples) // step_samples + 1)
