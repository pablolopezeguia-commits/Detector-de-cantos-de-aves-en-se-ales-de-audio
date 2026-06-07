"""Field-noise augmentation used by the final comparable training set."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from .audio import read_mono


EPS = 1e-12


@dataclass(frozen=True)
class MixMetadata:
    """Numerical details of one signal/noise mixture."""

    target_snr_db: float
    achieved_snr_db: float
    signal_power: float
    noise_power: float
    noise_gain: float


def match_noise_to_signal(noise: np.ndarray, noise_sr: int, target_sr: int, target_len: int) -> np.ndarray:
    """Resample, tile, or trim a noise window so it matches one signal window."""
    out = np.ascontiguousarray(noise, dtype=np.float32)
    if noise_sr != target_sr:
        out = librosa.resample(out, orig_sr=int(noise_sr), target_sr=int(target_sr)).astype(np.float32)
    if len(out) == 0:
        raise ValueError("Noise window is empty.")
    if len(out) < target_len:
        out = np.tile(out, int(math.ceil(target_len / len(out))))
    return np.ascontiguousarray(out[:target_len], dtype=np.float32)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, MixMetadata]:
    """Mix signal and noise with the requested signal-to-noise ratio."""
    signal = np.asarray(signal, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    signal_power = float(np.mean(signal.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if signal_power <= EPS:
        raise ValueError("Signal power is too low for augmentation.")
    if noise_power <= EPS:
        raise ValueError("Noise power is too low for augmentation.")

    target_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise_gain = float(np.sqrt(target_noise_power / (noise_power + EPS)))
    scaled_noise = noise * noise_gain
    mixed = signal + scaled_noise
    achieved_noise_power = float(np.mean(scaled_noise.astype(np.float64) ** 2))
    achieved_snr_db = float(10.0 * np.log10((signal_power + EPS) / (achieved_noise_power + EPS)))
    metadata = MixMetadata(float(snr_db), achieved_snr_db, signal_power, noise_power, noise_gain)
    return np.ascontiguousarray(mixed, dtype=np.float32), metadata


def write_augmented_window(
    signal_path: str | Path,
    noise_path: str | Path,
    output_path: str | Path,
    snr_db: float,
    headroom: float = 0.95,
) -> MixMetadata:
    """Create and save one augmented window from an existing signal/noise pair."""
    signal, signal_sr = read_mono(signal_path)
    noise, noise_sr = read_mono(noise_path)
    noise = match_noise_to_signal(noise, noise_sr, signal_sr, len(signal))
    mixed, metadata = mix_at_snr(signal, noise, snr_db)

    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > headroom:
        mixed = (mixed * (headroom / peak)).astype(np.float32)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), mixed, signal_sr)
    return metadata


def summarize_augmentation_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Summarize an augmentation manifest by target SNR and source groups."""
    group_cols = [col for col in ["target_snr_db", "base_tipo_audio", "noise_tipo_audio"] if col in manifest.columns]
    if not group_cols:
        raise ValueError("The manifest does not contain augmentation grouping columns.")
    return manifest.groupby(group_cols, dropna=False).size().reset_index(name="windows")
