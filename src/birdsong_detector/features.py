"""Manual acoustic features used by the classical SVM models.

The feature set follows the final memory narrative: 20 interpretable acoustic
features plus an optional distance to the positive centroid. The functions are
self-contained so the public repository does not depend on historical scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks, spectrogram


EPS = 1e-12
NPERSEG = 1024
NOVERLAP = 512

BASE_FEATURES = [
    "max_amplitude",
    "rms_amplitude",
    "max_time_waveform",
    "average_power_spectrogram",
    "max_power_peak_power",
    "delta_power_selection_spectrum",
    "energy_spectrogram",
    "aggregate_entropy_spectrogram",
    "average_entropy_spectrogram",
    "max_frequency_peak_frequency",
    "frequency_5_percent",
    "iqr_bandwidth",
    "center_time",
    "time_95_percent",
    "iqr_duration",
    "duration_90_percent",
]

ROBUST_FEATURES = [
    "snr_estimated",
    "band_ratio_bird_vs_low",
    "n_prominent_peaks",
    "spectral_concentration",
]

FINAL_FEATURES_20 = BASE_FEATURES + ROBUST_FEATURES
CENTROID_FEATURE = "distance_to_positive_centroid_final"
FINAL_FEATURES_21 = FINAL_FEATURES_20 + [CENTROID_FEATURE]


@dataclass(frozen=True)
class SelectionBounds:
    """Time-frequency rectangle used for Raven-like full-window measurements."""

    begin_time: float
    end_time: float
    low_freq: float
    high_freq: float


@dataclass(frozen=True)
class SpectrogramData:
    """Power spectral density representation for one audio window."""

    freqs: np.ndarray
    times: np.ndarray
    psd_linear: np.ndarray
    psd_db: np.ndarray


def _safe_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, EPS))


def _axis_indices(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.where((values >= low) & (values <= high))[0]


def _percentile_axis(axis: np.ndarray, energy: np.ndarray, percentile: float) -> float:
    total = float(np.sum(energy))
    if total <= EPS:
        return float(axis[0]) if len(axis) else 0.0
    idx = int(np.searchsorted(np.cumsum(energy), percentile * total, side="left"))
    idx = min(max(idx, 0), len(axis) - 1)
    return float(axis[idx])


def _entropy(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    total = float(np.sum(flat))
    if total <= EPS:
        return 0.0
    p = flat / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_spectrogram_data(
    audio: np.ndarray,
    sr: int,
    nperseg: int = NPERSEG,
    noverlap: int = NOVERLAP,
) -> SpectrogramData:
    """Compute a PSD spectrogram with the convention used by the SVM branch."""
    if len(audio) == 0:
        raise ValueError("Cannot extract features from an empty audio window.")

    effective_nperseg = min(int(nperseg), len(audio))
    effective_noverlap = min(int(noverlap), max(effective_nperseg - 1, 0))
    freqs, times, psd = spectrogram(
        x=np.asarray(audio, dtype=np.float32),
        fs=int(sr),
        window="hann",
        nperseg=effective_nperseg,
        noverlap=effective_noverlap,
        mode="psd",
        scaling="density",
    )
    psd = np.maximum(psd, EPS)
    return SpectrogramData(freqs=freqs, times=times, psd_linear=psd, psd_db=_safe_db(psd))


def default_full_selection(audio: np.ndarray, sr: int) -> SelectionBounds:
    """Create the full-window, full-band selection used by the final features."""
    return SelectionBounds(0.0, len(audio) / float(sr), 0.0, sr / 2.0)


def selection_view(spec: SpectrogramData, selection: SelectionBounds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected frequencies, times, and PSD matrix."""
    f_idx = _axis_indices(spec.freqs, selection.low_freq, selection.high_freq)
    t_idx = _axis_indices(spec.times, selection.begin_time, selection.end_time)
    if len(f_idx) == 0 or len(t_idx) == 0:
        raise ValueError("The selected time-frequency region contains no spectrogram bins.")
    return spec.freqs[f_idx], spec.times[t_idx], spec.psd_linear[np.ix_(f_idx, t_idx)]


def estimate_snr(audio: np.ndarray, sr: int, frame_s: float = 0.02, hop_s: float = 0.01) -> float:
    """Estimate SNR as the ratio between high and low frame-RMS percentiles."""
    frame_len = max(1, int(round(frame_s * sr)))
    hop_len = max(1, int(round(hop_s * sr)))
    if len(audio) < frame_len:
        frame_len = len(audio)
        hop_len = max(1, frame_len)

    starts = range(0, max(len(audio) - frame_len + 1, 1), hop_len)
    rms_values = []
    for start in starts:
        frame = audio[start : start + frame_len]
        if len(frame):
            rms_values.append(float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))))
    if not rms_values:
        return 0.0
    noise_floor = float(np.percentile(rms_values, 10))
    signal_peak = float(np.percentile(rms_values, 90))
    return float((signal_peak + EPS) / (noise_floor + EPS))


def extract_features_from_window(
    audio: np.ndarray,
    sr: int,
    snr_estimated: float | None = None,
    nperseg: int = NPERSEG,
    noverlap: int = NOVERLAP,
) -> dict[str, float]:
    """Extract the 20 final manual features from one complete audio window."""
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    spec = compute_spectrogram_data(audio, sr, nperseg=nperseg, noverlap=noverlap)
    selection = default_full_selection(audio, sr)
    freqs, times, psd = selection_view(spec, selection)

    spectrum_linear = np.mean(psd, axis=1)
    spectrum_db = _safe_db(spectrum_linear)
    freq_energy = np.sum(psd, axis=1)
    time_energy = np.sum(psd, axis=0)

    freq_25 = _percentile_axis(freqs, freq_energy, 0.25)
    freq_75 = _percentile_axis(freqs, freq_energy, 0.75)
    time_05 = _percentile_axis(times, time_energy, 0.05)
    time_25 = _percentile_axis(times, time_energy, 0.25)
    time_75 = _percentile_axis(times, time_energy, 0.75)
    time_95 = _percentile_axis(times, time_energy, 0.95)

    bird_band = (freqs >= 1000.0) & (freqs <= min(8000.0, sr / 2.0))
    low_band = freqs < 1000.0
    magnitude = np.sqrt(psd)
    bird_energy = float(np.mean(magnitude[bird_band, :])) if np.any(bird_band) else 0.0
    low_energy = float(np.mean(magnitude[low_band, :])) if np.any(low_band) else 0.0

    mean_spectrum = np.mean(magnitude, axis=1)
    prominence = float(np.std(mean_spectrum))
    peaks, _ = find_peaks(mean_spectrum, prominence=prominence if prominence > EPS else None)

    local_mean = np.mean(magnitude, axis=1, keepdims=True)
    spectral_snr = magnitude / (local_mean + EPS)

    row = {
        "max_amplitude": float(np.max(audio)),
        "rms_amplitude": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        "max_time_waveform": float(np.argmax(audio) / float(sr)),
        "average_power_spectrogram": float(np.mean(_safe_db(psd))),
        "max_power_peak_power": float(np.max(_safe_db(psd))),
        "delta_power_selection_spectrum": float(np.max(spectrum_db) - np.min(spectrum_db)),
        "energy_spectrogram": float(np.sum(psd)),
        "aggregate_entropy_spectrogram": _entropy(psd),
        "average_entropy_spectrogram": float(np.mean([_entropy(psd[:, idx]) for idx in range(psd.shape[1])])),
        "max_frequency_peak_frequency": float(freqs[int(np.argmax(freq_energy))]),
        "frequency_5_percent": _percentile_axis(freqs, freq_energy, 0.05),
        "iqr_bandwidth": float(freq_75 - freq_25),
        "center_time": _percentile_axis(times, time_energy, 0.50),
        "time_95_percent": time_95,
        "iqr_duration": float(time_75 - time_25),
        "duration_90_percent": float(time_95 - time_05),
        "snr_estimated": float(estimate_snr(audio, sr) if snr_estimated is None else snr_estimated),
        "band_ratio_bird_vs_low": float(bird_energy / (low_energy + EPS)),
        "n_prominent_peaks": float(len(peaks)),
        "spectral_concentration": float(np.max(spectral_snr) / (np.mean(spectral_snr) + EPS)),
    }
    return {name: float(row[name]) for name in FINAL_FEATURES_20}


def add_centroid_distance(
    features: np.ndarray,
    positive_centroid: np.ndarray,
    positive_std: np.ndarray,
) -> np.ndarray:
    """Append the normalized distance to the positive-class centroid."""
    z = (features - positive_centroid.reshape(1, -1)) / (positive_std.reshape(1, -1) + EPS)
    distance = np.sqrt(np.mean(z**2, axis=1)).reshape(-1, 1)
    return np.hstack([features, distance])


def dataframe_to_matrix(frame: Any, feature_columns: list[str] | tuple[str, ...] = FINAL_FEATURES_20) -> np.ndarray:
    """Convert a pandas-like feature table into a numeric matrix."""
    values = frame.loc[:, list(feature_columns)].replace([np.inf, -np.inf], np.nan)
    if values.isna().any().any():
        missing = values.columns[values.isna().any()].tolist()
        raise ValueError(f"Feature table contains NaN or infinite values in: {missing}")
    return values.to_numpy(dtype=np.float32)
