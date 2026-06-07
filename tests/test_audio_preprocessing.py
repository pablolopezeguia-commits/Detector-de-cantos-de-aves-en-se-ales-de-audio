import numpy as np
import pytest

from birdsong_detector.ecocanto.audio_preprocessing import (
    count_raw_windows,
    iter_raw_windows,
    prepare_window_for_yamnet,
    preprocess_window_v2,
)


def test_preprocess_window_v2_peak_headroom():
    audio = np.asarray([0.0, 0.25, -0.5, 0.125], dtype=np.float32)
    processed, metrics = preprocess_window_v2(audio, sr=100, frame_sec=0.02, hop_sec=0.01)
    assert np.max(np.abs(processed)) == pytest.approx(0.95, abs=1e-6)
    assert metrics["input_peak_abs"] == pytest.approx(0.5)
    assert metrics["output_peak_abs"] == pytest.approx(0.95, abs=1e-6)
    assert metrics["gain"] == pytest.approx(1.9)


def test_raw_window_count_uses_only_complete_windows():
    audio = np.ones(11, dtype=np.float32)
    windows = list(iter_raw_windows(audio, sr=10, window_s=0.4, overlap=0.5))
    assert count_raw_windows(len(audio), sr=10, window_s=0.4, overlap=0.5) == len(windows)
    assert len(windows) == 4
    assert windows[-1][1] == pytest.approx(1.0)
    assert len(windows[-1][2]) == 4


def test_180_second_audio_has_no_partial_tail_window():
    sr = 48000
    audio = np.ones(180 * sr, dtype=np.float32)
    windows = list(iter_raw_windows(audio, sr=sr, window_s=3.0, overlap=0.5))
    assert count_raw_windows(len(audio), sr=sr, window_s=3.0, overlap=0.5) == 119
    assert len(windows) == 119
    assert windows[-1][0] == pytest.approx(177.0)
    assert windows[-1][1] == pytest.approx(180.0)
    assert all((t_fin - t_ini) == pytest.approx(3.0) for t_ini, t_fin, _chunk in windows)


def test_short_audio_produces_no_windows():
    audio = np.ones(2 * 16000, dtype=np.float32)
    assert count_raw_windows(len(audio), sr=16000, window_s=3.0, overlap=0.5) == 0
    assert list(iter_raw_windows(audio, sr=16000, window_s=3.0, overlap=0.5)) == []


def test_prepare_window_for_yamnet_resamples_and_keeps_headroom():
    sr = 48000
    t = np.linspace(0.0, 3.0, sr * 3, endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    processed, metrics = prepare_window_for_yamnet(audio, source_sr=sr, target_sr=16000, window_s=3.0)
    assert len(processed) == 48000
    assert metrics["output_peak_abs"] == pytest.approx(0.95, abs=1e-6)
    assert np.max(np.abs(processed)) <= 0.96
    assert metrics["model_sample_rate"] == 16000
