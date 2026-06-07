import pathlib

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")
pytest.importorskip("librosa")
sf = pytest.importorskip("soundfile")

from birdsong_detector.ecocanto.engine import BirdSongEngine, verify_model


def test_verify_model():
    verify_model()


@pytest.mark.slow
def test_engine_loads():
    engine = BirdSongEngine()
    assert engine.threshold == pytest.approx(0.689144, abs=1e-4)
    assert engine.window_s == 3.0
    assert engine.sample_rate == 16000


@pytest.mark.slow
def test_silence_is_nocanto(tmp_path):
    engine = BirdSongEngine()
    wav = tmp_path / "silence.wav"
    sf.write(str(wav), np.zeros(10 * 16000, dtype=np.float32), 16000)
    df = engine.analyze_file(str(wav))
    assert (df["prediccion"] == 0).all()


@pytest.mark.slow
def test_audio_shorter_than_window_returns_empty_dataframe(tmp_path):
    engine = BirdSongEngine()
    wav = tmp_path / "short.wav"
    sf.write(str(wav), np.zeros(2 * 16000, dtype=np.float32), 16000)
    df = engine.analyze_file(str(wav))
    assert df.empty
    assert "p_canto" in df.columns


@pytest.mark.slow
def test_folder_with_only_short_audio_keeps_result_schema(tmp_path):
    engine = BirdSongEngine()
    wav = tmp_path / "short.wav"
    sf.write(str(wav), np.zeros(2 * 16000, dtype=np.float32), 16000)
    df, errors = engine.analyze_folder(tmp_path)
    assert errors == []
    assert df.empty
    assert "p_canto" in df.columns


@pytest.mark.slow
def test_threshold_out_of_range_is_rejected(tmp_path):
    engine = BirdSongEngine()
    wav = tmp_path / "silence.wav"
    sf.write(str(wav), np.zeros(3 * 16000, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="Threshold outside"):
        engine.analyze_file(str(wav), threshold=1.1)


@pytest.mark.slow
def test_missing_file():
    engine = BirdSongEngine()
    with pytest.raises(OSError):
        engine.analyze_file(pathlib.Path("no_existe.wav"))
