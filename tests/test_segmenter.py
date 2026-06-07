import pandas as pd

from birdsong_detector.ecocanto.segmenter import (
    fuse_windows_to_non_song_segments,
    fuse_windows_to_segments,
    fuse_windows_to_window_runs,
)


def make_windows(preds):
    step = 1.5
    return pd.DataFrame(
        [
            {
                "archivo": "t.wav",
                "t_inicio": i * step,
                "t_fin": i * step + 3.0,
                "p_canto": 0.8 if p else 0.2,
                "prediccion": p,
            }
            for i, p in enumerate(preds)
        ]
    )


def test_gap_fuses():
    segs = fuse_windows_to_segments(make_windows([1, 1, 0, 1, 1]), max_gap_windows=1)
    assert len(segs) == 1
    assert segs.iloc[0]["n_ventanas_canto"] == 4
    assert segs.iloc[0]["n_ventanas_gap"] == 1


def test_gap_splits():
    segs = fuse_windows_to_segments(make_windows([1, 1, 0, 0, 1, 1]), max_gap_windows=1)
    assert len(segs) == 2


def test_min_song_windows_filters_short_segments():
    segs = fuse_windows_to_segments(make_windows([1, 0, 0, 1, 1]), max_gap_windows=0, min_song_windows=2)
    assert len(segs) == 1
    assert segs.iloc[0]["n_ventanas_canto"] == 2


def test_empty():
    assert fuse_windows_to_segments(pd.DataFrame()).empty


def test_non_song_is_complement_of_song_segments():
    windows = make_windows([1, 0, 1])
    song = fuse_windows_to_segments(windows, max_gap_windows=1, min_song_windows=2)
    non_song = fuse_windows_to_non_song_segments(windows, song)
    assert len(song) == 1
    assert non_song.empty


def test_non_song_keeps_only_time_outside_song_segments():
    windows = make_windows([0, 0, 1, 1, 0])
    song = fuse_windows_to_segments(windows, max_gap_windows=0, min_song_windows=2)
    non_song = fuse_windows_to_non_song_segments(windows, song)
    assert len(non_song) == 2
    assert non_song.iloc[0]["t_inicio"] == 0.0
    assert non_song.iloc[0]["t_fin"] == 3.0
    assert non_song.iloc[1]["t_inicio"] == 7.5
    assert non_song.iloc[1]["t_fin"] == 9.0
    assert "confianza_no_canto" in non_song.columns


def test_window_runs_group_consecutive_predictions():
    runs = fuse_windows_to_window_runs(make_windows([0, 0, 1, 1, 0]))
    assert list(runs["tipo"]) == ["no_canto", "canto", "no_canto"]
    assert list(runs["n_ventanas"]) == [2, 2, 1]
