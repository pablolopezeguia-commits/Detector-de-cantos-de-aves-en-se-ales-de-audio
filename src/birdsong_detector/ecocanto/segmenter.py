"""Merge per-window predictions into reviewable song and non-song segments."""

from __future__ import annotations

import pandas as pd


SEGMENT_COLUMNS = [
    "archivo",
    "t_inicio",
    "t_fin",
    "duracion_s",
    "n_ventanas_canto",
    "n_ventanas_gap",
    "score_medio",
]

NON_SONG_SEGMENT_COLUMNS = [
    "archivo",
    "t_inicio",
    "t_fin",
    "duracion_s",
    "n_ventanas_no_canto",
    "n_ventanas_gap",
    "score_medio",
    "confianza_no_canto",
    "p_canto_max",
]

WINDOW_RUN_COLUMNS = [
    "archivo",
    "tipo",
    "t_inicio",
    "t_fin",
    "duracion_s",
    "n_ventanas",
    "p_canto_medio",
    "p_canto_min",
    "p_canto_max",
    "prediccion",
]


def fuse_windows_to_segments(
    windows_df: pd.DataFrame,
    max_gap_windows: int = 1,
    min_song_windows: int = 1,
) -> pd.DataFrame:
    if windows_df.empty:
        return _empty_segments_df()

    segments: list[dict] = []
    for archivo, group in windows_df.groupby("archivo", sort=False):
        ordered = group.sort_values("t_inicio").reset_index(drop=True)
        _process_file(ordered, str(archivo), segments, max_gap_windows, min_song_windows)
    return pd.DataFrame(segments, columns=SEGMENT_COLUMNS) if segments else _empty_segments_df()


def fuse_windows_to_non_song_segments(
    windows_df: pd.DataFrame,
    song_segments_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the temporal complement of the accepted song segments."""
    if windows_df.empty:
        return pd.DataFrame(columns=NON_SONG_SEGMENT_COLUMNS)

    segments: list[dict] = []
    song_lookup = _build_song_lookup(song_segments_df)
    for archivo, group in windows_df.groupby("archivo", sort=False):
        ordered = group.sort_values("t_inicio").reset_index(drop=True)
        song_group = _song_group_for_file(str(archivo), song_lookup)
        _process_non_song_file(ordered, str(archivo), segments, song_group)
    if not segments:
        return pd.DataFrame(columns=NON_SONG_SEGMENT_COLUMNS)
    return pd.DataFrame(segments, columns=NON_SONG_SEGMENT_COLUMNS)


def fuse_windows_to_window_runs(windows_df: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive windows with the same raw classifier decision."""
    if windows_df.empty:
        return pd.DataFrame(columns=WINDOW_RUN_COLUMNS)

    runs: list[dict] = []
    for archivo, group in windows_df.groupby("archivo", sort=False):
        ordered = group.sort_values("t_inicio").reset_index(drop=True)
        _process_window_runs_file(ordered, str(archivo), runs)
    if not runs:
        return pd.DataFrame(columns=WINDOW_RUN_COLUMNS)
    return pd.DataFrame(runs, columns=WINDOW_RUN_COLUMNS)


def _process_file(
    group: pd.DataFrame,
    archivo: str,
    segments: list[dict],
    max_gap: int,
    min_song: int,
) -> None:
    in_seg = False
    seg_start = 0.0
    seg_end = 0.0
    gap_count = 0
    song_count = 0
    gap_seg = 0
    scores: list[float] = []

    def close_segment() -> None:
        nonlocal in_seg, seg_start, seg_end, gap_count, song_count, gap_seg, scores
        if song_count >= min_song:
            segments.append(
                {
                    "archivo": archivo,
                    "t_inicio": round(seg_start, 3),
                    "t_fin": round(seg_end, 3),
                    "duracion_s": round(seg_end - seg_start, 3),
                    "n_ventanas_canto": int(song_count),
                    "n_ventanas_gap": int(gap_seg),
                    "score_medio": round(sum(scores) / len(scores), 4),
                }
            )
        in_seg = False
        seg_start = 0.0
        seg_end = 0.0
        gap_count = 0
        song_count = 0
        gap_seg = 0
        scores = []

    for _, row in group.iterrows():
        is_song = int(row["prediccion"]) == 1
        if not in_seg:
            if is_song:
                in_seg = True
                seg_start = float(row["t_inicio"])
                seg_end = float(row["t_fin"])
                gap_count = 0
                song_count = 1
                gap_seg = 0
                scores = [float(row["p_canto"])]
            continue

        if is_song:
            seg_end = float(row["t_fin"])
            song_count += 1
            gap_count = 0
            scores.append(float(row["p_canto"]))
        else:
            gap_count += 1
            if gap_count > max_gap:
                close_segment()
            else:
                gap_seg += 1
                seg_end = float(row["t_fin"])

    if in_seg:
        close_segment()


def _process_non_song_file(
    group: pd.DataFrame,
    archivo: str,
    segments: list[dict],
    song_group: pd.DataFrame | None,
) -> None:
    coverage_start = float(group["t_inicio"].min())
    coverage_end = float(group["t_fin"].max())
    if coverage_end <= coverage_start:
        return

    song_intervals = _song_intervals_from_segments(song_group)
    if song_group is None:
        song_intervals = _song_intervals_from_raw_positive_windows(group)
    song_intervals = _merge_intervals(song_intervals, coverage_start, coverage_end)

    cursor = coverage_start
    for song_start, song_end in song_intervals:
        if song_start > cursor:
            _append_non_song_interval(group, archivo, cursor, song_start, coverage_end, segments)
        cursor = max(cursor, song_end)
    if cursor < coverage_end:
        _append_non_song_interval(group, archivo, cursor, coverage_end, coverage_end, segments)


def _process_window_runs_file(group: pd.DataFrame, archivo: str, runs: list[dict]) -> None:
    current_pred: int | None = None
    run_start = 0.0
    run_end = 0.0
    scores: list[float] = []

    def close_run() -> None:
        if current_pred is None or not scores:
            return
        p_mean = sum(scores) / len(scores)
        runs.append(
            {
                "archivo": archivo,
                "tipo": "canto" if current_pred == 1 else "no_canto",
                "t_inicio": round(run_start, 3),
                "t_fin": round(run_end, 3),
                "duracion_s": round(run_end - run_start, 3),
                "n_ventanas": int(len(scores)),
                "p_canto_medio": round(p_mean, 4),
                "p_canto_min": round(min(scores), 4),
                "p_canto_max": round(max(scores), 4),
                "prediccion": int(current_pred),
            }
        )

    for _, row in group.iterrows():
        pred = int(row["prediccion"])
        score = float(row["p_canto"])
        if current_pred is None:
            current_pred = pred
            run_start = float(row["t_inicio"])
            run_end = float(row["t_fin"])
            scores = [score]
            continue
        if pred != current_pred:
            close_run()
            current_pred = pred
            run_start = float(row["t_inicio"])
            run_end = float(row["t_fin"])
            scores = [score]
            continue
        run_end = float(row["t_fin"])
        scores.append(score)

    close_run()


def _append_non_song_interval(
    group: pd.DataFrame,
    archivo: str,
    start: float,
    end: float,
    coverage_end: float,
    segments: list[dict],
) -> None:
    if end - start <= 1e-6:
        return
    centers = (group["t_inicio"].astype(float) + group["t_fin"].astype(float)) / 2.0
    if abs(end - coverage_end) <= 1e-6:
        selected = group[(centers >= start) & (centers <= end)]
    else:
        selected = group[(centers >= start) & (centers < end)]
    if selected.empty:
        selected = group[(group["t_fin"].astype(float) > start) & (group["t_inicio"].astype(float) < end)]

    scores = selected["p_canto"].astype(float).tolist()
    p_mean = sum(scores) / len(scores) if scores else 0.0
    p_max = max(scores) if scores else 0.0
    segments.append(
        {
            "archivo": archivo,
            "t_inicio": round(start, 3),
            "t_fin": round(end, 3),
            "duracion_s": round(end - start, 3),
            "n_ventanas_no_canto": int(len(selected)),
            "n_ventanas_gap": 0,
            "score_medio": round(p_mean, 4),
            "confianza_no_canto": round(1.0 - p_mean, 4),
            "p_canto_max": round(p_max, 4),
        }
    )


def _build_song_lookup(song_segments_df: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if song_segments_df is None or song_segments_df.empty or "archivo" not in song_segments_df.columns:
        return {}
    lookup: dict[str, pd.DataFrame] = {}
    for archivo, group in song_segments_df.groupby("archivo", sort=False):
        key = str(archivo)
        lookup[key] = group
        lookup.setdefault(key.replace("\\", "/"), group)
        lookup.setdefault(key.split("\\")[-1].lower(), group)
        lookup.setdefault(key.split("/")[-1].lower(), group)
    return lookup


def _song_group_for_file(archivo: str, lookup: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    if not lookup:
        return None
    for key in (
        archivo,
        archivo.replace("\\", "/"),
        archivo.split("\\")[-1].lower(),
        archivo.split("/")[-1].lower(),
    ):
        if key in lookup:
            return lookup[key]
    return None


def _song_intervals_from_segments(song_group: pd.DataFrame | None) -> list[tuple[float, float]]:
    if song_group is None or song_group.empty:
        return []
    return [
        (float(row["t_inicio"]), float(row["t_fin"]))
        for _, row in song_group.iterrows()
        if float(row["t_fin"]) > float(row["t_inicio"])
    ]


def _song_intervals_from_raw_positive_windows(group: pd.DataFrame) -> list[tuple[float, float]]:
    intervals = []
    for _, row in group.iterrows():
        if int(row["prediccion"]) == 1:
            intervals.append((float(row["t_inicio"]), float(row["t_fin"])))
    return intervals


def _merge_intervals(
    intervals: list[tuple[float, float]],
    coverage_start: float,
    coverage_end: float,
) -> list[tuple[float, float]]:
    clean = sorted(
        (
            (max(coverage_start, float(start)), min(coverage_end, float(end)))
            for start, end in intervals
        ),
        key=lambda value: value[0],
    )
    merged: list[tuple[float, float]] = []
    for start, end in clean:
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return merged


def _empty_segments_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SEGMENT_COLUMNS)
