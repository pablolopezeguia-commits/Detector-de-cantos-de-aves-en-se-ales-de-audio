"""EcoCanto result export helpers."""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pandas as pd


CSV_COLUMNS = [
    "archivo",
    "t_inicio",
    "t_fin",
    "duracion_s",
    "n_ventanas_canto",
    "n_ventanas_gap",
    "score_medio",
]

NON_SONG_CSV_COLUMNS = [
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

WINDOW_CSV_COLUMNS = [
    "archivo",
    "t_inicio",
    "t_fin",
    "p_canto",
    "prediccion",
]

WINDOW_PREPROCESS_COLUMNS = [
    "source_sample_rate",
    "preprocess_gain",
    "preprocess_snr_estimated",
    "preprocess_input_rms",
    "preprocess_input_peak_abs",
    "preprocess_output_peak_abs",
    "preprocess_noise_floor_rms",
]

WINDOW_RUN_CSV_COLUMNS = [
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


def export_csv(segments_df: pd.DataFrame, output_path: str | pathlib.Path, separator: str = ";") -> None:
    output = pathlib.Path(output_path)
    if segments_df.empty:
        output.write_text(separator.join(CSV_COLUMNS) + "\n", encoding="utf-8-sig")
        return

    df = segments_df[CSV_COLUMNS].copy()
    df["archivo"] = df["archivo"].apply(lambda p: pathlib.Path(str(p)).name)
    df.to_csv(output, sep=separator, index=False, encoding="utf-8-sig")


def export_non_song_csv(
    non_song_df: pd.DataFrame,
    output_path: str | pathlib.Path,
    separator: str = ";",
) -> None:
    output = pathlib.Path(output_path)
    if non_song_df.empty:
        output.write_text(separator.join(NON_SONG_CSV_COLUMNS) + "\n", encoding="utf-8-sig")
        return

    df = non_song_df[NON_SONG_CSV_COLUMNS].copy()
    df["archivo"] = df["archivo"].apply(lambda p: pathlib.Path(str(p)).name)
    df.to_csv(output, sep=separator, index=False, encoding="utf-8-sig")


def export_windows_csv(
    windows_df: pd.DataFrame,
    output_path: str | pathlib.Path,
    separator: str = ";",
) -> None:
    """Save window-level results with full paths so a run can be reloaded."""
    output = pathlib.Path(output_path)
    if windows_df.empty:
        output.write_text(separator.join(WINDOW_CSV_COLUMNS) + "\n", encoding="utf-8-sig")
        return

    columns = WINDOW_CSV_COLUMNS + [col for col in WINDOW_PREPROCESS_COLUMNS if col in windows_df.columns]
    df = windows_df[columns].copy()
    df.to_csv(output, sep=separator, index=False, encoding="utf-8-sig")


def export_window_runs_csv(
    window_runs_df: pd.DataFrame,
    output_path: str | pathlib.Path,
    separator: str = ";",
) -> None:
    output = pathlib.Path(output_path)
    if window_runs_df.empty:
        output.write_text(separator.join(WINDOW_RUN_CSV_COLUMNS) + "\n", encoding="utf-8-sig")
        return

    df = window_runs_df[WINDOW_RUN_CSV_COLUMNS].copy()
    df["archivo"] = df["archivo"].apply(lambda p: pathlib.Path(str(p)).name)
    df.to_csv(output, sep=separator, index=False, encoding="utf-8-sig")


def read_results_csv(path: str | pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


def export_session_log(
    output_path: str | pathlib.Path,
    params: dict,
    n_files: int,
    n_windows: int,
    n_segments: int,
    errors: list[dict],
    duration_s: float,
    extra: dict | None = None,
) -> None:
    log = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "app": "EcoCanto",
        "parametros": params,
        "archivos_procesados": int(n_files),
        "ventanas_analizadas": int(n_windows),
        "segmentos_detectados": int(n_segments),
        "duracion_analisis_s": round(float(duration_s), 1),
        "errores": errors,
    }
    if extra:
        log.update(extra)
    pathlib.Path(output_path).write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
