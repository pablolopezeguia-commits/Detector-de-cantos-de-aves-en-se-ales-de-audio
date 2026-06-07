"""EcoCanto main window."""

from __future__ import annotations

import json
import pathlib

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QWidget,
)

from ..config import APP_NAME, APP_VERSION, SUPPORTED_EXTENSIONS


class MainWindow(QMainWindow):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._worker = None
        self._windows_df = None
        self._segments_df = None
        self._non_song_df = None
        self._window_runs_df = None
        self._errors = []
        self._elapsed = 0.0
        self._analysis_folder = ""
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(780, 500)
        self._setup_ui()

    def _setup_ui(self) -> None:
        from .control_panel import ControlPanel
        from .segments_table import SegmentsTable
        from .spectrogram_viewer import SpectrogramViewer

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.control = ControlPanel()
        self.control.setMinimumWidth(220)
        self.control.analyze_requested.connect(self._start_analysis)
        self.control.cancel_requested.connect(self._cancel_analysis)
        self.control.export_requested.connect(self._export)
        self.control.import_requested.connect(self._import_results)

        control_scroll = QScrollArea()
        control_scroll.setObjectName("controlScroll")
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll.setFrameShape(QFrame.Shape.NoFrame)
        control_scroll.setWidget(self.control)
        control_scroll.setMinimumWidth(220)
        control_scroll.setMaximumWidth(292)

        self.table = SegmentsTable()
        self.table.segment_selected.connect(self._on_segment_selected)

        self.viewer = SpectrogramViewer(sample_rate=getattr(self.engine, "sample_rate", 16000))

        content_splitter = QSplitter(Qt.Orientation.Vertical)
        content_splitter.addWidget(self.viewer)
        content_splitter.addWidget(self.table)
        content_splitter.setStretchFactor(0, 55)
        content_splitter.setStretchFactor(1, 45)
        content_splitter.setSizes([360, 240])
        content_splitter.setChildrenCollapsible(False)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(control_scroll)
        main_splitter.addWidget(content_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([244, 900])
        main_splitter.setChildrenCollapsible(False)

        layout.addWidget(main_splitter)

    def _start_analysis(self, folder: str, params: dict) -> None:
        from ..workers.analysis_worker import AnalysisWorker

        self._windows_df = None
        self._segments_df = None
        self._non_song_df = None
        self._window_runs_df = None
        self._errors = []
        self._elapsed = 0.0
        self._analysis_folder = folder

        self._worker = AnalysisWorker(folder, params, self.engine)
        self._worker.file_started.connect(self.control.update_file_progress)
        self._worker.window_done.connect(self.control.update_window_progress)
        self._worker.analysis_done.connect(self._on_analysis_done)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.error_fatal.connect(self._on_fatal_error)
        self.control.set_analyzing(True)
        self._worker.start()

    def _cancel_analysis(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.control.set_status("Cancelando...")

    def _on_analysis_done(self, windows_df, segments_df, errors, elapsed: float) -> None:
        from ..segmenter import fuse_windows_to_non_song_segments, fuse_windows_to_window_runs

        self._windows_df = windows_df
        self._segments_df = segments_df
        self._non_song_df = fuse_windows_to_non_song_segments(windows_df, segments_df)
        self._window_runs_df = fuse_windows_to_window_runs(windows_df)
        self._errors = errors
        self._elapsed = elapsed
        self.table.load_data(segments_df, self._non_song_df, self._window_runs_df)
        self.control.set_analyzing(False)
        msg = f"{len(segments_df)} cantos y {len(self._non_song_df)} no cantos en {elapsed:.1f}s"
        if errors:
            msg += f" ({len(errors)} errores)"
        self.control.set_status(msg)

    def _on_cancelled(self) -> None:
        self.control.set_analyzing(False)
        self.control.set_status("Analisis cancelado")

    def _on_fatal_error(self, message: str) -> None:
        self.control.set_analyzing(False)
        QMessageBox.critical(self, "Error", message)

    def _on_segment_selected(self, row_data: dict) -> None:
        if self._windows_df is not None:
            self.viewer.load_segment(row_data, self._windows_df)

    def _export(self) -> None:
        from ..exporter import (
            export_csv,
            export_non_song_csv,
            export_session_log,
            export_window_runs_csv,
            export_windows_csv,
        )

        if self._segments_df is None:
            QMessageBox.warning(self, "Sin datos", "No hay resultados para exportar.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar CSV", "resultados_ecocanto.csv", "CSV (*.csv)")
        if not path:
            return

        params = self.control.get_params()
        if self._analysis_folder:
            params["carpeta_audio"] = self._analysis_folder
        export_csv(self._segments_df, path, params.get("csv_sep", ";"))
        output = pathlib.Path(path)
        windows_path = output.with_name(f"{output.stem}_ventanas.csv")
        window_runs_path = output.with_name(f"{output.stem}_ventanas_agrupadas.csv")
        non_song_path = output.with_name(f"{output.stem}_no_cantos.csv")
        log_path = output.with_name(f"{output.stem}_log.json")
        if self._windows_df is not None:
            export_windows_csv(self._windows_df, windows_path, params.get("csv_sep", ";"))
        if self._window_runs_df is not None:
            export_window_runs_csv(self._window_runs_df, window_runs_path, params.get("csv_sep", ";"))
        if self._non_song_df is not None:
            export_non_song_csv(self._non_song_df, non_song_path, params.get("csv_sep", ";"))
        n_files = self._windows_df["archivo"].nunique() if self._windows_df is not None and not self._windows_df.empty else 0
        n_windows = len(self._windows_df) if self._windows_df is not None else 0
        export_session_log(
            log_path,
            params,
            n_files=n_files,
            n_windows=n_windows,
            n_segments=len(self._segments_df),
            errors=self._errors,
            duration_s=self._elapsed,
            extra={
                "csv_segmentos": output.name,
                "csv_ventanas": windows_path.name if self._windows_df is not None else None,
                "csv_ventanas_agrupadas": window_runs_path.name if self._window_runs_df is not None else None,
                "csv_no_cantos": non_song_path.name if self._non_song_df is not None else None,
            },
        )
        QMessageBox.information(
            self,
            "Exportacion completa",
            "CSV guardado en:\n"
            f"{path}\n\nVentanas:\n{windows_path}"
            f"\n\nVentanas agrupadas:\n{window_runs_path}"
            f"\n\nNo cantos:\n{non_song_path}\n\nLog:\n{log_path}",
        )

    def _import_results(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Importar resultados EcoCanto",
            "",
            "CSV (*.csv)",
        )
        if not path:
            return

        try:
            windows_df, segments_df, non_song_df, notes = self._load_results(path)
        except Exception as exc:
            QMessageBox.critical(self, "Importacion fallida", str(exc))
            return
        from ..segmenter import fuse_windows_to_window_runs

        self._windows_df = windows_df
        self._segments_df = segments_df
        self._non_song_df = non_song_df
        self._window_runs_df = fuse_windows_to_window_runs(windows_df)
        self._errors = []
        self._elapsed = 0.0
        self._analysis_folder = self.control.selected_folder()
        self.table.load_data(segments_df, non_song_df, self._window_runs_df)
        self.control.set_status(
            f"Importados {len(segments_df)} cantos y {len(non_song_df)} no cantos"
            + (f" ({'; '.join(notes)})" if notes else "")
        )

    def _load_results(self, csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
        from ..exporter import CSV_COLUMNS, NON_SONG_CSV_COLUMNS, WINDOW_CSV_COLUMNS, read_results_csv
        from ..segmenter import fuse_windows_to_non_song_segments, fuse_windows_to_segments

        selected = pathlib.Path(csv_path)
        base_stem = self._result_base_stem(selected)
        segments_path = selected.with_name(f"{base_stem}.csv")
        windows_path = selected.with_name(f"{base_stem}_ventanas.csv")
        non_song_path = selected.with_name(f"{base_stem}_no_cantos.csv")
        log_path = selected.with_name(f"{base_stem}_log.json")

        params = self._params_from_log(log_path)
        df = read_results_csv(selected)
        columns = set(df.columns)
        non_song_required = {"archivo", "t_inicio", "t_fin", "duracion_s", "n_ventanas_no_canto"}
        notes: list[str] = []

        windows_df = pd.DataFrame(columns=WINDOW_CSV_COLUMNS)
        segments_df = pd.DataFrame(columns=CSV_COLUMNS)
        non_song_df = pd.DataFrame(columns=NON_SONG_CSV_COLUMNS)

        if set(WINDOW_CSV_COLUMNS).issubset(columns):
            windows_df = self._coerce_numeric(df, WINDOW_CSV_COLUMNS)
            if segments_path.exists() and segments_path != selected:
                segments_df = self._coerce_numeric(read_results_csv(segments_path), CSV_COLUMNS)
            else:
                segments_df = fuse_windows_to_segments(
                    windows_df,
                    max_gap_windows=int(params.get("max_gap", self.control.get_params()["max_gap"])),
                    min_song_windows=int(params.get("min_song_win", self.control.get_params()["min_song_win"])),
                )
                notes.append("segmentos recalculados desde ventanas")
        elif set(CSV_COLUMNS).issubset(columns):
            segments_df = self._coerce_numeric(df, CSV_COLUMNS)
            if windows_path.exists():
                windows_df = self._coerce_numeric(read_results_csv(windows_path), WINDOW_CSV_COLUMNS)
            else:
                notes.append("sin CSV de ventanas")
        elif non_song_required.issubset(columns):
            non_song_df = self._coerce_numeric(df, NON_SONG_CSV_COLUMNS)
            if segments_path.exists():
                segments_df = self._coerce_numeric(read_results_csv(segments_path), CSV_COLUMNS)
            if windows_path.exists():
                windows_df = self._coerce_numeric(read_results_csv(windows_path), WINDOW_CSV_COLUMNS)
            if segments_df.empty:
                notes.append("sin CSV de cantos")
        else:
            raise ValueError("El CSV no tiene columnas reconocibles de EcoCanto.")

        if not windows_df.empty:
            segments_df = self._copy_paths_from_windows(segments_df, windows_df)
            non_song_df = fuse_windows_to_non_song_segments(windows_df, segments_df)
            if non_song_path.exists():
                notes.append("no cantos recalculados como complemento de cantos")

        base_dirs = self._import_base_dirs(selected, params)
        segments_df = self._resolve_audio_paths(segments_df, base_dirs)
        windows_df = self._resolve_audio_paths(windows_df, base_dirs)
        non_song_df = self._resolve_audio_paths(non_song_df, base_dirs)
        return windows_df, segments_df, non_song_df, notes

    def _result_base_stem(self, path: pathlib.Path) -> str:
        stem = path.stem
        for suffix in ("_ventanas", "_no_cantos"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    def _params_from_log(self, log_path: pathlib.Path) -> dict:
        if not log_path.exists():
            return {}
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload.get("parametros", {}))

    def _coerce_numeric(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        clean = df.copy()
        for column in columns:
            if column not in clean.columns:
                clean[column] = ""
        clean = clean[columns].copy()
        for column in clean.columns:
            if column == "archivo":
                clean[column] = clean[column].astype(str)
            else:
                clean[column] = pd.to_numeric(clean[column], errors="coerce").fillna(0)
        if "prediccion" in clean.columns:
            clean["prediccion"] = clean["prediccion"].astype(int)
        return clean

    def _copy_paths_from_windows(self, df: pd.DataFrame, windows_df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or windows_df.empty or "archivo" not in df.columns:
            return df
        by_name: dict[str, str] = {}
        for value in windows_df["archivo"].dropna().astype(str).unique():
            by_name.setdefault(pathlib.Path(value).name.lower(), value)

        copied = df.copy()
        copied["archivo"] = copied["archivo"].apply(
            lambda value: by_name.get(pathlib.Path(str(value)).name.lower(), str(value))
        )
        return copied

    def _import_base_dirs(self, selected: pathlib.Path, params: dict) -> list[pathlib.Path]:
        candidates = [self.control.selected_folder(), str(params.get("carpeta_audio", "")), str(selected.parent)]
        dirs: list[pathlib.Path] = []
        for candidate in candidates:
            if not candidate:
                continue
            path = pathlib.Path(candidate)
            if path.exists() and path.is_dir() and path not in dirs:
                dirs.append(path)
        return dirs

    def _resolve_audio_paths(self, df: pd.DataFrame, base_dirs: list[pathlib.Path]) -> pd.DataFrame:
        if df.empty or "archivo" not in df.columns:
            return df

        recursive = self.control.get_params()["recursive"]
        cache: dict[str, str] = {}

        def resolve(value: str) -> str:
            raw = str(value)
            path = pathlib.Path(raw)
            if path.exists():
                return str(path)
            for base_dir in base_dirs:
                candidate = base_dir / path
                if candidate.exists():
                    return str(candidate)
            name = path.name
            if name in cache:
                return cache[name]
            for base_dir in base_dirs:
                iterator = base_dir.rglob(name) if recursive else base_dir.glob(name)
                for candidate in iterator:
                    if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                        cache[name] = str(candidate)
                        return cache[name]
            cache[name] = raw
            return raw

        resolved = df.copy()
        resolved["archivo"] = resolved["archivo"].apply(resolve)
        return resolved
