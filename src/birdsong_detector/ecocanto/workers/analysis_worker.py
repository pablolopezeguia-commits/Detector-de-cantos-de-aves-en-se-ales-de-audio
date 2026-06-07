"""Background analysis worker that keeps the UI responsive."""

from __future__ import annotations

import time

from PyQt6.QtCore import QThread, pyqtSignal

from ..segmenter import fuse_windows_to_segments


class AnalysisWorker(QThread):
    file_started = pyqtSignal(str, int, int)
    window_done = pyqtSignal(int, int)
    analysis_done = pyqtSignal(object, object, object, float)
    cancelled = pyqtSignal()
    error_fatal = pyqtSignal(str)

    def __init__(self, folder: str, params: dict, engine):
        super().__init__()
        self.folder = folder
        self.params = params
        self.engine = engine
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _is_cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        t0 = time.time()
        try:
            windows_df, errors = self.engine.analyze_folder(
                folder=self.folder,
                recursive=self.params["recursive"],
                threshold=self.params["threshold"],
                file_callback=lambda p, i, t: self.file_started.emit(p, i, t),
                progress_callback=lambda c, t: self.window_done.emit(c, t),
                cancel_callback=self._is_cancelled,
            )
            if self._cancelled:
                self.cancelled.emit()
                return

            segments_df = fuse_windows_to_segments(
                windows_df,
                max_gap_windows=self.params["max_gap"],
                min_song_windows=self.params["min_song_win"],
            )
            self.analysis_done.emit(windows_df, segments_df, errors, time.time() - t0)
        except Exception as exc:
            self.error_fatal.emit(str(exc))
