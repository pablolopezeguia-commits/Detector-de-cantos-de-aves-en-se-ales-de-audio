"""Spectrogram viewer and audio player."""

from __future__ import annotations

import pathlib

import librosa
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    COLOR_ACCENT,
    COLOR_CURSOR,
    COLOR_GAP_WINDOW,
    COLOR_SEGMENT_END,
    COLOR_SEGMENT_START,
    COLOR_SONG_WINDOW,
)

try:
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
except Exception:  # pragma: no cover - depende del backend multimedia de Qt
    QAudioOutput = None
    QMediaPlayer = None

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - depende del equipo de audio
    sd = None


class SpectrogramViewer(QWidget):
    ZOOM_RATIO = 0.62
    MIN_VIEW_S = 3.0
    PAN_SLIDER_STEPS = 10000

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self._audio = np.array([], dtype=np.float32)
        self._audio_path: pathlib.Path | None = None
        self._loaded_audio_path: pathlib.Path | None = None
        self._spec_cache_path: pathlib.Path | None = None
        self._spec_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._target_sr = int(sample_rate)
        self._sr = self._target_sr
        self._offset_s = 0.0
        self._play_index = 0
        self._stream = None
        self._cursor_line = None
        self._dragging_cursor = False
        self._media_player = None
        self._audio_output = None
        self._native_ready = False
        self._native_failed = False
        self._native_start_position_ms = 0
        self._audio_status = ""
        self._current_row_data: dict | None = None
        self._current_file_windows = pd.DataFrame()
        self._selection_start_s = 0.0
        self._selection_end_s = 0.0
        self._selection_is_full = True
        self._view_start_s = 0.0
        self._view_end_s = 0.0
        self._updating_pan_slider = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 12, 12)
        layout.setSpacing(0)

        self.figure = Figure(figsize=(7, 4), facecolor="#0B0F12")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.figure.add_subplot(111)
        self._style_axis()

        self.canvas_frame = QFrame()
        self.canvas_frame.setObjectName("spectrogramFrame")
        frame_layout = QGridLayout(self.canvas_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        frame_layout.addWidget(self.canvas, 0, 0)

        overlay = QFrame()
        overlay.setObjectName("spectrogramOverlayControls")
        overlay.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        controls = QHBoxLayout()
        controls.setContentsMargins(6, 6, 8, 6)
        controls.setSpacing(5)
        overlay.setLayout(controls)

        self.play_btn = QPushButton()
        self.pause_btn = QPushButton()
        self.stop_btn = QPushButton()
        self.zoom_in_btn = QPushButton()
        self.zoom_out_btn = QPushButton()
        for button in (self.play_btn, self.pause_btn, self.stop_btn, self.zoom_in_btn, self.zoom_out_btn):
            button.setObjectName("mediaBtn")
            button.setFixedSize(26, 26)
            button.setIconSize(QSize(13, 13))
        self.play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.zoom_in_btn.setIcon(self._make_zoom_icon(plus=True))
        self.zoom_out_btn.setIcon(self._make_zoom_icon(plus=False))
        self.play_btn.setToolTip("Reproducir")
        self.pause_btn.setToolTip("Pausar")
        self.stop_btn.setToolTip("Parar")
        self.zoom_in_btn.setToolTip("Acercar")
        self.zoom_out_btn.setToolTip("Alejar")
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.time_label = QLabel("Sin segmento")
        self.time_label.setObjectName("timeCode")
        self.selection_label = QLabel("sin seleccion")
        self.selection_label.setObjectName("selectionBadge")

        controls.addWidget(self.play_btn)
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.zoom_in_btn)
        controls.addWidget(self.zoom_out_btn)
        controls.addWidget(self.time_label)
        controls.addWidget(self.selection_label)
        frame_layout.addWidget(
            overlay,
            0,
            0,
            alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
        )

        self.pan_frame = QFrame()
        self.pan_frame.setObjectName("spectrogramPanBar")
        self.pan_frame.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        pan_layout = QHBoxLayout(self.pan_frame)
        pan_layout.setContentsMargins(10, 5, 10, 5)
        pan_layout.setSpacing(0)
        self.pan_slider = QSlider(Qt.Orientation.Horizontal)
        self.pan_slider.setObjectName("spectrogramPanSlider")
        self.pan_slider.setRange(0, self.PAN_SLIDER_STEPS)
        self.pan_slider.setMinimumWidth(320)
        self.pan_slider.setMaximumWidth(720)
        self.pan_slider.valueChanged.connect(self._on_pan_slider_changed)
        pan_layout.addWidget(self.pan_slider)
        self.pan_frame.hide()
        frame_layout.addWidget(
            self.pan_frame,
            0,
            0,
            alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        )

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)
        self._native_watchdog = QTimer(self)
        self._native_watchdog.setSingleShot(True)
        self._native_watchdog.setInterval(900)
        self._native_watchdog.timeout.connect(self._check_native_progress)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self._setup_native_player()

        layout.addWidget(self.canvas_frame, 1)
        self._set_controls_enabled(False)

    def _make_zoom_icon(self, plus: bool) -> QIcon:
        pixmap = QPixmap(18, 18)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#E8ECEB"), 1.7)
        painter.setPen(pen)
        painter.drawEllipse(3, 3, 8, 8)
        painter.drawLine(10, 10, 15, 15)
        painter.drawLine(5, 7, 9, 7)
        if plus:
            painter.drawLine(7, 5, 7, 9)
        painter.end()
        return QIcon(pixmap)

    def _setup_native_player(self) -> None:
        if QMediaPlayer is None or QAudioOutput is None:
            return
        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(1.0)
        self._media_player = QMediaPlayer(self)
        self._media_player.setAudioOutput(self._audio_output)
        self._media_player.positionChanged.connect(self._on_native_position_changed)
        self._media_player.durationChanged.connect(lambda _duration: self._update_time_label())
        self._media_player.errorOccurred.connect(self._on_native_error)

    def _style_axis(self) -> None:
        self.ax.set_facecolor("#0F1418")
        self.ax.tick_params(colors="#9EA9A6", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#2A343B")
        self.ax.xaxis.label.set_color("#E8ECEB")
        self.ax.yaxis.label.set_color("#E8ECEB")
        self.ax.title.set_color("#E8ECEB")
        self.ax.xaxis.label.set_size(9)
        self.ax.yaxis.label.set_size(9)
        self.ax.title.set_size(10)

    def load_segment(self, row_data: dict, windows_df: pd.DataFrame) -> None:
        self.stop()
        audio_path = pathlib.Path(str(row_data["archivo"]))
        self._current_row_data = dict(row_data)
        self._current_file_windows = pd.DataFrame()
        if not audio_path.exists():
            self._audio = np.array([], dtype=np.float32)
            self._audio_path = None
            self._loaded_audio_path = None
            self._spec_cache = None
            self._spec_cache_path = None
            self._native_ready = False
            self._native_failed = False
            if self._media_player is not None:
                self._media_player.stop()
                self._media_player.setSource(QUrl())
            self._set_selection(0.0, 0.0, full=True)
            self._fit_view_to_audio(redraw=False)
            self._set_controls_enabled(False)
            self._draw_message(f"No se encuentra el audio: {audio_path.name}")
            return
        self._audio_path = audio_path
        self._prepare_native_player(audio_path)

        if self._loaded_audio_path != audio_path:
            try:
                audio, sr = librosa.load(audio_path, sr=self._target_sr, mono=True)
            except Exception as exc:
                self._audio = np.array([], dtype=np.float32)
                self._loaded_audio_path = None
                self._spec_cache = None
                self._spec_cache_path = None
                self._native_ready = False
                self._native_failed = False
                if self._media_player is not None:
                    self._media_player.stop()
                    self._media_player.setSource(QUrl())
                self._set_selection(0.0, 0.0, full=True)
                self._fit_view_to_audio(redraw=False)
                self._set_controls_enabled(False)
                self._draw_message(f"No se puede abrir el audio: {audio_path.name}\n{exc}")
                return

            self._audio = audio.astype(np.float32, copy=False)
            self._sr = sr
            self._loaded_audio_path = audio_path
            self._spec_cache = None
            self._spec_cache_path = None
        file_windows = windows_df[windows_df["archivo"].astype(str) == str(audio_path)]
        self._current_file_windows = file_windows.copy()
        self._offset_s = 0.0
        self._set_selection(
            float(row_data.get("t_inicio", 0.0)),
            float(row_data.get("t_fin", self._audio_duration_s())),
            full=False,
            update_position=True,
        )
        if self._is_zoomed():
            self._center_current_zoom_on_selection(redraw=False)
        else:
            self._fit_view_to_audio(redraw=False)
        self._draw_spectrogram()
        self._set_controls_enabled(len(self._audio) > 0)

    def _prepare_native_player(self, audio_path: pathlib.Path) -> None:
        self._native_ready = False
        self._native_failed = False
        self._audio_status = ""
        if self._media_player is None:
            return
        self._media_player.stop()
        self._media_player.setSource(QUrl.fromLocalFile(str(audio_path.resolve())))
        self._native_ready = True

    def _draw_message(self, message: str) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self._style_axis()
        self.ax.set_axis_off()
        self.ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            color="#9EA9A6",
            transform=self.ax.transAxes,
            wrap=True,
        )
        self._cursor_line = None
        self._update_pan_slider()
        self.canvas.draw_idle()
        self._update_time_label()

    def _draw_spectrogram(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self._style_axis()

        if len(self._audio) == 0:
            self._draw_message("Sin audio")
            return

        n_fft = 512
        times, freqs, spec_db = self._spectrogram_data(n_fft=n_fft)

        self.ax.pcolormesh(times, freqs, spec_db, shading="auto", cmap="inferno", zorder=1)
        self.ax.set_ylim(0, 8000)
        view_start, view_end = self._view_bounds()
        self.ax.set_xlim(view_start, view_end)
        self.ax.set_xlabel("Tiempo (s)")
        self.ax.set_ylabel("Frecuencia (Hz)")
        title_path = self._audio_path or pathlib.Path(str((self._current_row_data or {}).get("archivo", "")))
        self.ax.set_title(pathlib.Path(str(title_path)).name)

        seg_start, seg_end = self._selection_bounds()
        current_data = self._current_row_data or {}
        selected_is_no_song = (
            "n_ventanas_no_canto" in current_data
            or str(current_data.get("tipo", "")).lower() == "no_canto"
            or int(current_data.get("prediccion", 1)) == 0
        )
        selected_color = COLOR_ACCENT
        if selected_is_no_song and not self._selection_is_full:
            selected_color = COLOR_GAP_WINDOW

        for _, win in self._current_file_windows.iterrows():
            is_song = int(win["prediccion"]) == 1
            if is_song:
                self.ax.axvspan(
                    float(win["t_inicio"]),
                    float(win["t_fin"]),
                    color=COLOR_SONG_WINDOW,
                    alpha=0.12,
                    linewidth=0,
                    zorder=2,
                )

        selection_alpha = 0.12 if self._selection_is_full else 0.28
        self.ax.axvspan(seg_start, seg_end, color=selected_color, alpha=selection_alpha, linewidth=0, zorder=3)
        self.ax.axvline(seg_start, color=COLOR_SEGMENT_START, linewidth=1.8, zorder=4)
        self.ax.axvline(seg_end, color=COLOR_SEGMENT_END, linewidth=1.8, zorder=4)

        current = self._current_play_position_s()
        self._cursor_line = self.ax.axvline(
            self._offset_s + current,
            color=COLOR_CURSOR,
            linewidth=1.1,
            zorder=5,
        )
        self.figure.subplots_adjust(left=0.065, right=0.995, bottom=0.13, top=0.92)
        self._update_pan_slider()
        self.canvas.draw_idle()
        self._update_time_label()

    def _spectrogram_data(self, n_fft: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._spec_cache is not None and self._spec_cache_path == self._loaded_audio_path:
            return self._spec_cache

        max_frames = 2200
        hop = max(128, int(np.ceil(len(self._audio) / max_frames)))
        spec = librosa.stft(self._audio, n_fft=n_fft, hop_length=hop)
        magnitude = np.abs(spec)
        ref_value = float(np.max(magnitude)) if magnitude.size else 1.0
        spec_db = librosa.amplitude_to_db(magnitude, ref=max(ref_value, 1e-9))
        times = librosa.times_like(spec_db, sr=self._sr, hop_length=hop) + self._offset_s
        freqs = librosa.fft_frequencies(sr=self._sr, n_fft=n_fft)
        self._spec_cache = (times, freqs, spec_db)
        self._spec_cache_path = self._loaded_audio_path
        return self._spec_cache

    def _audio_duration_s(self) -> float:
        if self._sr <= 0:
            return 0.0
        return len(self._audio) / self._sr

    def _view_bounds(self) -> tuple[float, float]:
        duration = self._audio_duration_s()
        if duration <= 0:
            return 0.0, 1.0
        start = min(max(self._view_start_s, 0.0), duration)
        end = min(max(self._view_end_s, start), duration)
        if end <= start:
            return 0.0, duration
        return start, end

    def _is_zoomed(self) -> bool:
        duration = self._audio_duration_s()
        if duration <= 0:
            return False
        start, end = self._view_bounds()
        return start > 1e-6 or end < duration - 1e-6

    def _fit_view_to_audio(self, *, redraw: bool = True) -> None:
        duration = self._audio_duration_s()
        self._view_start_s = 0.0
        self._view_end_s = duration
        self._update_pan_slider()
        if redraw and len(self._audio):
            self._draw_spectrogram()

    def _selection_center_s(self) -> float:
        duration = self._audio_duration_s()
        if duration <= 0:
            return 0.0
        start, end = self._selection_bounds()
        if self._selection_is_full:
            current = self._current_play_position_s()
            return min(max(current, 0.0), duration)
        return min(max((start + end) / 2.0, 0.0), duration)

    def _set_view_center_width(self, center_s: float, width_s: float, *, redraw: bool = True) -> None:
        duration = self._audio_duration_s()
        if duration <= 0:
            return
        min_width = min(duration, self.MIN_VIEW_S)
        width_s = min(max(width_s, min_width), duration)
        if width_s >= duration - 1e-6:
            self._fit_view_to_audio(redraw=redraw)
            return
        start = min(max(center_s - width_s / 2.0, 0.0), duration - width_s)
        self._view_start_s = start
        self._view_end_s = start + width_s
        self._update_pan_slider()
        if redraw:
            self._draw_spectrogram()

    def _center_current_zoom_on_selection(self, *, redraw: bool = True) -> None:
        start, end = self._view_bounds()
        self._set_view_center_width(self._selection_center_s(), end - start, redraw=redraw)

    def zoom_in(self) -> None:
        if len(self._audio) == 0:
            return
        start, end = self._view_bounds()
        self._set_view_center_width(self._selection_center_s(), (end - start) * self.ZOOM_RATIO)

    def zoom_out(self) -> None:
        if len(self._audio) == 0 or not self._is_zoomed():
            return
        start, end = self._view_bounds()
        self._set_view_center_width(self._selection_center_s(), (end - start) / self.ZOOM_RATIO)

    def _update_pan_slider(self) -> None:
        if not hasattr(self, "pan_frame"):
            return
        duration = self._audio_duration_s()
        zoomed = self._is_zoomed()
        self.pan_frame.setVisible(zoomed)
        self.zoom_out_btn.setEnabled(zoomed)
        if duration <= 0 or not zoomed:
            return
        start, end = self._view_bounds()
        max_start = max(duration - (end - start), 0.0)
        slider_value = 0 if max_start <= 0 else round((start / max_start) * self.PAN_SLIDER_STEPS)
        self._updating_pan_slider = True
        self.pan_slider.setValue(min(max(slider_value, 0), self.PAN_SLIDER_STEPS))
        self._updating_pan_slider = False

    def _on_pan_slider_changed(self, value: int) -> None:
        if self._updating_pan_slider or not self._is_zoomed():
            return
        duration = self._audio_duration_s()
        start, end = self._view_bounds()
        width = end - start
        max_start = max(duration - width, 0.0)
        self._view_start_s = (float(value) / self.PAN_SLIDER_STEPS) * max_start if max_start > 0 else 0.0
        self._view_end_s = self._view_start_s + width
        self._draw_spectrogram()

    def _selection_bounds(self) -> tuple[float, float]:
        duration = self._audio_duration_s()
        start = min(max(self._selection_start_s, 0.0), duration)
        end = min(max(self._selection_end_s, start), duration)
        if end <= start and duration > 0:
            start, end = 0.0, duration
        return start, end

    def _selection_sample_bounds(self) -> tuple[int, int]:
        start, end = self._selection_bounds()
        start_idx = min(max(int(start * self._sr), 0), len(self._audio))
        end_idx = min(max(int(end * self._sr), start_idx), len(self._audio))
        return start_idx, end_idx

    def _current_play_position_s(self) -> float:
        if len(self._audio) == 0:
            return 0.0
        if self._media_player is not None and self._native_ready and not self._native_failed:
            return max(0, self._media_player.position()) / 1000.0
        return self._play_index / self._sr

    def _set_selection(
        self,
        start_s: float,
        end_s: float,
        *,
        full: bool,
        update_position: bool = False,
        redraw: bool = False,
    ) -> None:
        duration = self._audio_duration_s()
        if full:
            start_s, end_s = 0.0, duration
        else:
            start_s = min(max(start_s, 0.0), duration)
            end_s = min(max(end_s, start_s), duration)
            if end_s <= start_s and duration > 0:
                end_s = min(duration, start_s + max(1.0 / max(self._sr, 1), 0.01))
        self._selection_start_s = start_s
        self._selection_end_s = end_s
        self._selection_is_full = full
        if update_position:
            self._set_play_position(start_s)
            self._update_cursor(start_s)
        self._update_selection_label()
        self._update_time_label()
        if redraw and len(self._audio):
            self._draw_spectrogram()

    def _update_selection_label(self) -> None:
        if not hasattr(self, "selection_label"):
            return
        if len(self._audio) == 0:
            self.selection_label.setText("sin seleccion")
            return
        start, end = self._selection_bounds()
        duration = max(0.0, end - start)
        if self._selection_is_full:
            self.selection_label.setText(f"audio completo | {duration:.1f}s")
        else:
            self.selection_label.setText(f"clip | {duration:.1f}s")

    def _set_controls_enabled(self, enabled: bool) -> None:
        audio_backend = self._native_ready or sd is not None
        self.play_btn.setEnabled(enabled and audio_backend)
        self.pause_btn.setEnabled(enabled and audio_backend)
        self.stop_btn.setEnabled(enabled and audio_backend)
        self.zoom_in_btn.setEnabled(enabled)
        self.zoom_out_btn.setEnabled(enabled and self._is_zoomed())

    def _on_native_error(self, _error, error_string: str = "") -> None:
        if not self._native_ready:
            return
        detail = error_string or "Qt no pudo abrir el audio"
        self._fallback_to_sounddevice(detail)

    def _on_canvas_press(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or len(self._audio) == 0:
            return
        if getattr(event, "dblclick", False):
            duration = self._audio_duration_s()
            self._dragging_cursor = False
            self._fit_view_to_audio(redraw=False)
            self._set_selection(0.0, duration, full=True, update_position=True, redraw=True)
            return
        self._dragging_cursor = True
        self._seek_to_time(float(event.xdata))

    def _on_canvas_motion(self, event) -> None:
        if not self._dragging_cursor or event.inaxes != self.ax or event.xdata is None:
            return
        self._seek_to_time(float(event.xdata))

    def _on_canvas_release(self, event) -> None:
        del event
        self._dragging_cursor = False

    def _seek_to_time(self, t_abs: float) -> None:
        if len(self._audio) == 0:
            return
        duration = len(self._audio) / self._sr
        t_rel = min(max(t_abs - self._offset_s, 0.0), duration)
        self._set_play_position(t_rel)
        self._update_cursor(t_rel)
        self._update_time_label()

    def _set_play_position(self, t_rel: float) -> None:
        self._play_index = min(int(t_rel * self._sr), max(len(self._audio) - 1, 0))
        if self._media_player is not None and self._native_ready:
            self._media_player.setPosition(int(t_rel * 1000))

    def play(self) -> None:
        if len(self._audio) == 0:
            return
        start_s, end_s = self._selection_bounds()
        if end_s <= start_s:
            return
        current = self._current_play_position_s()
        if current < start_s or current >= end_s - 0.03:
            self._set_play_position(start_s)
            self._update_cursor(start_s)
        if self._media_player is not None and self._native_ready and not self._native_failed:
            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)
            position_ms = self._media_player.position()
            if position_ms < start_ms or position_ms >= end_ms - 30:
                self._media_player.setPosition(start_ms)
            self._native_start_position_ms = self._media_player.position()
            self._media_player.play()
            self._native_watchdog.start()
            return
        self._play_with_sounddevice()

    def _play_with_sounddevice(self) -> None:
        if sd is None:
            self._audio_status = "Sin backend de audio"
            self._update_time_label()
            return
        start_idx, end_idx = self._selection_sample_bounds()
        if self._play_index < start_idx or self._play_index >= max(end_idx - 1, start_idx):
            self._play_index = start_idx
        if self._stream is None:
            try:
                self._stream = sd.OutputStream(
                    samplerate=self._sr,
                    channels=1,
                    dtype="float32",
                    callback=self._audio_callback,
                )
                self._stream.start()
            except Exception as exc:
                self._stream = None
                self._audio_status = f"Error de audio: {exc}"
                self._update_time_label()
                return
        self.timer.start()

    def _check_native_progress(self) -> None:
        if self._media_player is None or not self._native_ready or self._native_failed:
            return
        if self._media_player.position() <= self._native_start_position_ms + 20:
            self._fallback_to_sounddevice("Qt no inicio la reproduccion")

    def _fallback_to_sounddevice(self, reason: str) -> None:
        self._native_failed = True
        self._native_ready = False
        self._audio_status = f"Audio fallback: {reason}"
        if self._media_player is not None:
            self._media_player.stop()
        self._play_with_sounddevice()

    def pause(self) -> None:
        self.timer.stop()
        self._native_watchdog.stop()
        if self._media_player is not None and self._native_ready and not self._native_failed:
            self._media_player.pause()
            return
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def stop(self) -> None:
        self.pause()
        start_s, _end_s = self._selection_bounds()
        self._set_play_position(start_s)
        self._update_cursor(start_s)
        self._update_time_label()

    def _audio_callback(self, outdata, frames, time_info, status) -> None:
        del time_info, status
        _start_idx, selection_end_idx = self._selection_sample_bounds()
        end = min(self._play_index + frames, selection_end_idx)
        chunk = self._audio[self._play_index:end]
        outdata.fill(0)
        if len(chunk):
            outdata[: len(chunk), 0] = chunk
        self._play_index = end
        if self._play_index >= selection_end_idx:
            QTimer.singleShot(0, self._finish_selection_playback)

    def _tick(self) -> None:
        if len(self._audio):
            self._update_cursor(self._play_index / self._sr)
        self._update_time_label()

    def _on_native_position_changed(self, position_ms: int) -> None:
        if len(self._audio) == 0:
            return
        t_rel = max(0.0, float(position_ms) / 1000.0)
        self._play_index = min(int(t_rel * self._sr), max(len(self._audio) - 1, 0))
        _start_s, end_s = self._selection_bounds()
        if (
            self._media_player is not None
            and self._media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            and t_rel >= end_s - 0.02
        ):
            self._finish_selection_playback()
            return
        self._update_cursor(t_rel)
        self._update_time_label()

    def _finish_selection_playback(self) -> None:
        self.pause()
        start_s, _end_s = self._selection_bounds()
        self._set_play_position(start_s)
        self._update_cursor(start_s)
        self._update_time_label()

    def _update_cursor(self, t_rel: float) -> None:
        if self._cursor_line is None or len(self._audio) == 0:
            return
        duration = len(self._audio) / self._sr
        t_rel = min(max(t_rel, 0.0), duration)
        t_abs = self._offset_s + t_rel
        self._cursor_line.set_xdata([t_abs, t_abs])
        self.canvas.draw_idle()

    def _update_time_label(self) -> None:
        if len(self._audio) == 0:
            self.time_label.setText("Sin segmento")
            self._update_selection_label()
            return
        if self._audio_status and self._stream is None and (self._native_failed or sd is None):
            self.time_label.setText(self._audio_status)
            return
        current = self._current_play_position_s()
        total = self._audio_duration_s()
        start_s, end_s = self._selection_bounds()
        if self._selection_is_full:
            self.time_label.setText(f"{current:.1f}s / {total:.1f}s")
        else:
            clip_current = min(max(current - start_s, 0.0), max(end_s - start_s, 0.0))
            self.time_label.setText(f"{clip_current:.1f}s / {max(end_s - start_s, 0.0):.1f}s")
