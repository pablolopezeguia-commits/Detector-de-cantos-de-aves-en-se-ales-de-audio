"""EcoCanto control panel."""

from __future__ import annotations

import pathlib

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    APP_NAME,
    APP_LOGO_PATH,
    DEFAULT_CSV_SEP,
    DEFAULT_MAX_GAP,
    DEFAULT_MIN_SONG_WIN,
    DEFAULT_RECURSIVE,
    DEFAULT_THRESHOLD,
)


THRESHOLD_SCALE = 1_000_000


class ControlPanel(QWidget):
    analyze_requested = pyqtSignal(str, dict)
    cancel_requested = pyqtSignal()
    export_requested = pyqtSignal()
    import_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._selected_folder = ""
        self._setup_ui()
        self.set_analyzing(False)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(8)
        logo = QLabel()
        logo.setObjectName("brandLogo")
        logo.setFixedSize(32, 32)
        logo.setScaledContents(False)
        if APP_LOGO_PATH.exists():
            pixmap = QPixmap(str(APP_LOGO_PATH))
            logo.setPixmap(
                pixmap.scaled(
                    32,
                    32,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        title = QLabel(APP_NAME)
        title.setObjectName("brandTitle")
        brand_row.addWidget(logo)
        brand_row.addWidget(title, 1)
        subtitle = QLabel("Detector de cantos")
        subtitle.setObjectName("brandSubtitle")
        layout.addLayout(brand_row)
        layout.addWidget(subtitle)

        self._folder_box = self._build_folder_box()
        self._params_box = self._build_params_box()
        self._actions_box = self._build_actions_box()
        self._progress_box = self._build_progress_box()

        layout.addWidget(self._folder_box)
        layout.addWidget(self._params_box)
        layout.addWidget(self._actions_box)
        layout.addWidget(self._progress_box)
        layout.addStretch(1)

        model_note = QLabel("Modelo final YAMNet + MLP\nCSV: ;  |  Min. ventanas: 2")
        model_note.setObjectName("modelNote")
        model_note.setWordWrap(True)
        layout.addWidget(model_note)

    def _build_folder_box(self) -> QGroupBox:
        box = QGroupBox("Entrada")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setPlaceholderText("Carpeta de audios")

        pick_btn = QPushButton("Seleccionar carpeta")
        pick_btn.clicked.connect(self._pick_folder)

        self.recursive_check = QCheckBox("Incluir subcarpetas")
        self.recursive_check.setChecked(DEFAULT_RECURSIVE)

        layout.addWidget(self.folder_edit)
        layout.addWidget(pick_btn)
        layout.addWidget(self.recursive_check)
        return box

    def _build_params_box(self) -> QGroupBox:
        box = QGroupBox("Parametros")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        threshold_row = QHBoxLayout()
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.001)
        self.threshold_spin.setDecimals(6)
        self.threshold_spin.setValue(DEFAULT_THRESHOLD)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, THRESHOLD_SCALE)
        self.threshold_slider.setValue(round(DEFAULT_THRESHOLD * THRESHOLD_SCALE))
        self.threshold_spin.valueChanged.connect(self._sync_threshold_slider)
        self.threshold_slider.valueChanged.connect(self._sync_threshold_spin)
        threshold_row.addWidget(self.threshold_spin)
        threshold_row.addWidget(self.threshold_slider)

        form = QFormLayout()
        form.setSpacing(6)
        form.addRow("Umbral", threshold_row)

        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setRange(0, 5)
        self.max_gap_spin.setValue(DEFAULT_MAX_GAP)
        form.addRow("Gap max.", self.max_gap_spin)

        self.min_song_spin = QSpinBox()
        self.min_song_spin.setRange(1, 10)
        self.min_song_spin.setValue(DEFAULT_MIN_SONG_WIN)
        form.addRow("Min. ventanas", self.min_song_spin)

        layout.addLayout(form)
        return box

    def _build_actions_box(self) -> QFrame:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)

        self.analyze_btn = QPushButton("Analizar")
        self.analyze_btn.setObjectName("analyzeBtn")
        self.analyze_btn.clicked.connect(self._emit_analyze)

        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.setObjectName("cancelBtn")
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)

        self.export_btn = QPushButton("Exportar CSV")
        self.export_btn.setObjectName("exportBtn")
        self.export_btn.clicked.connect(self.export_requested.emit)

        self.import_btn = QPushButton("Importar resultados")
        self.import_btn.setObjectName("importBtn")
        self.import_btn.clicked.connect(self.import_requested.emit)

        layout.addWidget(self.analyze_btn)
        secondary_row = QHBoxLayout()
        secondary_row.setSpacing(8)
        secondary_row.addWidget(self.cancel_btn)
        secondary_row.addWidget(self.export_btn)
        layout.addLayout(secondary_row)
        layout.addWidget(self.import_btn)
        return frame

    def _build_progress_box(self) -> QGroupBox:
        box = QGroupBox("Progreso")
        layout = QVBoxLayout(box)
        layout.setSpacing(5)
        self.file_label = QLabel("Sin analisis")
        self.file_label.setObjectName("mutedLabel")
        self.file_label.setWordWrap(True)
        self.file_progress = QProgressBar()
        self.window_progress = QProgressBar()
        self.status_label = QLabel("Listo")
        self.status_label.setObjectName("mutedLabel")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.file_label)
        layout.addWidget(QLabel("Archivos"))
        layout.addWidget(self.file_progress)
        layout.addWidget(QLabel("Ventanas"))
        layout.addWidget(self.window_progress)
        layout.addWidget(self.status_label)
        return box

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de audios")
        if not folder:
            return
        self._selected_folder = folder
        self.folder_edit.setText(folder)

    def _emit_analyze(self) -> None:
        folder = self.folder_edit.text().strip()
        if not folder or not pathlib.Path(folder).exists():
            QMessageBox.warning(self, "Carpeta no valida", "Selecciona una carpeta de audios.")
            return
        self.analyze_requested.emit(folder, self.get_params())

    def _sync_threshold_slider(self, value: float) -> None:
        slider_value = round(float(value) * THRESHOLD_SCALE)
        if self.threshold_slider.value() != slider_value:
            self.threshold_slider.setValue(slider_value)

    def _sync_threshold_spin(self, value: int) -> None:
        spin_value = float(value) / THRESHOLD_SCALE
        if abs(self.threshold_spin.value() - spin_value) > 1e-7:
            self.threshold_spin.setValue(spin_value)

    def get_params(self) -> dict:
        return {
            "recursive": self.recursive_check.isChecked(),
            "threshold": round(float(self.threshold_spin.value()), 6),
            "max_gap": self.max_gap_spin.value(),
            "min_song_win": self.min_song_spin.value(),
            "csv_sep": DEFAULT_CSV_SEP,
        }

    def selected_folder(self) -> str:
        return self.folder_edit.text().strip()

    def set_analyzing(self, analyzing: bool) -> None:
        self.analyze_btn.setEnabled(not analyzing)
        self.cancel_btn.setEnabled(analyzing)
        self.export_btn.setEnabled(not analyzing)
        self.import_btn.setEnabled(not analyzing)
        self.recursive_check.setEnabled(not analyzing)
        self.threshold_spin.setEnabled(not analyzing)
        self.threshold_slider.setEnabled(not analyzing)
        self.max_gap_spin.setEnabled(not analyzing)
        self.min_song_spin.setEnabled(not analyzing)
        if analyzing:
            self.file_progress.setValue(0)
            self.window_progress.setValue(0)
            self.set_status("Analizando...")

    def update_file_progress(self, path: str, index: int, total: int) -> None:
        self.file_progress.setMaximum(max(total, 1))
        self.file_progress.setValue(index)
        name = pathlib.Path(path).name
        self.file_label.setText(f"{index}/{total}: {name}")
        self.window_progress.setValue(0)

    def update_window_progress(self, current: int, total: int) -> None:
        self.window_progress.setMaximum(max(total, 1))
        self.window_progress.setValue(current)

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)
