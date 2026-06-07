"""Tree table for detected result segments."""

from __future__ import annotations

import pathlib

import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QHBoxLayout,
    QLineEdit,
    QSizePolicy,
    QTabBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class SegmentsTable(QWidget):
    segment_selected = pyqtSignal(dict)

    SHEET_ORDER = ("song", "non_song", "runs")
    SHEET_COLUMNS = {
        "song": [
            ("Archivo / clip", "_label"),
            ("Inicio", "t_inicio"),
            ("Fin", "t_fin"),
            ("Duracion", "duracion_s"),
            ("Vent. canto", "n_ventanas_canto"),
            ("Gap", "n_ventanas_gap"),
            ("Score", "score_medio"),
        ],
        "non_song": [
            ("Archivo / tramo", "_label"),
            ("Inicio", "t_inicio"),
            ("Fin", "t_fin"),
            ("Duracion", "duracion_s"),
            ("Ventanas", "n_ventanas_no_canto"),
            ("Conf. no canto", "confianza_no_canto"),
            ("P canto max", "p_canto_max"),
        ],
        "runs": [
            ("Archivo / tramo", "_label"),
            ("Tipo", "tipo"),
            ("Inicio", "t_inicio"),
            ("Fin", "t_fin"),
            ("Duracion", "duracion_s"),
            ("Ventanas", "n_ventanas"),
            ("P canto medio", "p_canto_medio"),
            ("P canto max", "p_canto_max"),
        ],
    }
    ROW_DATA_ROLE = int(Qt.ItemDataRole.UserRole) + 1

    def __init__(self):
        super().__init__()
        self._song_df = pd.DataFrame()
        self._non_song_df = pd.DataFrame()
        self._window_runs_df = pd.DataFrame()
        self._active_sheet = "song"
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        self.sheet_tabs = QTabBar()
        self.sheet_tabs.setObjectName("sheetTabs")
        self.sheet_tabs.addTab("Cantos")
        self.sheet_tabs.addTab("No cantos")
        self.sheet_tabs.addTab("Ventanas")
        self.sheet_tabs.currentChanged.connect(self._switch_sheet)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filtrar archivo")
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.filter_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        top_row.addWidget(self.sheet_tabs, 0)
        top_row.addWidget(self.filter_edit, 1)

        self.tree = QTreeWidget()
        self.tree.setObjectName("resultsTree")
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setAnimated(True)
        self.tree.setIndentation(18)
        self.tree.setSortingEnabled(False)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.itemSelectionChanged.connect(self._emit_selection)
        self._configure_columns()

        layout.addLayout(top_row)
        layout.addWidget(self.tree)

    def load_segments(self, segments_df: pd.DataFrame) -> None:
        self.load_data(segments_df, pd.DataFrame(), pd.DataFrame())

    def load_data(
        self,
        segments_df: pd.DataFrame,
        non_song_df: pd.DataFrame,
        window_runs_df: pd.DataFrame | None = None,
    ) -> None:
        self._song_df = segments_df.copy()
        self._non_song_df = non_song_df.copy()
        self._window_runs_df = pd.DataFrame() if window_runs_df is None else window_runs_df.copy()
        self._update_tab_labels()
        self._apply_filter(self.filter_edit.text())

    def _switch_sheet(self, index: int) -> None:
        self._active_sheet = self.SHEET_ORDER[index]
        self._configure_columns()
        self._apply_filter(self.filter_edit.text())

    def _active_df(self) -> pd.DataFrame:
        if self._active_sheet == "song":
            return self._song_df
        if self._active_sheet == "non_song":
            return self._non_song_df
        return self._window_runs_df

    def _active_columns(self) -> list[tuple[str, str]]:
        return self.SHEET_COLUMNS[self._active_sheet]

    def _configure_columns(self) -> None:
        columns = self._active_columns()
        self.tree.setColumnCount(len(columns))
        self.tree.setHeaderLabels([label for label, _ in columns])
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(columns)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

    def _update_tab_labels(self) -> None:
        self.sheet_tabs.setTabText(0, f"Cantos ({len(self._song_df)})")
        self.sheet_tabs.setTabText(1, f"No cantos ({len(self._non_song_df)})")
        self.sheet_tabs.setTabText(2, f"Ventanas ({len(self._window_runs_df)})")

    def _apply_filter(self, text: str) -> None:
        self.tree.clear()
        df = self._active_df()
        if df.empty or "archivo" not in df.columns:
            return

        needle = text.strip().lower()
        columns = self._active_columns()
        for archivo, group in df.groupby("archivo", sort=False):
            file_name = pathlib.Path(str(archivo)).name
            if needle and needle not in file_name.lower():
                continue
            ordered = group.sort_values("t_inicio").reset_index(drop=True)
            parent = self._build_parent_item(str(archivo), file_name, ordered, columns)
            self.tree.addTopLevelItem(parent)
            for index, row in ordered.iterrows():
                child = self._build_child_item(index, row, columns)
                parent.addChild(child)
            parent.setExpanded(True)
        self.tree.resizeColumnToContents(1)

    def _build_parent_item(
        self,
        archivo: str,
        file_name: str,
        group: pd.DataFrame,
        columns: list[tuple[str, str]],
    ) -> QTreeWidgetItem:
        summary = self._summary_row(archivo, file_name, group)
        item = QTreeWidgetItem([self._display_value(key, summary.get(key, "")) for _, key in columns])
        item.setData(0, self.ROW_DATA_ROLE, summary)
        item.setFlags((item.flags() | Qt.ItemFlag.ItemIsSelectable) & ~Qt.ItemFlag.ItemIsEditable)

        font = QFont()
        font.setBold(True)
        for col in range(len(columns)):
            item.setFont(col, font)
            item.setForeground(col, QColor("#DCE7E4"))
        item.setForeground(0, QColor("#FFFFFF"))
        return item

    def _build_child_item(
        self,
        index: int,
        row: pd.Series,
        columns: list[tuple[str, str]],
    ) -> QTreeWidgetItem:
        row_dict = row.to_dict()
        row_dict["_label"] = self._child_label(index, row_dict)
        item = QTreeWidgetItem([self._display_value(key, row_dict.get(key, "")) for _, key in columns])
        item.setData(0, self.ROW_DATA_ROLE, row_dict)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._style_child_item(item, row_dict, columns)
        return item

    def _summary_row(self, archivo: str, file_name: str, group: pd.DataFrame) -> dict:
        data = {
            "archivo": archivo,
            "_label": f"{file_name}  ({len(group)} tramos)",
            "t_inicio": float(group["t_inicio"].min()),
            "t_fin": float(group["t_fin"].max()),
            "duracion_s": float(group["duracion_s"].sum()) if "duracion_s" in group else 0.0,
            "is_file_summary": True,
        }
        for key in ("n_ventanas_canto", "n_ventanas_gap", "n_ventanas_no_canto", "n_ventanas"):
            if key in group:
                data[key] = int(group[key].sum())
        for key in ("score_medio", "confianza_no_canto", "p_canto_medio", "p_canto_max"):
            if key in group:
                data[key] = float(group[key].mean())
        if self._active_sheet == "runs":
            data["tipo"] = "mixto"
        return data

    def _child_label(self, index: int, row: dict) -> str:
        if self._active_sheet == "song":
            return f"Canto {index + 1:02d}"
        if self._active_sheet == "non_song":
            return f"No canto {index + 1:02d}"
        tipo = "Canto" if str(row.get("tipo", "")) == "canto" else "No canto"
        return f"{tipo} {index + 1:02d}"

    def _style_child_item(
        self,
        item: QTreeWidgetItem,
        row: dict,
        columns: list[tuple[str, str]],
    ) -> None:
        tipo = str(row.get("tipo", ""))
        label_color = QColor("#43BFA2")
        if self._active_sheet == "non_song" or tipo == "no_canto":
            label_color = QColor("#D4A94F")
        item.setForeground(0, label_color)
        for col, (_, key) in enumerate(columns):
            if key in {"score_medio", "p_canto_medio", "p_canto_max"}:
                item.setForeground(col, self._score_color(float(row.get(key, 0.0))))
            elif key == "confianza_no_canto":
                item.setForeground(col, self._confidence_color(float(row.get(key, 0.0))))
            elif key == "tipo":
                item.setForeground(col, label_color)

    def _display_value(self, key: str, value) -> str:
        if key == "_label":
            return str(value)
        if key == "archivo":
            return pathlib.Path(str(value)).name
        if key == "tipo":
            mapping = {"canto": "canto", "no_canto": "no canto", "mixto": "mixto"}
            return mapping.get(str(value), str(value))
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if key in {"score_medio", "confianza_no_canto", "p_canto_medio", "p_canto_min", "p_canto_max"}:
            return f"{number:.4f}"
        if key.startswith("n_"):
            return str(int(number))
        return f"{number:.3f}"

    def _score_color(self, score: float) -> QColor:
        if score >= 0.9:
            return QColor("#45C49B")
        if score >= 0.7:
            return QColor("#D7A94C")
        return QColor("#AEBAB7")

    def _confidence_color(self, score: float) -> QColor:
        if score >= 0.9:
            return QColor("#45C49B")
        if score >= 0.7:
            return QColor("#AEBAB7")
        return QColor("#D7A94C")

    def _emit_selection(self) -> None:
        selected = self.tree.selectedItems()
        if not selected:
            return
        data = selected[0].data(0, self.ROW_DATA_ROLE)
        if data:
            self.segment_selected.emit(dict(data))
