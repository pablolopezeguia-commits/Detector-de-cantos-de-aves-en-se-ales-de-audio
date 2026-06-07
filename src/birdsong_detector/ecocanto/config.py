"""Central EcoCanto configuration."""

from __future__ import annotations

import pathlib
import sys


def get_base_dir() -> pathlib.Path:
    if getattr(sys, "frozen", False):
        return pathlib.Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return pathlib.Path(__file__).parent


BASE_DIR = get_base_dir()
MODEL_DIR = BASE_DIR / "model"
ASSETS_DIR = BASE_DIR / "assets"
APP_LOGO_PATH = ASSETS_DIR / "logo.png"
APP_ICON_PATH = ASSETS_DIR / "icon.ico"

# Analysis defaults.
DEFAULT_WINDOW_S = 3.0
DEFAULT_OVERLAP = 0.5
DEFAULT_THRESHOLD = 0.689144
DEFAULT_MAX_GAP = 1
DEFAULT_MIN_SONG_WIN = 2
DEFAULT_RECURSIVE = True
DEFAULT_CSV_SEP = ";"

# Audio preprocessing contract used by training and inference.
PREPROCESS_TARGET_RMS = 0.1
PREPROCESS_HEADROOM = 0.95
PREPROCESS_NORMALIZATION_MODE = "peak"
PREPROCESS_FRAME_SEC = 0.02
PREPROCESS_HOP_SEC = 0.01
PREPROCESS_EPS = 1e-8

# App metadata.
APP_NAME = "EcoCanto"
APP_VERSION = "1.0.0"
WINDOW_MARGIN_S = 0.5

# EcoCanto color palette.
COLOR_PRIMARY_DARK = "#0F5F53"
COLOR_PRIMARY = "#2F9D87"
COLOR_ACCENT = "#43BFA2"
COLOR_MINT = "#92D7C4"
COLOR_BG = "#0B0F12"
COLOR_PANEL = "#10161A"
COLOR_PANEL_ALT = "#151C21"
COLOR_BORDER = "#2A343B"
COLOR_TEXT = "#E8ECEB"
COLOR_MUTED = "#9EA9A6"
COLOR_WARNING = "#D3A34B"
COLOR_DANGER = "#E06C75"

# Visualization colors.
COLOR_SONG_WINDOW = COLOR_ACCENT
COLOR_GAP_WINDOW = COLOR_WARNING
COLOR_SEGMENT_START = COLOR_MINT
COLOR_SEGMENT_END = "#D7A94C"
COLOR_CURSOR = "#FF4D5E"

# Audio support.
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
SILENCE_RMS_THRESHOLD = 1e-7
