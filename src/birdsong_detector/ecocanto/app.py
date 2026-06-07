"""EcoCanto desktop application entry point."""

from __future__ import annotations

import pathlib
import sys

from .config import APP_ICON_PATH, APP_NAME, APP_VERSION, MODEL_DIR

WINDOWS_APP_ID = "BirdSongDetector.EcoCanto"


def _set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except Exception:
        pass


def main() -> None:
    _set_windows_app_id()

    # Import TensorFlow before Qt on Windows to avoid DLL loading conflicts.
    from .engine import BirdSongEngine, verify_model

    try:
        verify_model(MODEL_DIR)
        engine = BirdSongEngine()
    except Exception as exc:
        detail = str(exc)
        print(detail, file=sys.stderr)
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox

            app = QApplication.instance() or QApplication(sys.argv)
            app.setApplicationName(APP_NAME)
            app.setApplicationVersion(APP_VERSION)
            app.setOrganizationName("TFG")
            app.setDesktopFileName(WINDOWS_APP_ID)
            if APP_ICON_PATH.exists():
                from PyQt6.QtGui import QIcon

                app.setWindowIcon(QIcon(str(APP_ICON_PATH.resolve())))
            short = (
                "No se pudo cargar el modelo final.\n\n"
                "Revisa que el entorno tenga TensorFlow/Keras compatible. "
                "Con este proyecto usa `pip install -r requirements.txt` actualizado.\n\n"
                f"Detalle inicial:\n{detail[:1200]}"
            )
            if len(detail) > 1200:
                short += "\n\n[Mensaje recortado: el detalle completo aparece en la consola.]"
            QMessageBox.critical(None, "Error de modelo", short)
        except Exception:
            pass
        sys.exit(1)

    from PyQt6.QtWidgets import QApplication, QStyleFactory

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("TFG")
    app.setDesktopFileName(WINDOWS_APP_ID)
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    if APP_ICON_PATH.exists():
        from PyQt6.QtGui import QIcon

        app.setWindowIcon(QIcon(str(APP_ICON_PATH.resolve())))

    qss = pathlib.Path(__file__).parent / "ui" / "styles.qss"
    if qss.exists():
        app.setStyleSheet(qss.read_text(encoding="utf-8"))

    try:
        from .ui.mainwindow import MainWindow
    except Exception as exc:
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.critical(
            None,
            "Error de interfaz",
            "No se pudo cargar el modelo final.\n\n"
            f"Detalle:\n{str(exc)[:1200]}",
        )
        sys.exit(1)

    window = MainWindow(engine)
    if APP_ICON_PATH.exists():
        window.setWindowIcon(app.windowIcon())
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
