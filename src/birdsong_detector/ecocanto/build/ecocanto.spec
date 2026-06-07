# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_root = Path("..").resolve()
icon_path = project_root / "assets" / "icon.ico"

a = Analysis(
    [str(project_root / "app.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / "model" / "yamnet_final_mlp.keras"), "model"),
        (str(project_root / "model" / "yamnet_base"), "model/yamnet_base"),
        (str(project_root / "model" / "metadata.json"), "model"),
        (str(project_root / "ui" / "styles.qss"), "ui"),
        (str(project_root / "assets" / "logo.png"), "assets"),
        (str(project_root / "assets" / "icon.ico"), "assets"),
    ],
    hiddenimports=[
        "sounddevice",
        "soundfile",
        "librosa",
        "scipy",
        "PyQt6.QtMultimedia",
        "matplotlib.backends.backend_qtagg",
        "tensorflow",
        "tensorflow_hub",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EcoCanto",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path) if icon_path.exists() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="EcoCanto",
)
