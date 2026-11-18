# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Tiny9DTact desktop viewer (onedir build).

Build (from the repo root, on the Windows dev machine):

    python -m PyInstaller --noconfirm --clean Tiny9DTact.spec

Output:

    dist/Tiny9DTact/Tiny9DTact.exe   +   dist/Tiny9DTact/_internal/

The user-editable data folders (shape_reconstruction/shape_config.yaml,
shape_reconstruction/calibration/, web_viewer/index.html) are NOT
bundled -- build_release.bat copies them next to the exe so config and
calibration can be updated without rebuilding.

Notes:
  - console=True keeps the reconnect/capture log visible (useful on a
    lab machine). Flip to False in EXE(...) for a windowless build.
  - upx=False: UPX-compressed DLLs trigger more antivirus heuristics.
  - heavy upstream deps (torch / open3d / PyQt5 / vispy) are excluded;
    the runtime only needs numpy, opencv, yaml, websockets, pywebview.
"""

a = Analysis(
    ['_3_Shape_Reconstruction.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        # pywebview Windows backend (hooks-contrib usually collects
        # these; listed explicitly so a missing hook fails the build
        # loudly instead of producing an exe that dies at startup).
        'webview.platforms.winforms',
        'webview.platforms.edgechromium',
        'clr',
    ],
    excludes=[
        'torch', 'torchvision', 'open3d', 'vispy', 'pyqtgraph',
        'PyQt5', 'matplotlib', 'tkinter', 'ml_collections', 'IPython',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Tiny9DTact',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='Tiny9DTact',
)
