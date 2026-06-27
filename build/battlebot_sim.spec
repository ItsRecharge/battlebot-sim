# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the BattleBot Damage Simulator.

Build (from the project root):
    .venv\\Scripts\\pyinstaller build\\battlebot_sim.spec --noconfirm

Produces dist/BattleBotSim/BattleBotSim.exe (one-folder build). One-folder is
deliberate: VTK is large and a one-file build unpacks hundreds of MB to a temp
dir on every launch. Zip the dist/BattleBotSim folder to distribute.

NOTE: we deliberately do NOT collect_all() vtkmodules or pyvista. Those force
every VTK submodule to be imported at analysis time, and importing the OpenGL
rendering modules without a live graphics context segfaults the builder. The
bundled per-module VTK hooks (in _pyinstaller_hooks_contrib) bundle VTK
correctly from the normal import graph instead.
"""

import os
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
)

PROJECT_ROOT = os.path.abspath(os.getcwd())

datas, binaries, hiddenimports = [], [], []

# MuJoCo: collect its native library + data files WITHOUT importing its
# submodules. collect_all() would import every mujoco submodule (incl. the GL
# backends) into the analysis process alongside VTK/Qt/scipy native libs, and
# that mix of native DLLs triggers an access-violation crash during analysis.
# collect_dynamic_libs/collect_data_files only walk the filesystem.
binaries += collect_dynamic_libs("mujoco")
datas += collect_data_files("mujoco")
hiddenimports += ["mujoco"]

# pyvista / pyvistaqt: bundle their data files only. VTK binaries are handled
# by the built-in per-module hooks via the import graph.
datas += collect_data_files("pyvista")
datas += collect_data_files("pyvistaqt")

# Our bundled data.
datas += [
    (os.path.join(PROJECT_ROOT, "data", "materials.json"), "data"),
    (os.path.join(PROJECT_ROOT, "data", "sample_bots"), os.path.join("data", "sample_bots")),
]

hiddenimports += [
    "pyvistaqt",
    "PySide6.QtCharts",     # live metric-vs-time graphs (see ui/charts.py)
    "vtkmodules.qt.QVTKRenderWindowInteractor",
    "vtkmodules.util.numpy_support",
    # scipy 1.17 imports this Cython helper at startup but PyInstaller doesn't
    # auto-detect it; without it scipy reports itself "broken". Added as a plain
    # string (NOT collect_submodules, which would import scipy at spec-eval time
    # and dramatically raise the intermittent native-crash rate during analysis).
    "scipy._cyutility",
    "scipy.spatial.transform",
    "scipy.spatial._ckdtree",
]

a = Analysis(
    [os.path.join(PROJECT_ROOT, "battlebot_sim", "__main__.py")],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6", "pytest", "_pytest", "IPython",
              "trame", "notebook", "jupyter",
              # We only use PySide6 QtCore/QtGui/QtWidgets. Excluding the rest of
              # the Qt stack shrinks the bundle and reduces the number of native
              # DLLs loaded into PyInstaller's analysis process (which lowers the
              # rate of an intermittent native access-violation during analysis).
              "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuick3D",
              "PySide6.QtQuickWidgets", "PySide6.QtNetwork", "PySide6.QtMultimedia",
              "PySide6.QtMultimediaWidgets", "PySide6.QtWebEngineCore",
              "PySide6.QtWebEngineWidgets", "PySide6.QtWebChannel",
              "PySide6.QtWebSockets", "PySide6.Qt3DCore", "PySide6.Qt3DRender",
              # NOTE: QtCharts is NOT excluded — the live metric graphs use it.
              "PySide6.QtDataVisualization", "PySide6.QtSql",
              "PySide6.QtTest", "PySide6.QtBluetooth", "PySide6.QtPositioning",
              "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtPdf",
              "PySide6.QtPdfWidgets", "PySide6.QtDesigner", "PySide6.QtHelp"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BattleBotSim",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="BattleBotSim",
)
