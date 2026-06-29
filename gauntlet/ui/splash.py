"""Startup splash with a real loading bar, shown while the heavy native
libraries (NumPy/SciPy/VTK/MuJoCo + the matplotlib font cache) initialise.

PyInstaller's built-in ``Splash()`` is unsupported on macOS, so this is a plain
Qt widget instead — it renders identically on Windows and Mac and lets us drive
an actual progress bar from the entry point as each startup phase completes.

Used only by the GUI path in :mod:`gauntlet.__main__`; the headless ``--selftest``
path never constructs it.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class StartupSplash(QtWidgets.QWidget):
    """Frameless, centered splash: app icon, title, status line, progress bar."""

    def __init__(self, icon: QtGui.QIcon) -> None:
        super().__init__(
            None,
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.SplashScreen,
        )
        self.setFixedSize(420, 180)
        self.setStyleSheet("background:#1b1f24; border:1px solid #30363d; border-radius:8px;")

        icon_label = QtWidgets.QLabel()
        icon_label.setPixmap(icon.pixmap(48, 48))
        icon_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        title = QtWidgets.QLabel("Combat-Robot-Gauntlet")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color:#f0f6fc; font-size:18px; font-weight:600; border:none;")

        self._status = QtWidgets.QLabel("Starting…")
        self._status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color:#8b949e; font-size:12px; border:none;")

        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        self._bar.setStyleSheet(
            "QProgressBar{background:#30363d; border:none; border-radius:4px;}"
            "QProgressBar::chunk{background:#2f81f7; border-radius:4px;}"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(12)
        layout.addWidget(icon_label)
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(self._status)
        layout.addWidget(self._bar)

        self._center_on_screen()

    def _center_on_screen(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.move(geo.center().x() - self.width() // 2,
                      geo.center().y() - self.height() // 2)

    def step(self, pct: int, text: str) -> None:
        """Set the bar to ``pct`` and the status to ``text``, then repaint now.

        Called synchronously from the startup sequence, so it pumps the event
        loop to force an immediate paint before the next (blocking) phase runs.
        """
        self._bar.setValue(max(0, min(100, pct)))
        self._status.setText(text)
        QtWidgets.QApplication.processEvents()

    def finish(self, window: QtWidgets.QWidget) -> None:
        """Close the splash once ``window`` is up."""
        self.close()
