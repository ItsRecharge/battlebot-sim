"""Live metric-vs-time charts (PySide6.QtCharts — ships with PySide6, no new dep).

``MetricStreamer`` derives a ``MetricSample`` from each streamed chunk on the
worker thread (no Qt-widget work there); ``LiveCharts`` is a small dashboard of
rolling strip-charts updated on the UI thread as the battery runs. QtCharts (unlike
the VTK viewport) constructs fine under offscreen Qt, so this is headless-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

# Points kept per series (a rolling window so long runs stay responsive).
WINDOW = 600


@dataclass
class MetricSample:
    """One time-point of live run metrics (plain data, crosses a Qt signal)."""

    t: float            # battery time (s)
    peak_force: float   # peak contact normal force this window (N)
    cum_energy: float   # cumulative absorbed impact energy (J)
    max_margin: float   # worst failure margin so far (1.0 = yields)
    speed: float        # bot speed (m/s)
    hit_rate: float     # contacts per second this window


class MetricStreamer:
    """Turns streamed chunks into MetricSamples. Lives on the worker thread.

    Peak force and hit rate are accumulated over the window between emitted
    samples (call :meth:`flush` after each emit) so a brief spike between renders
    is never missed; cumulative energy and worst margin come straight from the
    damage accumulator; speed is finite-differenced from consecutive frame poses.
    """

    def __init__(self, bot):
        self.bot = bot
        self._last_pos = None
        self._last_t = None
        self._win_peak = 0.0
        self._win_hits = 0
        self._win_t0 = None

    def update(self, chunk, accum) -> MetricSample:
        t = float(chunk.t_global)
        if self._win_t0 is None:
            self._win_t0 = t
        for c in chunk.new_contacts:
            f = abs(c.normal_force)
            if f > self._win_peak:
                self._win_peak = f
        self._win_hits += len(chunk.new_contacts)

        pos = np.asarray(chunk.frame.pos, dtype=float)
        if self._last_pos is not None and self._last_t is not None and t > self._last_t:
            speed = float(np.linalg.norm(pos - self._last_pos) / (t - self._last_t))
        else:
            speed = 0.0
        self._last_pos, self._last_t = pos, t

        win_dt = max(t - self._win_t0, 1e-6)
        return MetricSample(
            t=t,
            peak_force=self._win_peak,
            cum_energy=float(accum.energy.sum()),
            max_margin=accum.current_max_margin(),
            speed=speed,
            hit_rate=self._win_hits / win_dt,
        )

    def flush(self) -> None:
        """Reset the per-window accumulators (call right after emitting)."""
        self._win_peak = 0.0
        self._win_hits = 0
        self._win_t0 = None


class _Strip:
    """A single rolling line-chart, optionally with a second (right-axis) series
    and a horizontal threshold marker."""

    def __init__(self, title, color, y_floor_max=None, threshold=None,
                 color2=None, name1=None, name2=None):
        self.chart = QChart()
        self.chart.setTitle(title)
        self.chart.setMargins(QtCore.QMargins(4, 2, 4, 2))
        self.chart.legend().setVisible(color2 is not None)
        self.chart.legend().setAlignment(QtCore.Qt.AlignmentFlag.AlignBottom)

        self.ax = QValueAxis()
        self.ax.setLabelFormat("%.0f")
        self.ay = QValueAxis()
        self.ay.setLabelFormat("%.3g")
        self.chart.addAxis(self.ax, QtCore.Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(self.ay, QtCore.Qt.AlignmentFlag.AlignLeft)

        self.series = QLineSeries()
        if name1:
            self.series.setName(name1)
        pen = QtGui.QPen(QtGui.QColor(color))
        pen.setWidth(2)
        self.series.setPen(pen)
        self.chart.addSeries(self.series)
        self.series.attachAxis(self.ax)
        self.series.attachAxis(self.ay)

        self.y_floor_max = y_floor_max
        self.xs: list[float] = []
        self.ys: list[float] = []

        self.threshold = threshold
        if threshold is not None:
            self.thr = QLineSeries()
            tp = QtGui.QPen(QtGui.QColor("#cf222e"))
            tp.setStyle(QtCore.Qt.PenStyle.DashLine)
            self.thr.setPen(tp)
            self.chart.addSeries(self.thr)
            self.thr.attachAxis(self.ax)
            self.thr.attachAxis(self.ay)

        self.series2 = None
        if color2 is not None:
            self.series2 = QLineSeries()
            if name2:
                self.series2.setName(name2)
            pen2 = QtGui.QPen(QtGui.QColor(color2))
            pen2.setWidth(2)
            self.series2.setPen(pen2)
            self.chart.addSeries(self.series2)
            self.ay2 = QValueAxis()
            self.ay2.setLabelFormat("%.2g")
            self.chart.addAxis(self.ay2, QtCore.Qt.AlignmentFlag.AlignRight)
            self.series2.attachAxis(self.ax)
            self.series2.attachAxis(self.ay2)
            self.ys2: list[float] = []

        self.view = QChartView(self.chart)
        self.view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.view.setMinimumHeight(120)

    def reset(self) -> None:
        self.series.clear()
        self.xs.clear()
        self.ys.clear()
        if self.threshold is not None:
            self.thr.clear()
        if self.series2 is not None:
            self.series2.clear()
            self.ys2.clear()

    @staticmethod
    def _trim(series, values):
        if series.count() > WINDOW:
            series.removePoints(0, series.count() - WINDOW)
        if len(values) > WINDOW:
            del values[:len(values) - WINDOW]

    def append(self, t, y, y2=None) -> None:
        t = float(t)
        self.series.append(t, float(y))
        self.xs.append(t)
        self.ys.append(float(y))
        self._trim(self.series, self.ys)
        self.xs = self.xs[-WINDOW:]

        x0, x1 = self.xs[0], self.xs[-1]
        if x1 <= x0:
            x1 = x0 + 1e-3
        self.ax.setRange(x0, x1)

        ymax = max(self.ys)
        if self.y_floor_max is not None:
            ymax = max(ymax, self.y_floor_max)
        self.ay.setRange(0.0, (ymax if ymax > 0 else 1.0) * 1.1)

        if self.threshold is not None:
            self.thr.clear()
            self.thr.append(x0, self.threshold)
            self.thr.append(x1, self.threshold)

        if self.series2 is not None and y2 is not None:
            self.series2.append(t, float(y2))
            self.ys2.append(float(y2))
            self._trim(self.series2, self.ys2)
            ymax2 = max(self.ys2) if self.ys2 else 1.0
            self.ay2.setRange(0.0, (ymax2 if ymax2 > 0 else 1.0) * 1.1)


class LiveCharts(QtWidgets.QWidget):
    """A 2x2 dashboard of live metric-vs-time strip charts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.force = _Strip("Peak contact force (N)", "#fb8500")
        self.energy = _Strip("Cumulative impact energy (J)", "#1f77b4")
        # Yield line at margin = 1.0 (ties to the heatmap's red->=-yield reading).
        self.margin = _Strip("Max failure margin", "#8338ec",
                             y_floor_max=1.1, threshold=1.0)
        self.motion = _Strip("Bot speed & hit rate", "#2a9d8f",
                            color2="#e76f51", name1="speed (m/s)",
                            name2="hits/s")
        self._strips = [self.force, self.energy, self.margin, self.motion]

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setSpacing(4)
        grid.addWidget(self.force.view, 0, 0)
        grid.addWidget(self.energy.view, 0, 1)
        grid.addWidget(self.margin.view, 1, 0)
        grid.addWidget(self.motion.view, 1, 1)

    def reset(self) -> None:
        """Clear every series (call at the start of a run)."""
        for s in self._strips:
            s.reset()

    def append(self, m: MetricSample) -> None:
        """Add one live sample to every chart."""
        self.force.append(m.t, m.peak_force)
        self.energy.append(m.t, m.cum_energy)
        self.margin.append(m.t, m.max_margin)
        self.motion.append(m.t, m.speed, m.hit_rate)
