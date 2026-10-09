"""CueIt graphical interface (PySide6).

Layout:
  * left:   batch job queue (files or whole folders), worker count, run button
  * center: waveform with beat grid, section shading and draggable cue markers
  * right:  analysis readout, cue table, export actions

Analysis runs in a background multiprocessing pool hosted on a QThread so the
UI never blocks and every core can be used.
"""

from __future__ import annotations

import os
import sys
import threading

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPointF, QRectF, QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPalette, QPolygonF
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QFileDialog, QGraphicsItem,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from .batch import AUDIO_EXTS, default_workers, discover, run_batch, Job
from .libraries import LibrarySource, detect_libraries, load_tracks
from . import session as session_mod
from .cue import Cue
from .exporters import (
    TrackData, export_csv, export_json, export_mixxx, export_rekordbox,
    export_serato, export_traktor,
)

pg.setConfigOptions(antialias=True)


# --------------------------------------------------------------------------- #
# themes
# --------------------------------------------------------------------------- #

THEMES = {
    "dark": {
        "window": "#16171d",
        "base": "#20222b",
        "text": "#e6e6ea",
        "plot_bg": "#121319",
        "wave_fill": "#1aa3c4",
        "wave_edge": "#0c5f79",
        "playhead": "#ffffff",
        "downbeat": "#3a3d49",
        "grid": 0.06,
        # fill = translucent area tint, line = bright boundary/section marker
        "sections": {
            "drop":      {"fill": (245, 158, 11, 60),  "line": (250, 190, 40, 230)},
            "breakdown": {"fill": (168, 139, 250, 60), "line": (190, 160, 255, 230)},
            "intro":     {"fill": (99, 102, 241, 70),  "line": (140, 145, 255, 220)},
            "outro":     {"fill": (20, 184, 166, 60),  "line": (60, 220, 200, 220)},
            "verse":     {"fill": (100, 116, 139, 45), "line": (150, 165, 185, 160)},
            "section":   {"fill": (90, 96, 110, 40),   "line": (130, 136, 150, 120)},
        },
    },
    "light": {
        "window": "#eef0f4",
        "base": "#ffffff",
        "text": "#14161c",
        "plot_bg": "#ffffff",
        "wave_fill": "#1d4ed8",
        "wave_edge": "#1e3a8a",
        "playhead": "#111111",
        "downbeat": "#b8bcc6",
        "grid": 0.10,
        "sections": {
            "drop":      {"fill": (245, 158, 11, 80),  "line": (217, 119, 6, 230)},
            "breakdown": {"fill": (168, 139, 250, 80), "line": (124, 58, 237, 220)},
            "intro":     {"fill": (99, 102, 241, 80),  "line": (67, 56, 202, 200)},
            "outro":     {"fill": (20, 184, 166, 80),  "line": (13, 148, 136, 200)},
            "verse":     {"fill": (148, 163, 184, 60), "line": (100, 116, 139, 170)},
            "section":   {"fill": (148, 163, 184, 50), "line": (100, 116, 139, 140)},
        },
    },
}



# --------------------------------------------------------------------------- #
# background batch runner
# --------------------------------------------------------------------------- #

class SessionSaver(QThread):
    """Debounced off-thread session writer.

    The GUI thread only serializes each finished track (O(1) per track) and
    hands the already-encoded dicts here; JSON encoding of the whole (possibly
    thousands-of-tracks) session happens off the UI thread, so dragging cue
    selectors during a long batch stays smooth.
    """

    def __init__(self, path: str | None = None, parent=None):
        super().__init__(parent)
        self._path = path
        self._lock = threading.Lock()
        self._pending: dict | None = None
        self._wake = threading.Event()
        self._stop = False

    def request(self, tracks_parts: dict):
        with self._lock:
            self._pending = tracks_parts
        self._wake.set()

    def run(self):
        while not self._stop:
            self._wake.wait(timeout=2.0)
            self._wake.clear()
            with self._lock:
                pending = self._pending
                self._pending = None
            if pending is None:
                continue
            try:
                session_mod.save_parts(pending, self._path)
            except Exception:  # noqa: BLE001
                pass

    def stop(self):
        self._stop = True
        self._wake.set()


class LibraryLoader(QThread):
    """Read a DJ library off the GUI thread so the window never freezes."""

    progress = Signal(str)
    found = Signal(int)
    loaded = Signal(list)
    failed = Signal(str)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.source = source

    def run(self):
        try:
            self.progress.emit(f"reading {self.source.name}…")
            tracks = load_tracks(self.source)
            self.found.emit(len(tracks))
            self.loaded.emit(tracks)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class BatchRunner(QThread):
    job_started = Signal(int)
    job_done = Signal(int, object)
    failed = Signal(str)
    finished_batch = Signal(bool)

    def __init__(self, paths: list[str], workers: int, use_downbeats: bool = True, parent=None):
        super().__init__(parent)
        self.paths = paths
        self.workers = workers
        self.use_downbeats = use_downbeats
        self._index = {p: i for i, p in enumerate(paths)}
        self._stop = threading.Event()
        self._pool = None
        self._lock = threading.Lock()

    def cancel(self):
        """Terminate immediately: sets the flag and kills the pool right away."""
        self._stop.set()
        with self._lock:
            pool = self._pool
        if pool is not None:
            try:
                pool.terminate()
            except Exception:  # noqa: BLE001
                pass

    def _on_pool(self, pool):
        with self._lock:
            self._pool = pool

    def run(self):
        try:
            result = run_batch(
                self.paths, workers=self.workers, use_downbeats=self.use_downbeats,
                on_start=lambda job: self.job_started.emit(self._index[job.path]),
                on_done=lambda job: self.job_done.emit(self._index[job.path], job),
                should_stop=self._stop.is_set,
                on_pool=self._on_pool,
            )
            self.finished_batch.emit(result.cancelled)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


# --------------------------------------------------------------------------- #
# waveform
# --------------------------------------------------------------------------- #

class WaveformView(pg.PlotWidget):
    seekRequested = Signal(float)
    cueMoved = Signal(int, float)

    def __init__(self):
        # Plain state must exist before Qt calls resizeEvent during super().__init__().
        self._duration = 1.0
        self._theme = THEMES["dark"]
        self._track: TrackData | None = None
        self._downbeats: list[float] = []
        self._cue_lines: list[WaveformView._CueMarker] = []
        super().__init__()
        self.setMouseEnabled(x=False, y=False)
        self.hideAxis("left")
        self.setLabel("bottom", "time", units="s")
        self._playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ffffff", width=2))
        self.addItem(self._playhead)
        self.apply_theme(self._theme)

    def apply_theme(self, theme: dict, rerender: bool = True):
        self._theme = theme
        self.setBackground(theme["plot_bg"])
        if self._playhead is not None:
            self._playhead.setPen(pg.mkPen(theme["playhead"], width=2))
        self._update_grid()
        if rerender and self._track is not None:
            self.plot_track(self._track, rerender=False)

    def _update_grid(self):
        """Only show the vertical beat grid when it will not look cluttered."""
        item = self.getPlotItem() if hasattr(self, "getPlotItem") else None
        if item is None:
            return
        if not self._downbeats:
            self.showGrid(x=True, y=False, alpha=self._theme["grid"])
            return
        vb = item.vb
        visible = [d for d in self._downbeats if 0 <= d <= self._duration]
        width_px = max(1.0, float(vb.width()))
        px_per_bar = width_px / max(1e-6, (visible[-1] - visible[0])) if len(visible) > 1 else 0.0
        if 4 <= len(visible) <= 400 and px_per_bar >= 11.0:
            self.showGrid(x=True, y=False, alpha=self._theme["grid"])
        else:
            self.showGrid(x=False, y=False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_grid()

    def set_duration(self, duration: float):
        self._duration = max(duration, 1.0)
        self.setXRange(0, self._duration, padding=0)

    def plot_track(self, track: TrackData, rerender: bool = True):
        self._track = track
        self.clear()
        self._cue_lines = []
        self._playhead = pg.InfiniteLine(angle=90, movable=False,
                                         pen=pg.mkPen(self._theme["playhead"], width=2))
        a = track.analysis
        self._downbeats = list(a.downbeats)
        self.set_duration(a.duration)

        # Section overlays sit behind the waveform: a translucent fill for the
        # energy/colour area plus a bright 1px boundary line so boundaries stay
        # legible even where the opaque waveform covers the fill.
        sec = self._theme["sections"]
        for s in a.sections:
            spec = sec.get(s.kind, sec["section"])
            region = pg.LinearRegionItem(
                values=(s.start, s.end), movable=False,
                brush=pg.mkBrush(*spec["fill"]), pen=pg.mkPen(None),
            )
            region.setZValue(-20)
            self.addItem(region)
        for s in a.sections:
            spec = sec.get(s.kind, sec["section"])
            line = pg.InfiniteLine(
                pos=s.start, angle=90, movable=False,
                pen=pg.mkPen(QColor(*spec["line"]), width=1.0),
            )
            line.setZValue(6)  # above the waveform so section colours stay readable
            self.addItem(line)

        mins = np.asarray(a.peak_mins)
        maxs = np.asarray(a.peak_maxs)
        if mins.size >= 2:
            norm = max(float(np.abs(mins).max()), float(np.abs(maxs).max()), 1e-6)
            mins = mins / norm
            maxs = maxs / norm
            x = np.linspace(0.0, a.duration, mins.size)
            lower = pg.PlotCurveItem(x, mins, pen=pg.mkPen(self._theme["wave_edge"], width=1))
            upper = pg.PlotCurveItem(x, maxs, pen=pg.mkPen(self._theme["wave_edge"], width=1))
            fill = pg.FillBetweenItem(lower, upper,
                                      brush=pg.mkBrush(QColor(self._theme["wave_fill"])))
            self.addItem(fill)
            self.addItem(lower)
            self.addItem(upper)
            self.setYRange(-1.28, 1.28, padding=0)
        else:
            self.plot(np.asarray(a.rms_times), np.asarray(a.rms),
                      pen=pg.mkPen(self._theme["wave_fill"], width=1))

        self.addItem(self._playhead)
        for cue in track.cues:
            self._add_cue_line(cue)
        self._update_grid()

    class _CueMarker(pg.GraphicsObject):
        """A thin, precise cue line with fat draggable arrow tabs at both ends."""

        def __init__(self, time: float, color: str, on_moved):
            super().__init__()
            self._time = float(time)
            self._color = color
            self._on_moved = on_moved
            self._tab_w = 15.0
            self._tab_h = 13.0
            self._hover = False
            self.setAcceptHoverEvents(True)
            self.setZValue(30)
            self.setFlag(pg.GraphicsObject.ItemIsMovable, False)
            # Let Qt repaint a little around the marker (antialiased edges).
            self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption, True)

        def set_cue(self, cue: Cue):
            self.prepareGeometryChange()
            self._color = cue.color
            self._time = cue.time
            self.update()

        def value(self) -> float:
            return self._time

        def set_time(self, t: float):
            if abs(t - self._time) < 1e-9:
                return
            self.prepareGeometryChange()
            self._time = float(t)
            self.update()

        def _metrics(self):
            vb = self.getViewBox()
            if vb is None:
                return 0.0, 0.0, 0.0, 0.0
            tl = vb.mapSceneToView(QPointF(0, 0))
            br = vb.mapSceneToView(QPointF(0, vb.height()))
            top, bottom = max(tl.y(), br.y()), min(tl.y(), br.y())
            span = top - bottom
            px = span / max(1.0, float(vb.height()))     # view units per pixel (vertical)
            ux = (vb.viewRange()[0][1] - vb.viewRange()[0][0]) / max(1.0, float(vb.width()))
            return top, bottom, span, px, ux

        def _bounding(self) -> QRectF:
            if self.getViewBox() is None:
                return QRectF(self._time - 1.0, -2.0, 2.0, 4.0)
            top, bottom, span, px, ux = self._metrics()
            pad_x = (self._tab_w + 8.0) * ux
            pad_y = (self._tab_h + 8.0) * px
            # The whole plot area, plus a margin so old pixels are always repainted.
            return QRectF(self._time - pad_x, bottom - pad_y, 2 * pad_x, span + 2 * pad_y)

        def boundingRect(self) -> QRectF:
            return self._bounding()

        def paint(self, painter, option, widget=None):
            if self.getViewBox() is None:
                return
            top, bottom, span, px, ux = self._metrics()

            # Full plot height (so the thin line never leaves a short stale stub).
            region = QRectF(self._time - 4 * ux, bottom - 3 * px, 8 * ux, span + 6 * px)
            painter.fillRect(region, Qt.transparent)

            # Thin, precise full-height line (1px).
            painter.setPen(pg.mkPen(self._color, width=1.0))
            painter.drawLine(QPointF(self._time, bottom), QPointF(self._time, top))

            tab_w = self._tab_w * ux
            tab_h = self._tab_h * px
            painter.setBrush(pg.mkBrush("#ffffff" if self._hover else self._color))
            painter.setPen(pg.mkPen("#000000", width=1.0))
            top_pts = [QPointF(self._time - tab_w / 2, top),
                       QPointF(self._time + tab_w / 2, top),
                       QPointF(self._time, top - tab_h)]
            bottom_pts = [QPointF(self._time - tab_w / 2, bottom),
                          QPointF(self._time + tab_w / 2, bottom),
                          QPointF(self._time, bottom + tab_h)]
            painter.drawPolygon(QPolygonF(top_pts))
            painter.drawPolygon(QPolygonF(bottom_pts))

        def hoverEnterEvent(self, event):
            self._hover = True
            self.setCursor(Qt.SizeHorCursor)
            self.update()
            super().hoverEnterEvent(event)

        def hoverLeaveEvent(self, event):
            self._hover = False
            self.unsetCursor()
            self.update()
            super().hoverLeaveEvent(event)

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self.setCursor(Qt.SizeHorCursor)
                event.accept()
            else:
                super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if event.buttons() & Qt.LeftButton:
                p = self.mapToScene(event.pos())
                new_t = self.getViewBox().mapSceneToView(p).x()
                if abs(new_t - self._time) > 1e-9:
                    # Must tell Qt the bounding rect is about to move, or the old
                    # region is left behind as a ghost trail while dragging.
                    self.prepareGeometryChange()
                    self._time = new_t
                    self._on_moved(self)
                    self.update()
                event.accept()
            else:
                super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            self.unsetCursor()
            self._on_moved(self)
            event.accept()

    def _add_cue_line(self, cue: Cue, index: int = None):
        if index is None:
            index = len(self._cue_lines)
        marker = self._CueMarker(
            cue.time, cue.color,
            on_moved=lambda m, i=index: self.cueMoved.emit(i, m.value()),
        )
        self.addItem(marker)
        self._cue_lines.append(marker)

    def update_cue(self, index: int, cue: Cue):
        if 0 <= index < len(self._cue_lines):
            self._cue_lines[index].set_cue(cue)

    def set_playhead(self, t: float):
        self._playhead.setValue(t)

    def zoom(self, factor: float):
        """Zoom the time axis by *factor* (<1 zooms in) around the view centre."""
        (x0, x1), _ = self.getPlotItem().vb.viewRange()
        span = (x1 - x0) * factor
        span = max(self._duration / 500.0, min(span, self._duration))
        center = (x0 + x1) / 2.0
        lo = max(0.0, center - span / 2.0)
        hi = min(self._duration, lo + span)
        lo = max(0.0, hi - span)
        self.setXRange(lo, hi, padding=0)
        self._update_grid()

    def zoom_to(self, t: float, span: float = 8.0):
        """Zoom in on a specific time (used when jumping to a cue)."""
        span = max(self._duration / 500.0, min(span, self._duration))
        lo = max(0.0, t - span / 2.0)
        hi = min(self._duration, lo + span)
        lo = max(0.0, hi - span)
        self.setXRange(lo, hi, padding=0)
        self._update_grid()

    def zoom_reset(self):
        self.setXRange(0, self._duration, padding=0)
        self._update_grid()

    def center_on(self, t: float):
        """Scroll so *t* is centred, keeping the current zoom level."""
        (x0, x1), _ = self.getPlotItem().vb.viewRange()
        half = (x1 - x0) / 2.0
        lo = max(0.0, t - half)
        hi = min(self._duration, lo + 2 * half)
        lo = max(0.0, hi - 2 * half)
        self.setXRange(lo, hi, padding=0)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.getPlotItem().vb.mapSceneToView(event.position())
            self.seekRequested.emit(max(0.0, min(pos.x(), self._duration)))
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        # Mouse wheel zooms the time axis around the cursor (Shift = fine).
        delta = event.angleDelta().y()
        if delta:
            pos = self.getPlotItem().vb.mapSceneToView(event.position())
            self.zoom_around(pos.x(), 0.8 if delta > 0 else 1.25)
            event.accept()
            return
        super().wheelEvent(event)

    def zoom_around(self, t: float, factor: float):
        (x0, x1), _ = self.getPlotItem().vb.viewRange()
        span = max(self._duration / 500.0, min((x1 - x0) * factor, self._duration))
        frac = (t - x0) / max(1e-6, (x1 - x0))
        lo = max(0.0, t - frac * span)
        hi = min(self._duration, lo + span)
        lo = max(0.0, hi - span)
        self.setXRange(lo, hi, padding=0)
        self._update_grid()


# --------------------------------------------------------------------------- #
# main window
# --------------------------------------------------------------------------- #

class MainWindow(QMainWindow):
    def __init__(self, initial: list[str] | None = None):
        super().__init__()
        self.setWindowTitle("CueIt — automatic cue points")
        self.resize(1340, 780)
        self.setAcceptDrops(True)

        self.track: TrackData | None = None
        self.queue_paths: list[str] = []
        self._queue_set: set[str] = set()
        self.results: dict[str, TrackData] = {}
        self.results_parts: dict[str, dict] = {}
        self._runner: BatchRunner | None = None
        self._saver = SessionSaver()
        self._saver.start()
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(1500)
        self._save_timer.timeout.connect(self._flush_session)

        self.player = QMediaPlayer()
        self.audio_out = QAudioOutput()
        self.player.setAudioOutput(self.audio_out)
        self.audio_out.setVolume(0.8)
        self.player.positionChanged.connect(self._on_position)

        self._build_ui()
        self.set_theme(self._settings().value("theme", "dark"))
        self._apply_shortcut_hints()
        QApplication.instance().installEventFilter(self)
        self._load_session()

        if initial:
            self._add_paths(initial)

    # ---- ui ----
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("Batch queue"))
        self.queue = QListWidget()
        self.queue.setSelectionMode(QListWidget.ExtendedSelection)
        self.queue.itemActivated.connect(self._load_selected)
        self.queue.itemClicked.connect(self._load_selected)
        lv.addWidget(self.queue)
        row = QHBoxLayout()
        add_files_btn = QPushButton("Add files…")
        add_files_btn.clicked.connect(self.add_files)
        add_folder_btn = QPushButton("Add folder…")
        add_folder_btn.clicked.connect(self.add_folder)
        row.addWidget(add_files_btn)
        row.addWidget(add_folder_btn)
        lv.addLayout(row)
        load_lib_btn = QPushButton("Load DJ library…")
        load_lib_btn.clicked.connect(self.load_library)
        lv.addWidget(load_lib_btn)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Workers"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, os.cpu_count() or 4)
        self.workers_spin.setValue(default_workers())
        row2.addWidget(self.workers_spin)
        self.run_btn = QPushButton("Run batch (all)")
        self.run_btn.clicked.connect(self.run_batch)
        row2.addWidget(self.run_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_batch)
        row2.addWidget(self.cancel_btn)
        lv.addLayout(row2)
        row3 = QHBoxLayout()
        self.run_sel_btn = QPushButton("Run selected")
        self.run_sel_btn.clicked.connect(self.run_selected)
        row3.addWidget(self.run_sel_btn)
        self.clear_btn = QPushButton("Clear queue")
        self.clear_btn.clicked.connect(self.clear_queue)
        row3.addWidget(self.clear_btn)
        lv.addLayout(row3)
        splitter.addWidget(left)

        center = QWidget()
        cv = QVBoxLayout(center)
        self.waveform = WaveformView()
        self.waveform.seekRequested.connect(self._seek)
        self.waveform.cueMoved.connect(self._on_cue_moved)
        cv.addWidget(self.waveform, stretch=1)
        controls = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)
        self.time_label = QLabel("00:00.0 / 00:00.0")
        controls.addWidget(self.time_label)
        controls.addStretch(1)
        self.add_cue_btn = QPushButton("+ Cue at playhead")
        self.add_cue_btn.clicked.connect(self._add_cue_at_playhead)
        controls.addWidget(self.add_cue_btn)
        cv.addLayout(controls)
        splitter.addWidget(center)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.info = QLabel("Add files or a folder, then Run batch.")
        self.info.setWordWrap(True)
        rv.addWidget(self.info)
        self.cue_table = QTableWidget(0, 2)
        self.cue_table.setHorizontalHeaderLabels(["Cue", "Time"])
        self.cue_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cue_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cue_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.cue_table.cellChanged.connect(self._on_cell_changed)
        rv.addWidget(self.cue_table)
        self.del_btn = QPushButton("Delete selected cue")
        self.del_btn.clicked.connect(self._delete_selected)
        rv.addWidget(self.del_btn)
        self.redetect_btn = QPushButton("Re-detect cues")
        self.redetect_btn.clicked.connect(self._redetect_cues)
        rv.addWidget(self.redetect_btn)
        rv.addWidget(QLabel("Export (all analyzed tracks)"))
        for label, fn in [
            ("Rekordbox XML", self._export_rekordbox),
            ("Traktor NML", self._export_traktor),
            ("Mixxx library", self._export_mixxx),
            ("Serato tags", self._export_serato),
            ("JSON", lambda: self._export_one("json")),
            ("CSV", lambda: self._export_one("csv")),
        ]:
            b = QPushButton(label)
            b.clicked.connect(fn)
            rv.addWidget(b)
        splitter.addWidget(right)

        splitter.setSizes([300, 700, 320])
        self.setCentralWidget(splitter)

        self.progress = QProgressBar()
        self.statusBar().addPermanentWidget(self.progress, 1)

        file_menu = self.menuBar().addMenu("&File")
        for label, slot, shortcut in [
            ("Add &files…", self.add_files, QKeySequence.Open),
            ("Add f&older…", self.add_folder, "Ctrl+Shift+O"),
            ("Load DJ &library…", self.load_library, "Ctrl+Shift+L"),
        ]:
            act = QAction(label, self)
            act.setShortcut(shortcut)
            act.triggered.connect(slot)
            file_menu.addAction(act)
        file_menu.addSeparator()
        clear_session_act = QAction("Clear saved session", self)
        clear_session_act.triggered.connect(self._clear_session)
        file_menu.addAction(clear_session_act)

        view_menu = self.menuBar().addMenu("&View")
        self._theme_actions = {}
        for name in THEMES:
            act = QAction(f"{name.capitalize()} mode", self, checkable=True)
            act.setShortcut("Ctrl+D" if name == "dark" else "Ctrl+L")
            act.triggered.connect(lambda _=False, n=name: self.set_theme(n))
            view_menu.addAction(act)
            self._theme_actions[name] = act
        view_menu.addSeparator()
        self._add_action(view_menu, "Zoom in", "Ctrl+=", lambda: self._zoom(0.8))
        self._add_action(view_menu, "Zoom out", "Ctrl+-", lambda: self._zoom(1.25))
        self._add_action(view_menu, "Zoom to fit", "Ctrl+0", lambda: self.waveform.zoom_reset())

        batch_menu = self.menuBar().addMenu("&Batch")
        self._add_action(batch_menu, "Run batch (all pending)", "Ctrl+R", self.run_batch)
        self._add_action(batch_menu, "Run selected", "Ctrl+Shift+Return", self.run_selected)
        self._add_action(batch_menu, "Cancel batch", "Esc", self.cancel_batch)
        self._add_action(batch_menu, "Re-detect cues", "Ctrl+Shift+R", self._redetect_cues)

        play_menu = self.menuBar().addMenu("&Playback")
        self._add_action(play_menu, "Play / Pause", "Space", self._toggle_play)
        self._add_action(play_menu, "Jump back 10s", "Left", lambda: self._nudge(-10))
        self._add_action(play_menu, "Jump forward 10s", "Right", lambda: self._nudge(10))
        self._add_action(play_menu, "Previous track", "Ctrl+Up", lambda: self._select_delta(-1))
        self._add_action(play_menu, "Next track", "Ctrl+Down", lambda: self._select_delta(1))
        self._add_action(play_menu, "Previous cue", "Ctrl+Left", lambda: self._goto_cue(-1))
        self._add_action(play_menu, "Next cue", "Ctrl+Right", lambda: self._goto_cue(1))

        cues_menu = self.menuBar().addMenu("&Cues")
        self._add_action(cues_menu, "Add cue at playhead", "C", self._add_cue_at_playhead)
        self._add_action(cues_menu, "Delete selected cue", "Delete", self._delete_selected)
        self._add_action(cues_menu, "Save session", "Ctrl+S", self._save_now)

        help_menu = self.menuBar().addMenu("&Help")
        self._add_action(help_menu, "Keyboard shortcuts", "F1", self._show_shortcuts)

    def _add_action(self, menu, text: str, shortcut, slot, bind: bool = True):
        act = QAction(text if bind else f"{text}\t{shortcut}", self)
        if shortcut and bind:
            act.setShortcut(shortcut)
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    def eventFilter(self, obj, event):
        """Global, focus-aware keyboard handling.

        Arrow/Space/Delete are only intercepted when the user is not editing text
        (cue table cell, spin box, line edit), so inline editing keeps working.
        """
        if event.type() == QEvent.KeyPress and self.isActiveWindow() and not self._text_input_active():
            key = event.key()
            mods = event.modifiers()
            ctrl = bool(mods & Qt.ControlModifier)
            handled = True
            if key == Qt.Key_Space:
                self._toggle_play()
            elif key == Qt.Key_Left and ctrl:
                self._goto_cue(-1)
            elif key == Qt.Key_Right and ctrl:
                self._goto_cue(1)
            elif key == Qt.Key_Left:
                self._nudge(-10)
            elif key == Qt.Key_Right:
                self._nudge(10)
            elif key == Qt.Key_Up:
                self._select_delta(-1)
            elif key == Qt.Key_Down:
                self._select_delta(1)
            elif key == Qt.Key_Home:
                self._seek(0.0)
            elif key == Qt.Key_End and self.track:
                self._seek(self.track.analysis.duration)
            elif key in (Qt.Key_Delete, Qt.Key_Backspace):
                self._delete_selected()
            elif key == Qt.Key_C and not ctrl:
                self._add_cue_at_playhead()
            else:
                handled = False
            if handled:
                return True
        return super().eventFilter(obj, event)

    def _text_input_active(self) -> bool:
        fw = QApplication.focusWidget()
        if fw is None:
            return False
        if isinstance(fw, (QLineEdit, QAbstractSpinBox)):
            return True
        if isinstance(fw, QTableWidget) and fw.state() == QTableWidget.EditingState:
            return True
        return False

    def _apply_shortcut_hints(self):
        """Buttons must not steal Space/arrow keys from the global handler."""
        for b in self.findChildren(QPushButton):
            b.setFocusPolicy(Qt.NoFocus)

    # ---- queue ----
    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add audio files", os.path.expanduser("~"),
            "Audio (*.wav *.aiff *.aif *.mp3 *.flac *.m4a *.aac *.ogg *.opus)",
        )
        self._add_paths(paths)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add folder", os.path.expanduser("~"))
        if folder:
            self._add_paths([folder])

    def load_library(self):
        libs = detect_libraries()
        source = self._pick_library(libs)
        if source is None:
            return
        # Load off the GUI thread so a huge library never looks like a freeze.
        self._lib_loader = LibraryLoader(source, self)
        self._lib_loader.progress.connect(self.statusBar().showMessage)
        self._lib_loader.found.connect(
            lambda n: self.statusBar().showMessage(f"{source.name}: {n} tracks found — adding to queue…"))
        self._lib_loader.loaded.connect(self._on_library_loaded)
        self._lib_loader.failed.connect(
            lambda m: QMessageBox.warning(self, "Library error", m))
        self._lib_loader.start()
        self.progress.setRange(0, 0)  # indeterminate "busy" bar

    def _on_library_loaded(self, tracks: list[str]):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        if not tracks:
            QMessageBox.information(self, "Library empty", "No playable tracks found in this library.")
            return
        self._add_paths(tracks)

    def _pick_library(self, libs: list[LibrarySource]) -> LibrarySource | None:
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Load DJ library")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Detected libraries:" if libs else "No libraries auto-detected. Browse for a file:"))
        combo = QComboBox()
        for lib in libs:
            combo.addItem(lib.label(), lib)
        if not libs:
            combo.setEnabled(False)
        v.addWidget(combo)
        browse = QPushButton("Browse…")
        chosen: dict = {}

        def do_browse():
            path, _ = QFileDialog.getOpenFileName(
                dlg, "Select library file", os.path.expanduser("~"),
                "Libraries (*.sqlite *.xml *.nml *.db)",
            )
            if not path:
                return
            if path.endswith(".sqlite") or path.endswith(".db"):
                kind, name = "mixxx", "Mixxx library"
            elif path.endswith(".nml"):
                kind, name = "traktor", "Traktor collection"
            else:
                kind, name = "rekordbox", "Rekordbox XML"
            chosen["src"] = LibrarySource(name, kind, path)
            combo.addItem(chosen["src"].label(), chosen["src"])
            combo.setCurrentIndex(combo.count() - 1)

        browse.clicked.connect(do_browse)
        v.addWidget(browse)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)
        if dlg.exec() != QDialog.Accepted:
            return None
        return chosen.get("src") or (combo.currentData() if combo.isEnabled() else None)

    def _add_paths(self, paths: list[str]):
        files = discover(paths)
        # Set membership keeps this O(n) instead of O(n^2) on huge libraries.
        new = [p for p in files if p not in self._queue_set]
        if not new:
            self.statusBar().showMessage("no new audio files found")
            return
        self._queue_set.update(new)
        self.queue_paths.extend(new)
        # Disable sorting/per-item overhead while adding, and show progress for
        # very large libraries so the app doesn't look frozen.
        big = len(new) > 400
        if big:
            self.progress.setRange(0, len(new))
        self.queue.setUpdatesEnabled(False)
        try:
            for i, p in enumerate(new):
                item = QListWidgetItem(f"{os.path.basename(p)}   · queued")
                item.setData(Qt.UserRole, p)
                self.queue.addItem(item)
                if big and i % 250 == 0:
                    self.progress.setValue(i)
                    QApplication.processEvents()
        finally:
            self.queue.setUpdatesEnabled(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        if self.queue.count() and self.queue.currentRow() < 0:
            self.queue.setCurrentRow(0)
        self.statusBar().showMessage(
            f"loaded {len(new)} track(s) — select rows and press Run selected, "
            f"or Run batch (all) to analyze everything")

    def _clear_session(self):
        if self._runner and self._runner.isRunning():
            self.cancel_batch()
        session_mod.clear()
        self.queue.clear()
        self.queue_paths.clear()
        self._queue_set.clear()
        self.results.clear()
        self.progress.setValue(0)
        self.track = None
        self.waveform.clear()
        self.cue_table.setRowCount(0)
        self.info.setText("Add files or a folder, then Run batch.")
        self.statusBar().showMessage("cleared saved session")

    def clear_queue(self):
        if self._runner and self._runner.isRunning():
            return
        self.queue.clear()
        self.queue_paths.clear()
        self._queue_set.clear()
        self.results.clear()
        self.progress.setValue(0)
        self._save_session()

    def _set_row(self, index: int, text: str):
        item = self.queue.item(index)
        if item is not None:
            item.setText(text)

    def _load_selected(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)
        track = self.results.get(path)
        if track is not None:
            self.load_track(track)
        else:
            self.statusBar().showMessage(f"{os.path.basename(path)}: not analyzed yet — run the batch")

    def _load_session(self):
        """Restore previously analyzed tracks so a long run can resume."""
        try:
            saved = session_mod.load()
        except Exception:  # noqa: BLE001
            saved = {}
        if not saved:
            return
        self.results.update(saved)
        for p, t in saved.items():
            self.queue_paths.append(p)
            self._queue_set.add(p)
            a = t.analysis
            item = QListWidgetItem(
                f"{os.path.basename(p)}   · {a.bpm:.1f} BPM  {a.key} {a.scale}  ({len(t.cues)} cues)"
            )
            item.setData(Qt.UserRole, p)
            self.queue.addItem(item)
        self.statusBar().showMessage(f"resumed {len(saved)} analyzed track(s) from previous session")

    # ---- batch ----
    def run_batch(self):
        """Analyze every not-yet-analyzed track in the queue."""
        if self._runner and self._runner.isRunning():
            return
        pending = [p for p in self.queue_paths if p not in self.results]
        if not pending:
            self.statusBar().showMessage("nothing to analyze — everything is already done")
            return
        self._start_batch(pending)

    def run_selected(self):
        """Analyze only the selected queue rows (skips ones already done)."""
        if self._runner and self._runner.isRunning():
            return
        rows = sorted({i.row() for i in self.queue.selectedIndexes()})
        selected = [self.queue.item(r).data(Qt.UserRole) for r in rows]
        selected = [p for p in selected if p and p not in self.results]
        if not selected:
            self.statusBar().showMessage("no unanalyzed tracks selected — select rows first")
            return
        # Analyze in queue order.
        selected = [p for p in self.queue_paths if p in set(selected)]
        self._start_batch(selected)

    def _start_batch(self, pending: list[str]):
        self.progress.setRange(0, len(pending))
        self.progress.setValue(0)
        self._completed = 0
        self._runner = BatchRunner(pending, self.workers_spin.value())
        self._runner.job_started.connect(self._on_job_started)
        self._runner.job_done.connect(self._on_job_done)
        self._runner.finished_batch.connect(self._on_batch_finished)
        self._runner.failed.connect(lambda m: QMessageBox.critical(self, "Batch failed", m))
        self._set_running(True)
        self.statusBar().showMessage(f"analyzing {len(pending)} track(s)…")
        self._runner.start()

    def cancel_batch(self):
        if self._runner and self._runner.isRunning():
            self.statusBar().showMessage("cancelling…")
            self._runner.cancel()

    def _set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)

    def _on_batch_finished(self, cancelled: bool):
        self._set_running(False)
        self._save_session()
        if cancelled:
            done = sum(1 for p in self.queue_paths if p in self.results)
            self.statusBar().showMessage(
                f"cancelled — {done}/{len(self.queue_paths)} analyzed and saved; press Run batch to resume")

    def _touch(self, track: TrackData):
        """Record a track's latest state (cheap, per-track) and debounce the flush."""
        try:
            self.results_parts[track.path] = session_mod.track_to_dict(track)
        except Exception:  # noqa: BLE001
            return
        self._save_timer.start()

    def _save_session(self):
        """Serialize any not-yet-recorded results, then debounce a flush."""
        for p, t in self.results.items():
            if p not in self.results_parts:
                try:
                    self.results_parts[p] = session_mod.track_to_dict(t)
                except Exception:  # noqa: BLE001
                    pass
        self._save_timer.start()

    def _flush_session(self):
        """Hand the pre-serialized parts to the background writer."""
        try:
            self._saver.request(dict(self.results_parts))
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"could not save session: {exc}")

    def _on_job_started(self, index: int):
        p = self._index_path(index)
        self._set_row(self.queue.row(self._item_for(p)), f"{os.path.basename(p)}   · analyzing…")
        self.statusBar().showMessage(f"analyzing {os.path.basename(p)}")

    def _on_job_done(self, index: int, job: Job):
        p = job.path
        item = self._item_for(p)
        if job.track is not None:
            self.results[p] = job.track
            self._touch(job.track)  # per-track serialize + debounced save
            a = job.track.analysis
            self._set_row(self.queue.row(item),
                          f"{os.path.basename(p)}   · {a.bpm:.1f} BPM  {a.key} {a.scale}  ({len(job.track.cues)} cues)")
            if self.track is None:
                self.load_track(job.track)
        elif job.status == "cancelled":
            self._set_row(self.queue.row(item), f"{os.path.basename(p)}   · cancelled")
        else:
            self._set_row(self.queue.row(item), f"{os.path.basename(p)}   · FAILED: {job.error}")
        self._completed += 1
        self.progress.setValue(self._completed)

    def _index_path(self, index: int) -> str:
        return self._runner.paths[index] if self._runner else ""

    def _item_for(self, path: str) -> QListWidgetItem:
        for i in range(self.queue.count()):
            if self.queue.item(i).data(Qt.UserRole) == path:
                return self.queue.item(i)
        raise KeyError(path)

    # ---- track ----
    def load_track(self, track: TrackData):
        self.track = track
        a = track.analysis
        self.info.setText(
            f"<b>{track.title}</b><br>"
            f"{a.bpm:.1f} BPM &nbsp;·&nbsp; {a.key} {a.scale} ({a.camelot or '—'})<br>"
            f"duration {a.duration:.1f}s · {len(a.sections)} sections · {len(a.downbeats)} bars"
            f"<br>grid: {a.downbeat_phase} phase · confidence {a.downbeat_confidence:.2f}"
            f" · {a.phrase_length}-bar phrases"
        )
        self.waveform.plot_track(track)
        self._refresh_cue_table()
        self.player.setSource(QUrl.fromLocalFile(track.path))
        self.statusBar().showMessage(f"loaded {os.path.basename(track.path)}")

    def _refresh_cue_table(self):
        self.cue_table.blockSignals(True)
        self.cue_table.setRowCount(0)
        if self.track:
            for cue in self.track.cues:
                r = self.cue_table.rowCount()
                self.cue_table.insertRow(r)
                self.cue_table.setItem(r, 0, QTableWidgetItem(cue.name))
                it = QTableWidgetItem(f"{cue.time:.3f}")
                it.setForeground(QColor(cue.color))
                self.cue_table.setItem(r, 1, it)
        self.cue_table.blockSignals(False)

    # ---- cues ----
    def _redetect_cues(self):
        if not self.track:
            return
        from .cue import infer_cues
        self.track.cues = infer_cues(self.track.analysis).cues
        self.results[self.track.path] = self.track
        self.waveform.plot_track(self.track)
        self._refresh_cue_table()
        self._touch(self.track)
        self.statusBar().showMessage(f"re-detected {len(self.track.cues)} cue(s)")

    def _add_cue_at_playhead(self):
        if not self.track:
            return
        t = self.player.position() / 1000.0
        cue = Cue(f"Cue {len(self.track.cues) + 1}", round(t, 3), "#00BCD4")
        self.track.cues.append(cue)
        self.track.cues.sort(key=lambda c: c.time)
        self.waveform.plot_track(self.track)
        self._refresh_cue_table()
        self._touch(self.track)

    def _on_cue_moved(self, index: int, time: float):
        if not self.track or index >= len(self.track.cues):
            return
        # Update a single cell only — a full table rebuild on every mouse-move is
        # what made dragging feel laggy, especially while audio was playing.
        cue = self.track.cues[index]
        cue.time = round(time, 3)
        self._update_cue_row(index, cue)
        self._touch(self.track)

    def _update_cue_row(self, row: int, cue: Cue):
        if row < 0 or row >= self.cue_table.rowCount():
            return
        self.cue_table.blockSignals(True)
        self.cue_table.item(row, 0).setText(cue.name)
        self.cue_table.item(row, 1).setText(f"{cue.time:.3f}")
        self.cue_table.blockSignals(False)

    def _on_cell_changed(self, row: int, col: int):
        if not self.track or row >= len(self.track.cues):
            return
        cue = self.track.cues[row]
        if col == 0:
            cue.name = self.cue_table.item(row, 0).text()
        else:
            try:
                cue.time = float(self.cue_table.item(row, 1).text())
            except ValueError:
                return
            self.waveform.update_cue(row, cue)
        self._touch(self.track)

    def _delete_selected(self):
        if not self.track:
            return
        rows = sorted({idx.row() for idx in self.cue_table.selectedIndexes()}, reverse=True)
        if not rows:
            row = self.cue_table.currentRow()
            rows = [row] if row >= 0 else []
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self.track.cues):
                del self.track.cues[row]
        self.waveform.plot_track(self.track)
        self._refresh_cue_table()
        self._touch(self.track)

    # ---- playback ----
    def _toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.play_btn.setText("Play")
        else:
            self.player.play()
            self.play_btn.setText("Pause")

    def _seek(self, t: float):
        if self.track:
            self.player.setPosition(int(t * 1000))

    def _nudge(self, seconds: float):
        if not self.track:
            return
        t = self.player.position() / 1000.0 + seconds
        t = max(0.0, min(t, self.track.analysis.duration))
        self.player.setPosition(int(t * 1000))
        self.waveform.center_on(t)

    def _zoom(self, factor: float):
        self.waveform.zoom(factor)

    def _select_delta(self, delta: int):
        if self.queue.count() == 0:
            return
        row = self.queue.currentRow()
        row = 0 if row < 0 else row
        row = max(0, min(self.queue.count() - 1, row + delta))
        self.queue.setCurrentRow(row)
        self.queue.scrollToItem(self.queue.item(row))
        self._load_selected(self.queue.item(row))

    def _goto_cue(self, delta: int):
        if not self.track or not self.track.cues:
            return
        cues = sorted(self.track.cues, key=lambda c: c.time)
        t = self.player.position() / 1000.0
        if delta > 0:
            nxt = next((c for c in cues if c.time > t + 0.05), cues[-1])
        else:
            nxt = next((c for c in reversed(cues) if c.time < t - 0.05), cues[0])
        self.player.setPosition(int(nxt.time * 1000))
        self.waveform.center_on(nxt.time)

    def _save_now(self):
        self._save_session()
        self._flush_session()
        self.statusBar().showMessage("session saved")

    def _show_shortcuts(self):
        QMessageBox.information(
            self, "Keyboard shortcuts",
            "Playback\n"
            "  Space            Play / Pause\n"
            "  ← / →            Jump back / forward 10s\n"
            "  Ctrl+← / →       Previous / next cue\n"
            "  Ctrl+↑ / ↓       Previous / next track\n\n"
            "View\n"
            "  Ctrl+= / -       Zoom in / out\n"
            "  Wheel            Zoom around cursor\n"
            "  Ctrl+0           Zoom to fit\n"
            "  Ctrl+D / Ctrl+L  Dark / light mode\n\n"
            "Cues\n"
            "  C                Add cue at playhead\n"
            "  Delete           Delete selected cue\n"
            "  Ctrl+Shift+R     Re-detect cues\n\n"
            "Batch\n"
            "  Ctrl+R           Run batch (all pending)\n"
            "  Ctrl+Shift+Enter Run selected rows\n"
            "  Esc              Cancel batch\n"
            "  Ctrl+S           Save session\n"
            "  Ctrl+O           Add files\n"
            "  Ctrl+Shift+O     Add folder\n"
            "  Ctrl+Shift+L     Load DJ library\n"
        )

    def _on_position(self, ms: int):
        t = ms / 1000.0
        self.waveform.set_playhead(t)
        if self.track:
            self.time_label.setText(f"{_fmt(t)} / {_fmt(self.track.analysis.duration)}")

    # ---- drag & drop ----
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self._add_paths(paths)

    # ---- export ----
    def _ask_dir(self) -> str | None:
        return QFileDialog.getExistingDirectory(self, "Export to folder", os.path.expanduser("~")) or None

    def _all_tracks(self) -> list[TrackData]:
        return list(self.results.values())

    def _export_one(self, kind: str):
        tracks = self._all_tracks()
        if not tracks:
            return
        d = self._ask_dir()
        if not d:
            return
        fn = export_json if kind == "json" else export_csv
        for t in tracks:
            base = os.path.splitext(os.path.basename(t.path))[0]
            fn(t, os.path.join(d, f"{base}.cueit.{kind}"))
        self.statusBar().showMessage(f"wrote {len(tracks)} {kind} file(s) to {d}")

    def _export_rekordbox(self):
        tracks = self._all_tracks()
        if not tracks:
            return
        d = self._ask_dir()
        if d:
            out = os.path.join(d, "rekordbox.xml")
            export_rekordbox(tracks, out)
            self.statusBar().showMessage(f"wrote {out}")

    def _export_traktor(self):
        tracks = self._all_tracks()
        if not tracks:
            return
        d = self._ask_dir()
        if d:
            out = os.path.join(d, "collection.nml")
            export_traktor(tracks, out)
            self.statusBar().showMessage(f"wrote {out}")

    def _export_mixxx(self):
        tracks = self._all_tracks()
        if not tracks:
            return
        ok, fail = 0, 0
        for t in tracks:
            try:
                export_mixxx(t)
                ok += 1
            except Exception:  # noqa: BLE001
                fail += 1
        self.statusBar().showMessage(f"Mixxx: {ok} updated, {fail} skipped (backup saved).")

    def _export_serato(self):
        tracks = self._all_tracks()
        if not tracks:
            return
        ok, fail = 0, 0
        for t in tracks:
            try:
                export_serato(t)
                ok += 1
            except Exception:  # noqa: BLE001
                fail += 1
        self.statusBar().showMessage(f"Serato: {ok} tagged, {fail} skipped (MP3/AIFF only).")

    def closeEvent(self, event):
        if self._runner and self._runner.isRunning():
            self._runner.cancel()
            self._runner.wait(5000)
        self._save_session()
        self._flush_session()
        self._saver.stop()
        self._saver.wait(3000)
        super().closeEvent(event)

    # ---- theme ----
    def set_theme(self, name: str):
        theme = THEMES.get(name, THEMES["dark"])
        self._theme_name = name
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(theme["window"]))
        pal.setColor(QPalette.WindowText, QColor(theme["text"]))
        pal.setColor(QPalette.Base, QColor(theme["base"]))
        pal.setColor(QPalette.Text, QColor(theme["text"]))
        pal.setColor(QPalette.Button, QColor(theme["base"]))
        pal.setColor(QPalette.ButtonText, QColor(theme["text"]))
        pal.setColor(QPalette.Highlight, QColor("#3f7fd0"))
        pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        app = QApplication.instance()
        app.setPalette(pal)
        self.menuBar().setPalette(pal)
        self.waveform.apply_theme(theme)
        for n, act in self._theme_actions.items():
            act.setChecked(n == name)
        settings = self._settings()
        settings.setValue("theme", name)
        self.statusBar().showMessage(f"{name.capitalize()} mode")

    def _settings(self):
        from PySide6.QtCore import QSettings
        return QSettings("CueIt", "CueIt")


def _fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


def app_icon() -> QIcon:
    """Load the CueIt window/taskbar icon (falls back to a drawn glyph)."""
    from . import asset_path
    for name in ("cueit.svg", "cueit.png", "cueit_512.png", "cueit.ico"):
        path = asset_path(name)
        if os.path.exists(path):
            icon = QIcon(path)
            if not icon.isNull():
                return icon
    # Last-resort: draw a simple pin so the window is never icon-less.
    from PySide6.QtGui import QPixmap, QPainter, QBrush
    pm = QPixmap(64, 64)
    pm.fill(QColor("#0d1424"))
    p = QPainter(pm)
    p.setBrush(QBrush(QColor("#fbbf24")))
    p.setPen(Qt.NoPen)
    p.drawEllipse(20, 14, 24, 24)
    p.end()
    return QIcon(pm)


def main(initial: list[str] | None = None):
    app = QApplication(sys.argv)
    app.setApplicationName("CueIt")
    app.setApplicationDisplayName("CueIt")
    app.setOrganizationName("CueIt")
    icon = app_icon()
    app.setWindowIcon(icon)
    win = MainWindow(initial)
    win.setWindowIcon(icon)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
